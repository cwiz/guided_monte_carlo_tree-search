import uuid
import numpy as np
import gym
import torch.optim as optim
import torch


MCTS_N_SIMULATIONS = 20
MCTS_ROLLOUT_DEPTH = 20
MCTS_C = 1.0
LR = 3e-4

"""Random Play Tree plays go game randomly. Base class for Monte Carlo Tree."""
class RandomPlayTree:
    
    def __init__(self, board_size):
        self.env = gym.make('gym_go:go-v0', size=board_size, reward_method='real')
        self.root_node = Node(None, self.env.reset(), None, 1.0)
        self.board_size = board_size
        
    """Pick next move"""
    def pick_move(self, node):
        possible_moves = node.possible_moves()
        if len(possible_moves) == 0:
            return None, None
        index = np.random.choice(len(possible_moves))
        return tuple(possible_moves[index]), 1./len(possible_moves)
    
    """Find tree node by UUID"""
    def find_node(self, uuid):
        def _find_node(parent_node):
            if parent_node.uuid == uuid:
                return parent_node
            for node in parent_node.children.values():
                return _find_node(node)
        return _find_node(self.root_node)

    """Make a move"""
    def move(self, node, move, prob):
        # reset environment to node's state. useful in MCTS unroll situation
        self.env.reset(state=node.state)
        state, _, _, _ = self.env.step(move)
        new_node = Node(node, state, move, prob)
        node.add_child(new_node)

        return new_node

    """
    Simulation is MCTS is a sequence of moves that starts in current node and ends in terminal node. 
    During simulation moves are chosen wrt rollout policy function which in usually uniform random.
    
    """
    def simulate(self, node):
        current_node = node
        game_has_ended = False
        
        while not game_has_ended:
            move, prob = self.pick_move(current_node)   

            if move is None: # no possible moves
                game_has_ended = True
                break
       
            current_node = self.move(current_node, move, prob)
        
        return current_node

    """
    Return node score (who's winning):
     1 - black
     0 - draw
    -1 - white
    """
    def evaluate_node(self, node):
        self.env.reset(state=node.state)
        return self.env.get_winning()

"""Random Play Tree plays go game randomly. Base class for Monte Carlo Tree."""
class MonteCarloPlayTree(RandomPlayTree):

    """Pick next move"""
    def pick_move(self, node):

        for _ in range(MCTS_N_SIMULATIONS):
            leaf = self.traverse(node)  
            terminal_node = self.rollout(leaf, 0)
            simulation_result = self.evaluate_node(terminal_node)
            self.backpropagate(leaf, simulation_result)

        if node.is_terminal():
            return None, None

        best_child = node.best_child()
        return best_child.move, best_child.prob

    """Traverse policy in uniform random"""
    def traverse_policy(self, node):
        unexplored_moves = list(node.possible_unexplored_moves())
        index = np.random.choice(len(unexplored_moves))
        move = unexplored_moves[index]
        return move, 1. / len(node.possible_moves())

    """
    Traverse a node. Pick a path prioritizing highest UTC for fully explored nodes 
    and random uniform otherwise.
    """
    def traverse(self, node):
        if node.is_terminal():
            return node

        if node.is_fully_expanded():
            return self.traverse(node.best_uct(self.uct))

        move, prob = self.traverse_policy(node)
        return self.move(node, move, prob)

    """
    A Policy used to pick next best move
    In Non-Neural Monte Carlo Tree we are using random uniform
    """
    def rollout_policy(self, node):
        return self.traverse_policy(node)

    """Rollout a node according to a rollout policy."""
    def rollout(self, node, depth):
        if depth > MCTS_ROLLOUT_DEPTH:
            return node
        if node.is_terminal():
            return node
        move, prob = self.rollout_policy(node)
        return self.rollout(self.move(node, move, prob), depth+1)

    """Backpropagate node's statistics all the way up to root node"""
    def backpropagate(self, node, result):
        node.update_stats(result)
        if node.is_root():
            return
        self.backpropagate(node.parent, result)

    """
    UCT is a core of MCTS. It allows us to choose next node among visited nodes.
    
    Q_v/N_v                               - exploitation component (favors nodes that were winning)
    torch.sqrt(torch.log(N_v_parent)/N_v) - exploration component (favors node that weren't visited)
    c                                     - tradeoff
    
    In competetive games Q is always computed relative to player who moves.

    Parameters
	----------
    player: int
         1 for blacks
        -1 for whites
    c: float
        Constant for exploration/exploitation tradeoff
    """
    def uct(self, node, c=MCTS_C):
        if node.current_player() == 1:
            Q_v = node.q_black
        else:
            Q_v = node.q_white
        N_v = node.number_of_visits + 1
        N_v_parent = node.parent.number_of_visits + 1
        
        return Q_v/N_v + c*np.sqrt(np.log(N_v_parent)/N_v) 


"""MCTS with neural augmentations"""
class GuidedMonteCarloPlayTree(MonteCarloPlayTree):

    def __init__(self, tree_size, actor_critic_network, device):
        super(GuidedMonteCarloPlayTree, self).__init__(tree_size)
        self.actor_critic_network = actor_critic_network
        self.optimizer = optim.Adam(self.actor_critic_network.parameters(), lr=LR)
        self.device = device
    
    def rollout_policy(self, node):
        
        state = node.prepared_game_state()
        state_tensor = torch.from_numpy(state).float().to(self.device).unsqueeze(0).unsqueeze(0)
        prob, _ = self.actor_critic_network(state_tensor)
        mask = node.possible_moves_mask().astype(float)

        # TODO: test

        prob = prob.detach().cpu().numpy()

        table = prob*mask
        probs = table / table.sum()

        options = np.argwhere(table!=np.nan).tolist()
        index = np.random.choice(np.arange(self.board_size**2), p=probs.flatten())
        move = options[index]
        prob = table[move[0], move[1]]

        return move, prob

    def uct(self, node, c=MCTS_C):
        N_v = node.number_of_visits
        N_v_parent = node.parent.number_of_visits

        # TODO: test
        V_current = self.estimate_node_value(node)
        for child in node.children:
            V_current -= self.estimate_node_value(child)
        
        result = V_current/N_v + c * np.sqrt(np.log(N_v_parent)/N_v) * node.prob

        return result

    """Estimate node value with neural network"""
    def estimate_node_value(self, node):
        state = node.prepared_game_state(node.current_player())
        state_tensor = torch.from_numpy(state).float().to(self.device).unsqueeze(0).unsqueeze(0)
        _, v = self.actor_critic_network(state_tensor)
        return v.detach().cpu().numpy().sum()

    """
    Train guided MCTS
    AlphaZero algorithm:
    1. Initialize actor-critic
    2. Simulate a game
    3. Compute loss
    4. Repeat
    """
    def train(self, n_iterations):
        for i in range(n_iterations):
            terminal_node = self.simulate(self.root_node)
            last_player = terminal_node.current_player()
            winning_player = self.evaluate_node(last_player)
            if last_player == winning_player:
                score = 1
            else:
                score = -1

            trajectory = terminal_node.unroll()
            states = np.array([node.prepared_game_state(terminal_node.current_player()) for node in trajectory])
            states_tensor = torch.from_numpy(states).float().to(self.device)
            probs, values = self.actor_critic_network(states_tensor)
            # Loss function. Core of alpha-zero
            loss = (values - score)**2 - (probs * torch.log(probs)).sum()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print("Iteration #", i, " loss:", loss)


"""Game Tree Node"""
class Node:
    
    def __init__(self, parent, state, move, prob):
        self.uuid = uuid.uuid1()
        self.parent = parent
        self.children = []
        # State
        self.state = state
        # MCT node properties
        self.number_of_visits = 0
        self.q_black = 0
        self.q_white = 0
        # Traversal properties
        self.prob = 0
        self.move = move
        
    def add_child(self, node):
        self.children.append(node)

    """
    Prepare game state from perspective of current player
    [
        [ 1 -1  0]
        [ 1  0  0]
        [ 0  0  0]
    ]

    Where  1 is current player
          -1 is opposing player
           0 available moves
    """
    def prepared_game_state(self, player=None):
        if player == None:
            player = self.current_player()
        if self.current_player() == 1:
            state = self.blacks() - self.whites()
        else:
            state = self.whites() - self.blacks()
        
        return state

    """White figures on board"""
    def whites(self):
        return self.state[1]
    
    """Black figures on board"""
    def blacks(self):
        return self.state[0]

    """Return 1 if current player plays black, and -1 for whites"""
    def current_player(self):
        if np.any(self.state[2]):
            return 1
        return -1

    """Moves blocked by go rules"""
    def invalid_moves(self):
        return self.state[3]

    """Return whether previous move was a pass"""
    def previous_pass(self):
        return self.state[4][0]

    """Return whether game has ended"""
    def game_over(self):
        return self.state[5][0]

    """List of possible next moves"""
    def possible_moves(self):
        return np.argwhere(self.possible_moves_mask()).tolist()

    """Return list of possible next moves as int mask"""
    def possible_moves_mask(self):
        whites = self.whites()
        blacks = self.blacks()
        invalid_moves = self.invalid_moves()
        return np.logical_not(np.logical_or(whites, blacks, invalid_moves))

    """How far node from root"""
    def depth(self):
        return len(self.unroll())

    """Whether node is last in the game"""        
    def is_terminal(self):
        return len(self.possible_moves()) == 0

    """Whether node all of node's children were expanded"""
    def is_fully_expanded(self):
        return len(self.possible_unexplored_moves()) == 0

    """Pick child node with highest UCT"""
    def best_uct(self, uct_func):
        return sorted(self.children, key=lambda node: uct_func(node), reverse=True)[0]

    """Pick unvisited child node"""
    def possible_unexplored_moves(self):
        possible_moves_set = set([tuple(m) for m in self.possible_moves()])
        explored_moves_set = set([tuple(m.move) for m in self.children])
        return possible_moves_set - explored_moves_set

    """Return best child nod"""
    def best_child(self):
        return sorted(self.children, key=lambda node: node.number_of_visits, reverse=True)[0]

    def is_root(self):
        return self.parent == None

    """Update node statistics"""
    def update_stats(self, result):
        self.number_of_visits += 1
        if result == 1:
            self.q_black += 1
        if result == -1:
            self.q_white += 1
    
    """Return list of nodes to root"""
    def unroll(self):
        nodes = [self]
        node = self
        while node.parent is not None:
            node = node.parent
            nodes.append(node)

        return nodes