import uuid
import numpy as np
import gym
import torch


MCTS_N_SIMULATIONS = 20
MCTS_ROLLOUT_DEPTH = 20
MCTS_C = 1.0

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
            return None
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
        self.env.reset(node.state)
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
            return None

        return self.node_action(node.best_child())

    """
    Traverse a node. Pick a path prioritizing highest UTC for fully explored nodes 
    and random uniform otherwise.
    """
    def traverse(self, node):
        if node.is_terminal():
            return node

        if node.is_fully_expanded():
            return self.traverse(node.best_uct())
        
        possible_moves = list(node.possible_unexplored_moves())
        index = np.random.choice(len(possible_moves))
        move = possible_moves[index]
        return self.move(node, move)

    """Rollout a node according to a rollout policy."""
    def rollout(self, node, depth):
        if depth > MCTS_ROLLOUT_DEPTH:
            return node
        if node.is_terminal():
            return node
        return self.rollout(self.rollout_policy(node), depth+1)

    """
    A Policy used to pick next best move
    In Non-Neural Monte Carlo Tree we are using random uniform
    """
    def rollout_policy(self, node):
        possible_moves = list(node.possible_unexplored_moves())
        index = np.random.choice(len(possible_moves))
        move = possible_moves[index]
        return self.move(node, move)

    """Backpropagate node's statistics all the way up to root node"""
    def backpropagate(self, node, result):
        node.update_stats(result)
        if node.is_root():
            return
        self.backpropagate(node.parent, result)

    """Get action corresponding to node"""
    def node_action(self, node):
        for (action, child_node) in node.parent.children.items():
            if child_node.uuid == node.uuid:
                return action
        return None

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
            Q_v = self.q_black
        else:
            Q_v = self.q_white
        N_v = node.number_of_visits
        N_v_parent = node.parent.number_of_visits
        
        return Q_v/N_v + c*np.sqrt(np.log(N_v_parent)/N_v) 


"""MCTS with neural augmentations"""
class GuidedMonteCarloPlayTree(MonteCarloPlayTree):

    def __init__(self, tree_size, actor_critic_network, device):
        super(GuidedMonteCarloPlayTree, self).__init__(tree_size)
        self.actor_critic_network = actor_critic_network
        self.device = device
    
    def rollout_policy(self, node):
        
        state = node.prepared_game_state(node.current_player())
        state_tensor = torch.from_numpy(state).float().to(self.device).unsqueeze(0).unsqueeze(0)
        prob, _ = self.actor_critic_network(state_tensor)
        mask = torch.from_numpy(node.possible_moves_mask()).to(self.device)

        # TODO: test
        table = prob[mask].view(-1).detach().cpu().numpy()
        coordinates = np.argwhere(table > 0)
        move = np.random.choice(coordinates, p=table[coordinates])

        return self.move(node, move)

    def uct(self, node, c=MCTS_C):
        N_v = node.number_of_visits
        N_v_parent = node.parent.number_of_visits

        # TODO: test
        V_current = self.estimate_node_value(node)
        for child in node.children.values():
            V_current -= self.estimate_node_value(child)
        
        return V_current/N_v + c * np.sqrt(np.log(N_v_parent)/N_v) * node.probability

    """Estimate node value with neural network"""
    def estimate_node_value(self, node):
        state = node.prepared_game_state(node.current_player())
        state_tensor = torch.from_numpy(state).float().to(self.device).unsqueeze(0).unsqueeze(0)
        _, v = self.actor_critic_network(state_tensor)
        return v.detach().cpu().numpy()

    """Train actor critic network"""
    def train(self):
        # TODO: implement
        pass

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
    def prepared_game_state(self):
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
        depth = 0
        parent = self.parent

        while parent.parent is not None:
            parent = parent.parent
            depth += 1

        return depth

    """Whether node is last in the game"""        
    def is_terminal(self):
        return len(self.possible_moves()) == 0

    """Whether node all of node's children were expanded"""
    def is_fully_expanded(self):
        possible_moves_set = set([tuple(m) for m in self.possible_moves()])
        explored_moves_set = set([tuple(m) for (m, n) in self.children])
        return len(possible_moves_set-explored_moves_set) == 0

    """Pick child node with highest UCT"""
    def best_uct(self):
        children = list(self.children.values())
        return sorted(children, key=lambda node: node.uct(self.current_player()), reverse=True)[0]

    """Pick unvisited child node"""
    def possible_unexplored_moves(self):
        possible_moves_set = set([tuple(m) for m in self.possible_moves()])
        explored_moves_set = set([tuple(m.move) for m in self.children])
        return possible_moves_set - explored_moves_set

    """Return best child nod"""
    def best_child(self):
        children = list(self.children.values())
        return sorted(children, key=lambda node: node.number_of_visits, reverse=True)[0]

    def is_root(self):
        return self.parent == None

    """Update node statistics"""
    def update_stats(self, result):
        self.number_of_visits += 1
        if result == 1:
            self.q_black += 1
        if result == -1:
            self.q_white += 1
