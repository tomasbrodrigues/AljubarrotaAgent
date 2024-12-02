# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='MyTeamOffensive', second='MyTeamDefensive', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def init(self, index, time_for_computing=.1):
        super().init(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}
    
    


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}

class MyTeamBaseAgent(CaptureAgent):
    """
    Base class for MyTeam agents with shared methods.
    """

    def init(self, index, time_for_computing=.1):
        super().init(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

class MyTeamOffensive(MyTeamBaseAgent):
    """
    Offensive strategy for MyTeamAgent.
    """

    def choose_action(self, game_state):
        """
        Choose the best defensive action.
        """
        actions = game_state.get_legal_actions(self.index)
        
        best_action = None
        highest_q_value = -9999       
        if not actions:
            return None
        else:
            for action in actions:
                current_q_value = self.get_offensive_features(game_state,action)
                if current_q_value > highest_q_value:
                    highest_q_value = current_q_value
                    best_action = action
                if current_q_value == highest_q_value:
                    best_action = random.choice([best_action,action])
            
            return best_action




    def get_offensive_features(self,game_state,action):
        #oFeatures = util.Counter()
        successor = self.get_successor(game_state, action)
       
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        food_list = self.get_food(successor).as_list()
        num_food = len(food_list)
        #oFeatures['next_food_score'] = -len(food_list)  # self.getScore(successor)
        capsules_list = self.get_capsules(game_state)
        
        '''
        width = game_state.data.layout.width
        height = game_state.data.layout.height
        middle_width = 0
        
        

        if(game_state.is_on_red_team(self.index)):
            middle_width = width/2 - 1
            
        else:
            middle_width = width/2
            
        #middle_of_board = [(middle_width,y) for y in range(height)]'''
        
        distance_to_food = 0
        distance_to_capsule = 0
        distance_to_base = 0
        stop_penalty = 0
        reverse_penalty = 0
        successor_score = 0
        num_op_ghosts = 0
        op_ghost_dists = 0
        invader_distance = 0

        successor_score = -len(food_list) * 100.0
            
        
        if num_food > 0:  # This should always be True,  but better safe than sorry
            distance_to_food = min([self.get_maze_distance(my_pos, food) for food in food_list]) * -3.0
            #oFeatures['distance_to_food'] = min_distance
        
        if len(capsules_list) > 0:
            distance_to_capsule = min([self.get_maze_distance(my_pos, capsule) for capsule in capsules_list]) * -1.0
            #oFeatures['distance_to_capsule'] = min_distance
        
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        opponents_ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        num_op_ghosts = len(opponents_ghosts)
        if len(opponents_ghosts) > 0 :
            op_ghost_dists = min([self.get_maze_distance(my_pos, a.get_position()) for a in opponents_ghosts])
            if successor.get_agent_state(self.index).is_pacman and op_ghost_dists <= 5:
                op_ghost_dists *=  25
                distance_to_capsule *= 5.0

        scared_ghosts = [agent for agent in self.get_opponents(game_state) 
                     if game_state.get_agent_state(agent).scared_timer > 0]  
    
        if scared_ghosts:
            distance_to_food *= 25.0
            op_ghost_dists *= -0.2
            #oFeatures['distance_to_food_scared'] = min_distance
            
        if len(invaders) > 1:
            invader_distance = min([self.get_maze_distance(my_pos, a.get_position()) for a in invaders]) * -100.0
        elif len(invaders) == 1 and len(invaders) and min([self.get_maze_distance(my_pos, a.get_position()) for a in invaders]) <= 5:
            op_ghost_dists *= -1.0
        
        
        if game_state.get_agent_state(self.index).num_carrying == 1: 
            #dsitance_to_base = min([self.get_maze_distance(my_pos,position)]) * -15.0
            distance_to_base = min([self.get_maze_distance(my_pos,self.start)]) * -120.0
            distance_to_capsule *= 5.0
            op_ghost_dists *= 2.5
            
            #oFeatures['distance_to_base'] = min_distance

        if action == Directions.STOP: stop_penalty = -1000
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: reverse_penalty = -10
        
        score = successor_score + distance_to_food + distance_to_capsule + distance_to_base + num_op_ghosts + op_ghost_dists + stop_penalty + reverse_penalty + invader_distance
        return score
        


class MyTeamDefensive(MyTeamBaseAgent):
    """
    Defensive strategy for MyTeamAgent.
    """

    def choose_action(self, game_state):
        """
        Choose the best defensive action.
        """
        actions = game_state.get_legal_actions(self.index)
        
        best_action = None
        highest_q_value = -9999       
        if not actions:
            return None
        else:
            for action in actions:
                current_q_value = self.get_defensive_features(game_state,action)
                if current_q_value > highest_q_value:
                    highest_q_value = current_q_value
                    best_action = action
                if current_q_value == highest_q_value:
                    best_action = random.choice([best_action,action])
            
            return best_action



    def get_defensive_features(self,game_state,action):
        #dFeatures = util.Counter()
        successor = self.get_successor(game_state, action)
        #dFeatures['successor_score'] = self.get_score(successor)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        food_list = self.get_food_you_are_defending(successor).as_list()
        
        
        width = game_state.data.layout.width
        height = game_state.data.layout.height
        middle_width = 0
        middle_height = 0

        if(game_state.is_on_red_team(self.index)):
            middle_width = width/2 - 1
            middle_height = height/2 - 1
        else:
            middle_width = width/2
            middle_height = height/2
        
        
        distance_to_food = 0
        num_invaders = 0
        invader_distance = 0
        distance_to_middle = 0
        distance_to_middle_scared = 0
        stop_penalty = 0
        reverse_penalty = 0
        #patrol_reward = 0
            
        
        
        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            distance_to_food = min([self.get_maze_distance(my_pos, food) for food in food_list]) * -1.0
        
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        num_invaders = len(invaders) * -1000.0 
        if len(invaders) > 0:
            invader_distance = min([self.get_maze_distance(my_pos, a.get_position()) for a in invaders]) * -100.0
        else:      
            distance_to_middle = min([self.get_maze_distance(my_pos,(middle_width, middle_height))]) * -3.0

        
        if my_state.scared_timer > 0:
            distance_to_middle = min([self.get_maze_distance(my_pos, (middle_width, middle_height))]) * 5.0
            

        if action == Directions.STOP: stop_penalty = -100
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: reverse_penalty = -10
        
       
        score = distance_to_food + num_invaders + invader_distance + distance_to_middle + distance_to_middle_scared + stop_penalty + reverse_penalty #+ patrol_reward
        return score
   