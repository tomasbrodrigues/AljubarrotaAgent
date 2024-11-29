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

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
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

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.food_carried = 0

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        self.food_carried = 0

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
        Choose the best offensive action.
        """
        actions = game_state.get_legal_actions(self.index)
        best_actions = []
        best_action = None
        highest_q_value = -9999       
        if not actions:
            return None
        else:
            for action in actions:
                current_q_value = self.evaluate(game_state,action)
                if current_q_value >= highest_q_value:
                    highest_q_value = current_q_value
                    best_actions.append(action) 
            best_action = random.choice(best_actions)
            if(game_state.get_agent_state(self.index).is_pacman):
                self.update_food_carried(game_state,best_action)

            return best_action


    def evaluate(self, game_state, action):
        """
        Evaluate the state for offensive features.
        """
        weights = self.get_weights(game_state, action)
        if(game_state.get_agent_state(self.index).is_pacman):
            return self.get_offensive_features(game_state,action) * weights


    def get_offensive_features(self,game_state,action):
        oFeatures = util.Counter()
        successor = self.get_successor(game_state, action)
        oFeatures['successor_score'] = self.get_score(successor)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        food_list = self.get_food(successor).as_list()
        num_food = len(food_list)
        oFeatures['next_food_score'] = -len(food_list)  # self.getScore(successor)
        capsules_list = self.get_capsules(game_state).as_list()
        width = game_state.data.layout.width
        height = game_state.data.layout.height
        
        middle_of_board = [(width/2 - 1, y) for y in range(height)]
        
        
        if num_food > 0:  # This should always be True,  but better safe than sorry
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            oFeatures['distance_to_food'] = min_distance
        
        if len(capsules_list) > 0:
            min_distance = min([self.get_maze_distance(my_pos, capsule) for capsule in capsules_list])
            oFeatures['distance_to_capsule'] = min_distance
        
        scared_ghosts = [agent for agent in self.get_opponents(game_state) 
                     if game_state.get_agent_state(agent).scared_timer > 0]  
    
        if scared_ghosts:
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            oFeatures['distance_to_food_scared'] = min_distance
        
        if self.food_carried > 0.05 * num_food:
            min_distance = min([self.get_maze_distance(my_pos,position) for position in middle_of_board])
            oFeatures['distance_to_base'] = min_distance

        if action == Directions.STOP: oFeatures['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: oFeatures['reverse'] = 1
        
        return oFeatures


    def get_offensive_weights(self):
        return {'successor_score': 1.0, 'next_food_score':100.0,
            'distance_to_food': -1.0, 'distance_to_capsule':-2.0 , 'distance_to_food_scared':-5.0 , 'distance_to_base': -3.0, 'stop': -100.0, 'reverse': -2.0}

        
    def update_food_carried(self,game_state,action):
        food_list = self.get_food(game_state).as_list()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        if my_pos in food_list:
            self.food_carried += 1

class MyTeamDefensive(MyTeamBaseAgent):
    """
    Defensive strategy for MyTeamAgent.
    """

    def choose_action(self, game_state):
        """
        Choose the best defensive action.
        """
        actions = game_state.get_legal_actions(self.index)
        best_actions = []
        best_action = None
        highest_q_value = -9999       
        if not actions:
            return None
        else:
            for action in actions:
                current_q_value = self.evaluate(game_state,action)
                if current_q_value >= highest_q_value:
                    highest_q_value = current_q_value
                    best_actions.append(action) 
            best_action = random.choice(best_actions)
            
            return best_action

    def evaluate(self, game_state, action):
        """
        Evaluate the state for defensive features.
        """
        weights = self.get_weights(game_state, action)
        return  self.get_defensive_features(game_state,action) * weights



    def get_defensive_features(self,game_state,action):
        dFeatures = util.Counter()
        successor = self.get_successor(game_state, action)
        dFeatures['successor_score'] = self.get_score(successor)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        food_list = self.get_food_you_are_defending(successor).as_list()
        
        
        width = game_state.data.layout.width
        height = game_state.data.layout.height
        
        
        
        
        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            dFeatures['distance_to_your_food'] = min_distance
        
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        dFeatures['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            dFeatures['invader_distance'] = min(dists)
        
        dFeatures['distance_to_middle'] = min([self.get_maze_distance(my_pos,(width/2 - 1, height / 2 - 1))])

        scared_team = [agent for agent in self.get_team(game_state) 
                     if game_state.get_agent_state(agent).scared_timer > 0]  
    
        if scared_team:
            min_distance = min([self.get_maze_distance(my_pos, (width/2 - 1, height / 2 - 1))])
            dFeatures['distance_to_middle_scared'] = min_distance


        if action == Directions.STOP: dFeatures['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: dFeatures['reverse'] = 1
        
        # Add defensive boundary and wall penalties
        walls = game_state.get_walls()
        wall_positions = [(x, y) for x in range(walls.width) for y in range(walls.height) if walls[x][y]]
        if len(wall_positions) > 0:
            min_wall_dist = min(self.get_maze_distance(my_pos, wall) for wall in wall_positions)
            dFeatures['distance_to_wall'] = min_wall_dist

        return dFeatures

    def get_defensive_weights(self):
        return {'successor_score': 1.0,'distance_to_your_food': 5.0,
            'num_invaders': -1000.0,'invader_distance': -10.0, 'distance_to_middle': -3.0, 'distance_to_middle_scared': -5.0, 'stop': -100.0, 'reverse': -2.0, 'distance_to_wall': -1.0 }
