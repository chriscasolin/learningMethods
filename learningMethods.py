#!/usr/bin/env python3

# learningMethods.py
# Written by Christopher Casolin (z5420849)
# April, 2025
#
# Program demonstrates the learning methods:
#   Value Function
#       Move according to policy π
#           Random action is chosen until Reward or Penalty is found.
#       Update value of all states in path.
#       This process will iteratively approximate V*.
#   Q Learning
#       Move according to Policy π
#           Random action is chosen epsilon * 100 % of the time.
#           Otherwise, max Q(state, actions) is chosen.
#       Maintain Q(s, a) --> Value.
#           Value represents weighted expected reward from state if action
#           is chosen.
#       This process will iteratively approximate Q*.
#
# Program reads in map files. Expected Values:
#   @       - Player (Agent)
#   T       - Target
#   X       - Obstacle
#   ~       - File separator for Target values
#   Other   - Empty
#   [x,y,z] - List of possible target values. Each list corresponds with
#               unique target. Ordered Top Left to Bottom Right
# E.g:
#   ---T
#   -X-T
#   @---
#   ~
#   [1]
#   [-1, -2]
# 
# Produces a map where:
# The player starts in the bottom left.
# There is an obstacle at coord (1, 1).
# There are two targets.
#   The first always gives reward 1.
#   The second gives either reward -1 or -2 at random.

from ast import literal_eval
from collections import defaultdict
import random
import re
import readline
import sys
import time

################################################################################
# Globals

MAX_STEPS = 10000
# Assume infinite loop early if user selects --show
SHOWN_MAX_STEPS = 30

print_coordinates = False

# Maintain Value Function:   V(state) --> Value | Target
V = {}
# Maintain Q Function:       Q(s,a) --> Value
Q = defaultdict(lambda: 0.0)

################################################################################
# Classes

class Value():
    '''Value represents the average expected reward if random actions are chosen from a specific state.
    Used only by Value Function.
    '''
    def __init__(self, value):
        self.total = value
        self.times_visited = 0

    def get_value(self):
        if self.times_visited == 0:
            return self.total
        else:
            return self.total / self.times_visited
    
    def update_value(self, reward):
        '''Value is refined by adding the reward to total and dividing by the number of times visited.'''
        self.total += reward
        self.times_visited += 1

    @staticmethod
    def update_values(coords, reward):
        for coord in coords:
            V[coord].update_value(reward)

class Player():
    '''Player (Agent) is moved between states to update Value/Q Functions.'''
    def __init__(self, start):
        self.position = start

    def get_position(self):
        return self.position
    
    def set_position(self, coord):
        self.position = coord

class GameMap():
    '''Reading, maintaining, and displaying map (states).'''
    def __init__(self, map_file):
        # Default values
        self.X_SIZE = None
        self.Y_SIZE = None
        self.TARGETS = {}
        self.OBSTACLES = set()
        self.START = None
        try:
            if map_file: self.read_map(map_file)
        except Exception as e:
            print(f"Error Reading Map File '{map_file}': {e}")
            sys.exit(1)

        for y in range(0, self.Y_SIZE):
            for x in range(0, self.X_SIZE):
                V[(x, y)] = Value(0)

    def read_map(self, map_file):
        with open(map_file, 'r') as f:
            file_lines = [l.strip() for l in f.readlines() if l.strip()]
            sep = file_lines.index('~')
            # Reverse so map origin (0, 0) is bottom-left, matching display
            map_lines = list(reversed(file_lines[:sep]))
            try:
                target_values_strs = list(reversed(file_lines[sep + 1:]))
                for str in target_values_strs:
                    # Eval functions are not consistent enough with errors and type checking
                    # So we check each target value input line against this regex
                    if not re.fullmatch(r'\[(\s*-?\d+\s*,)*(\s*-?\d+\s*),?\s*\]', str.strip()):
                        syntax_error = SyntaxError()
                        syntax_error.text = str
                        raise syntax_error
                
                target_values = [literal_eval(l) for l in target_values_strs]
            except SyntaxError as e:
                print(f"Error: Target values expects list of integers - Found '{e.text}'")
                sys.exit(1)

            self.Y_SIZE = len(map_lines)
            for y, line in enumerate(map_lines):
                for x, char in enumerate(line):
                    if char == 'X':
                        self.OBSTACLES.add((x, y))
                    elif char == '@':
                        self.START = (x, y)
                    if char == 'T':
                        try:
                            self.TARGETS[(x, y)] = target_values.pop(0)
                        except IndexError:
                            print(f"Error: Not enough Target value lists in map file: '{map_file}'")
                            sys.exit(1)
            self.X_SIZE = len(line)

    def get_neighbours(self, coord):
        neighbours = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_x = coord[0] + dx
                new_y = coord[1] + dy
                if (new_x >= 0 and new_x < self.X_SIZE 
                    and new_y >= 0 and new_y < self.Y_SIZE
                    and (new_x, new_y) not in self.OBSTACLES):
                    neighbours.append((new_x, new_y))
        return neighbours

    def display_v(self, player):
        WIDTH = 6
        horizontal_border = ('+' + '-'*WIDTH)*self.X_SIZE + '+'
        print(horizontal_border)
        for y in reversed(range(self.Y_SIZE)):
            values = []
            coords = []
            objects = []
            for x in range(self.X_SIZE):
                coord = (x, y)
                coords.append(f"{str(coord):{WIDTH}}")
                value = f"{V[coord].get_value():.3f}"
                values.append(f"{value:{WIDTH}}")
                if coord == player.position:
                    objects.append(f"{'PLAYER':{WIDTH}}")
                elif coord in self.TARGETS:
                    objects.append(f"{'T':{WIDTH}}")
                elif coord in self.OBSTACLES:
                    objects.append(f"{'X'*WIDTH}")
                else:
                    objects.append(f"{'':{WIDTH}}")
            
            # Print state information
            if print_coordinates: print('', *coords, '', sep='|',)
            print('', *objects, '', sep='|',)
            print('', *values, '', sep='|',)
            print(horizontal_border)

    def display_q(self, player):
        WIDTH = 8
        horizontal_border = ('+' + '-'*WIDTH)*self.X_SIZE + '+'
        print(horizontal_border)
        for y in reversed(range(self.Y_SIZE)):
            coords = []
            objects = []
            ups = []
            downs = []
            lefts = []
            rights = []
            for x in range(self.X_SIZE):
                coord = (x, y)
                coords.append(f"{str(coord):{WIDTH}}")

                def maintain_direction(list, value, char):
                    if value:
                        value = f"{value:.3f}"
                        list.append(f"{char + ' ' + value:{WIDTH}}")
                    else:
                        list.append(f"{char + ' ':{WIDTH}}")
                maintain_direction(ups, Q.get((coord, (x, y + 1))), '^')
                maintain_direction(downs, Q.get((coord, (x, y - 1))), 'v')
                maintain_direction(lefts, Q.get((coord, (x - 1, y))), '<')
                maintain_direction(rights, Q.get((coord, (x + 1, y))), '>')

                if coord == player.position:
                    objects.append(f"{'PLAYER':{WIDTH}}")
                elif coord in self.TARGETS:
                    objects.append(f"{f'T: {random.choice(self.TARGETS[coord])}':{WIDTH}}")
                elif coord in self.OBSTACLES:
                    objects.append(f"{'X'*WIDTH}")
                else:
                    objects.append(f"{'':{WIDTH}}")

            # Print state information
            if print_coordinates: print('', *coords, '', sep='|',)
            print('', *objects, '', sep='|',)
            print('', *ups, '', sep='|',)
            print('', *downs, '', sep='|',)
            print('', *lefts, '', sep='|',)
            print('', *rights, '', sep='|',)
            print(horizontal_border)
            
class Game():
    '''Game maintains a map and a player (agent).
    Contains work for performing exploration and exploitation.
    '''
    def __init__(self, map_file):
        self.map = GameMap(map_file)
        self.player = Player(self.map.START)
    
    def run_q(self, learning_rate=0.1, discount_factor=0.9, random_chance=0.2):
        '''Perform Q Learning algorithm.
        Policy may choose action randomly, otherwise choose best known action according to Q(s,a).
        '''
        player = self.player
        game_map = self.map
        state = game_map.START
        player.set_position(state)

        while state not in game_map.TARGETS:
            neighbours = game_map.get_neighbours(state)

            # Policy:
            # Chance that random action is taken.
            # Otherwise best known action taken
            if random.random() < random_chance:
                action = random.choice(neighbours)
            else:
                action = max(neighbours, key=lambda a: Q[(state, a)])

            player.set_position(action)
            next_state = action

            if next_state in game_map.TARGETS:
                reward = random.choice(game_map.TARGETS[next_state])
            else:
                reward = 0

            # Using stochastic approximation for Q
            next_neighbours = game_map.get_neighbours(next_state)
            max_q_next = max([Q[(next_state, a)] for a in next_neighbours], default=0)
            Q[(state, action)] += learning_rate * (reward + discount_factor * max_q_next - Q[(state, action)])

            state = next_state
        return reward
    
    def follow_q(self, show=False):
        '''Follow Q Values to try and find a goal. There is no guarantee that this works because
        it is not part of the Q learning process.
        '''
        player = self.player
        game_map = self.map
        player.set_position(game_map.START)
        state = player.get_position()
        path = []

        goal_found = False
        step = 0
        step_limit = SHOWN_MAX_STEPS if show else MAX_STEPS
        while step < step_limit:
            if state in game_map.TARGETS:
                goal_found = True
                break

            if show:
                time.sleep(0.2)
                game_map.display_q(player)

            path.append(state)
            actions = game_map.get_neighbours(state)

            if not actions:
                break

            action = max(actions, key=lambda a: Q[(state, a)])
            player.set_position(action)
            state = action
            step += 1

        game_map.display_q(player)
        if not goal_found:
            print("Goal not found")
        else:
            print(f'Found Q-path of length {len(path)}')

    def run_v(self):
        '''Exploration:
        Agent will move randomly until reward or penalty is found.
        Then, every state visited has its value updated.
        '''
        player = self.player
        game_map = self.map
        state = game_map.START

        path = []
        self.player.set_position(game_map.START)
        while state not in game_map.TARGETS:
            path.append(state)
            neighbours = game_map.get_neighbours(state)
            random_neighbour = random.choice(neighbours)
            player.set_position(random_neighbour)
            state = player.get_position()
        path.append(state)
        
        if state in game_map.TARGETS:
            reward = random.choice(game_map.TARGETS[state])
        else:
            reward = 0

        Value.update_values(path, reward)
        return reward
    
    def follow_v(self, show=False):
        '''Exploitation:
        Determine next action by max(V(neighbours)).
        '''
        player = self.player
        game_map = self.map
        player.set_position(game_map.START)
        state = player.get_position()
        path = []

        goal_found = False
        step = 0
        step_limit = SHOWN_MAX_STEPS if show else MAX_STEPS
        while step < step_limit:
            if state in game_map.TARGETS:
                goal_found = True
                break

            if show:
                time.sleep(0.2)
                game_map.display_v(player)

            path.append(state)
            actions = game_map.get_neighbours(state)
            if not actions:
                break

            action = max(actions, key=lambda a: V[a].get_value())
            player.set_position(action)
            state = action
            step += 1

        game_map.display_v(player)
        if not goal_found:
            print("Goal not found")
        else:
            print(f'Found path of length {len(path) + 1}')

################################################################################
# Command Parsing

def parse_run(game: Game, params, args):
    usage_error = lambda: print('Usage: run|r <v|q> [num_iterations (default: 1)]')

    # Default case if num_iterations not specified
    args.append(1)

    if len(args) < 2:
        usage_error()
        return
    
    mode = args[0]
    
    try:
        num_iterations = int(args[1])
        if num_iterations < 1:
            raise ValueError
    except ValueError:
        print("'num_iterations' must be a positive integer")
        return
    
    if mode == 'v':
        run_func = game.run_v
        display_func = game.map.display_v
    elif mode == 'q':
        run_func = game.run_q
        display_func = game.map.display_q
    else:
        usage_error()
        return
    
    while num_iterations > 0:
        reward = run_func()
        num_iterations -= 1
    display_func(game.player)
    print(f"Collected reward of {reward}")

def parse_follow(game: Game, params, args):
    usage_error = lambda: print('Usage: follow|f [-s] <v|q>')
    if len(args) < 1:
        usage_error()
        return
    
    show = '-s' in params
    mode = args[0]
    if mode == 'v':
        game.follow_v(show)
    elif mode == 'q':
        game.follow_q(show)
    else:
        usage_error()
        return

def parse_display(game: Game, args):
    usage_error = lambda: print("Usage: display|d <v|q>")
    if len(args) != 1:
        usage_error()
        return
    
    mode = args[0]
    player = game.player
    game_map = game.map
    player.set_position(game_map.START)

    if mode == 'v':
        game_map.display_v(player)
    elif mode == 'q':
        game_map.display_q(player)
    else:
        usage_error()
        return

def parse_toggle(args: list):
    if not args: print('Usage: toggle <option> ...')
    args = [arg.upper() for arg in args]
    for arg in args:
        if arg == 'COORDS':
            global print_coordinates
            print_coordinates = not print_coordinates
            print(f"{arg} = {print_coordinates}")
        else:
            print(f"Unknown toggle option '{arg}'. Type 'help' for more.")
            return

def parse_help():
    print('''Available Commands:
    display|d <v|q>
        Display the current map with either Value Function or Q values.

    run|r <v|q> [num_iterations (default: 1)]
        Perform num_iterations refining Value Function or Q Learning.
        'v' (Value Function)
            Exploration:
            Agent will move randomly until target is found.
            All visited states are updated using the target value, approximating V*.

        'q' (Q Learning)
            Agent will move according to Q Learning Algorithm.
            Iteratively refining Q, approximating Q*.
          
    follow|f [-s] <v|q>
        Agent will navigate to reward.
        'v' (Value Function)
            Exploitation:
            Agent will prioritise max(ValueFunction(actions)).
            Path is optimised as V approaches V*.
        'q' (Q Learning)
            **This command has little significance but is interesting to see.
             It can be thought of as an exploitation stage, although there is
             no guarantee that the goal will be found as it is not part of
             the Q Learning process.
            Agent will prioritise max(Q(state, actions)).
            Path is optimised as Q approaches Q*.
        '-s' (show all) will briefly pause between each step to make movement visible.
          
    toggle|t <option> ...
        Toggle options about the program:
            COORDS --> Print state coordinates in display.
        
    help|h
        Display the help menu
    
    quit|q
        Exit the program''')

################################################################################
# Main

def input_loop(game: Game):
    while True:
        user_input = input('Enter Command: ')

        parts = user_input.split()

        if len(parts) < 1:
            continue

        command = parts[0].upper()
        params = set()
        args= []
        for p in parts[1:]:
            if p.startswith('-'):
                params.add(p)
            else:
                args.append(p)

        if command == 'D' or command == 'DISPLAY':
            parse_display(game, args)
        elif command == 'T' or command == 'TOGGLE':
            parse_toggle(args)
        elif command == 'R' or command == 'RUN':
            parse_run(game, params, args)
        elif command == 'F' or command == 'FOLLOW':
            parse_follow(game, params, args)
        elif command == 'H' or command == 'HELP':
            parse_help()
        elif command == 'Q' or command == 'QUIT':
            sys.exit(0)
        else:
            print(f"Unknown command: '{parts[0]}'. Try 'help'")

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <map_file>")
        sys.exit(1)
    
    game = Game(sys.argv[1])

    try:
        input_loop(game)
    except (KeyboardInterrupt, EOFError):
        print()
        sys.exit(0)

if __name__ == "__main__":
    main()
    