# Learning Methods
Written by Christopher Casolin (z5420849)
April, 2025

# Functionality
## Value Function
- Move according to policy π
	- Random action is chosen until Reward or Penalty is found.
	- Update value of all states in path.
- This process will iteratively approximate V*.

## Q Learning
- Move according to Policy π
	- Random action is chosen epsilon * 100 % of the time.
	- Otherwise, max Q(state, actions) is chosen.
- Maintain Q(s, a) --> Value.
	-  Value represents weighted expected reward from state if action is chosen.
- This process will iteratively approximate Q*.

# Input
Program reads in map files.
### Expected Values
| Value | Meaning |
|-|-|
| '@' | Player |
| T | Target |
| X | Obstacle |
| ~ | File separator for Target values |
| Other | Empty |
| [x,y,z] | List of possible target values.Each list corresponds with unique target. Ordered Top Left to Bottom Right|

## Example
```text
---T
-X-T
@---
~
[1]
[-1, -2]
```

Produces a map where

- The player starts in the bottom left.

- There is an obstacle at coord (1, 1).

- There are two targets.

- The first always gives reward 1.

- The second gives either reward -1 or -2 at random.

  

# Available Commands
1. display|d <v|q>
	- Display the current map with either Value Function or Q values.
2. run|r <v|q> [num_iterations (default: 1)]
	- Perform num_iterations refining Value Function or Q Learning.
	- 'v' (Value Function)
		- Exploration Phase
		-  Agent will move randomly until target is found.
		-  All visited states are updated using the target value, approximating V*.
	- 'q' (Q Learning)
		- Agent will move according to Q Learning Algorithm.
		- Iteratively refining Q, approximating Q*.
3. follow|f [-s] <v|q>
	- Agent will navigate to reward.
	- 'v' (Value Function)
		- Exploitation
		- Agent will prioritise max(ValueFunction(actions)).
		- Path is optimised as V approaches V*.
	- 'q' (Q Learning)
		- This command has little significance but is interesting to see. It can be thought of as an exploitation phase, although there is no guarantee that the goal will be found as it is not part of the Q Learning process.
		- Agent will prioritise max(Q(state, actions)).
		- Path is optimised as Q approaches Q*.
	-  '-s' (show all) will briefly pause between each step to make movement visible.
4. toggle|t \<option\> ...
	- Toggle options about the program:
		- COORDS --> Print state coordinates in display.
5. help|h
	- Display the help menu
6. quit|q
	- Exit the program