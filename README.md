# simple_qlearning
This is a basic working qlearning example from the mountain-car environment in gym

## Parameters
Here are some of the paramaters you may wish to tinker around wth.
* EPISODES = 1 + 2000
  * Number of episodes for the environment simulation, change the last number not the 1. The higher the number the more learning, and the better the agent gets.
* SHOW_EVERY = 500
  * When to show the environment render
* SAVE_EVERY = 20
  * Important for the graphs, the lower the number the significantly lower the whole simulation is
* RENDER = True
  * Set to False if you don't want to render the environment. This is useful if you want to run this in jupyter notebook or Ipython.
* LEARNING_RATE = 0.1
* DISCOUNT = 0.95
* EPSILON = 0.1
* EPSILON_START = 0
* EPSILON_END = EPISODES
* EPSILON_DECAY = EPSILON / (EPSILON_END - EPSILON_START)
* Q_SIZE = [20, 20]
  * The size of the Q table, the bigger it is the slower the whole process becomes, the smaller it is the lower the precision.
