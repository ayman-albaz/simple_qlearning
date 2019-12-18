import os
import cv2
import gym
import shutil
import keyboard
import numpy as np
import matplotlib.pyplot as plt


"""
BACKGROUND
----------
VARS: DIS, VEL
HIGH: 0.6, 0.07
LOW: -1.2,-0.07
ADJ:  1.8, 0.14

TERMINOLOGY
-----------
DISCRETE STATE: Discrete integer from 0 to Q_SIZE. It will be useful for indexing the q_table.
DISCOUNT: How much to care about future rewards vs current rewards.
EPSILON: If less than epislon take a random step, in order to discover new paths.

OUTLINE
-------
Load environment.
Declare variables.
Double simulation for loop.
Take action, or take random action if lower then epsilon.
Obtain old q-value with new q-value
Enter in the q-value equation.
Increase q-value if we achieve goal, decrease it if don't reach goal.
Register variables like discrete state and epsilon to get raady for next tick.

FEATURES
--------
Quit by pressing q
Show ever X episodes
Graph q values
Video of the q values change over time

"""

#Environement
env = gym.make("MountainCar-v0")
np.random.seed(7)
env.seed(7)

#The q_table for each 3 actions will be stored here
q_dir = 'q_heatmaps'
if os.path.exists(q_dir):
    shutil.rmtree(q_dir, ignore_errors=True)
    os.makedirs(q_dir)
else:
    os.makedirs(q_dir)

#The argmax of the q_table will be stored here
maxq_dir = 'maxq_heatmaps'
if os.path.exists(maxq_dir):
    shutil.rmtree(maxq_dir, ignore_errors=True)
    os.makedirs(maxq_dir)
else:
    os.makedirs(maxq_dir)

#ENV variables
EPISODES = 1 + 2000
SHOW_EVERY = 500
SAVE_EVERY = 20
RENDER = True

#LEARNING variables
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPSILON = 0.1
EPSILON_START = 0
EPSILON_END = EPISODES
EPSILON_DECAY = EPSILON / (EPSILON_END - EPSILON_START)

#Q_TABLE variables
Q_SIZE = [20, 20]
OS_ACTIONS = env.action_space.n
OS_GOAL = env.goal_position
OS_SIZE_HIGH = env.observation_space.high
OS_SIZE_LOW = env.observation_space.low
OS_SIZE = (OS_SIZE_HIGH - OS_SIZE_LOW).round(2)
OS_STEP = (OS_SIZE / (np.array(Q_SIZE) - 1)).round(3)

#VARIABLE variables. Do not modify.
quit = False
q_table = np.random.uniform(low=-2, 
							high=0, 
							size=(Q_SIZE + [env.action_space.n]))

#Simulations
for episode in range(EPISODES):
    discrete_state = tuple(((env.reset() - OS_SIZE_LOW) / OS_STEP).astype(int))
    done = False
    
    #Single simulation
    while not done:

        #User control
        if keyboard.is_pressed('q'):
            quit = True
            break
            
        #Graphics render
        if episode % SHOW_EVERY == 0:
            if RENDER == True:
                env.render()
        
        #What action to take
        if np.random.random() > EPSILON:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, OS_ACTIONS)
        observation, reward, done, info = env.step(action)
        new_discrete_state = tuple(((observation - OS_SIZE_LOW) / OS_STEP).astype(int))

        #Q calculations
        if observation[0] >= OS_GOAL:
            print(f'We made it on episode {episode}')
            q_table[discrete_state + (action, )] = 0
        
        else:
            current_q = q_table[discrete_state + (action, )] 
            max_future_q = np.max(q_table[new_discrete_state])
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action, )] = new_q

        #Registering the current state to the future state
        discrete_state = new_discrete_state
    
    #Registering new epsilon
    if EPSILON_START < episode < EPSILON_END:
        EPSILON = EPSILON - EPSILON_DECAY

    #Graphs
    if episode % SAVE_EVERY == 0:
        fig = plt.figure(figsize=(15,30))
        plt.subplot(311)
        plt.title('Left')
        plt.imshow(q_table[:,:,0], aspect='auto')
        plt.ylabel('Distance')
        plt.xlabel('Velocity')
        plt.clim(-2, 0)
        plt.yticks(np.arange(Q_SIZE[0]), np.arange(OS_SIZE_LOW[0], OS_SIZE_HIGH[0] + OS_STEP[0], OS_STEP[0]).round(3))
        plt.xticks(np.arange(Q_SIZE[1]), np.arange(OS_SIZE_LOW[1], OS_SIZE_HIGH[1] + OS_STEP[1], OS_STEP[1]).round(3))
        plt.subplot(312)
        plt.title('Stop')
        plt.imshow(q_table[:,:,1], aspect='auto')
        plt.ylabel('Distance')
        plt.xlabel('Velocity')
        plt.clim(-2, 0)
        plt.yticks(np.arange(Q_SIZE[0]), np.arange(OS_SIZE_LOW[0], OS_SIZE_HIGH[0] + OS_STEP[0], OS_STEP[0]).round(3))
        plt.xticks(np.arange(Q_SIZE[1]), np.arange(OS_SIZE_LOW[1], OS_SIZE_HIGH[1] + OS_STEP[1], OS_STEP[1]).round(3))
        plt.subplot(313)
        plt.title('Right')
        im = plt.imshow(q_table[:,:,2], aspect='auto')
        plt.ylabel('Distance')
        plt.xlabel('Velocity')
        plt.clim(-2, 0)
        plt.yticks(np.arange(Q_SIZE[0]), np.arange(OS_SIZE_LOW[0], OS_SIZE_HIGH[0] + OS_STEP[0], OS_STEP[0]).round(3))
        plt.xticks(np.arange(Q_SIZE[1]), np.arange(OS_SIZE_LOW[1], OS_SIZE_HIGH[1] + OS_STEP[1], OS_STEP[1]).round(3))
        cbar_ax = fig.add_axes([0.91, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        fig.suptitle(f'car_episode_{episode}')
        fig.savefig(f'{q_dir}/car_{episode}')
        
        fig = plt.figure(figsize=(15,10))
        plt.title('Move')
        im = plt.imshow(np.argmax(q_table, axis=2), aspect='auto')
        plt.ylabel('Distance')
        plt.xlabel('Velocity')
        plt.yticks(np.arange(Q_SIZE[0]), np.arange(OS_SIZE_LOW[0], OS_SIZE_HIGH[0] + OS_STEP[0], OS_STEP[0]).round(3))
        plt.xticks(np.arange(Q_SIZE[1]), np.arange(OS_SIZE_LOW[1], OS_SIZE_HIGH[1] + OS_STEP[1], OS_STEP[1]).round(3))
        fig.colorbar(im)
        fig.suptitle(f'car_episode_{episode}')
        fig.savefig(f'{maxq_dir}/car_{episode}')
        
    if quit == True:
        break

env.close()


#Writing videos from heatmaps
fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter('qlearn.avi', fourcc, 20.0, (1500, 3000))
for i in range(0, EPISODES, SAVE_EVERY):
    frame = cv2.imread(os.path.join(q_dir, f'car_{i}.png'))
    out.write(frame)
out.release()

out = cv2.VideoWriter('qmaxlearn.avi', fourcc, 20.0, (1500, 1000))
for i in range(0, EPISODES, SAVE_EVERY):
    frame = cv2.imread(os.path.join(maxq_dir, f'car_{i}.png'))
    out.write(frame)
out.release()
