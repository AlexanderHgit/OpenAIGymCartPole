import numpy as np
import gymnasium as gym
from collections import defaultdict
import random
import math
import time

#finds and returns the action associated with the most reward along with the reward itself
#if there is no reward for the current state then return a random action
def getHighestReward(q_table, state):
    if state in q_table:
        action = max(q_table[state], key=q_table[state].get)
        max_q_value = q_table[state][action]
    else:
        action = random.randint(0, 1)
        max_q_value = 0
    return action, max_q_value

env = gym.make("CartPole-v1")
observable_stateervation_bins = [30, 30, 50, 50]
np_array_win_size = np.array([0.25, 0.25, 0.01, 0.1])
q_table = {}
epsilon = 1.0
epsilon_decay_value = 0.99995
min_epsilon = 0.05
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 10000
def getDiscreteState(state):
    discrete_state = state / np_array_win_size + np.array([15, 10, 1, 10])
    return tuple(discrete_state.astype(int))

for episode in range(1, EPISODES+1):
    state = env.reset()[0]
    episode_reward=0
    done = False
    #getting starting state with observable_stateerbvable values
    observable_state=getDiscreteState(state)

    while not done:
        
        #choosing action based on epison
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            #chooses a random action if there is no action chosen for the current state
            action, _ = getHighestReward(q_table, observable_state)
        #get new state
        next_state, reward, done, truncated, info = env.step(action) 
        #getting an observable_stateervavble next state
        next_observable_state=getDiscreteState(next_state)
        #reward is lowered when the game ends early
        if done:
            reward=-100


        #add table entries if they dont exist    
        if observable_state not in q_table:
            q_table[observable_state] = {0: 0, 1: 0}
        
        if next_observable_state not in q_table:
            q_table[next_observable_state] = {0: 0, 1: 0}

        #getting the q_value of the current state
        current_q=q_table[observable_state][action]
        #getting the reward of the best action in the next state
        _,max_future_q=getHighestReward(q_table,next_observable_state)
        #new_q uses the q learning formula to set a reward value
        new_q = (1-LEARNING_RATE)*current_q+LEARNING_RATE*(reward+DISCOUNT*max_future_q)
        #updating the q table and adding new_q 
        q_table[observable_state][action]=new_q
        observable_state=next_observable_state
        episode_reward+=reward
    #epsilon starts at 1 so the agent can learn the environment better but as the episodes go on the agent will rely on the q table more
    if epsilon > min_epsilon:
        epsilon *= epsilon_decay_value

    if episode % 1000 == 0:
        print(f"Episode: {episode}, Epsilon: {epsilon:.5f}")



env.close()
# testing trained ai
input("press enter to continue")
env = gym.make("CartPole-v1", render_mode="human")
episodes = 100

for _ in range(episodes):
    state = env.reset()[0]
    done = False
    while not done:
        discrete_state = getDiscreteState(state)
        if discrete_state in q_table:
            action = getHighestReward(q_table, discrete_state)[0]
        else:
            action = env.action_space.sample()
        state, reward, done, truncated, info = env.step(action)

env.close()
print(q_table)