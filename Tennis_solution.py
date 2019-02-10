import torch
import numpy as np
import pandas as pd
from unityagents import UnityEnvironment
from ddpg_agent import Agent
from collections import deque
import time


#
# Setup Environment
#

env = UnityEnvironment(file_name="Tennis.app")
# env = UnityEnvironment(file_name="/data/Tennis_Linux_NoVis/Tennis")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents 
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
print(type(states))
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


#
# Train DDPG Agent
#

random_seed = 4
train_mode = True

agent = Agent(state_size=state_size, action_size=action_size, num_agents=2, random_seed=random_seed)
print(agent)

def ddpg(n_episodes=5000, max_t=1000, goal_score=0.5):
    scores_window = deque(maxlen=100)
    learning_stats = pd.DataFrame(index=range(1, n_episodes+1), 
                                  columns=['Max Score', 'Moving Avg. (100 eps)'])
    learning_stats.index.name = 'Episode'
    
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=train_mode)[brain_name] 
        states = env_info.vector_observations
        scores = np.zeros(num_agents)
        agent.reset()
        
        start_time = time.time()
        
        for t in range(max_t):
            actions = agent.act(states)
            
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            
            agent.step(states, actions, rewards, next_states, dones)
            
            scores += rewards
            states = next_states
            
            if np.any(dones):
                break
    
        learning_stats.loc[i_episode, 'Max Score'] = max_score = np.max(scores)
        scores_window.append(max_score)
        learning_stats.loc[i_episode,'Moving Avg. (100 eps)'] = moving_average_score = np.mean(scores_window)
        
        duration = time.time() - start_time
        
        print('\rEpisode {}\tMoving Avg. Score: {:.2f}\tMax Score: {:.2f}\tDur.: {:.2f}\tSteps: {}'
              .format(i_episode, moving_average_score, max_score, duration, t))
            
        if moving_average_score >= goal_score and i_episode >= 100:
            print('Problem Solved after {} epsisodes! Moving Average score: {:.2f}'.format(i_episode, moving_average_score))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break
        
    learning_stats.dropna(inplace=True)
    learning_stats.to_csv('learning_stats.csv')

ddpg()
env.close()
