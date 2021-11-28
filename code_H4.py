import gym
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time

## 1
## 1.1 load MDP problem enviornment
env = gym.make("FrozenLake8x8-v1")

## 1.2   define policy testing function
def test_policy(p)
    e = 0
    for i_episode in range(10000):
        c = env.reset()
        while True:
            c,reward,done,info = env.step(int(p[c]))
            #time.sleep(0.5)
            #env.render()
            if done:
                if reward == 1:
                    e +=1
                break
    print(e/10000)

## 2 model-based solutions
## 2.1 implement value iteration
def value_iteration(env, max_iterations=100000, lmbda=0.99, theta=1e-20):
    env.reset()
    stateValue = np.zeros(env.nS)
    bestPolicy = np.zeros(env.nS)
    for i in range(max_iterations):
        newStateValue = np.copy(stateValue)
        for state in range(env.nS):
            action_values = []      
            for action in range(env.nA):
                next_state_rewards = []
                for next_sr in env.P[state][action]:
                    prob, next_state, reward, done = next_sr
                    next_state_rewards.append(prob * (reward + lmbda*newStateValue[next_state]))      #the value of each action
                action_values.append(np.sum(next_state_rewards))
            stateValue[state] = max(action_values)  #update the value of the state
            bestPolicy[state] = action_values.index(max(action_values))
        
        if (np.sum(np.fabs(newStateValue-stateValue)) <= theta):
            print("Value-iteration converged at iteration #",i)
            break
    
    return stateValue, bestPolicy

env.reset()  
v, p = value_iteration(env)
print(v)
print(np.reshape(p,(8,8)))

test_policy(p)

## 2.2 implement policy iteration
def evaluate(policy, lmbda=0.99, theta=1e-5):
    stateValue = np.zeros(env.nS)
    i = 0
    while True:
        newStateValue = np.copy(stateValue)
        for state in range(env.nS):
            action = policy[state]
            stateValue[state] = sum([prob * (reward + lmbda*newStateValue[next_state]) for prob, next_state, reward, done in env.P[state][action]])
        i += 1
        if (np.sum(np.fabs(newStateValue-stateValue)) <= theta):
            print("compute_value_function at iteration #",i)
            break
    return stateValue

def extract_policy(stateValue, lmbda=0.99):
    policy = np.zeros(env.nS)
    for state in range(env.nS):
        action_values = np.zeros(env.nA)
        for action in range(env.nA):
            for next_sr in env.P[state][action]:
                    prob, next_state, reward, done = next_sr
                    action_values[action] += prob * (reward + lmbda*stateValue[next_state])
        policy[state] = np.argmax(action_values)
    return policy    

def policy_iterations(env, max_iterations=100000, lmbda=0.99,):
    env.reset()  
    random_policy = np.zeros(env.nS)
    for i in range(max_iterations):
        value = evaluate(random_policy,lmbda)
        new_policy = extract_policy(value,lmbda)
        if (np.all(random_policy == new_policy)):
            print("policy-iteration converged at iteration #",i)
            break
        random_policy = new_policy
    return new_policy

p_new = policy_iterations(env)
print(p_new)
print(p)
print(p==p_new)
test_policy(p)

## 3 Rreinforcement learning solutions
## 3.1 Q-learning
class QAgent(object):
    def __init__(self,obs_n,act_n,learning_rate,gamma,e_greed):
        self.obs_n = obs_n
        self.act_n = act_n
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = e_greed
        self.Q = np.zeros([env.nS,env.nA])
    
    def predict(self,state):
        Q_list = self.Q[state,:]
        max_q = np.max(Q_list)
        action_list = np.where(Q_list==max_q)[0]
        action = np.random.choice(action_list)
        return action
        
    def sample(self,state):
        if np.random.uniform(0,1) < (1.0-self.epsilon):
            action = self.predict(state)
        else:
            action = np.random.choice(self.act_n)
        return action
    
    def learn(self,state,action,reward,next_state,done):
        predict_Q = self.Q[state,action]
        if done:
            target_Q = reward
        else:
            target_Q = reward + self.gamma * np.max(self.Q[next_state,:])
        
        self.Q[state,action] += self.lr * (target_Q - predict_Q)

def run_episode_Q(env,agent,is_render=False):
    total_steps = 0
    total_reward = 0
    state = env.reset()
    while True:
        action = agent.sample(state)
        next_state, reward, done, _ = env.step(action)
        agent.learn(state,action,reward,next_state,done)
        state = next_state
        total_steps += 1
        total_reward += reward
        if is_render:
            env.render()
        if done:
            break
    return total_reward,total_steps

def test_episode(env,agent,is_render=False):
    total_reward = 0
    state = env.reset()
    
    while True:
        action = agent.predict(state)
        next_state,reward,done,_ = env.step(action)
        total_reward += reward
        state = next_state
        if is_render:
            time.sleep(0.5)
            env.render()
        if done:
            #print('test_reward', total_reward)
            break
    return total_reward

env.reset()
Q_agent = QAgent(obs_n = env.nS, act_n = env.nA, learning_rate = 0.1, gamma = 0.9, e_greed = 0.1)  
reward_his = []
for episode in range(10000):
    ep_reward, ep_steps = run_episode_Q(env,Q_agent)
    reward_his.append(ep_reward)
    #print("Episode %s: steps = %s, reward = %1.f"%(episode, ep_steps, ep_reward))
for i in range(10):
    print(sum(reward_his[i*1000:(i+1)*1000-1]))


## 3.2 SARSA
class SarsaAgent(object):
    def __init__(self,obs_n,act_n,learning_rate,gamma,e_greed):
        self.obs_n = obs_n
        self.act_n = act_n
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = e_greed
        self.Q = np.zeros([env.nS,env.nA])
    
    def predict(self,state):
        Q_list = self.Q[state,:]
        max_q = np.max(Q_list)
        action_list = np.where(Q_list==max_q)[0]
        action = np.random.choice(action_list)
        return action
        
    def sample(self,state):
        if np.random.uniform(0,1) < (1.0-self.epsilon):
            action = self.predict(state)
        else:
            action = np.random.choice(self.act_n)
        return action
    
    def learn(self,state,action,reward,next_state,next_action,done):
        predict_Q = self.Q[state,action]
        if done:
            target_Q = reward
        else:
            target_Q = reward + self.gamma * self.Q[next_state,next_action]
        self.Q[state,action] += self.lr * (target_Q - predict_Q)

def run_episode_Sarsa(env,agent,is_render=False):
    total_steps = 0
    total_reward = 0
    state = env.reset()
    action = agent.sample(state)
    while True:
        next_state, reward, done, _ = env.step(action)
        next_action = agent.sample(next_state)
        # update Q table
        agent.learn(state,action,reward,next_state,next_action,done)
        # s <- s'   a <- a'
        state = next_state
        action = next_action
        total_steps += 1
        total_reward += reward
        if is_render:
            env.render()
        if done:
            break
    return total_reward,total_steps

env.reset()
S_agent = SarsaAgent(obs_n = env.nS, act_n = env.nA, learning_rate = 0.1, gamma = 0.9, e_greed = 0.1)  
reward_his = []
for episode in range(10000):
    ep_reward, ep_steps = run_episode(env,S_agent)
    reward_his.append(ep_reward)
    #print("Episode %s: steps = %s, reward = %1.f"%(episode, ep_steps, ep_reward))
for i in range(10):
    print(sum(reward_his[i*1000:(i+1)*1000-1]))
