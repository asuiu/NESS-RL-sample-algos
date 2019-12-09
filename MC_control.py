#!/usr/bin/env python
# coding:utf-8
# Author: ASU --<andrei.suiu@gmail.com>
# Purpose: 
# Created: 12/8/2019
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import List, NamedTuple, Dict

import numpy as np

from ScoreLogger import ScoreLogger, PLOT_REFRESH
from blackjack import BlackjackEnv2

epsilon = 0.05
env = BlackjackEnv2()
nA = env.action_space.n


@dataclass
class V:
    returns: float = 0.0
    state_cnt: int = 0
    
    @property
    def val(self) -> float:
        return self.returns / self.state_cnt if self.state_cnt > 0 else 0.0


class State(NamedTuple):
    my_sum: int
    dealers_card: int
    usable_ace: bool


def maxes(itr, key=lambda x: x):
    """Returns a list of MAX values"""
    itr = list(itr)
    mval = max(map(key, itr))
    return list(filter(lambda x: key(x) == mval, itr))


@dataclass
class StateActions:
    actions: List[V] = field(default_factory=lambda: [V() for i in range(nA)])
    
    @property
    def best_action(self):
        '''Similar with np.argmax but if there's multiple max values, returns a random one out of them'''
        max_idxes = maxes(range(len(self.actions)), key=lambda i: self.actions[i].val)
        rnd_best_action = np.random.choice(max_idxes)
        
        return rnd_best_action


class GreedyMCPlayer:
    def __init__(self, rounds: int = 30) -> None:
        self.observation_space = env.observation_space.shape
        self.action_space = env.action_space.n
        
        self._N = rounds
        self.sl = ScoreLogger(str(self.__class__), success_rounds=50)
        self._STOP_THRESHOLD = 1.0  # 0.86- with RP
        self._round = 0
        self._score = 0
        self._total_score = 0
        self._max_avg_score = -100
    
    def get_epsilon_greedy_action_policy(self, observation: State):
        def_vals = epsilon / nA
        A = np.ones(nA, dtype=float) * def_vals
        best_action = self.P[observation].best_action
        A[best_action] += (1.0 - epsilon)
        
        return A
    
    def generate_episode(self):
        episode = []
        current_state = env.reset()
        current_state = State(*current_state)
        
        while True:
            prob_scores = self.get_epsilon_greedy_action_policy(current_state)
            action = np.random.choice(np.arange(len(prob_scores)), p=prob_scores)  # 0 or 1
            
            next_state, reward, done, _ = env.step(action)
            next_state = State(*next_state)
            episode.append((current_state, action, reward))
            if done:
                self._round += 1
                self._total_score += int(reward)
                
                self.sl.add_score(int(reward), self._round, refresh=self._round % PLOT_REFRESH == 0)
                
                break
            current_state = next_state
        
        return episode
    
    def mc_control_epsilon_greedy(self):
        self.P: Dict[State, StateActions] = defaultdict(StateActions)
        
        for k in range(self._N):
            episode = self.generate_episode()
            
            for i, (state, action, reward) in enumerate(episode):
                G = sum([_rew for _, _, _rew in episode[i:]])
                v = self.P[state].actions[action]
                v.returns += G
                v.state_cnt += 1
        
        return self.P


class Agent(ABC):
    @abstractmethod
    def action(self, state: State) -> int:
        ...


class RandomAgent(Agent):
    def __init__(self, action_space: int):
        self._action_space = action_space
    
    def action(self, state: State) -> int:
        return np.random.randint(self._action_space)


class PredictAgent(Agent):
    
    def __init__(self, policy: Dict[State, StateActions]):
        self._P = policy
    
    def action(self, state: State) -> int:
        return self._P[state].best_action


def evaluate_agent(agent: Agent, rounds: int):
    total_reward = 0
    for i in range(rounds):
        current_state = env.reset()
        current_state = State(*current_state)
        done = False
        while not done:
            action = agent.action(current_state)
            next_state, reward, done, _ = env.step(action)
            next_state = State(*next_state)
            
            if done:
                total_reward += reward
                break
            current_state = next_state
    
    return total_reward


if __name__ == '__main__':
    player = GreedyMCPlayer(2000)
    policy = player.mc_control_epsilon_greedy()
    random_agent = RandomAgent(nA)
    q_agent = PredictAgent(policy)
    n_rounds = 1000
    print(evaluate_agent(random_agent, n_rounds))
    print(evaluate_agent(q_agent, n_rounds))
