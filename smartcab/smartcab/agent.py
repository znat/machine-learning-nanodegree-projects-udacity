import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(
            env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.counter = 0
        self.gamma = 0.4
        self.epsilon = 0.05
        self.powert = 1
        self.previous_state = None
        self.previous_action = None
        self.q = defaultdict(dict)
        self.score = 0
        self.penalties = 0
        self.counter = 0
        self.df = self.df = pd.DataFrame(columns=('inputs', 'waypoint', 'action'))


    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.previous_state = None
        self.previous_action = None
        self.score = 0
        self.penalties = 0
        self.counter += 1



    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        # TODO: Select action according to your policy
        alpha = float(1) / (t**self.powert + 1)
        # TODO: Update state
        self.state = (self.next_waypoint, inputs['light'], inputs['oncoming'], inputs['right'], inputs['left'])
        action = take_action(self, self.state)
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        qsa = self.q[self.previous_state][self.previous_action] if self.previous_action in self.q[self.previous_state] else 0
        qsa_prime = self.q[self.state][action] if action in self.q[self.state] else 0
        self.q[self.state][action] = (1 - alpha) * qsa + alpha * (reward + self.gamma * qsa_prime)
        # print "deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        # TODO: update previous state and action
        self.previous_action = action
        self.previous_state = self.state

        # for statistics
        self.score += reward
        self.penalties += reward if reward < 0 else 0

        # Add a row with current state, waypoint and action to dataframe
        if self.counter == 100:
            self.df.loc[len(self.df)] = [inputs, self.next_waypoint, action]
            print (t, self.env.get_deadline(self), reward)

        # save DataFrame when done
        if self.counter == 100 and (reward >= 10 or t == self.env.get_deadline(self)):
            print "saving"
            self.df.to_csv("actions.csv")


def take_action(self, state):

    # pick one
    self.env.get_deadline(self)
    self.epsilon -= self.epsilon / 1000  # arbitrarily decreasing
    # print self.epsilon
    if random.uniform(0, 1) <= self.epsilon:
        return random.choice(self.env.valid_actions)
    else:
        # find q values associated to each valid action. If no q value is found use 0
        q_values = [self.q[state][a] if a in self.q[state] else 0 for a in self.env.valid_actions]
        # find the valid actions with highest q values
        best = [i for i in range(len(self.env.valid_actions)) if q_values[i] == max(q_values)]
        return self.env.valid_actions[random.choice(best)]

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

"""Below is the code for the simulation discussed in the report. For this simulation to work the code of environment.py must be changed to keep the results """

# def run():
#     """Run the agent for a finite number of trials."""
#     import matplotlib.pyplot as plt
#     start_time = time.time()
#     df = pd.DataFrame(columns=('values', 'gamma', 'powert', 'epsilon', 'average_penalty_last_10', 'average_score_last_10', 'average_outcome_last_10', 'last_outcome'))
#     for gamma in [x * 0.05 for x in range(0, 11)]:
#         for epsilon in [x * 0.05 for x in range(0, 11)]:
#             for powert in range(1, 3):
#                 n_instances = 11
#                 last_10_penalties = np.zeros((n_instances, 10))
#                 last_10_scores = np.zeros((n_instances, 10))
#                 last_10_outcomes = np.zeros((n_instances, 10))
#                 for val in range(0, n_instances):
#                     # Set up environment and agent
#                     e = Environment()  # create environment (also adds some dummy traffic)
#                     a = e.create_agent(LearningAgent)  # create agent
#                     a.gamma = gamma
#                     a.powert = powert
#                     a.epsilon = epsilon
#                     e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
#                     # NOTE: You can set enforce_deadline=False while debugging to allow longer trials
#
#                     # Now simulate it
#                     sim = Simulator(e, update_delay=0,
#                                     display=False)  # create simulator (uses pygame when display=True, if available)
#                     # NOTE: To speed up simulation, reduce update_delay and/or set display=False
#
#
#
#                     n = 100
#                     sim.run(n_trials=n)  # run for a specified number of trials
#                     # print sim.env.results
#                     indexes = np.array(range(0, n))
#                     scores = np.array([x[1] for x in sim.env.results])
#                     penalties = np.array([x[2] for x in sim.env.results])
#                     outcomes = np.array([x[0] for x in sim.env.results])
#                     last_10_penalties[val] = penalties[-10:]
#                     last_10_scores[val] = scores[-10:]
#                     last_10_outcomes[val] = outcomes[-10:]
#
#
#                 #averaging
#                 apenalty = np.average(np.average(last_10_penalties, axis=0), axis=0)
#                 ascore = np.average(np.average(last_10_scores, axis=0), axis=0)
#                 aoutcome = np.average(np.average(last_10_outcomes, axis=0), axis=0)
#                 values = "Eps:{:.2f} Gamma:{:.2f} PowT:{:d}".format(epsilon, gamma, powert)
#                 print values
#                 print [values, gamma, powert, epsilon, apenalty, ascore, aoutcome]
#                 df.loc[len(df)] = [values, gamma, powert, epsilon, apenalty, ascore, aoutcome, last_10_outcomes[9]]
#     df.to_csv("results.csv")
#     elapsed_time = time.time() - start_time
#     print "Simulation took {} seconds".format(elapsed_time)






    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
