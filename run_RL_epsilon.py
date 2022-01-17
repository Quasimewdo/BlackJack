import blackjack_extended as bjk #The extended Black-Jack environment
import blackjack_base as bjk_base #The standard sum-based Black-Jack environment
from math import inf
import RL as rl
import sys
import os

import plotting as pl
import time
import matplotlib
import pandas as pd
from collections import defaultdict

matplotlib.style.use('ggplot')

if __name__ == "__main__":
    directory = "{}/data".format(sys.path[0])
    if not os.path.exists(directory):
        os.makedirs(directory)
    path_fun = lambda x: "{}/{}_{}.txt".format(directory,x, decks)
    # init constants
    omega = 0.77
    epsilon_list = [0.001,0.01,0.05,0.1,0.5,0.9,0.99]      # power decay of the learning rate
    n_sims = 10 ** 2    # Number of episodes generated
    epsilon = 0.05      # Probability in epsilon-soft strategy
    init_val = 0.0
    warmup = n_sims//10
    # Directory to save plots in
    plot_dir = "{}/figures/".format(sys.path[0])
    frame = pd.DataFrame(columns=['epsilon','decks','MC','Q1','Q2'])


    for epsilon in epsilon_list:
        for decks in [1,2,6,8,inf]:
            print("----- deck number equal to {} -----".format(decks))
            # set seed
            seed = 31233
            # init envs.
            env = bjk.BlackjackEnvExtend(decks=decks, seed=seed)
            sum_env = bjk_base.BlackjackEnvBase(decks=decks, seed=seed)


            print("----- Starting MC training on expanded state space -----")
            # MC-learning wit expanded state representation
            start_time_MC = time.time()
            Q_MC, MC_avg_reward, state_action_count, MC_avg_rewards = rl.learn_MC(
                env, n_sims, gamma = 1, epsilon = epsilon, init_val = init_val,
                episode_file=path_fun("hand_MC_state"), warmup=warmup)
            print("Number of explored states: " + str(len(Q_MC)))
            print("Cumulative avg. reward = " + str(MC_avg_reward))
            time_to_completion_MC = time.time() - start_time_MC


            print("----- Starting Q-learning on expanded state space -----")
            # Q-learning with expanded state representation
            start_time_expanded = time.time()
            Q, avg_reward, state_action_count, avg_rewards = rl.learn_Q(
                env, n_sims, gamma = 1, omega = omega, epsilon = epsilon, init_val = init_val,
                episode_file=path_fun("hand_state"), warmup=warmup)
            print("Number of explored states: " + str(len(Q)))
            print("Cumulative avg. reward = " + str(avg_reward))
            time_to_completion_expanded = time.time() - start_time_expanded



            print("----- Starting Q-learning for sum-based state space -----")
            # Q-learning with player sum state representation
            start_time_sum = time.time()
            sumQ, sum_avg_reward, sum_state_action_count, sum_avg_rewards = rl.learn_Q(
                sum_env, n_sims, omega = omega, epsilon = epsilon, init_val = init_val,
                episode_file=path_fun("sum_state"), warmup=warmup)
            time_to_completion_sum = time.time() - start_time_sum
            print("Number of explored states (sum states): " + str(len(sumQ)))
            print("Cumulative avg. reward = " + str(sum_avg_reward))

            print("Training time: \n " +
                  "Expanded state space MC: {} \n Expanded state space: {} \n Sum state space: {}".format(
                     time_to_completion_MC, time_to_completion_expanded, time_to_completion_sum))

            frame = frame.append(pd.Series([epsilon,decks,MC_avg_reward,avg_reward,sum_avg_reward],index = frame.columns),ignore_index=True)
            # Convert Q (extended state) to sum state representation and make 3D plots
            # Extended state MC-learning
            frame.to_csv("epsilon.csv")