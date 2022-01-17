import blackjack_extended as bjk #The extended Black-Jack environment
import blackjack_base as bjk_base #The standard sum-based Black-Jack environment
import RL as rl

import sys
import os
import time
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

def train_model():
    """Train the model, return Q. Use fixed values """
    directory = "{}/data".format(sys.path[0])
    if not os.path.exists(directory):
        os.makedirs(directory)
    path_fun = lambda x: "{}/{}_{}.txt".format(directory,x, decks)

    sum_env = bjk_base.BlackjackEnvBase(decks = 8, seed = 31233)

    n_sims = int(1e6)
    # Q-learning with player sum state representation
    sumQ, sum_avg_reward, sum_state_action_count, sum_avg_rewards = rl.learn_Q(
        sum_env, n_sims, omega = 0.77, epsilon = 0.05, init_val = 0.0,
        episode_file=path_fun("sum_state"), warmup = n_sims//10)

    print("Model trained")
    print("Number of explored states (sum states): " + str(len(sumQ)))
    print("Cumulative avg. reward = " + str(sum_avg_reward))

    return sumQ

# def calculate_best_actions_alt():
#     for dealer_open in dealer_open_pos:
#         for card1 in range(1,11):
#             for card2 in range(1,11):
#                 #construct a new deck
#                 sum_env.construct_deck()
#                 if sum_env.can_be_drawn([dealer_open, card1, card2]): #I.e. is possible to draw these 3 cards
#                     state = sum_env.set_state([card1,card2],dealer_open)
#                     sum_player[state[0]] += 1
#                     actions
#
#     deck = sum_env.cards_in_deck
#     return
def calculate_best_actions(sumQ, savetype = 'numpy', save = True):
    """ Calculate the best actions given each state.
    savetype denotes how the values should be saved. Options: all in one file('df'), separate files('numpy') or 'both'
    if save == False, the values will not be saved, simply printed"""
    #state has form (12, 1, False): (env.sum_hand(self.player), env.dealer_show_cards(), env.usable_ace(self.player))
    #print("State ", state, type(state))
    if save:#check that the correct folder exists, otherwise create folder
        directory = "{}/saved_optimal".format(sys.path[0])
        if not os.path.exists(directory):
            os.makedirs(directory)

    sum_player_pos = range(22) #posible player sums
    sum_dealer_pos = range(11) #posible dealer sums
    ace_pos = [True, False]

    if (savetype == 'numpy' or savetype == 'both'):
        actions = np.full([len(sum_player_pos),len(sum_dealer_pos)], np.nan)
        Qmax = np.full([len(sum_player_pos),len(sum_dealer_pos)], np.nan)
        actions_string = np.full([len(sum_player_pos),len(sum_dealer_pos)], ' ')

        for ace_TF in ace_pos:
            for sum_dealer in sum_dealer_pos:
                for sum_player in sum_player_pos:
                    state = (sum_player,sum_dealer, ace_TF)
                    act = np.argmax(sumQ[state]) # Take the best possible action
                    Qmax[sum_player,sum_dealer] = sumQ[state][act]
                    actions[sum_player,sum_dealer] = act
            if save:
                if ace_TF:
                    string = "T"
                else:
                    string = "F"
                csvfile = "saved_optimal/actions_ace"+ string + ".csv"
                np.savetxt(csvfile, actions, delimiter = ',')

                csvfile_Qmax = "saved_optimal/Qmax_ace"+ string + ".csv"
                np.savetxt(csvfile_Qmax, Qmax, delimiter = ',')
            else:
                print(actions)

    if (savetype == 'df' or savetype == 'both'):
        max_index = len(sum_dealer_pos)*len(sum_player_pos)*len(ace_pos)
        deal_list = ['']*max_index
        play_list = ['']*max_index
        act_list = ['']*max_index
        act_TF_list = [0]*max_index
        ace_list = ['']*max_index

        i = 0
        for ace_TF in ace_pos:
            for sum_dealer in sum_dealer_pos:
                for sum_player in sum_player_pos:
                    state = (sum_player,sum_dealer, ace_TF)
                    act = np.argmax(sumQ[state]) # Take the best possible action
                    #values for DataFrame
                    if act:
                        #actions_string[sum_player,sum_dealer] = 'hit'
                        act_list[i] = 'hit'
                    else:
                        #actions_string[sum_player,sum_dealer] = 'stay'
                        act_list[i] = 'stay'
                    deal_list[i] = sum_dealer
                    play_list[i] = sum_player
                    #act_list[i] = actions_string[sum_player,sum_dealer]
                    ace_list[i] = ace_TF
                    act_TF_list[i] = act
                    i+=1


        df = pd.DataFrame({'deal_sum': deal_list,'play_sum': play_list, 'action':act_list, 'action_TF': act_TF_list, 'usable_ace(TF)':ace_list } )
        if save:
            df.to_csv('saved_optimal/actionDF.csv')
        else:
            print(df)

    return

def show_action_df():
    df = pd.read_csv('actionDF.csv', index_col = 0)
    #sub_df = df[(df['usable_ace(TF)'] == ace_TF)]
    #fig = px.imshow(df)
    #print(sub_df)

    fig = px.density_heatmap(df, x='play_sum', y='deal_sum', z = 'action_TF',nbinsx=35, nbinsy=28, facet_col="usable_ace(TF)", histfunc="sum")
    fig.show()
    return

def load_action_Qmax(ace_TF):
    """Helpfunction to load actions/Qmax from csvfiles. Used by show_action_plotly, show_action_matplot"""
    if ace_TF:
        string = "T"
    else:
        string = "F"

    directory = "{}/".format(sys.path[0])

    csvfile = "saved_optimal/actions_ace"+ string + ".csv"
    if not os.path.isfile(directory+csvfile):
        raise ValueError("Can not load {} \n file does not exist".format(csvfile))
    actions= np.genfromtxt(csvfile, delimiter=',', dtype=None)

    Qfile = "saved_optimal/Qmax_ace"+ string + ".csv"
    if not os.path.isfile(directory+csvfile):
        raise ValueError("Can not load {} \n file does not exist".format(Qfile))
    Qmax = np.genfromtxt(Qfile, delimiter=',', dtype=None)

    return actions, Qmax

def show_action_matplot(ace_TF):
    """Show optimal action using matplotlib"""
    actions, Qmax = load_action_Qmax(ace_TF)
    plt.matshow(actions)

    #plt.axis('equal')
    plt.axis([0.5,10.5,3.5,21.5])
    #plt.grid(visible = True)
    plt.xlabel("Dealers open card")
    plt.ylabel("Sum of players hand")

    plt.show()
    return

def show_action_plotly(ace_TF, save = False, Q_color = False):
    """Visualize the best policies(from calculated files) using plotly.
    ace_TF (True/False) denotes uasble ace,
    save == True: saves heatmap as png. save == False: shows plot in interactive window
    Q_color == False: color is optimal action. Q_color == True: color is optimal Q-value"""
    actions, Qmax = load_action_Qmax(ace_TF)

    #get the strings for annotation
    actions_string = np.full([len(actions),len(actions[0])], ' ', dtype = 'U5')
    for i in range((len(actions))):
        for j in range(len(actions[0])):
            if actions[i,j]:
                actions_string[i, j] = 'hit'
                #print("hit when player %d, dealer %d" %(i,j))

            elif not actions[i,j]:
                actions_string[i, j] = 'stay'

    #create annotated heatmap
    #set x/ylabels
    x = list(range(len(actions[0])))
    y = list(range(len(actions)))

    if Q_color: #Want the color to represent Q-value
        zero = -np.min(Qmax)/(np.max(Qmax)-np.min(Qmax)) #find zero for colorscale
        #fig = ff.create_annotated_heatmap(Qmax, x = x, y =y, annotation_text=actions_string, showscale = True,  colorscale='ReYlGn')
        fig = ff.create_annotated_heatmap(
            Qmax,
            x = x,
            y =y,
            annotation_text=actions_string,
            showscale = True,
            colorscale=[[0,'red'],[zero, 'white'],[1,'green']],
            )
        fig.update_layout(legend = dict(title= "Q-value"), showlegend = True)
    else:
        fig = ff.create_annotated_heatmap(actions, x = x, y =y, annotation_text=actions_string, colorscale=[[0, 'lightgreen'], [1, 'lightyellow']])

    fig.update_layout(
        #title = "usable ace = "+ str(ace_TF),
        font = dict(size = 14),
        width = 600,
        yaxis_title = "Sum of players hand",
        xaxis_title = "Dealers open card"
        )
    fig.update_xaxes(
        gridwidth = 1,
        range = [0.5,10.5]
    )
    if ace_TF:
        fig.update_yaxes(
            gridwidth = 1,
            range = [12.5,21.5]
        )
    else:
        fig.update_yaxes(
            gridwidth = 1,
            range = [3.5,21.5]
        )
    if save:
        if ace_TF:
            string = "T"
        else:
            string = "F"
        if Q_color:
            figname = "saved_optimal/open_actionsQ_ace"+ string +".png"
        else:
            figname = "saved_optimal/open_actions_ace"+ string +".png"
        fig.write_image(figname)
    else:
        fig.show()
    return


if __name__ == "__main__":
    # #train model
    # sumQ = train_model() #add variables if wanted
    # #get best actions
    # calculate_best_actions(sumQ, savetype = 'numpy', save = True)

    #visualize actions
    for ace_TF  in [True, False]:
        show_action_plotly(ace_TF, save = True, Q_color=True)
        #show_action_matplot(ace_TF)
    #show_action_df()
