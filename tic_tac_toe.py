import random
import numpy as np
from tqdm import tqdm
import sys
import copy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
import pandas as pd


import torch
import pyro
from pyro.contrib.autoname import scope

class State():
    def __init__(self, board, outcome=None):
        if len(board) != 9:
            raise ValueError("Not a valid board")
        for entry in board:
            if entry not in [None, "x", "o"]:
                raise ValueError("Invalid entry: " + str(entry))

        if outcome not in [None, "x", "o", "draw"]:
            raise ValueError("Invalid outcome: " + str(outcome))

        self.board = board
        self.outcome = outcome
    def update_outcome(self):
        # outcome only set once
        if self.outcome != None:
            return

        # check for draw
        all_filled = True
        for entry in self.board:
            if entry == None:
                all_filled = False
                break
        if all_filled:
            self.outcome = "draw"

        # check horizontals
        for player in ["x", "y"]:
            if ((self.board[0] == player and self.board[1] == player and self.board[2] == player) or
                (self.board[3] == player and self.board[4] == player and self.board[5] == player) or
                (self.board[6] == player and self.board[7] == player and self.board[8] == player) or
                (self.board[0] == player and self.board[3] == player and self.board[6] == player) or
                (self.board[1] == player and self.board[4] == player and self.board[7] == player) or
                (self.board[2] == player and self.board[5] == player and self.board[8] == player) or
                (self.board[0] == player and self.board[4] == player and self.board[8] == player) or
                (self.board[2] == player and self.board[4] == player and self.board[6] == player)):
                self.outcome = player

    def __hash__(self):
        return hash(str(self))
    def __eq__(self, other):
        return self.board == other.board and self.outcome == other.outcome
    def __str__(self):
        return str(self.board) + " " + str(self.outcome)
    def __repr__(self):
        return "State({})".format(self)

class Action():
    def __init__(self, player, location):
        if player not in ["x", "o"]:
            raise ValueError("Invalid player: " + str(player))
        if location not in range(9):
            raise ValueError("Invalid location: " + str(location))

        self.player = player
        self.location = location
    def __hash__(self):
        return hash(str(self))
    def __eq__(self, other):
        return self.player == other.player and self.location == other.location
    def __str__(self):
        return self.player + " " + str(self.location)
    def __repr__(self):
        return "Action(%s)" % self.name
    

def apply_transistion(state, action, normalized=False, **kwargs):
    # no changes after game is over
    if state.outcome != None:
        return state

    # illegal moves end in instant loss
    if state.board[action.location] != None:
        if action.player == "o":
            state.outcome = "x"
        else:
            state.outcome = "o"
        return state

    state.board[action.location] = action.player

    state.update_outcome()

# Reward Model
def reward_probability(outcome, player, normalized=False, **kwargs):
    # if we are unloaded things, give reward 100, otherwise give -1
    if outcome == player:
        return 10
    if outcome == other_player(player):
        return -10
    elif outcome == "draw":
        return 1
    else:
        return -1

def other_player(player):
    return "o" if player == "x" else "x"


def sample_outcome(state, action, player, max_depth=0, t=0):
    next_state = copy.deepcopy(state)
    apply_transistion(next_state, action)
    if next_state.outcome != None:
        return next_state.outcome

    # just assume we've lost if we haven't won yet, this is not very sofisticated
    # and resticts us to working with x as the main player
    if t >= max_depth:
        return "o"
    
    with scope(prefix="o{}{}".format(next_state, t)):
        return sample_outcome(next_state, 
                              sample_action(next_state, other_player(player), max_depth=max_depth, t=t+1),
                              other_player(player), max_depth=max_depth, t=t+1)


def sample_action(state, player, max_depth=0, t=0):
    action_weights = torch.zeros(9)
    for i in range(9):
        action = Action(player, i)
        with scope(prefix="a{}{}{}".format(state, t, action)):
            outcome = sample_outcome(state, action, player, max_depth=max_depth, t=t)
        expected_reward = reward_probability(outcome, player)
        action_weights[i] = np.exp(expected_reward)


    location = pyro.sample("a{}{}".format(state, t), pyro.distributions.Categorical(action_weights))
    return Action(player, location.item())
    

def sample_action_guide(state, player, max_depth=0, t=0):
    params = pyro.param(str(state) + player + "params", torch.ones(9), torch.distributions.constraints.positive)
    location = pyro.sample("a{}{}".format(state, t), pyro.distributions.Categorical(params))
    action = Action(player, location.item())

    for i in range(9):
        action = Action(player, i)
        with scope(prefix="a{}{}{}".format(state, t, action)):
            outcome = sample_outcome(state, action, player, max_depth=max_depth, t=t)

    return action

def Infer(svi, *args, num_steps=100, print_losses=True, **kwargs):
    losses = []
    for t in range(num_steps):
        losses.append(svi.step(*args, **kwargs))
        if print_losses:
            print("Loss [%d] = %.3f" % (t, losses[-1]))
    
    
if __name__ == "__main__":
    board = ["x", "o", "x", "o", None, None, None, None, None]
    state = State(board)

    max_depth = 1

    adam_params = {"lr": 0.005}
    optimizer = pyro.optim.Adam(adam_params)
    svi = pyro.infer.SVI(sample_action, sample_action_guide, optimizer, loss=pyro.infer.Trace_ELBO())
    Infer(svi, state, "x", max_depth=max_depth, print_losses=True)
    action_weights = pyro.param(str(state) + "x" + "params")
    df = pd.DataFrame({"actions": np.array(range(9)),
                       "action_weights": action_weights.detach().numpy()})
    sns.barplot(data=df,
                x="actions",
                y="action_weights")
    plt.title("state = %s" % state)
    plt.show()


