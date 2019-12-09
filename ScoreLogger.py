#!/usr/bin/env python
# coding:utf-8
# Author: ASU --<andrei.suiu@gmail.com>
# Purpose: 
# Created: 12/9/2019
from os.path import join
from typing import List

TRAIN_EPISODES = 100000

import threading
import time
from statistics import mean

import matplotlib
from plotly.subplots import make_subplots

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
import os
import csv
import numpy as np
import pandas as pd
from plotly import graph_objects as go

SCORES_DIR = "./scores"
SCORES_CSV_PATH = join(SCORES_DIR, "scores.csv")
SCORES_PNG_PATH = join(SCORES_DIR, "scores.png")
SOLVED_CSV_PATH = join(SCORES_DIR, ".solved.csv")
SOLVED_PNG_PATH = join(SCORES_DIR, "solved.png")
AVERAGE_SCORE_TO_SOLVE = 0.1
CONSECUTIVE_RUNS_TO_SOLVE = 200
PLOT_REFRESH = 100
TRAIN_EACH_STEP = True

# Stochastic training - ie train with batches of size 1, thus offering faster feedback loop of predicted Q-values
# Convergence is significantly faster and starts converging after 7 runs
# As higher the BATCH_SIZE, as higher might be difference in conv speed offered by this, but slowing-down the training
STOCHASTIC_TRAIN = False


class ScoreLogger:
    
    def __init__(self, env_name, success_rounds=50):
        self.scores = deque(maxlen=CONSECUTIVE_RUNS_TO_SOLVE)
        self.averages = deque(maxlen=CONSECUTIVE_RUNS_TO_SOLVE)
        self.last_20_avg = deque(maxlen=CONSECUTIVE_RUNS_TO_SOLVE)
        self._N = success_rounds
        self.last20_scores = deque(maxlen=self._N)
        self.exp_rates = deque(maxlen=CONSECUTIVE_RUNS_TO_SOLVE)
        self.time_hist = deque(maxlen=CONSECUTIVE_RUNS_TO_SOLVE)
        self.t1 = time.time()
        
        self.env_name = env_name
        if not os.path.exists(SCORES_DIR):
            os.makedirs(SCORES_DIR)
        if os.path.exists(SCORES_PNG_PATH):
            os.remove(SCORES_PNG_PATH)
        if os.path.exists(SCORES_CSV_PATH):
            os.remove(SCORES_CSV_PATH)
        self.cached_scores = []
    
    def show_graph(self, y: pd.DataFrame):
        self.fig = make_subplots(specs=[[{"secondary_y": True}]])
        self.fig.add_trace(go.Scatter(x=y.index, y=y.score, name="score"))
        self.fig.add_trace(go.Scatter(x=y.index, y=y.m, name="mean"))
        self.fig.add_trace(go.Scatter(x=y.index, y=y.m20, name=f"mean_last{self._N}"))
        # self.fig.add_trace(go.Scatter(x=y.index, y=y.expl, name="expl"))
        self.fig.add_trace(go.Scatter(x=y.index, y=y.time, name="time"), secondary_y=True)
        self.fig.show()
    
    def add_score(self, score: int, run: int, refresh=False):
        self.cached_scores.append(score)
        self.scores.append(score)
        self.last20_scores.append(score)
        last_20mean = mean(self.last20_scores)
        self.last_20_avg.append(last_20mean)
        mean_score = mean(self.scores)
        self.averages.append(mean_score)
        self.exp_rates.append(0)
        td = time.time() - self.t1
        self.time_hist.append(td)
        
        if refresh:
            self._save_csv(SCORES_CSV_PATH, self.cached_scores)
            self.cached_scores = []
            self._save_png(input_path=SCORES_CSV_PATH,
                           output_path=SCORES_PNG_PATH,
                           x_label="runs",
                           y_label="scores",
                           average_of_n_last=CONSECUTIVE_RUNS_TO_SOLVE,
                           show_goal=True,
                           show_trend=True,
                           show_legend=True)
            
            # Here we start a new thread as because of a bug in Plotly, sometimes the fig.show() doesn't return at all and process freezes
            y = pd.DataFrame(zip(self.scores, self.averages, self.last_20_avg, self.exp_rates, self.time_hist),
                             columns=['score', 'm', 'm20', 'expl', 'time'])
            
            threading.Thread(target=self.show_graph, args=(y,)).start()
        print(
            f"Run {run:3}: (avg: {mean_score:2.3f}, last{self._N}_avg: {last_20mean:2.3f}, "
            f"time: {td:3.1f})\n")
        if mean_score >= AVERAGE_SCORE_TO_SOLVE and len(self.scores) >= CONSECUTIVE_RUNS_TO_SOLVE:
            solve_score = run - CONSECUTIVE_RUNS_TO_SOLVE
            print("Solved in " + str(solve_score) + " runs, " + str(run) + " total runs.")
            self.cached_scores.append(solve_score)
            self._save_csv(SOLVED_CSV_PATH, self.cached_scores)
            self.cached_scores = []
            self._save_png(input_path=SOLVED_CSV_PATH,
                           output_path=SOLVED_PNG_PATH,
                           x_label="trials",
                           y_label="steps before solve",
                           average_of_n_last=None,
                           show_goal=False,
                           show_trend=False,
                           show_legend=False)
            exit()
    
    def _save_png(self, input_path, output_path, x_label, y_label, average_of_n_last, show_goal, show_trend,
                  show_legend):
        x = []
        y = []
        with open(input_path, "r") as scores:
            reader = csv.reader(scores)
            data = list(reader)
            j = 0
            for i in range(0, len(data)):
                if len(data[i]) == 0:
                    continue
                x.append(int(j))
                y.append(int(data[i][0]))
                j += 1
        
        plt.subplots()
        #plt.plot(x, y, label="score per run")
        
        average_range = average_of_n_last if average_of_n_last is not None else len(x)
        plt.plot(x[-average_range:], [np.mean(y[-average_range:])] * len(y[-average_range:]), linestyle="--",
                 label="last " + str(average_range) + " runs average")
        
        if show_goal:
            plt.plot(x, [AVERAGE_SCORE_TO_SOLVE] * len(x), linestyle=":",
                     label=str(AVERAGE_SCORE_TO_SOLVE) + " score average goal")
        
        if show_trend and len(x) > 1:
            trend_x = x[1:]
            z = np.polyfit(np.array(trend_x), np.array(y[1:]), 1)
            p = np.poly1d(z)
            plt.plot(trend_x, p(trend_x), linestyle="-.", label="trend")
        
        plt.title(self.env_name)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        
        if show_legend:
            plt.legend(loc="upper left")
        
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
    
    def _save_csv(self, path, scores: List[float]):
        if not os.path.exists(path):
            with open(path, "w"):
                pass
        scores_file = open(path, "a")
        with scores_file:
            writer = csv.writer(scores_file)
            for score in scores:
                writer.writerow([score])


if __name__ == '__main__':
    pass
