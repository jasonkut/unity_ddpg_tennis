import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

df = pd.read_csv('learning_stats.csv')

fig, ax = plt.subplots(1)
ax.plot(df.index, df['Max Score'], label='Max Score of Episode', color='blue')
ax.plot(df.index, df['Moving Avg. (100 eps)'], label='Moving Avg. (last 100 eps)', color='red')
ax.axhline(.50, linestyle='--', color='black', linewidth=.5)
ax.legend(loc='upper left')
ax.set_xlabel('Episode')
ax.set_ylabel('Tennis Env. Score')