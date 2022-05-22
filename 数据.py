import os
import pandas as pd
import numpy as np

home_dir = os.getcwd()

mode = 'fine'
f_list = []
describes = ['SleepStage']
for i in range(len(describes)):
    exp_dir = os.path.join(home_dir, 'experiments_logs', describes[i])
    for root, dirs, files in os.walk(exp_dir):
        for file in files:
            if mode in file:
                if 'xlsx' in file:
                    f_list.append(os.path.join(root, file))

res_list = []
for i in range(len(f_list)):
    df = pd.read_excel(f_list[i])
    res_list.append(df.loc[4]["W"])
    # accuracy.append(df.loc[4]["W"])
    # cohen.append(df.loc[5]["W"])
    # f1.append(df["macro avg"][2])
arr_mean = np.mean(res_list)
arr_var = np.std(res_list)
print("Mean: " + str(arr_mean))
print("Var: " + str(arr_var))