from pandas import DataFrame
from pandas import read_csv
import matplotlib.pyplot as plt
import numpy as np
import os
runs = False

if runs == True:
    # Read the csv files in the 10, 15 or 20 folder
    # The csv files are the output of the training

    folder = "20" # 10, 15 or 20
    path = os.path.join("output", "runs_graph" , folder)

    # Read the csv files in the folder
    files = os.listdir(path)
    files = [file for file in files if file.endswith(".csv")]

    # Read the csv files
    for file in files:
        if file.endswith("(1).csv"):
            sr = read_csv(os.path.join(path, file), delimiter="," )
        else: 
            r = read_csv(os.path.join(path, file))

    # Rewrite the data into arrays for manipulation
    steps_sr = sr["Step"].to_numpy()
    steps_r = r["Step"].to_numpy()

    srs = sr["Value"].to_numpy()
    rs = r["Value"].to_numpy()

    # Run a moving average on the values 
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

    aver=5
    # Plot both the moving average of the reward and the reward
    plt.plot( steps_r[:-aver+1],moving_average(rs, aver), label="Smoothed Reward" , color="red")
    plt.plot( steps_r, rs,label="Reward", color="purple")
    plt.legend()
    plt.xlabel("Step")
    plt.ylabel("Reward [-]")
    plt.title("Average reward over episodes for " + folder + "x" + folder + " AO system")
    plt.savefig("output/runs_graph/" + folder + "/reward"+ folder+ ".png")
    plt.show()


    # Plot both the moving average of the strehl ratio and strehl ratio
    plt.plot(steps_sr[:-aver+1],moving_average(srs, aver),  label="Smoothed Strehl Ratio", color="green")
    plt.plot( steps_sr, srs,label="Strehl Ratio" , color="blue")
    plt.legend()
    plt.xlabel("Step")
    plt.ylabel("Strehl Ratio [-]")
    plt.title("Strehl Ratio at end of episodes for " + folder + "x" + folder + " AO system")
    plt.savefig("output/runs_graph/" + folder + "/strehl_ratio"+ folder+ ".png")
    plt.show()

else: 
    # Read the csv files in the same folder
    # The csv files are the output of the training: one is the strehl ratio with RL and the other is the strehl ratio without RL for both 15x15 and 20x20

    folder_1 = "15" # 15 or 20
    folder_2 = "20" # 15 or 20
    path_1 = os.path.join("output", "runs_graph" , folder_1)
    path_2 = os.path.join("output", "runs_graph" , folder_2)

    linear_sr = "lin_sr.csv"
    rl_sr = "rl_sr.csv"

    # Read the csv files
    lin = []
    rl = []
    steps = []

    for path in [path_1, path_2]:
        i=0
        for file in [linear_sr, rl_sr]:
            df = read_csv(os.path.join(path, file))
            if file == linear_sr:
                lin.append(df["Value"].to_numpy())
            else:
                rl.append(df["Value"].to_numpy())
            if i==0:
                steps.append(df["Step"].to_numpy())
                i=1 
        
    # Plot both the strehl ratio for the 15x15 and strehl ratio for the 20x20
    plt.plot(steps[0], lin[0], label="Linear AO 15x15", color="red")
    plt.plot(steps[0], rl[0], label="RL AO 15x15", color="blue")
    plt.xlabel("Step")
    plt.ylabel("Strehl Ratio [-]")
    plt.title("Long Exposure Strehl Ratio at end of episodes for 15x15 AO system")
    plt.legend()
    plt.savefig("output/runs_graph/15/rl_lin_sr_15.png")
    plt.show()



    plt.plot(steps[1], lin[1], label="Linear AO 20x20", color="purple")
    plt.plot(steps[1], rl[1], label="RL AO 20x20", color="green")
    plt.xlabel("Step")
    plt.ylabel("Strehl Ratio [-]")
    plt.title("Long Exposure Strehl Ratio at end of episodes for 20x20 AO system")
    plt.legend()
    plt.savefig("output/runs_graph/20/rl_lin_sr_20.png")
    plt.show()

    