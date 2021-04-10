import os
import logging
# libraries for data storage using np array and panel data of pandas. also library for scientific and statistical calculations
import numpy as np
import pandas as pd
import scipy.stats as stats
from datetime import datetime

# to store objects as file
import pickle

# libraries for visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go

# changing working directory
work_dir = "D:/OneDrive - Jacobs University/Thesis/Master-Thesis"
data_dir = "D:/OneDrive - Jacobs University/Thesis/data"
os.chdir(work_dir)

# set file names to read
actual_generation_filename = data_dir + \
    "/Actual_generation_202001010000_202012312359.csv"
phase_angle_filename = data_dir + \
    "/All_2020-01-01T00_00_00.0-2020-12-31T23_59_59.9.csv"
afrr_filename = data_dir + \
    "/Automatic_Frequency_Restoration_Reserve_202001010000_202012312359.csv"

# function to read the csv file


def read_csv_data_as_char(filename, sep=';', chunk_size=10000):
    chunkTemp = []

    for chunk in pd.read_csv(filename, sep=sep, dtype=str, chunksize=chunk_size, iterator=True, low_memory=False):
        # BUFFER THE CHUNKS IN ORDER TO LOAD YOUR WHOLE DATASET OR DO YOUR PROCESSING OVER A CHUNK AND STORE THE RESULT OF IT
        chunkTemp.append(chunk)

    #!  NEVER DO pd.concat OR pd.DataFrame() INSIDE A LOOP
    print("Concatinating chunks of data into a signle dataframe")
    df = pd.concat(chunkTemp)

    print("{} is loaded into dataframe".format(filename.split('/')[-1]))
    return(df)

# function to directly read all datasets


def read_all_generation_data(read_pickle_file = False, chunk_size=10000):
    # Read actual generation data
	global actual_generation_filename
	print(actual_generation_filename)
	actual_generation_df = read_csv_data_as_char(actual_generation_filename)

    # convert str type to date
	actual_generation_df = convert_str_to_date(actual_generation_df)

    # convert str type to int
	actual_generation_df = convert_str_to_float(actual_generation_df, ["Wind offshore[MWh]", "Wind onshore[MWh]"])

    # create month columns for analysis
	actual_generation_df['Month'] = actual_generation_df.Datetime.dt.month
    
	keep_cols = ["Wind offshore[MWh]","Wind onshore[MWh]", "Datetime", "Month"]
	actual_generation_df = actual_generation_df[keep_cols]
	actual_generation_df['total_wind[MWh]'] = actual_generation_df["Wind offshore[MWh]"] + actual_generation_df["Wind onshore[MWh]"]
	del actual_generation_df['Wind offshore[MWh]']
	del actual_generation_df['Wind onshore[MWh]']
	
	actual_generation_df_pkl = open("actual_generation_df.pickle", "wb")
	pickle.dump(actual_generation_df, actual_generation_df_pkl)
	
	if(read_pickle_file):
	# open the saved pickle file
		actual_generation_df_pkl = open("actual_generation_df.pickle", "rb")
		actual_generation_df = pickle.load(actual_generation_df_pkl)
	return actual_generation_df

# function to read phase angle data between Bremen and Schondorf
# 246 --> Schondorf
# 248 --> Herzogenrath
# 252 --> Bremen
# 294 --> Dresden

def read_phase_angle_data(bus1="Bremen", bus2="Schondorf", read_pickle_file=False, chunk_size=10000):
	mapping = {"Schondorf": "246_phasors", "Herzogenrath": "248_phasors", "Bremen": "252_phasors", "Dresden": "294_phasors"}
	bus1 = mapping.get(bus1)
	bus2 = mapping.get(bus2)
	chunkTemp = []
	for chunk in pd.read_csv(phase_angle_filename, usecols=("time", bus1, bus2), skiprows=1, parse_dates=['time'], keep_date_col=True, chunksize=chunk_size, iterator=True, low_memory=False):
		# create month columns for analysis
		chunk['Month'] = chunk.time.dt.month
		chunk['Week'] = chunk.time.dt.week
		chunk = chunk.set_index(['time'])
		chunk["phase_diff"] = (chunk[bus1] - chunk[bus2])
		chunk["sin_delta"] = np.sin(np.deg2rad(chunk["phase_diff"]))
		chunk = chunk.resample('15 min').mean()
		chunk['Datetime'] = chunk.index
		chunkTemp.append(chunk)
		
	print("Concatinating chunks of data into a single dataframe")
	phase_angle_df = pd.concat(chunkTemp)
	phase_angle_df.rename(columns={'248_phasors': 'Schondorf', '294_phasors': 'Bremen'}, inplace=True)
	print("Phase angle data is loaded")
    # save the df as .pickle
	phase_angle_df_pkl = open("phase_angle_df.pickle", "wb")
	pickle.dump(phase_angle_df, phase_angle_df_pkl)
	
	if(read_pickle_file):
	# open the saved pickle file
		phase_angle_df_pkl = open("phase_angle_df.pickle", "rb")
		phase_angle_df = pickle.load(phase_angle_df_pkl)
	
	return phase_angle_df

# function to convert string to float


def convert_str_to_float(df, column_names):
    for column_name in column_names:
        if df[column_name].dtype != np.number:
            df[column_name] = df[column_name].str.replace(
                ',', '').astype(float)
        else:
            print('column is already a number')
    return df

# function to convert string to date


def convert_str_to_date(df):
    df['Date'] = pd.to_datetime(
        df['Date'].str.replace(',', ''), format='%b %d %Y')
    df['Datetime'] = df['Date'].astype(str) + " " + df['Time of day']
    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d %I:%M %p')
    df = df.set_index(pd.DatetimeIndex(df['Datetime']))
    #del df['Datetime']
    return df

# function to plot time series data


def plot_series(time, series, format="-", start=0, end=None, label=None):
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)

# function to plot time series data using plotly


def plot_series_with_plotly(df, time, col_name, start=0, end=None, rangeslider=True):
    fig = px.line(df[start:end], x=time, y=col_name,
                  title='Volume of activated balancing services (+)[MWh]')
    fig.update_xaxes(
        rangeslider_visible=rangeslider,
        rangeselector=dict(
            buttons=list([
                dict(count=9, label="9m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    fig.show()
    #fig = px.line(x=time[start:end], y=series[start:end])
    #fig.update_xaxes(rangeslider_visible = rangeslider)
    # fig.show()

# function to plot calender heat map of time series data


def plot_heatmap(df, col_name, cmap="YlGnBu", v_min=None, v_max=None):
    # create Day and Hour columns
    df['day'] = [i.dayofyear for i in df.index]
    df['hour'] = [i.hour for i in df.index]
    # group by month and year, get the average
    df = df.groupby(['day', 'hour']).mean()
    df = df[col_name].unstack(level=0)
    fig, ax = plt.subplots(figsize=(24, 10))
    sns.heatmap(df, cmap=cmap, vmin=v_min, vmax=v_max)
    plt.xlabel("Day of the Year", fontsize=16)
    plt.ylabel("Time of the Day", fontsize=16)


if __name__ == "__main__":
    # changing working directory
    work_dir = "D:/OneDrive - Jacobs University/Thesis/Master-Thesis"
    data_dir = "D:/OneDrive - Jacobs University/Thesis/data"
    os.chdir(work_dir)

    # set file names to read
    actual_generation_filename = data_dir + \
        "/Actual_generation_202001010000_202012312359.csv"
    phase_angle_filename = data_dir + \
        "/All_2020-01-01T00_00_00.0-2020-12-31T23_59_59.9.csv"
    afrr_filename = data_dir + \
        "/Automatic_Frequency_Restoration_Reserve_202001010000_202012312359.csv"

    mpl.rcParams['figure.figsize'] = (20, 7)
    mpl.rcParams['axes.grid'] = False

    # Gets or creates a logger
    logger = logging.getLogger(__name__)

    # set log level
    logger.setLevel(logging.WARNING)

    # define file handler and set formatter
    file_handler = logging.FileHandler('logfile.log')
    formatter = logging.Formatter(
        '%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    file_handler.setFormatter(formatter)

    # add file handler to logger
    logger.addHandler(file_handler)
