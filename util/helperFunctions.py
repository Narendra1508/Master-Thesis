# libraries for data storage using np array and panel data of pandas. also library for scietific and statistical calculations
import numpy as np
import pandas as pd
import scipy.stats as stats
from datetime import datetime

# to store objects as file
import pickle

# libraries for visaulization
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go


# function to read the csv file
def read_csv_data_as_char(filename, sep=';', chunk_size = 10000):
    chunkTemp = []

    for chunk in pd.read_csv(filename, sep = sep, dtype=str, chunksize=chunk_size, iterator=True, low_memory=False):
        # BUFFER THE CHUNKS IN ORDER TO LOAD YOUR WHOLE DATASET OR DO YOUR PROCESSING OVER A CHUNK AND STORE THE RESULT OF IT
        chunkTemp.append(chunk)

    #!  NEVER DO pd.concat OR pd.DataFrame() INSIDE A LOOP
    print("Concatinating chunks of data into a signle dataframe")
    df = pd.concat(chunkTemp)
    
    print("{} is loaded into dataframe".format(filename.split('/')[-1]))
    return(df)

# function to convert string to float
def convert_str_to_float(df, column_names):
    for column_name in column_names:
        if df[column_name].dtype != np.number:
            df[column_name] = df[column_name].str.replace(',', '').astype(float)
        else:
            print('column is already a number')
    return df

# function to convert string to date
def convert_str_to_date(df):
	df['Date'] = pd.to_datetime(df['Date'].str.replace(',',''), format='%b %d %Y')
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
def plot_series_with_plotly(df, time, col_name, start=0, end=None, rangeslider = True):
	fig = px.line(df[start:end], x=time, y=col_name, title='Volume of activated balancing services (+)[MWh]')
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
    #fig.show()

# function to plot calender heat map of time series data
def plot_heatmap(df, col_name, cmap = "YlGnBu", v_min=None, v_max=None):
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
