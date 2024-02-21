import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from warnings import filterwarnings

filterwarnings('ignore')

df = pd.read_excel(r'Flight_Price_resources\Data_Train.xlsx')
df.dropna(inplace=True)

def change_to_datetime(col):
    df[col] = pd.to_datetime(df[col])

def extract_hour_min(df, col):
    df[col+'_hour'] = df[col].dt.hour
    df[col+'_min'] = df[col].dt.minute

def flight_dep_time(x):
    if (x>4) and (x<=8):
        return 'Early_Morning'
    elif (x>8) and (x<=12):
        return 'Morning'
    elif (x>12) and (x<=16):
        return 'Noon'
    elif (x>16) and (x<=20):
        return 'Evening'
    elif (x>20) and (x<=24):
        return 'Night'
    else:
        return 'Late_Night'
    
lt1 = ['Date_of_Journey', 'Dep_Time', 'Arrival_Time']

for i in lt1:
    change_to_datetime(i)

df['Journey_Date']  = df['Date_of_Journey'].dt.date
df['Journey_Month']  = df['Date_of_Journey'].dt.month
df['Journey_Year']  = df['Date_of_Journey'].dt.year

lt2 = ['Dep_Time','Arrival_Time']

for i in lt2:
    extract_hour_min(df, i)
    df.drop(i, axis=1, inplace=True)

df['Dep_Time_hour'].apply(flight_dep_time).value_counts().plot(kind='bar')
plt.show()