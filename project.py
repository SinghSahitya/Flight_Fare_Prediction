import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from warnings import filterwarnings
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import pickle

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
    
def change_duration(x):
    if 'h' not in x:
        x = '0h '+ x
    elif 'm' not in x:
        x = x + ' 0m'

    return x
    
lt1 = ['Date_of_Journey', 'Dep_Time', 'Arrival_Time']

for i in lt1:
    change_to_datetime(i)

df['Journey_Date']  = df['Date_of_Journey'].dt.day
df['Journey_Month']  = df['Date_of_Journey'].dt.month
df['Journey_Year']  = df['Date_of_Journey'].dt.year

lt2 = ['Dep_Time','Arrival_Time']

for i in lt2:
    extract_hour_min(df, i)
    df.drop(i, axis=1, inplace=True)

df['Dep_Time_hour'].apply(flight_dep_time).value_counts().plot(kind='bar')
# plt.show()

df['Duration'] = df['Duration'].apply(change_duration)
df['Duration_Hour'] = df['Duration'].apply(lambda x : int(x.split(' ')[0][0:-1]))
df['Duration_Minute'] = df['Duration'].apply(lambda x : int(x.split(' ')[1][0:-1]))

df['Duration'] = df['Duration'].str.replace('h', '*60').str.replace(' ', '+').str.replace('m', '*1').apply(eval)
# print(df['Duration'])

sns.lmplot(x='Duration', y='Price', data=df)
# plt.show()

df['Route'] = df['Route'].str.replace(r'[^\x00-\x7F]', '-')
# print(df[df['Airline']=='Jet Airways'].groupby('Route').size().sort_values(ascending=False))

sns.boxplot(y='Price', x='Airline', data=df.sort_values('Price', ascending=False))
plt.xticks(rotation='vertical')
# plt.show()

cat_col = [col for col in df.columns if df[col].dtype=='object']
num_col = [col for col in df.columns if df[col].dtype!='object']

#one_hot_encoding
for city in df['Source']:
    df['Source'+'_'+city] = df['Source'].apply(lambda x: 1 if x==city else 0)

#target_encoding
airl = df.groupby('Airline')['Price'].mean().sort_values().index

dict_air = {key:index for index, key in enumerate(airl, 0)}
df['Airline'] = df['Airline'].map(dict_air)

df['Destination'].replace('New Delhi', 'Delhi', inplace=True)
dest = df.groupby('Destination')['Price'].mean().sort_values().index

dict_dest = {key:index for index, key in enumerate(dest, 0)}
df['Destination'] = df['Destination'].map(dict_dest)

#manual_label_encoding for ordinal values

stops = {'non-stop':0, '2 stops':2, '1 stop':1, '3 stops':3,'4 stops':4 }
df['Total_Stops'] = df['Total_Stops'].map(stops)

df.drop(columns=['Date_of_Journey', 'Route', 'Duration','Additional_Info','Journey_Year', 'Source'], axis=1, inplace=True)

#outliers

def plot(df, col):
    fig, (ax1, ax2, ax3) = plt.subplots(3,1)
    sns.displot(df[col], ax=ax1)
    sns.boxplot(df[col], ax=ax2)
    sns.displot(df[col], ax=ax3, kde=False)
    # plt.show()

plot(df, 'Price')

#replacing outliers with median

q1 = df['Price'].quantile(0.25)
q3 = df['Price'].quantile(0.75)
iqr = q3 - q1
maximum = q3 + 1.5*iqr
minimum = q1 - 1.5*iqr

df['Price'] = np.where(df['Price']>35000, df['Price'].median(), df['Price']) #using np to replace all values greater than 35k with median

#feature selection
X = df.drop('Price', axis=1)
Y= df['Price']

imp = mutual_info_regression(X, Y)
imp_df = pd.DataFrame(imp, index=X.columns)
imp_df.columns = ['importance']


#building model

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
ml_model = RandomForestRegressor()
ml_model.fit(X_train, Y_train)

def mape(y_true , y_pred):
    y_true , y_pred = np.array(y_true) , np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def predict(ml_model): # automating prediction for other models
    model = ml_model.fit(X_train , Y_train)
    print('Training score : {}'.format(model.score(X_train , Y_train)))
    y_predection = model.predict(X_test)
    print('predictions are : {}'.format(y_predection))
    print('\n')
    r2_score = metrics.r2_score(Y_test , y_predection)
    print('r2 score : {}'.format(r2_score))
    print('MAE : {}'.format(metrics.mean_absolute_error(Y_test , y_predection)))
    print('MSE : {}'.format(metrics.mean_squared_error(Y_test , y_predection)))
    print('RMSE : {}'.format(np.sqrt(metrics.mean_squared_error(Y_test , y_predection))))
    print('MAPE : {}'.format(mape(Y_test , y_predection)))
    

with open('rf_random.pickle', 'wb') as file:
    pickle.dump(ml_model, file)

with open('rf_random.pickle', 'rb') as model:
    forest = pickle.load(model)
    Y_pred = forest.predict(X_test)
    print('Accuracy Score: ',metrics.r2_score(Y_test, Y_pred)) 
    print('Percentage Error: ',mape(Y_test , Y_pred))

# Hypertuning for better score
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start =100 , stop=1200 , num=6)]

# Number of features to consider at every split
max_features = ["auto", "sqrt"]

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(start =5 , stop=30 , num=4)]

# Minimum number of samples required to split a node
min_samples_split = [5,10,15,100]

random_grid = {
    'n_estimators' : n_estimators , 
    'max_features' : max_features , 
    'max_depth' : max_depth , 
    'min_samples_split' : min_samples_split
}

rf_random = RandomizedSearchCV(estimator=ml_model , param_distributions=random_grid , cv=3 , n_jobs=-1 , verbose=2)
rf_random.fit(X_train , Y_train)

print('Best Params: ',rf_random.best_params_)
print('Best Estimators: ',rf_random.best_estimator_)
print('Best Score: ',rf_random.best_score_)

