import pandas as pd
from pandas import read_csv
import pickle
from datetime import datetime

file = read_csv('ipl.csv')
# print(file.head())

# ----data cleaning
coloumns_to_remove = ['mid', 'venue', 'batsman', 'bowler', 'striker', 'non-striker']
file.drop(coloumns_to_remove, axis=1, inplace=True)

# print(file['bat_team'].unique())

consistent_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals', 'Mumbai Indians',
                    'Kings XI Punjab', 'Royal Challengers Bangalore', 'Delhi Daredevils', 'Sunrisers Hyderabad']

file = file[(file['bat_team'].isin(consistent_teams)) & (file['bowl_team'].isin(consistent_teams))]
file = file[file['overs'] >= 5.0]
# print(file['bat_team'].unique())
# print(file.head())
file['date'] = file['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
encoded_file = pd.get_dummies(data=file, columns=['bat_team', 'bowl_team'])
# print(encoded_file.head())


encoded_file = encoded_file[
    ['date', 'bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils', 'bat_team_Kings XI Punjab',
     'bat_team_Kolkata Knight Riders', 'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
     'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
     'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils', 'bowl_team_Kings XI Punjab',
     'bowl_team_Kolkata Knight Riders', 'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
     'bowl_team_Royal Challengers Bangalore', 'bowl_team_Sunrisers Hyderabad',
     'overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5', 'total']]
X_train=encoded_file.drop(labels='total',axis=1)[file['date'].dt.year<=2016]
X_test=encoded_file.drop(labels='total',axis=1)[file['date'].dt.year>=2017]

Y_train=encoded_file[encoded_file['date'].dt.year <=2016]['total'].values
Y_test=encoded_file[encoded_file['date'].dt.year>=2017]['total'].values

X_test.drop(labels='date',axis=True,inplace=True)
X_train.drop(labels='date',axis=True,inplace=True)



# ----- Model Building -----
# ----- Linear Regression Model -----

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

# Saving Model

filename='Linear-Model_for-IPL-Prediction.pkl'
pickle.dump(regressor,open(filename,'wb'))

# ------- Another model ---------

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,20,10,30,40,35]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(X_train,Y_train)

prediction=ridge_regressor.predict(X_test)
#print(X_test)
filename2="RidgeModel.pkl"
pickle.dump(ridge_regressor,open(filename2,'wb'))