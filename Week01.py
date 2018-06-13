# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 21:17:56 2018

@author: akulap
"""

import pandas as pd
from sklearn import model_selection as ms
import math
import numpy as np

#Get raw data
tv_df = pd.read_csv('https://raw.githubusercontent.com/ncooper76/MSDA/master/TV_Show_Preferences.csv')
tv_df = tv_df.drop(["Timestamp"],axis=1)
tv_df.head()

#Add user column and unpivot the dataframe
tv_df['user'] = tv_df.index + 1
tv_df_ColToRow = pd.melt(tv_df, id_vars='user')
tv_df_ColToRow.columns = ['user', 'tv_show','rating']

#Get number of users
total_user = tv_df_ColToRow.user.unique().shape[0]

#Get number of tv shows
total_shows = tv_df_ColToRow.tv_show.unique().shape[0]

#Replace nan with zero
tv_df_ColToRow = tv_df_ColToRow.fillna(0)

#Get clean dataset
tv_df_ColToRow_NoMissing = tv_df_ColToRow[tv_df_ColToRow.rating != 0]

#Create training and testing dataset
train_data, test_data = ms.train_test_split(tv_df_ColToRow_NoMissing, test_size=0.30, random_state=345)

train_data_pivot = train_data.pivot(index='user', columns='tv_show', values='rating')
test_data_pivot = test_data.pivot(index='user', columns='tv_show', values='rating')

#tv_shows
tv_shows = list(train_data.columns.values)

#Raw mean for training dataframe
df_rawmean = train_data['rating'].mean()

#RMSE train
train_data['ratingMeanDiffSq'] = (train_data['rating'] - df_rawmean)**2
train_rmse = math.sqrt(train_data['ratingMeanDiffSq'].sum() / len(train_data))
print(train_rmse)

#RMSE test
test_data['ratingMeanDiffSq'] = (test_data['rating'] - df_rawmean)**2
test_rmse = math.sqrt(test_data['ratingMeanDiffSq'].sum() / len(test_data))
print(test_rmse)

#Raw mean for user - user bias
user_rawmean = pd.DataFrame(train_data.groupby(['user'])['rating'].mean() - df_rawmean)
user_rawmean.columns = ['userBias']

#Raw mean for tv_show -show bias
tv_show_rawmean = pd.DataFrame(train_data.groupby(['tv_show'])['rating'].mean() - df_rawmean)
tv_show_rawmean.columns = ['showBias']
tv_show_rawmean['tv_show'] = tv_show_rawmean.index
tv_show_bias = tv_show_rawmean

tv_show_rawmean = tv_show_rawmean.reset_index(drop=True)
tv_show_rawmean['id'] = 'showBias'
tv_show_rawmean = tv_show_rawmean.pivot(index = 'id', columns='tv_show', values='showBias')
tv_show_rawmean['userBias'] = np.NaN

#append user bias
train_data_pivot = pd.concat([train_data_pivot, user_rawmean], axis=1)

#append tv_show bias
train_data_pivot = pd.concat([train_data_pivot, tv_show_rawmean], axis=0)


#Baseline predictors Training
user_rawmean['user'] = user_rawmean.index
train_data = pd.merge(train_data, user_rawmean, on='user', how='inner')
train_data = pd.merge(train_data, tv_show_bias, on='tv_show', how='inner')
train_data['baseline_pred'] = (df_rawmean + train_data['userBias'] + train_data['showBias'])
train_data.loc[train_data['baseline_pred'] < 1, 'baseline_pred'] = 1
train_data.loc[train_data['baseline_pred'] > 5, 'baseline_pred'] = 5
train_data['rmse_bl'] = (train_data['rating'] - train_data['baseline_pred'])**2
train_data_rmse = math.sqrt(train_data['rmse_bl'].sum() / len(train_data))
print(train_data_rmse)

#Baseline predictors Testing
test_data = pd.merge(test_data, user_rawmean, on='user', how='inner')
test_data = pd.merge(test_data, tv_show_bias, on='tv_show', how='inner')
test_data['baseline_pred'] = (df_rawmean + test_data['userBias'] + test_data['showBias'])
test_data.loc[test_data['baseline_pred'] < 1, 'baseline_pred'] = 1
test_data.loc[test_data['baseline_pred'] > 5, 'baseline_pred'] = 5
test_data['rmse_bl'] = (test_data['rating'] - test_data['baseline_pred'])**2
test_data_rmse = math.sqrt(test_data['rmse_bl'].sum() / len(test_data))
print(test_data_rmse)

