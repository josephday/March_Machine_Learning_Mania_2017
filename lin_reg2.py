
#Joseph Day
#3/5/17
#Cleaning the data and building a classifier 

import random
import csv
import pandas as pd
import pickle
import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier as gbc
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.linear_model import LinearRegression

df_tour = pd.read_csv('TourneyCompactResults.csv')
df_tour.drop(labels=['Daynum', 'Wscore', 'Lscore', 'Wloc', 'Numot'], inplace=True, axis=1)
df_teams = pd.read_csv('season_averages.csv')
df_seeds = pd.read_csv('TourneySeeds.csv')

def seed_to_int(seed):
    """Get just the digits from the seeding. Return as int"""
    s_int = int(seed[1:3])
    return s_int

df_seeds['n_seed'] = df_seeds.Seed.apply(seed_to_int)
df_seeds.drop(labels=['Seed'], inplace=True, axis=1) 


df_teams['fgp'] = df_teams['fgm']/df_teams['fga']
df_teams['3p'] = df_teams['fgm3']/df_teams['fga3']
df_teams['ftp'] = df_teams['ftm']/df_teams['fta']
#df_teams['seed'] = df_seeds['n_seed']

df_winseeds = df_seeds.rename(columns={'Team':'Wteam', 'n_seed':'win_seed'})
df_lossseeds = df_seeds.rename(columns={'Team':'Lteam', 'n_seed':'loss_seed'})
df_dummy = pd.merge(left=df_tour, right=df_winseeds, how='left', on=['Season', 'Wteam'])
#print(df_dummy.head)
df_concat1 = pd.merge(left=df_dummy, right=df_lossseeds, on=['Season', 'Lteam'])
df_concat1['seed_diff'] = df_concat1.win_seed - df_concat1.loss_seed
#print(df_concat.head)

#df_teams = pd.merge(left=df_teams, right=df_concat, how='left', on=['Season', 'Wteam'])

df_winteams = df_teams.rename(columns={'Team_Id':'Wteam', 'score':'w_score', 'fgp':'w_fgp', '3p':'w_3p',
                'ftp':'w_ftp', 'or':'w_or', 'dr':'w_dr', 'ast':'w_ast', 'to':'w_to', 'stl':'w_stl','blk':'w_blk',
                'pf':'w_pf'})
df_lossteams = df_teams.rename(columns={'Team_Id':'Lteam', 'score':'l_score','fgp':'l_fgp', '3p':'l_3p',
                'ftp':'l_ftp', 'or':'l_or', 'dr':'l_dr', 'ast':'l_ast', 'to':'l_to', 'stl':'l_stl','blk':'l_blk',
                'pf':'l_pf'})
df_dummy = pd.merge(left=df_tour, right=df_winteams, how='left', on=['Season', 'Wteam'])
df_concat = pd.merge(left=df_dummy, right=df_lossteams, on=['Season', 'Lteam'])
df_concat.drop(labels=['Unnamed: 0_x', 'Unnamed: 0_y', 'fgm_x', 'fgm_y', 'fga_x', 'fga_y',
                'fgm3_x', 'fgm3_y', 'fga3_x', 'fga3_y', 'ftm_x', 'ftm_y', 'fta_y', 'fta_x',
                'Team_Name_x', 'Team_Name_y'], inplace=True, axis=1) # This is the string label

df_concat = pd.merge(left=df_concat, right = df_concat1, how='left', on=['Season', 'Wteam', 'Lteam'])
#print(df_concat.head)
df_concat.drop(labels=['win_seed','loss_seed'], inplace=True, axis=1) 

df_concat['pt_diff'] = df_concat.w_score - df_concat.l_score
df_concat['ast_diff'] = df_concat.w_ast - df_concat.l_ast
df_concat['or_diff'] = df_concat.w_or - df_concat.l_or
df_concat['dr_diff'] = df_concat.w_dr - df_concat.l_dr
df_concat['to_diff'] = df_concat.w_to - df_concat.l_to
df_concat['stl_diff'] = df_concat.w_stl - df_concat.l_stl
df_concat['blk_diff'] = df_concat.w_blk - df_concat.l_blk
df_concat['pf_diff'] = df_concat.w_pf - df_concat.l_pf
df_concat['fgp_diff'] = df_concat.w_fgp - df_concat.l_fgp
df_concat['3p_diff'] = df_concat.w_3p - df_concat.l_3p
df_concat['ftp_diff'] = df_concat.w_ftp - df_concat.l_ftp
#df_concat['seed_diff'] = df_concat.w_seed - df_concat.l_seed

df_wins = df_concat[['pt_diff', 'ast_diff', 'or_diff', 'dr_diff', 'to_diff', 'stl_diff', 'blk_diff',
'pf_diff', 'fgp_diff', '3p_diff', 'ftp_diff', 'seed_diff', 'Season', 'Wteam', 'Lteam']]
df_wins['result'] = 1
df_losses = -df_concat[['pt_diff', 'ast_diff', 'or_diff', 'dr_diff', 'to_diff', 'stl_diff', 'blk_diff',
'pf_diff', 'fgp_diff', '3p_diff', 'ftp_diff', 'seed_diff', 'Season', 'Wteam', 'Lteam']]
df_losses['result'] = 0

df_for_predictions = pd.concat((df_wins, df_losses))

columns = df_for_predictions.columns.tolist()
columns = columns[:-4]

given_data = df_for_predictions.as_matrix(columns = ['seed_diff','fgp_diff','pt_diff'])
target_data = np.array(df_for_predictions['result'].tolist())

#X_train = df_for_predictions.seed_diff.values.reshape(-1,1)
#y_train = df_for_predictions.result.values
#X_train, y_train = shuffle(X_train, y_train)

clf = LinearRegression()
#params = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}
#clf = GridSearchCV(logreg, params, cv=None)
clf.fit(given_data, target_data)
print(clf.coef_, clf.intercept_)

df_sample_sub = pd.read_csv('sample_submission.csv')
n_test_games = len(df_sample_sub)

def get_year_t1_t2(id):
    """Return a tuple with ints `year`, `team1` and `team2`."""
    return (int(x) for x in id.split('_'))

X_test = np.zeros(shape=(n_test_games, 3))
for ii, row in df_sample_sub.iterrows():
    year, t1, t2 = get_year_t1_t2(row.id)
    # There absolutely must be a better way of doing this!
    t1_score = df_teams[(df_teams.Team_Id == t1) & (df_teams.Season == year)][['score','fgp']].values[0]
    t2_score = df_teams[(df_teams.Team_Id == t2) & (df_teams.Season == year)][['score','fgp']].values[0]
    t1_seed = df_seeds[(df_seeds.Team == t1) & (df_seeds.Season == year)].n_seed.values[0]
    t2_seed = df_seeds[(df_seeds.Team == t2) & (df_seeds.Season == year)].n_seed.values[0]
    for i in range(0,2):
        #diff_score.append(t1_score[i] - t2_score[i])
        X_test[ii, i] = t1_score[i] - t2_score[i]
    X_test[ii,2]= t1_seed - t2_seed


preds = clf.predict(X_test)
clipped_preds = np.clip(preds, 0.05, 0.95)
df_sample_sub.pred = preds
df_sample_sub.head()

df_sample_sub.to_csv('lin_reg2.csv', index=False)
