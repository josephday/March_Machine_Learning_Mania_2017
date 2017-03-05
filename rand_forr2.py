#Uptake Data Science Case Study
#Joseph Day
#1/7/17
#Cleaning the data and building a classifier for 
#probability of response

import random
import csv
import pandas as pd
import pickle
import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier as gbc


df_tour = pd.read_csv('TourneyCompactResults.csv')
df_tour.drop(labels=['Daynum', 'Wscore', 'Lscore', 'Wloc', 'Numot'], inplace=True, axis=1)
df_teams = pd.read_csv('season_averages.csv')

df_teams['fgp'] = df_teams['fgm']/df_teams['fga']
df_teams['3p'] = df_teams['fgm3']/df_teams['fga3']
df_teams['ftp'] = df_teams['ftm']/df_teams['fta']
#print(df_teams.head)

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

df_wins = df_concat[['pt_diff', 'ast_diff', 'or_diff', 'dr_diff', 'to_diff', 'stl_diff', 'blk_diff',
'pf_diff', 'fgp_diff', '3p_diff', 'ftp_diff', 'Season', 'Wteam', 'Lteam']]
df_wins['result'] = 1
df_losses = -df_concat[['pt_diff', 'ast_diff', 'or_diff', 'dr_diff', 'to_diff', 'stl_diff', 'blk_diff',
'pf_diff', 'fgp_diff', '3p_diff', 'ftp_diff', 'Season', 'Wteam', 'Lteam']]
df_losses['result'] = 0

df_for_predictions = pd.concat((df_wins, df_losses))

columns = df_for_predictions.columns.tolist()
columns = columns[:-4]

df_for_test = df_for_predictions[df_for_predictions['Season'] >= 2013]
df_for_predictions = df_for_predictions[df_for_predictions['Season'] <= 2012]


given_data = df_for_predictions.as_matrix(columns = columns)
target_data = np.array(df_for_predictions['result'].tolist())


given_test = df_for_test.as_matrix(columns = columns)
#target_test = np.array(df_for_test['result'].tolist())

clf = gbc(n_estimators = 10)
clf = clf.fit(given_data, target_data)

df_sample_sub = pd.read_csv('sample_submission.csv')
n_test_games = len(df_sample_sub)

def get_year_t1_t2(id):
    """Return a tuple with ints `year`, `team1` and `team2`."""
    return (int(x) for x in id.split('_'))

X_test = np.zeros(shape=(n_test_games, 11))
for ii, row in df_sample_sub.iterrows():
    year, t1, t2 = get_year_t1_t2(row.id)
    # There absolutely must be a better way of doing this!
    t1_score = df_teams[(df_teams.Team_Id == t1) & (df_teams.Season == year)][['score','ast','or','dr','to','stl','blk','pf','fgp','3p','ftp']].values[0]
    t2_score = df_teams[(df_teams.Team_Id == t2) & (df_teams.Season == year)][['score','ast','or','dr','to','stl','blk','pf','fgp','3p','ftp']].values[0]
    #diff_score=[]
    for i in range(0,11):
        #diff_score.append(t1_score[i] - t2_score[i])
        X_test[ii, i] = t1_score[i] - t2_score[i]
    
    


preds = clf.predict_proba(X_test)
#clipped_preds = np.clip(preds, 0.05, 0.95)
df_sample_sub.pred = preds
df_sample_sub.head()

df_sample_sub.to_csv('rand_forr1.csv', index=False)

def calc_errors(forest, test_file):

    '''Calculates typeI and typeII errors,
    useful in ensuring not too many of either
    '''

    clf = forest
    test_given, test_target = process(test_file, 'responded')
    predictions = clf.predict(test_given)
    typeI = 0
    typeII = 0
    total = 0
    for i in range(1, len(test_target)):
        if predictions[i] == test_target[i]:
            pass
        else:
            if predictions[i] == 'yes':
                typeI += 1
            else: 
                typeII += 1
        total+=1
    typeI = typeI/total
    typeII = typeII/total
    return typeI,typeII


def roc_auc(forest, test_file):

    '''Returns area under ROC curve for 
    classifier provided / built from test data provided
    '''

    clf = forest
    test_given, test_target = process(test_file, 'responded')
    predictions = clf.predict_proba(test_given)
    test_target = pd.DataFrame(test_target)
    test_target.replace(['yes', 'no'], [1,0], inplace=True)
    test_target = np.array(test_target[0].tolist())
    predictions = predictions[:,1]
    return roc_auc_score(test_target, predictions)


def score(forest, test_file):

    '''Function takes in a forest/gbc and a test file
    and returns various metrics for evaluation
    '''

    roc_score = roc_auc(forest, test_file)

    test_given, test_target = process(test_file, 'responded')
    predictions = forest.predict(test_given)
    
    typeI, typeII = calc_errors(forest, test_file)
    accuracy = accuracy_score(test_target, predictions)
    
    print("Roc_Auc_Score is ", roc_score)
    print("Type I Error Rate is ", typeI)
    print("Type II Error Rate is ", typeII)
    print("Accuracy Score is ", accuracy)
