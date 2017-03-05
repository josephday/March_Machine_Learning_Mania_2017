#rand_forr3.py



import pandas as pd
import numpy as np
#from sklearn.model_selection import GridSearchCV
#from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

TourneySeeds = pd.read_csv("TourneySeeds.csv")
sample_submission = pd.read_csv("sample_submission.csv")
TourneyCompactResults = pd.read_csv("TourneyCompactResults.csv")

def stringtonumber(id):
    return(int(x) for x in id.split("_"))
def extractfeatures(df):
    X_test = np.zeros(shape=(len(df),3), dtype=int)
    for i, row in df.iterrows():
        year,t1,t2 = stringtonumber(row.id)
        t1_wscore = TourneyCompactResults[(TourneyCompactResults.Season <= year) & 
                                                (TourneyCompactResults.Wteam == t1)].Wscore.values.mean()
        t2_wscore = TourneyCompactResults[(TourneyCompactResults.Season <= year) & 
                                                (TourneyCompactResults.Wteam == t2)].Wscore.values.mean()
        t1_seed = TourneySeeds[(TourneySeeds.Season <= year) & 
                                                (TourneySeeds.Team == t1)].Seed.str.extract('(\d\d)').astype(int).values.sum()
        t2_seed = TourneySeeds[(TourneySeeds.Season <= year) & 
                                                (TourneySeeds.Team == t2)].Seed.str.extract('(\d\d)').astype(int).values.sum()
        t1_win = len(TourneyCompactResults[(TourneyCompactResults.Season <= year) & 
                                                (TourneyCompactResults.Wteam == t1)].index)
        t2_win = len(TourneyCompactResults[(TourneyCompactResults.Season <= year) & 
                                                (TourneyCompactResults.Wteam == t2)].index)
        if np.isnan(t1_wscore):
            t1_wscore = 0
        if np.isnan(t2_wscore):
            t2_wscore = 0
        if np.isnan(t1_seed):
            t1_seed = 0
        if np.isnan(t2_seed):
            t2_seed = 0
        if np.isnan(t1_win):
            t1_win = 0
        if np.isnan(t2_win):
            t2_win = 0
        ScoreDistance = t1_wscore - t2_wscore
        SeedDistance = t1_seed - t2_seed
        WinDistance = t1_win - t2_win
        X_test[i,0] = ScoreDistance
        X_test[i,1] = SeedDistance
        X_test[i,2] = WinDistance
    return X_test

NCAA = pd.DataFrame()
tdata = TourneyCompactResults[["Season","Wteam","Lteam"]].copy()
tdata["status"] = 1
tdata1 = TourneyCompactResults[["Season","Lteam","Wteam"]].copy()
tdata1["status"] = 0
traindata = pd.concat([tdata,tdata1]).reset_index(drop=True)
traindata = traindata[["Season","Wteam","Lteam","status"]]
NCAA.target = traindata.status.values
traindata.drop(["status"],axis=1,inplace=True)
traindata["id"] = traindata.apply(lambda x: '_'.join(x.values.astype(str).tolist()), axis=1)
traindata.drop(["Season","Wteam","Lteam"], axis=1, inplace=True)
NCAA.data = extractfeatures(traindata)
NCAA.source = extractfeatures(sample_submission)

rf = RandomForestClassifier(random_state=1)
rfc = rf.fit(NCAA.data, NCAA.target)

preds = rfc.predict_proba(NCAA.source)
clipped_preds = np.clip(preds, 0.05, 0.95)
sample_submission.pred = clipped_preds[0:,:1].ravel()
sample_submission.to_csv('rand_forr6.csv', index=False)
