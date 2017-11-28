import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

train=pd.read_csv("/home/sohom/Desktop/AV_CrossSell_Oct17/train.csv",low_memory=False,header=0)
test=pd.read_csv("/home/sohom/Desktop/AV_CrossSell_Oct17/test.csv",low_memory=False,header=0)

train=pd.read_csv("/index/sohom_experiment/av_crossshell_oct17/train.csv",low_memory=False,header=0)
test=pd.read_csv("/index/sohom_experiment/av_crossshell_oct17/test.csv",low_memory=False,header=0)


set(train.columns) - set(test.columns)
#{'RESPONDERS'}

train['RESPONDERS'].value_counts()
#N    295388
#Y      4612

train['RESPONDERS']=train['RESPONDERS'].replace(to_replace={'N':0,'Y':1})

features=test.columns
test['RESPONDERS']=np.nan
train_test=train.append(test)

'''
fp=open("categorical_variable_distribution.txt",'w')
for i in train.columns:
	if train[i].dtype=='object':
		#print(train[i].value_counts())
		fp.write(str(train[i].value_counts())+"\n")


#OCCUP_ALL_NEW #All same so remove from test, train


for i in train.columns:
	if train[i].dtype!='object':
		print(i)

#EEG_TAG - All nan
'''
del train_test['OCCUP_ALL_NEW']
del train_test['EEG_TAG']
features=list(set(features)-set(['OCCUP_ALL_NEW','EEG_TAG']))

for i in train_test.columns:
	if train_test[i].dtype=='object':
		lbl=LabelEncoder()
		lbl.fit(list(train_test[i].values))
		train_test[i] = lbl.transform(list(train_test[i].values))

####Feature Engineering

X_train_all=train_test[0:len(train.index)]

X_train=X_train_all.sample(frac=0.80, replace=False)
X_valid=pd.concat([X_train_all, X_train]).drop_duplicates(keep=False)
X_test=train_test[len(train.index):len(train_test.index)]

dtrain = xgb.DMatrix(X_train[features], X_train['RESPONDERS'], missing=np.nan)
dvalid = xgb.DMatrix(X_valid[features], X_valid['RESPONDERS'], missing=np.nan)
dtest = xgb.DMatrix(X_test[features], missing=np.nan)

nrounds = 700
watchlist = [(dtrain, 'train')]

params = {"objective": "binary:logistic","booster": "gbtree", "nthread": 16, "silent": 1,
                "eta": 0.08, "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.7,
                "min_child_weight": 1,
                "seed": 2016, "tree_method": "exact"}

bst = xgb.train(params, dtrain, num_boost_round=nrounds, evals=watchlist, verbose_eval=20)

valid_preds = bst.predict(dvalid)
test_preds = bst.predict(dtest)

####Check function to find The evaluation metric for this competition is the maximum lift in the first decile which means capture maximum responders in the first decile from all the responders. Evaluating lift is widely used in campaign targeting problems. This tells us till which decile can we target customers for a specific campaign. Also, it tells you how much response do you expect from the new target base. 

#Step 1 : Calculate probability for each observation
valid_preds

#Step 2 : Rank these probabilities in decreasing order.
ranked_valid_probab=sorted(set(valid_preds),reverse=True)

#Step 3 : Build deciles with each group having almost 10% of the observations.
deciles=list(pd.qcut(x=ranked_valid_probab,q=10,labels=np.arange(10, 0, -1)))
indices = [i for i, x in enumerate(deciles) if x == 1]
reqd_probab=[ranked_valid_probab[i] for i in indices]
valid_indices=[i for i,x in enumerate(valid_preds) if x in reqd_probab]

#Step 4 : Calculate the response rate at each deciles for Good (Responders) ,Bad (Non-responders) corresponding to reqd_probab
predicted=[valid_preds[i] for i in valid_indices]
actual=[list(X_valid['label'])[i] for i in valid_indices]
print(f1_score(predicted,actual,average='weighted'))


submit = pd.DataFrame({'CUSTOMER_ID': test['CUSTOMER_ID'], 'RESPONDERS': test_preds})
submit[['CUSTOMER_ID','RESPONDERS']].to_csv("XGB3.csv", index=False)



###########################################################################################################################
#Plot variable importance using SRK code
def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i,feat))
    outfile.close()

create_feature_map(features)
bst.dump_model('xgbmodel.txt', 'xgb.fmap', with_stats=True)
importance = bst.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)
imp_df = pd.DataFrame(importance, columns=['feature','fscore'])
imp_df['fscore'] = imp_df['fscore'] / imp_df['fscore'].sum()
imp_df.to_csv("imp_feat.txt", index=False)


# create a function for labeling #
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.02*height,
                '%f' % float(height),
                ha='center', va='bottom')


#imp_df = pd.read_csv('imp_feat.txt')
labels = np.array(imp_df.feature.values)
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(12,6))
rects = ax.bar(ind, np.array(imp_df.fscore.values), width=width, color='y')
ax.set_xticks(ind+((width)/2.))
ax.set_xticklabels(labels, rotation='vertical')
ax.set_ylabel("Importance score")
ax.set_title("Variable importance")
autolabel(rects)
plt.savefig('dummy_feature_imp_diagram.png',dpi=1000)
plt.show()
###########################################################################################################################
