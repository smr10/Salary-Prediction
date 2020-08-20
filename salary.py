import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix,accuracy_score

#UPLOADING CSV FILE

train = pd.read_csv('AdultData_train.csv')
test = pd.read_csv('AdultData_eval.csv',header=None)

#GETTING THE COLUMN HEADER OF TEST TO TRAIN

test.columns=train.columns

#CONVERTING THE CATEGORICAL VALUES TO NUMERICAL VALUE(ENCODING)




from sklearn import preprocessing




le_native=preprocessing.LabelEncoder()
le_native.fit(train["native"].values)
train["native"] = le_native.transform(train["native"])

le_native1=preprocessing.LabelEncoder()
le_native1.fit(test["native"].values)
test["native"] = le_native1.transform(test["native"])


le_workclass=preprocessing.LabelEncoder()
le_workclass.fit(train["workclass"].values)
train["workclass"] = le_workclass.transform(train["workclass"])

le_workclass1=preprocessing.LabelEncoder()
le_workclass1.fit(test["workclass"].values)
test["workclass"] = le_workclass1.transform(test["workclass"])


le_education=preprocessing.LabelEncoder()
le_education.fit(train["eduaction"].values)
train["eduaction"] = le_education.transform(train["eduaction"])

le_education1=preprocessing.LabelEncoder()
le_education1.fit(test["eduaction"].values)
test["eduaction"] = le_education1.transform(test["eduaction"])

le_marital=preprocessing.LabelEncoder()
le_marital.fit(train["marital-status"].values)
train["marital-status"] = le_marital.transform(train["marital-status"])

le_marital1=preprocessing.LabelEncoder()
le_marital1.fit(test["marital-status"].values)
test["marital-status"] = le_marital1.transform(test["marital-status"])

le_occupation=preprocessing.LabelEncoder()
le_occupation.fit(train["occupation"].values)
train["occupation"] = le_occupation.transform(train["occupation"])

le_occupation1=preprocessing.LabelEncoder()
le_occupation1.fit(test["occupation"].values)
test["occupation"] = le_occupation1.transform(test["occupation"])

le_sex=preprocessing.LabelEncoder()
le_sex.fit(train["sex"].values)
train["sex"] = le_sex.transform(train["sex"])

le_sex1=preprocessing.LabelEncoder()
le_sex1.fit(test["sex"].values)
test["sex"] = le_sex1.transform(test["sex"])

le_relationship=preprocessing.LabelEncoder()
le_relationship.fit(train["relationship"].values)
train["relationship"] = le_relationship.transform(train["relationship"])

le_relationship1=preprocessing.LabelEncoder()
le_relationship1.fit(test["relationship"].values)
test["relationship"] = le_relationship1.transform(test["relationship"])

le_race=preprocessing.LabelEncoder()
le_race.fit(train["race"].values)
train["race"] = le_race.transform(train["race"])

le_race1=preprocessing.LabelEncoder()
le_race1.fit(test["race"].values)
test["race"] = le_race1.transform(test["race"])


le_income=preprocessing.LabelEncoder()
le_income.fit(train["income"].values)
train["income"] = le_income.transform(train["income"])

le_income1=preprocessing.LabelEncoder()
le_income1.fit(test["income"].values)
test["income"] = le_income1.transform(test["income"])



#DATA CLEANING

train[train==' ?']=np.nan
train=train.dropna()
test[test==' ?']=np.nan
test=test.dropna()

def out_lier(df,colname):
    qua1 = df[colname].quantile(0.25)
    qua2 = df[colname].quantile(0.75)
    t=3

    z = np.abs(stats.zscore(df[colname]))
    df[colname][z>t] = qua2
    df[colname][z<-t] = qua1

out_lier(train,'age')
out_lier(train,'hours-per-week')

out_lier(test,'age')
out_lier(test,'hours-per-week')


Y_res1 = train['income'].values
Y_res2 = test['income'].values

x = train[['age', 'workclass','fnlwgt','eduaction', 'eduaction-num', 'marital-status', 'occupation', 'relationship', 'race','sex', 'capital-gain', 'capital-loss','hours-per-week']] .values
y = test[['age', 'workclass','fnlwgt','eduaction', 'eduaction-num', 'marital-status', 'occupation', 'relationship', 'race','sex', 'capital-gain', 'capital-loss','hours-per-week']] .values



#train = train.values
#test  = test.values

train = preprocessing.StandardScaler().fit(train).transform(train.astype(float))
test = preprocessing.StandardScaler().fit(test).transform(test.astype(float))





#FEATURE SELECTION

a=x
b=y
X_train,X_test,Y_train,Y_test=a,b,Y_res1,Y_res2

#NAIVE BAYES ALGORITHM

from sklearn.naive_bayes import GaussianNB
naive=GaussianNB()
naive.fit(X_train,Y_train)
n=naive.predict(X_test)
nacc=accuracy_score(Y_test,n)
print('naive accuracy: ',nacc)
print('naive confusion matrix: ',confusion_matrix(Y_test,n))

#KNN ALGORITHM

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,Y_train)
k=knn.predict(X_test)
kacc=accuracy_score(Y_test,k)
print('knn accuracy: ',kacc)
print('knn confusion matrix: ',confusion_matrix(Y_test,k))

#RANDOM FOREST ALGORITHM

from sklearn.ensemble import RandomForestClassifier
ran=RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=2)
ran.fit(X_train,Y_train)
r=ran.predict(X_test)
racc=accuracy_score(Y_test,r)
print('random forest accuracy: ',racc)
print('randon forest confusion matrix: ',confusion_matrix(Y_test,r))


#DECISION TREE ALGORITHM

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion="entropy", max_depth = 5)
tree.fit(X_train,Y_train)
t=tree.predict(X_test)
tacc=accuracy_score(Y_test,t)
print('decision tree accuracy: ',tacc)
print('decision tree confusion matrix: ',confusion_matrix(Y_test,t))

#SUPPORT VECTOT MACHINE

from sklearn import svm
sv = svm.SVC(kernel='rbf')
sv.fit(X_train, Y_train)
s=sv.predict(X_test)
sacc=accuracy_score(Y_test,s)
print('svm accuracy: ',sacc)
print('svm confusion matrix: ',confusion_matrix(Y_test,s))


#LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(C=0.01, solver='liblinear')
LR.fit(X_train,Y_train)
L=LR.predict(X_test)
lracc=accuracy_score(Y_test,L)
print('LG accuracy: ',lracc)
print('LG confusion matrix: ',confusion_matrix(Y_test,L))


#BARPLOT

import matplotlib.pyplot as plt

objects=['NAIVEBAYES','KNN','RANDOMFOREST','DECISIONTREE','SVM']
y_pos=np.arange(len(objects))
nacc=nacc*100
kacc=kacc*100
racc=racc*100
tacc=tacc*100
sacc=sacc*100
accuracy=[nacc,kacc,racc,tacc,sacc]

plt.bar(y_pos, accuracy, align='center', alpha=0.5,color=('red','blue','green','orange','purple'))
plt.xticks(y_pos, objects)
plt.xlabel('ALGORITHM')
plt.ylabel('ACCURACY')
plt.title('Comparisson Between Algorithms')

plt.show()

