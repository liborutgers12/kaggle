# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from pylab import savefig
import matplotlib.pyplot as plt

PATH = './'
train = pd.read_csv(PATH + 'train.csv')
test = pd.read_csv(PATH + 'test.csv')
testPId = test['PassengerId']
print(train.head(10))

dataFull = [train, test]

for dataset in dataFull:
    # Fill missing Embarked feature with the most occurred value ( 'S' ).
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    # Fill missing Fare feature with the median value
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
    # Log transform to reduce the skewness
    dataset['Fare'] = np.log1p(dataset['Fare'])
    # Fill missing Age with random values
    ageAvg 	   = dataset['Age'].mean()
    ageStd 	   = dataset['Age'].std()
    ageNullCount = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(ageAvg - ageStd, ageAvg + ageStd, size=ageNullCount)
    #dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].fillna(train['Age'].median())
    dataset['Age'] = dataset['Age'].astype(int)

for dataset in dataFull:
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    # Mapping Fare
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

    # Mapping Age
    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']                           = 4

# Feature Selection
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']
train = train.drop(drop_elements, axis = 1)
test  = test.drop(drop_elements, axis = 1)
print (train.head(10))
print(test.head(10))

train = train.values
test  = test.values

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

X = train[0::, 1::]
y = train[0::, 0]

# Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits=5)

### SVC classifier
SVMCLF = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'],
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}

gsSVMCLF = GridSearchCV(SVMCLF,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs=-1, verbose = 1)

gsSVMCLF.fit(X, y)
svmCLF = gsSVMCLF.best_estimator_
print('Best score for SVM CLF: ', gsSVMCLF.best_score_)
svmPredicts = svmCLF.predict(test)

# RFC Parameters tunning
RFCLF = RandomForestClassifier()
## Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
              "max_features": [3],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}

gsRFCLF = GridSearchCV(RFCLF, param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs=-1, verbose = 1)
gsRFCLF.fit(X,y)
rfCLF = gsRFCLF.best_estimator_
print('Best score for RandomForestCLF: ', gsRFCLF.best_score_)
rfPredicts = rfCLF.predict(test)

# Gradient boosting classifier
GBCLF = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1]
              }
gsGBCLF = GridSearchCV(GBCLF,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs=-1, verbose = 1)
gsGBCLF.fit(X, y)
gboostCLF = gsGBCLF.best_estimator_
print('Best score for GradientBoostCLF: ', gsGBCLF.best_score_)
"""
aboostCLF = AdaBoostClassifier()
aboostCLF.fit(X,y)
aboostPredicts = aboostCLF.predict(test)
"""

votingC = VotingClassifier(estimators=[('rfc', rfCLF), ('svc', svmCLF), ('gbc', gboostCLF)], voting='soft', n_jobs=-1)
votingC = votingC.fit(X, y)
votingPredicts = votingC.predict(test)

predictions = svmPredicts       #votingPredicts
myPreditionsDF = pd.DataFrame({'PassengerId': testPId, 'Survived': predictions})
myPreditionsDF.to_csv('submission.csv', index=False)
