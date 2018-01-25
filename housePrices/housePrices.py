import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

dataTrain = pd.read_csv('train.csv', index_col='Id')
dataTest = pd.read_csv('test.csv', index_col='Id')

targetTrain = dataTrain['SalePrice']
# Log-transform the target to make it normally distributed
targetTrain = np.log1p(targetTrain)
dataTrain = dataTrain.drop('SalePrice', axis=1)

# Join the train and test data together to carry out same type pre-processing
dataTrain['training_set'] = True
dataTest['training_set'] = False
dataFull = pd.concat([dataTrain, dataTest])
#log transform skewed numeric features:
numeric_feats = dataFull.dtypes[dataFull.dtypes != "object"].index
skewed_feats = dataTrain[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
dataFull[skewed_feats] = np.log1p(dataFull[skewed_feats])

#dataFull = dataFull.interpolate()               # Interpolate to fill the missing values
dataFull = dataFull.fillna(dataFull.mean())     #filling NA's with the mean of the column:
dataFull = pd.get_dummies(dataFull)             # Transform string categories using One-Hot encoding
# Separate the train and test data after the pre-processing
dataTrain = dataFull[dataFull['training_set']==True]
dataTrain = dataTrain.drop('training_set', axis=1)
dataTest = dataFull[dataFull['training_set']==False]
dataTest = dataTest.drop('training_set', axis=1)

from sklearn.model_selection import cross_val_score
def rmseCV(model):
    rmse = np.sqrt(-cross_val_score(model, dataTrain, targetTrain, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

# Train the RandomForestRegressor
rfRegressor = RandomForestRegressor(n_estimators=200, n_jobs=-1)
rfRegressor.fit(dataTrain, targetTrain)
rfRegressorPredicts = np.expm1(rfRegressor.predict(dataTest))
print('The rfRegressor achieves RMSE of ', rmseCV(rfRegressor).mean())

# Train the Lasso Regressor
from sklearn.linear_model import Lasso, LassoCV                # Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
lasso = LassoCV(alphas=[1, 0.1, 0.001, 0.00075, 0.0005, 0.0004], cv=5).fit(dataTrain, targetTrain)
print('The amount of penalization in LASSO chosen by cross validation is', lasso.alpha_)
# Make it more robust to outliers using the sklearn's Robustscaler() method on pipeline
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
lasso.fit(dataTrain, targetTrain)
print('The lasso model achieves RMSE of ', rmseCV(lasso).mean())
lassoPredicts = np.expm1(lasso.predict(dataTest))

# Train the GradientBoostingRegressor (using huber loss for robustness to outliers)
gboost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state=5)
gboost.fit(dataTrain, targetTrain)
gboostPredicts = np.expm1(gboost.predict(dataTest))
# print('The GradientBoostingRegressor achieves RMSE of ', rmseCV(gboost))

predictions = lassoPredicts # 1.0/3.0*(rfRegressorPredicts + lassoPredicts + gboostPredicts)
myPreditionsDF = pd.DataFrame({'Id': dataTest.index, 'SalePrice': predictions})
myPreditionsDF.to_csv('submission.csv', index=False)
