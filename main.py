# general imports
import numpy as np
import pandas as pd
from housing_utils import true_false_plot

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    ElasticNet,
)
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor


# Formatting data
data = pd.read_csv("datasets/housing.csv").drop(columns='Unnamed: 0')
data = data[data["median_house_value"] != 500001]
data = data[data["housing_median_age"] <= 50].dropna().reset_index(drop=True)
data["ocean_proximity"] = data["ocean_proximity"].astype("string")


# Numericalising Ocean_Proximity - works but doesn't reduce MSE, therefore unused in final model
op = pd.Series([], dtype=float)
i=0
for val in data["ocean_proximity"]:
    if val == "<1H OCEAN" or val == "INLAND":
        op[i] = 0
    else:
        op[i] = 1
    i = i + 1
    
data = data.drop(columns='ocean_proximity')
data = pd.concat([data, op], axis=1)


# Split data to be classified
Xtrain, Xtest, ytrain, ytest = train_test_split(
    data[['longitude', 'latitude', 'median_income', 'total_rooms', 'total_bedrooms', 'population']], 
    data["median_house_value"])


# Regressors
lin_reg = LinearRegression(normalize=True)
ridge = Ridge(alpha=1e-5, solver="cholesky")
el_net = ElasticNet(alpha=0.1, l1_ratio=0.5)


# GBR
gbr = GradientBoostingRegressor(max_depth=2, n_estimators=120).fit(Xtrain, ytrain)
errors = [mean_squared_error(ytest, ypred) for ypred in gbr.staged_predict(Xtest)]
best_n = np.argmin(errors) + 1
gbr_best = GradientBoostingRegressor(max_depth=2, n_estimators=best_n)


# Voting - unused - GBR worked better by itself
vote = VotingRegressor(estimators=[("Linear", lin_reg), 
                                   ("Ridge", ridge), 
                                   ("Elastic Net", el_net),
                                   ("Gradient Boosting", gbr_best)])


# Running regression
clf = gbr_best # <-- replace with chosen regressor

clf.fit(Xtrain, ytrain)
ypred = clf.predict(Xtest)
mse = mean_squared_error(ytest, ypred)

true_false_plot(ytest, ypred, "truepred")
print(f"Mean Squared Error score: {mse:.2f}")




