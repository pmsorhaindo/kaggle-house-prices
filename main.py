# import numpy as np
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBRegressor
from ml_metrics import rmsle

print('Script running..')


def read_data(training_path, test_path):
    train = pd.read_csv(training_path)
    train['train'] = 1
    test = pd.read_csv(test_path)
    test['train'] = 0
    data = train.append(test, ignore_index=True, sort=False)
    return train, test, data


def populate_missing_values(data, variable, new_value):
    data[variable] = data[variable].fillna(new_value)


def fixing_ordinal_variables(data, variable):
    data.loc[data[variable] == 'Ex', variable] = 5
    data.loc[data[variable] == 'Gd', variable] = 4
    data.loc[data[variable] == 'TA', variable] = 3
    data.loc[data[variable] == 'Fa', variable] = 2
    data.loc[data[variable] == 'Po', variable] = 1
    data.loc[data[variable] == 'None', variable] = 0


train, test, data = read_data('data/train.csv', 'data/test.csv')
data = data.set_index('Id')

populate_missing_values(data, 'GarageCond', 'None')
populate_missing_values(data, 'GarageQual', 'None')
populate_missing_values(data, 'FireplaceQu', 'None')
populate_missing_values(data, 'BsmtCond', 'None')
populate_missing_values(data, 'BsmtQual', 'None')
populate_missing_values(data, 'PoolQC', 'None')
populate_missing_values(data, 'MiscFeature', 'None')


fixing_ordinal_variables(data, 'ExterQual')
fixing_ordinal_variables(data, 'ExterCond')
fixing_ordinal_variables(data, 'BsmtCond')
fixing_ordinal_variables(data, 'BsmtQual')
fixing_ordinal_variables(data, 'HeatingQC')
fixing_ordinal_variables(data, 'KitchenQual')
fixing_ordinal_variables(data, 'FireplaceQu')
fixing_ordinal_variables(data, 'GarageQual')
fixing_ordinal_variables(data, 'GarageCond')
fixing_ordinal_variables(data, 'PoolQC')

data.loc[data['PavedDrive'] == 'Y', 'PavedDrive'] = 3
data.loc[data['PavedDrive'] == 'P', 'PavedDrive'] = 2
data.loc[data['PavedDrive'] == 'N', 'PavedDrive'] = 1

# colu = data.columns[(data.isnull().sum() > 50) & (data.isnull().sum() > 0)]
# for i in colu:
#     print(data[colu].isnull().sum())

# populate data for Garage related features when house doesn't have a garage
# populate_missing_values(data, 'GarageArea', 0
# populate_missing_values(data, 'GarageCars', 0)
# data['GarageFinish'][
#   (data.GarageFinish.isnull() is True) & (data.GarageCond == 0)
#   ] = 0
# data['GarageType'][
#   (data.GarageType.isnull() is True) & (data.GarageCond == 0)
#   ] = 0
# data['GarageYrBlt'][
#   (data.GarageYrBlt.isnull() is True) & (data.GarageCond == 0)
#   ] = 0

# print(data.describe())

# trainX, testX, trainY, testY = train_test_split(data[data.SalePrice.isnull()==False].drop('SalePrice',axis=1),data.SalePrice[data.SalePrice.isnull()==False],test_size=0.30, random_state=2019)
# trainY = np.log(trainY)

# model = XGBRegressor(
#   learning_rate=0.001,
#   n_estimators=4600,
#   max_depth=7,
#   min_child_weight=0,
#   gamma=0,
#   subsample=0.7,
#   colsample_bytree=0.7,
#   scale_pos_weight=1,
#   seed=27,
#   reg_alpha=0.00006
# )

# model.fit(trainX, trainY)
# y_pred = model.predict(testX)
# y_pred = np.exp(y_pred)

# print(rmsle(testY, y_pred))
