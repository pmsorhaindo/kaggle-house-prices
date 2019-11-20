# import numpy as np
import pandas as pd

print('Script running..')


def read_data(training_path, test_path):
    train = pd.read_csv(training_path)
    train['train'] = 1
    test = pd.read_csv(test_path)
    test['train'] = 0
    data = train.append(test, ignore_index=True)
    return train, test, data


def populate_missing_values(data, variable, new_value):
    data[variable] = data[variable].fillna(new_value)


def fixing_ordinal_variables(data, variable):
    data[variable][data[variable] == 'Ex'] = 5
    data[variable][data[variable] == 'Gd'] = 4
    data[variable][data[variable] == 'TA'] = 3
    data[variable][data[variable] == 'Fa'] = 2
    data[variable][data[variable] == 'Po'] = 1
    data[variable][data[variable] == 'None'] = 0


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

data['PavedDrive'][data['PavedDrive'] == 'Y'] = 3
data['PavedDrive'][data['PavedDrive'] == 'P'] = 2
data['PavedDrive'][data['PavedDrive'] == 'N'] = 1

colu = data.columns[(data.isnull().sum() < 50) & (data.isnull().sum() > 0)]
for i in colu:
    print(data[colu].isnull().sum())
