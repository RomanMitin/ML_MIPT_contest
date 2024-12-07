#!/bin/env python3
import numpy as np             
import pandas as pd             
import seaborn as sns           
import matplotlib.pyplot as plt
from sklearn.preprocessing import *
from sklearn.ensemble import *
from sklearn.metrics import roc_auc_score, accuracy_score

def drop_useless_features(data_frame):
    return data_frame.drop(columns=["EmployeeCount", "StandardHours", "Over18", "id"])

def drop_correlated_features(data_frame):
    return data_frame.drop(columns=["JobLevel", "PerformanceRating", "YearsAtCompany", "YearsWithCurrManager", "TotalWorkingYears"])

def encode_category_features(data_frame):
    data_frame['Gender'] = (data_frame['Gender'] == 'Male').astype(int) * 2 - 1
    data_frame['OverTime'] = (data_frame['OverTime'] == 'Yes').astype(int) * 2 - 1

    enc = OrdinalEncoder()
    data_frame['BusinessTravel'] = enc.fit_transform(np.array(data_frame['BusinessTravel']).reshape(-1, 1)).astype(int)

    columns_to_encode = ['Department', 'JobRole', 'EducationField', 'MaritalStatus']
    one_hot_encode = OneHotEncoder(drop='first', sparse_output=False)
    one_hot_encode.fit(data_frame[columns_to_encode])

    dummies = pd.DataFrame(one_hot_encode.transform(data_frame[columns_to_encode]),
                           columns=one_hot_encode.get_feature_names_out(), index=data_frame.index)

    data_frame = pd.concat((data_frame, dummies), axis=1).drop(columns_to_encode, axis=1)
    
    return data_frame

def normalize_data(data_frame):
    scaler = StandardScaler()
    scaler.fit(data_frame)
    data_frame = pd.DataFrame(scaler.transform(data_frame), columns=data_frame.columns, index=data_frame.index)

    return data_frame


class BaseLineClassifier:
    def __init__(self) -> None:
        self.model = RandomForestClassifier()
        pass

    def fit(self, X, y):
        # # print(data_raw.columns)
        # print(data_raw.info())

        # # В выборке нет NA и null
        # print(f"Total NA in data = {data_raw.isna().sum().sum()}")
        # print(f"Total null in data = {data_raw.isnull().sum().sum()}")

        # # Видно что EmployeeCount и StandardHours можно удалить
        # print(f"Всего семплов в выборке: {data_raw['id'].count()}, (data_raw['EmployeeCount'] == 1).sum() == {(data_raw['EmployeeCount'] == 1).sum()}")
        # print(f"Всего семплов в выборке: {data_raw['id'].count()}, (data_raw['StandardHours'] == 1).sum() == {(data_raw['StandardHours'] == 80).sum()}")
        # print(f"Всего семплов в выборке: {data_raw['id'].count()}, (data_raw['Over18'] == 1).sum() == {(data_raw['Over18'] == 'Y').sum()}")

        data_raw = drop_useless_features(X)
        
        corr_mat = data_raw.corr(numeric_only=True)
        # print(corr_mat)
        
        # mask = corr_mat.abs() > 0.7
        # print(f"Количество сильно коррелирующий параметров: \n{mask.sum()}")
        
        # Красивый вывод корреляционной матрицы
        # heatmap = sns.heatmap(corr_mat, square=True, vmin=-1, vmax=1, cmap='coolwarm')
        # plt.show()
        
        # Удалим все сильно коррелирующие параметры из выборки
        data_raw = drop_correlated_features(data_raw)

        # Красивый вывод корреляционной матрицы после удаления
        # corr_mat = data_raw.corr(numeric_only=True)
        # heatmap = sns.heatmap(corr_mat, square=True, vmin=-1, vmax=1, cmap='coolwarm')

        # plt.show()

        # print(data_raw.info())

        # Обработка выбросов
        for name in data_raw:
            if name != 'id' and data_raw[name].dtype == int:
                col = data_raw[name]
                data_raw.loc[data_raw[name] > col.quantile(0.99), name] = col.quantile(0.99)

                col = data_raw[name]
                data_raw.loc[data_raw[name] < col.quantile(0.01), name] = col.quantile(0.01)
        
        # Перекодируем не целочисленные признаки
        data_raw = encode_category_features(data_raw)
        # print(data_raw.info())

        # Нормализация признаков
        data = normalize_data(data_raw)

        self.model.fit(data, y)

        return self

    def preprocess(self, X):
        X = drop_correlated_features(drop_useless_features(X))
        X = normalize_data(encode_category_features(X))

        return X

    def predict_proba(self, X):
        return self.model.predict_proba(self.preprocess(X))
    
    def predict(self, X):
        return self.model.predict(self.preprocess(X))



data_raw = pd.read_csv('train.csv', low_memory=False)
X_test = pd.read_csv('test.csv', low_memory=False)

validation_start = 1200

train_data = data_raw.iloc[:validation_start]
validation_data = data_raw.iloc[validation_start:]

y_train = train_data['Attrition']
X_train = train_data.drop(columns='Attrition')

y_val = validation_data['Attrition']
X_val = validation_data.drop(columns='Attrition')

base_model = BaseLineClassifier()

base_model.fit(X_train, y_train)

y_pred = base_model.predict_proba(X_val)[:, 1]

y_final = base_model.predict_proba(X_test)[:, 1]

score = roc_auc_score(y_val, y_pred)
print(score)


col_to_drop = X_test.columns
col_to_drop = col_to_drop.drop('id')
final_data = X_test.drop(columns=col_to_drop)
print(final_data)

final_data = final_data.assign(Attrition=pd.DataFrame(y_final,columns=['Attrition'], index = final_data.index))
print(final_data)

final_data.to_csv("result.csv", index=False)