# First Phase of Prediction 
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# loading the data set
dataset = pd.read_csv('/home/gfg/Documents/dataset/diabetes.csv')
# print(dataset.head(10))

# Plot Distribution of the target variable ‘Outcome’
labels = 'Diabetic', 'Non-Diabetic'
dataset.Outcome.value_counts().plot.pie(labels=labels, autopct='%1.1f%%')
# plt.show()
# plt.savefig("before balancing.png")

# grouping the target variable ‘Outcome’ into 'Diabetic', 'Non-Diabetic'
outcome = dataset.groupby('Outcome').size()
# print(outcome)


#  plot Distribution of the feature variables
dataset.hist(figsize=(15,10))
# plt.show()
# plt.savefig("feature.png")

d=dataset[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']]=dataset[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.nan)
# print(d.head(5))

d['Glucose'].fillna(d['Glucose'].median(),inplace=True)
d['BloodPressure'].fillna(d['BloodPressure'].median(),inplace=True)
d['SkinThickness'].fillna(d['SkinThickness'].median(),inplace=True)
d['Insulin'].fillna(d['Insulin'].median(),inplace=True)
d['BMI'].fillna(d['BMI'].median(),inplace=True)
# print(d.head())

f = dataset[["Age", "Pregnancies", "DiabetesPedigreeFunction", "Outcome"]]

data = pd.concat([d , f],axis=1)
# print(data.isnull().sum())

x = data.drop("Outcome" , axis= 1)
y = data["Outcome"]


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3 , random_state=45, stratify=y) 

from sklearn.preprocessing import StandardScaler

scale = StandardScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.transform(x_test)


from imblearn.over_sampling import RandomOverSampler
sm = RandomOverSampler(random_state=42)
x_train, y_train = sm.fit_resample(x_train, y_train)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=45, criterion= 'gini', max_depth= 40, n_estimators=450)
model.fit(x_train, y_train)


from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_pred,y_test)
print(accuracy)


# Second Phase of Prediction

import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import pickle
import matplotlib
matplotlib.use('Agg')
dataset = pd.read_csv('/home/fuck_love/Documents/DataSet/pima-indians-diabetes.csv')

from sklearn.model_selection import train_test_split
X = dataset
y = dataset['Outcome']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42, stratify=y)

# Filling nulls and zeros
X_train.loc[(X_train['Outcome'] == 0 ) & (X_train['Glucose'] == 0), 'Glucose'] = X_train.loc[X_train["Outcome"]==0, 'Glucose'].median()
X_train.loc[(X_train['Outcome'] == 1 ) & (X_train['Glucose'] == 0), 'Glucose'] = X_train.loc[X_train["Outcome"]==1, 'Glucose'].median()

X_test.loc[(X_test['Outcome'] == 0 ) & (X_test['Glucose'] == 0), 'Glucose'] = X_train.loc[X_train["Outcome"]==0, 'Glucose'].median()
X_test.loc[(X_test['Outcome'] == 1 ) & (X_test['Glucose'] == 0), 'Glucose'] = X_train.loc[X_train["Outcome"]==1, 'Glucose'].median()


X_train.loc[(X_train['Outcome'] == 0 ) & (X_train['BloodPressure'] == 0), 'BloodPressure'] = X_train.loc[X_train["Outcome"]==0, 'BloodPressure'].median()
X_train.loc[(X_train['Outcome'] == 1 ) & (X_train['BloodPressure'] == 0), 'BloodPressure'] = X_train.loc[X_train["Outcome"]==1, 'BloodPressure'].median()

X_test.loc[(X_test['Outcome'] == 0 ) & (X_test['BloodPressure'] == 0), 'BloodPressure'] = X_train.loc[X_train["Outcome"]==0, 'BloodPressure'].median()
X_test.loc[(X_test['Outcome'] == 1 ) & (X_test['BloodPressure'] == 0), 'BloodPressure'] = X_train.loc[X_train["Outcome"]==1, 'BloodPressure'].median()

X_train.loc[(X_train['Outcome'] == 0 ) & (X_train['SkinThickness'] == 0), 'SkinThickness'] = X_train.loc[X_train["Outcome"]==0, 'SkinThickness'].median()
X_train.loc[(X_train['Outcome'] == 1 ) & (X_train['SkinThickness'] == 0), 'SkinThickness'] = X_train.loc[X_train["Outcome"]==1, 'SkinThickness'].median()

X_test.loc[(X_test['Outcome'] == 0 ) & (X_test['SkinThickness'] == 0), 'SkinThickness'] = X_train.loc[X_train["Outcome"]==0, 'SkinThickness'].median()
X_test.loc[(X_test['Outcome'] == 1 ) & (X_test['SkinThickness'] == 0), 'SkinThickness'] = X_train.loc[X_train["Outcome"]==1, 'SkinThickness'].median()


X_train.loc[(X_train['Outcome'] == 0 ) & (X_train['Insulin'] == 0), 'Insulin'] = X_train.loc[X_train["Outcome"]==0, 'Insulin'].median()
X_train.loc[(X_train['Outcome'] == 1 ) & (X_train['Insulin'] == 0), 'Insulin'] = X_train.loc[X_train["Outcome"]==1, 'Insulin'].median()

X_test.loc[(X_test['Outcome'] == 0 ) & (X_test['Insulin'] == 0), 'Insulin'] = X_train.loc[X_train["Outcome"]==0, 'Insulin'].median()
X_test.loc[(X_test['Outcome'] == 1 ) & (X_test['Insulin'] == 0), 'Insulin'] = X_train.loc[X_train["Outcome"]==1, 'Insulin'].median()

X_train.loc[(X_train['Outcome'] == 0 ) & (X_train['BMI'] == 0), 'BMI'] = X_train.loc[X_train["Outcome"]==0, 'BMI'].median()
X_train.loc[(X_train['Outcome'] == 1 ) & (X_train['BMI'] == 0), 'BMI'] = X_train.loc[X_train["Outcome"]==1, 'BMI'].median()

X_test.loc[(X_test['Outcome'] == 0 ) & (X_test['BMI'] == 0), 'BMI'] = X_train.loc[X_train["Outcome"]==0, 'BMI'].median()
X_test.loc[(X_test['Outcome'] == 1 ) & (X_test['BMI'] == 0), 'BMI'] = X_train.loc[X_train["Outcome"]==1, 'BMI'].median()


X_train = X_train.drop("Outcome", axis=1)
X_test = X_test.drop("Outcome", axis=1)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train.mean(axis=0)


# Performing oversampling using SMOTE
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)


# Random forest model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_predicted = model.predict(X_test)

# Evaluation
from sklearn.metrics import classification_report

report = classification_report(y_test,y_predicted)
# print(report)





from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators" :[100, 200, 300],
    "criterion" :[ "gini", "entropy", "log_loss"],
    "max_depth" : [10,20,30]

}

grid_search = GridSearchCV( estimator= model , param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train,y_train)

best_model = grid_search.best_estimator_
best_param = grid_search.best_params_
best_score = grid_search.best_score_

y_pred = grid_search.predict(X_test)
best = classification_report(y_pred, y_test)
# print( best, best_score, best_model, best_param)


with open("model.pkl", 'wb') as file:
        pickle.dump(model, file)

# with open("standard_scaler.pkl", 'wb') as file:
#     pickle.dump(scaler, file)