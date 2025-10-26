import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
import joblib


df = pd.read_csv('data/shops.csv')
pd.set_option('display.max_columns', None)


df.drop(columns='id', inplace=True)

# Data Cleaning

df['Arrival Delay in Minutes'].fillna(df['Arrival Delay in Minutes'].mean(), inplace=True)

# Charts
plt.pie(df['satisfaction'].value_counts(), labels=['neutral or dissatisfied', 'satisfied'], autopct='%1.1f%%')


cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']
plt.figure(figsize=(15, 15))
for i, col in enumerate(cols):
    plt.subplot(3, 2, i+1)
    sns.countplot(x=col, data=df)

df.hist(bins=20, figsize=(20, 20), color='green')

# Column Data Encoding

# df.replace({
#     'Gender': {
#         'Male': 1,
#         'Female': 2,
#     },
#     'Customer Type': {
#         'First-time': 1,
#         'Returning': 2,
#     },
#     'Type of Travel': {
#         'Business': 1,
#         'Personal': 2,
#     },
#     'Class': {
#         'Business': 1,
#         'Economy': 2,
#         'Economy Plus': 3,
#     }
# }, inplace=True)

label_encoder = LabelEncoder()
columns = df.select_dtypes(include='object').drop(columns='satisfaction').columns

for column in columns:
    df[column] = label_encoder.fit_transform(df[column])


# Additional Charts
plt.figure(figsize=(16 ,8))
sns.heatmap(df.drop(columns='satisfaction').corr(), annot=True, fmt='.2f')

sns.catplot(data=df, x='Age', height=4, aspect=4, kind='count', hue='satisfaction')

# plt.show()

# Models

X = df.drop(columns='satisfaction')
y = df['satisfaction']

# Decision Tree

model = DecisionTreeClassifier()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)

predictions = model.predict(X_test)


model_score = accuracy_score(y_test, predictions)

# Random Forest
model_r = RandomForestClassifier()
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y, test_size=0.2)
model_r.fit(X_train_r, y_train_r)

predictions_r = model_r.predict(X_test_r)
model_score_r = accuracy_score(y_test_r, predictions_r)

# KNeighborsClassifier
model_k = KNeighborsClassifier()
X_train_k, X_test_k, y_train_k, y_test_k = train_test_split(X, y, test_size=0.2)
model_k.fit(X_train_k, y_train_k)

predictions_k = model_k.predict(X_test_k)
model_score_k = accuracy_score(y_test_k, predictions_k)

# Logistic Regression
model_l = LogisticRegression(max_iter=10000)
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X, y, test_size=0.2)
model_l.fit(X_train_l, y_train_l)

predictions_l = model_l.predict(X_test_l)
model_score_l = accuracy_score(y_test_l, predictions_l)

print(model_score)
print(model_score_r)
print(model_score_k)
print(model_score_l)

# Saving Prediction Model
joblib.dump(model, 'shop_user_satisfaction.joblib')

# print(df.dtypes)
# print(df.isnull().sum())
# print(df['Arrival Delay in Minutes'].mean())
# print(f'{df.shape} \n {df.describe().round(2)}')