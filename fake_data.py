import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import seaborn as sns
import matplotlib.pyplot as plt

# LOADING DATA
df = pd.read_csv('data/goods.csv')

# CLEANING
df.isnull().sum()
df.fillna({'Income': 0.0}, inplace=True)

# ENCODING
# df.replace({
#     'Gender': {
#         'male': 0,
#         'female': 1,
#     },
# }, inplace=True)
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])

# MODEL
X = df.drop(columns='Favorite Transport')
Y = df['Favorite Transport']

model = DecisionTreeClassifier()
model.fit(X, Y)

# PREDICTION
test_df = pd.DataFrame({
    'Age': [12, 30, 75],
    'Gender': [0, 0, 1],
    'Income': [0.0, 4000, 50000]
})

# EVALUATION
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


model = DecisionTreeClassifier()
model.fit(X_train, Y_train)


predictions  = model.predict(X_test)

model_score = accuracy_score(Y_test, predictions)

# Exporting to the DOT file
tree.export_graphviz(model, out_file='decission_tree_model.dot',
                     feature_names=['Age', 'Gender', 'Income'],
                     filled=True,
                     class_names=sorted(Y.unique()))


# Charts
sns.countplot(x=df['Gender'], hue=df['Favorite Transport'])
# plt.show()

sns.histplot(x=df['Income'], hue=df['Favorite Transport'])
plt.show()

# print(df)
# print(type(model.predict(test_df)))
# print(model.predict(test_df))