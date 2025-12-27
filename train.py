import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# 1. Load your data
df = pd.read_csv('insurance.csv')

df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
df['sex'] = df['sex'].map({'male': 1, 'female': 0})
df = pd.get_dummies(df, columns=['region'], drop_first=True)
df['High_Claim'] = df['charges'] > df['charges'].median()

X = df.drop(['High_Claim','charges'], axis=1) # Features
y = df['High_Claim']                # Target

model_features = X.columns.tolist()

model = LogisticRegression(max_iter=1000)

model.fit(X, y)


with open('insurance_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model_features.pkl', 'wb') as f:
    pickle.dump(model_features, f)

print("Model saved successfully as .pkl!")