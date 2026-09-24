import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from seaborn import load_dataset

df = load_dataset("iris")
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = RandomForestClassifier()
model.fit(X_train,y_train)

pd.DataFrame(X_test).to_csv("./data/X_test.csv",index=False)
joblib.dump(model, 'artifacts/random_forest_model.pkl')