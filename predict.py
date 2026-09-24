import joblib
import pandas as pd

X_test = pd.read_csv("X_test.csv")
trained_model = joblib.load("./random_forest_model.pkl")

predictions = trained_model.predict(X_test)
print(predictions)