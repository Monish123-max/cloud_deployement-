from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
import json

trained_model = joblib.load("artifacts/random_forest_model.pkl")
X_test = pd.read_csv("data/X_test.csv")

app = FastAPI(title="ML Model API")


@app.get("/predict")
def get_predictions():
    predictions = trained_model.predict(X_test)
    print(predictions)
    return {'predictions':predictions.tolist()}

@app.post("/predict-one/{payload}")
def get_single_predictions(payload):
    predictions = trained_model.predict(np.array(list(payload)).reshape(1, -1))
    print(predictions)
    return {'predictions':predictions.tolist()}


if __name__=="__main__":
    app.run()