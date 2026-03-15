import joblib
import pandas as pd


# Load saved model
model = joblib.load("models/model.pkl")


def predict_survival(data: dict):

    df = pd.DataFrame([data])

    prediction = model.predict(df)

    if prediction[0] == 1:
        return "Survived"
    else:
        return "Did not survive"
