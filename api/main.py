from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import predict_survival


app = FastAPI()


class PassengerData(BaseModel):
    Pclass: int
    Sex: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float


@app.get("/")
def home():
    return {"message": "Titanic ML Model Deployment API is running"}


@app.post("/predict")
def predict(data: PassengerData):
    result = predict_survival(data.dict())
    return {"prediction": result}
