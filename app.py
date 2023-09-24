from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = FastAPI()

# Load the dataset
df = pd.read_csv('diabetes.csv')

# X AND Y DATA
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Define a Pydantic model for input data
class UserData(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float  # Use float for BMI since it can be a decimal
    DiabetesPedigreeFunction: float
    Age: int

# Initialize the machine learning model
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

@app.post("/predict/")
async def predict_diabetes(user_data: UserData):
    user_data_dict = user_data.dict()
    user_data_df = pd.DataFrame([user_data_dict])

    user_result = rf.predict(user_data_df)

    if user_result[0] == 0:
        output = 'You are not Diabetic'
    else:
        output = 'You are Diabetic'

    # Calculate accuracy using the test data
    test_prediction = rf.predict(x_test)
    accuracy = accuracy_score(y_test, test_prediction) * 100

    return {
        "result": output,
        "accuracy": accuracy
    }
