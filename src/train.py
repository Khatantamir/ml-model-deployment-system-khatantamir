import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load dataset
df = pd.read_csv("data/titanic.csv")

# Select useful features
df = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Survived"]]

# Handle missing values
df["Age"].fillna(df["Age"].median(), inplace=True)

# Encode categorical column
le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])

# Features and target
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print("Model accuracy:", acc)

# Save model
joblib.dump(model, "models/model.pkl")

print("Model saved to models/model.pkl")
