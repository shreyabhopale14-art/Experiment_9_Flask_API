
import pandas as pd
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Test Model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Save Model
joblib.dump(model, "model.pkl")

print("Model Trained Successfully!")
print("Model Accuracy:", accuracy)
print("Model Saved as model.pkl")
