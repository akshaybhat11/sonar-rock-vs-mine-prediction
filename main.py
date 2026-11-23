# Sonar Rock vs Mine Prediction using Logistic Regression
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------- Data collection and processing ----------
sonar_data = pd.read_csv('sonar data.csv', header=None)

print("First 5 rows:")
print(sonar_data.head())
print("\nShape (rows, columns):", sonar_data.shape)
print("\nStatistical description:")
print(sonar_data.describe())

print("\nClass distribution (M = Mine, R = Rock):")
print(sonar_data[60].value_counts())

print("\nMean values grouped by class:")
print(sonar_data.groupby(60).mean())

# ---------- Separate features and label ----------
X = sonar_data.drop(columns=60)
Y = sonar_data[60]

# ---------- Train-test split ----------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, stratify=Y, random_state=1
)
print(f"\nX: {X.shape}, train: {X_train.shape}, test: {X_test.shape}")

# ---------- Model training ----------
model = LogisticRegression()
model.fit(X_train, Y_train)

# ---------- Model evaluation ----------
train_accuracy = accuracy_score(model.predict(X_train), Y_train)
test_accuracy = accuracy_score(model.predict(X_test), Y_test)
print(f"\nAccuracy on training data: {train_accuracy:.4f}")
print(f"Accuracy on test data: {test_accuracy:.4f}")

# ---------- Predictive system ----------
input_data = (
    0.0200, 0.0371, 0.0428, 0.0207, 0.0954, 0.0986, 0.1539, 0.1601, 0.3109, 0.2111,
    0.1609, 0.1582, 0.2238, 0.0645, 0.0660, 0.2273, 0.3100, 0.2999, 0.5078, 0.4797,
    0.5783, 0.5071, 0.4328, 0.5550, 0.6711, 0.6415, 0.7104, 0.8080, 0.6791, 0.3857,
    0.1307, 0.2604, 0.5121, 0.7547, 0.8537, 0.8507, 0.6692, 0.6097, 0.4943, 0.2744,
    0.0510, 0.2834, 0.2825, 0.4256, 0.2641, 0.1386, 0.1051, 0.1343, 0.0383, 0.0324,
    0.0232, 0.0027, 0.0065, 0.0159, 0.0072, 0.0167, 0.0180, 0.0084, 0.0090, 0.0032
)

input_data_reshaped = np.asarray(input_data).reshape(1, -1)
prediction = model.predict(input_data_reshaped)

print("\n--- Predictive system result ---")
if prediction[0] == 'R':
    print("The object is a Rock ('R')")
else:
    print("The object is a Mine ('M')")
