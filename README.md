# Sonar Rock vs Mine Prediction

An end-to-end Python machine learning project that classifies whether an underwater object detected by sonar is a **Rock (R)** or a **Mine (M)**, using a Logistic Regression classifier.

Sonar signals are bounced off rocks and metal cylinders (representing mines) in a laboratory setting. The frequency response of each return is recorded across 60 features, and the model learns to tell the two classes apart.

## Dataset
- **Instances:** 208 (111 Mines, 97 Rocks)
- **Features:** 60 numerical values (sonar signal energy across frequency bands)
- **Target:** `M` (Mine) or `R` (Rock), stored in column index 60
- **File:** `sonar data.csv` (no header row)

## Workflow
1. **Data collection:** import dependencies and load `sonar data.csv`
2. **Data analysis:** check shape, descriptive statistics and class distribution, and compare feature means for Rocks and Mines
3. **Feature/label split:** separate the 60 features (X) from the label (Y)
4. **Train-test split:** 90% training and 10% testing, stratified so both sets keep the same class balance
5. **Model training:** fit a Logistic Regression model
6. **Model evaluation:** measure accuracy on training and test data
7. **Predictive system:** reshape a single unseen sample to `(1, -1)` and predict Rock or Mine

## Results
| Dataset | Accuracy |
|---------|----------|
| Training data | ~83.4% |
| Test data | ~76.2% |

## Project structure
```text
sonar-rock-vs-mine-prediction/
├── sonar data.csv
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and usage
```bash
git clone https://github.com/<YOUR_USERNAME>/sonar-rock-vs-mine-prediction.git
cd sonar-rock-vs-mine-prediction
pip install -r requirements.txt
python main.py
```

## Predicting on a new sample
Provide 60 sonar readings as a tuple, convert them to a NumPy array and reshape to a single row:

```python
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = model.predict(input_data_reshaped)

print("Rock" if prediction[0] == 'R' else "Mine")
```

## Tech stack
Python, NumPy, pandas, scikit-learn

## Future improvements
- Compare other models (SVM, KNN, Random Forest) with cross-validation
- Scale features and tune hyperparameters with GridSearchCV
- Report precision, recall and a confusion matrix, since missing a mine is costlier than a false alarm
- Save the trained model with joblib and wrap it in a small web app

## License
This project is for educational purposes.
