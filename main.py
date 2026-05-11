import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
# Sample datase
data = {
    'StudyHours': [2, 3, 5, 6, 8, 1, 4, 7],
    'Attendance': [60, 70, 80, 90, 95, 50, 75, 85],
    'Assignments': [3, 4, 6, 7, 8, 2, 5, 7],
    'PreviousMarks': [50, 55, 65, 70, 90, 40, 60, 80],
    'FinalResult': [55, 60, 70, 75, 95, 45, 68, 85]
}
# Create DataFrame
df = pd.DataFrame(data)
# Features and target
X = df[['StudyHours', 'Attendance', 'Assignments', 'PreviousMarks']]
y = df['FinalResult']
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train model
model = LinearRegression()
model.fit(X_train, y_train)
# Prediction
prediction = model.predict(X_test)
print('Predicted Results:', prediction)
