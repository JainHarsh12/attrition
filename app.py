import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the attrition dataset
file_path = 'Attrition.csv'
attrition_data = pd.read_csv(file_path)

# Encode the target column 'Attrition' (Yes/No) to 1/0
le = LabelEncoder()
attrition_data['Attrition'] = le.fit_transform(attrition_data['Attrition'])

# Encoding other categorical columns
categorical_columns = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']

# Apply one-hot encoding for these categorical columns
attrition_data = pd.get_dummies(attrition_data, columns=categorical_columns, drop_first=True)

# Dropping irrelevant columns that won't contribute to the model
attrition_data.drop(columns=['EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18'], inplace=True)

# Separate features and target variable
X = attrition_data.drop('Attrition', axis=1)
y = attrition_data['Attrition']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features for better model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Logistic Regression model
log_reg = LogisticRegression(max_iter=10000)

# Train the model
log_reg.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = log_reg.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print the results
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)