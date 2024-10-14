import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# Function to load and preprocess the dataset
def load_and_preprocess_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Encode the target column 'Attrition' (Yes/No) to 1/0
    le = LabelEncoder()
    data['Attrition'] = le.fit_transform(data['Attrition'])
    
    # Encoding other categorical columns
    categorical_columns = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
    
    # Dropping irrelevant columns
    data.drop(columns=['EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18'], inplace=True)
    
    # Separate features and target variable
    X = data.drop('Attrition', axis=1)
    y = data['Attrition']
    
    return X, y

# Test if the data loads correctly and preprocessing works as expected
def test_load_and_preprocess_data():
    file_path = 'attrition.csv'
    X, y = load_and_preprocess_data(file_path)
    
    # Check if X and y have the correct length
    assert len(X) == len(y), "Features and target length mismatch"
    
    # Check if there are no missing values
    assert X.isnull().sum().sum() == 0, "There are missing values in the features"
    assert y.isnull().sum() == 0, "There are missing values in the target"

# Function to train and evaluate the model
def train_and_evaluate_model(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scale the features
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
    
    return accuracy

# Test if the model can be trained and if accuracy is reasonable
def test_model_accuracy():
    file_path = 'attrition.csv'
    X, y = load_and_preprocess_data(file_path)
    accuracy = train_and_evaluate_model(X, y)

    # Assert that the model achieves at least 80% accuracy
    assert accuracy > 0.80, f"Model accuracy is below acceptable level: {accuracy:.2f}"

# Run all tests
if __name__ == "__main__":
    pytest.main()