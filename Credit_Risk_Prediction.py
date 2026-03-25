import pandas as pd

# Load datasets
loan_train_data = pd.read_csv("archive/train_u6lujuX_CVtuZ9i.csv")
loan_test_data = pd.read_csv("archive/test_Y3wMUE5_7gLdaTN.csv")

# First rows
print(loan_train_data.head())

# Dataset size
print("Shape:", loan_train_data.shape)

# Column names
print("Columns:", loan_train_data.columns)

# Data types
print(loan_train_data.info())

# Check missing values
print("\nMissing Values Per Column:")
print(loan_train_data.isnull().sum())

# Loan approval distribution
print("\nLoan Status Distribution:")
print(loan_train_data["Loan_Status"].value_counts())

# Visualize loan_status
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x="Loan_Status", data=loan_train_data)
plt.title("Loan Approval Distribution")
plt.show()

#  DATA CLEANING

# Drop unnecessary column
loan_train_data = loan_train_data.drop('Loan_ID', axis=1)

# Fill missing values
loan_train_data['LoanAmount'].fillna(loan_train_data['LoanAmount'].mean(), inplace=True)
loan_train_data['Loan_Amount_Term'].fillna(loan_train_data['Loan_Amount_Term'].mean(), inplace=True)
loan_train_data['Credit_History'].fillna(loan_train_data['Credit_History'].mode()[0], inplace=True)

loan_train_data['Gender'].fillna(loan_train_data['Gender'].mode()[0], inplace=True)
loan_train_data['Married'].fillna(loan_train_data['Married'].mode()[0], inplace=True)
loan_train_data['Dependents'].fillna(loan_train_data['Dependents'].mode()[0], inplace=True)
loan_train_data['Self_Employed'].fillna(loan_train_data['Self_Employed'].mode()[0], inplace=True)

print("\nData Cleaning Completed")
print("Remaining Missing Values:\n", loan_train_data.isnull().sum())


# ENCODING

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

cols = ['Gender', 'Married', 'Dependents', 'Education',
        'Self_Employed', 'Property_Area', 'Loan_Status']

for col in cols:
    loan_train_data[col] = le.fit_transform(loan_train_data[col])

print("\nCategorical Encoding Completed")
print("Sample Data:\n", loan_train_data.head())


# FEATURE & TARGET

X = loan_train_data.drop('Loan_Status', axis=1)
y = loan_train_data['Loan_Status']


# TRAIN-TEST SPLIT

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nData Split Completed")
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)


# MODEL TRAINING

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("\nModel Training Completed (Logistic Regression)")


# MODEL EVALUATION

from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)

print("\nModel Evaluation Results")
print("Accuracy Score:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

#confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show(block=False)
plt.pause(3)
plt.close()