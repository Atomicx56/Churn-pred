import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Read the customer churn dataset
churn_data = pd.read_csv('E:\\churn\\Customer_churn_dataset.csv')

# Data preprocessing: Convert 'TotalCharges' to numeric and handle errors
churn_data['TotalCharges'] = pd.to_numeric(churn_data['TotalCharges'], errors='coerce')
churn_data.fillna(churn_data['TotalCharges'].mean(), inplace=True)

# Map senior citizen column to readable format
churn_data['SeniorCitizen'] = churn_data['SeniorCitizen'].map({0: "No", 1: "Yes"})

# Convert categorical features to numeric using Label Encoding
def encode_categorical_columns(df):
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = LabelEncoder().fit_transform(df[column])
    return df

churn_data = encode_categorical_columns(churn_data)

# Split the data into features (X) and target (y)
features = churn_data.drop(columns=['Churn'])
target = churn_data['Churn']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.30, random_state=42, stratify=target)

# Apply standard scaling to numerical features
numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# Function to train and evaluate a model
def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    print(f"Accuracy of {model.__class__.__name__}: {accuracy:.4f}")
    print(classification_report(y_test, predictions))
    return model, predictions

# Train and evaluate K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier(n_neighbors=11)
train_and_evaluate(knn_model, X_train, y_train, X_test, y_test)

# Train and evaluate Support Vector Machine (SVM)
svm_model = SVC(random_state=42)
train_and_evaluate(svm_model, X_train, y_train, X_test, y_test)

# Train and evaluate Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=500, oob_score=True, random_state=42)
train_and_evaluate(rf_model, X_train, y_train, X_test, y_test)

# Confusion Matrix for Random Forest model
def plot_confusion_matrix(y_test, predictions, model_name):
    plt.figure(figsize=(4, 3))
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt="d", linecolor="k", linewidths=3)
    plt.title(f"{model_name} Confusion Matrix", fontsize=14)
    plt.show()

# Random Forest confusion matrix visualization
rf_predictions = rf_model.predict(X_test)
plot_confusion_matrix(y_test, rf_predictions, "Random Forest")

# Train and evaluate Logistic Regression model
log_reg_model = LogisticRegression(max_iter=1000)
train_and_evaluate(log_reg_model, X_train, y_train, X_test, y_test)

# Confusion Matrix for Logistic Regression model
log_reg_predictions = log_reg_model.predict(X_test)
plot_confusion_matrix(y_test, log_reg_predictions, "Logistic Regression")

# Visualizing the distribution of features
def plot_feature_distribution(df, feature_name, hue_column=None, color_palette='coolwarm'):
    plt.figure(figsize=(8, 5))
    sns.kdeplot(df[feature_name][df[hue_column] == 1], color='blue', shade=True, label="Churn: Yes")
    sns.kdeplot(df[feature_name][df[hue_column] == 0], color='red', shade=True, label="Churn: No")
    plt.title(f"Distribution of {feature_name} by Churn", fontsize=14)
    plt.xlabel(feature_name)
    plt.ylabel("Density")
    plt.legend()
    plt.show()

# Visualize distributions for 'MonthlyCharges' and 'TotalCharges' by churn
plot_feature_distribution(churn_data, 'MonthlyCharges', hue_column='Churn')
plot_feature_distribution(churn_data, 'TotalCharges', hue_column='Churn')

# Churn Rate Graph (Pie Chart)
def plot_churn_rate(df):
    churn_labels = ['Churn: No', 'Churn: Yes']
    churn_values = df['Churn'].value_counts()

    fig = go.Figure(data=[go.Pie(labels=churn_labels, values=churn_values, hole=.3)])
    fig.update_layout(title_text="Churn Rate Distribution")
    fig.show()

plot_churn_rate(churn_data)

# Feature Graph for Internet Service vs Churn
def plot_internet_service_vs_churn(df):
    internet_service_churn = df.groupby(['InternetService', 'Churn']).size().unstack().fillna(0)

    internet_service_churn.plot(kind='bar', stacked=True, figsize=(10, 6), color=['#FF6347', '#4682B4'])
    plt.title("Churn Distribution by Internet Service Type", fontsize=14)
    plt.xlabel("Internet Service")
    plt.ylabel("Count of Customers")
    plt.xticks(rotation=0)
    plt.legend(title="Churn Status", labels=['Not Churned', 'Churned'])
    plt.tight_layout()
    plt.show()

plot_internet_service_vs_churn(churn_data)
