import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

path = r"C:\Users\SAYANDEEP\Desktop\DATA ANLYTICS CSV\iris.csv"
df = pd.read_csv(path)
print("Sample Data:\n", df.head())

if df['species'].dtype == 'object':
    df['species'] = df['species'].astype('category').cat.codes

X = df.drop('species', axis=1)
y = df['species']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=200),
    'Random Forest': RandomForestClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n🔍 {name} Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='macro'))
    print("Recall:", recall_score(y_test, y_pred, average='macro'))
    print("F1 Score:", f1_score(y_test, y_pred, average='macro'))

param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 3, 5, 10]
}
grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, scoring='accuracy')
grid.fit(X_train, y_train)

print("\n✅ Best Parameters (Random Forest):", grid.best_params_)
best_rf = grid.best_estimator_
y_pred_best = best_rf.predict(X_test)

print("\n📊 Final Random Forest Evaluation:")
print(classification_report(y_test, y_pred_best))
