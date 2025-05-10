import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Create the dataset with features and label
data = pd.DataFrame({
    "Feature_A": [1.2, 1.0, 1.1, 2.0, 2.1, 2.2, 1.9, 2.3, 1.8,
                  5.0, 5.2, 5.1, 4.9, 5.3, 5.4],
    "Feature_B": [3.5, 3.7, 3.6, 3.9, 4.0, 4.1, 3.8, 4.2, 3.9,
                  1.1, 1.0, 1.2, 0.9, 1.3, 1.4],
    "Feature_C": [0.5, 0.6, 0.4, 0.7, 0.8, 0.9, 0.7, 0.8, 0.6,
                  2.1, 2.2, 2.3, 2.0, 2.4, 2.5],
    "Label": [
        'banana', 'banana', 'banana', 'banana', 'banana',
        'banana', 'banana', 'banana', 'banana',
        'apple', 'apple', 'apple', 'apple', 'apple', 'apple'
    ]
})

# Step 2: Convert string labels into 0, 1
# drop_first=True can remove 'banana' and keep only 'Label_apple'
#    which will be T F, T is apple and F is banana
# in this example we will not remvoe the column...
data_encoded = pd.get_dummies(data, columns=['Label'], drop_first=False)

print(data_encoded)

# Choose 'Label_apple' explicitly
y = data_encoded['Label_apple'].astype(int)
X = data_encoded.drop(['Label_apple', 'Label_banana'], axis=1)

print(y)
print()
print("-"*50,"\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)
pipeline = make_pipeline(StandardScaler(),SVC(kernel='linear', C=1))
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"After using StandardScaler() and an SVC model with (kernel='linear', C=1), Model's Accuracy is: {accuracy:.2f}")

new_fruit = pd.DataFrame({
    "Feature_A": [1.9],
    "Feature_B": [4.0],
    "Feature_C": [0.7]})

new_fruit_pred = pipeline.predict(new_fruit)
fruit = 'Apple' if new_fruit_pred == [1] else 'Banana'
print(f"According to our SVM model, the new fruit is : [{fruit}]")

param_grid = {
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'C': [0.1, 1, 10, 100]
}

svm = SVC()
grid = GridSearchCV(svm, param_grid)
grid.fit(X,y)

print(f"\n Best parameters from Grid Search:\nKernel: {grid.best_params_['kernel']}, C: {grid.best_params_['C']}")
