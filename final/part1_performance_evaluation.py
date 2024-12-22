
# Part 1: Performance Evaluation

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Question 1: Import the dataset
url = "https://raw.githubusercontent.com/jackty9/Handling_Imbalanced_Data_in_Python/master/bank-full-encoded.csv"
data = pd.read_csv(url)
print(data.dtypes)

# Question 2: Check for missing values
print(data.isnull().sum())

# Question 3: Split the dataset
X = data.drop('y', axis=1)
y = data['y']

# Question 4: Convert categorical variables to numeric
X_encoded = pd.get_dummies(X)

# Question 5: Divide the dataset
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Question 6: Normalize the dataset
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Question 7: Use KNN to predict
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)

# Question 8: Display confusion matrix and compute accuracy
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Confusion Matrix:\n", conf_matrix)
print("Accuracy:", accuracy)
