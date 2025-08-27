import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the dataset
data = pd.read_csv('online_shoppers_intention.csv')

# Fill missing values
data.fillna(0, inplace=True)

# ---------------------
# PART 1: K-MEANS CLUSTERING
# ---------------------
x = data[['BounceRates', 'ProductRelated_Duration']].values

# Elbow Method to find optimal number of clusters
wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    km.fit(x)
    wcss.append(km.inertia_)

# Elbow Method Plot (first graph)
plt.figure(figsize=(12, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal K', fontsize=16)
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()  # Show the first graph individually

# Apply KMeans with chosen cluster number (3)
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=0)
y_kmeans = kmeans.fit_predict(x)

# Plot clusters (second graph)
plt.figure(figsize=(12, 6))
colors = ['red', 'green', 'blue']
for i in range(3):
    plt.scatter(x[y_kmeans == i, 0], x[y_kmeans == i, 1],
                s=50, c=colors[i], label=f'Cluster {i}')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=200, c='yellow', label='Centroids', marker='X')
plt.title('KMeans Clustering Results')
plt.xlabel('BounceRates')
plt.ylabel('ProductRelated_Duration')
plt.legend()
plt.grid(True)
plt.show()  # Show the second graph individually

# ---------------------
# PART 2: SUPERVISED LEARNING & CONFUSION MATRIX
# ---------------------
# Encode categorical variables
le = LabelEncoder()
data['VisitorType'] = le.fit_transform(data['VisitorType'])
data['Month'] = le.fit_transform(data['Month'])
data['Weekend'] = data['Weekend'].astype(int)
data['Revenue'] = data['Revenue'].astype(int)

# Select features and target
X = data.drop('Revenue', axis=1)
y = data['Revenue']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier (Random Forest)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Purchase', 'Purchase'])

# Plot confusion matrix (third graph)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()  # Show the third graph individually

# ---------------------
# Display All Plots Together
# ---------------------
# Create a 2x2 subplot layout
fig, axs = plt.subplots(2, 2, figsize=(14, 12))

# Elbow Method Plot
axs[0, 0].plot(range(1, 11), wcss, marker='o')
axs[0, 0].set_title('Elbow Method for Optimal K', fontsize=16)
axs[0, 0].set_xlabel('Number of Clusters')
axs[0, 0].set_ylabel('WCSS')
axs[0, 0].grid(True)

# Plot KMeans Clusters
for i in range(3):
    axs[0, 1].scatter(x[y_kmeans == i, 0], x[y_kmeans == i, 1],
                      s=50, c=colors[i], label=f'Cluster {i}')
axs[0, 1].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                  s=200, c='yellow', label='Centroids', marker='X')
axs[0, 1].set_title('KMeans Clustering Results')
axs[0, 1].set_xlabel('BounceRates')
axs[0, 1].set_ylabel('ProductRelated_Duration')
axs[0, 1].legend()
axs[0, 1].grid(True)

# Confusion Matrix
disp.plot(cmap=plt.cm.Blues, ax=axs[1, 0])
axs[1, 0].set_title('Confusion Matrix')

# Hide empty subplot
axs[1, 1].axis('off')

# Display all plots
plt.tight_layout()
plt.show()