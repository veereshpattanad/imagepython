import numpy as np
import matplotlib. pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_curve ,auc
from sklearn.decomposition import PCA

# load the LFW dataset
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# Extract features (eignfaces in this case) amd labels
X = lfw_people.data
y = lfw_people.target

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# standardize featurees by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply principal component analysis (pca) for dimensionality  reduction
n_components = 150
pca = PCA(n_components=n_components, whiten=True, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# initialize svm classifer
svm_classifier = SVC(kernel='rbf', C =1000.0, gamma=0.001, probability=True, random_state=42)

# Train the svm classifier
svm_classifier.fit(X_train_pca, y_train)

# Predict the probabilites for each class for test set
y_score = svm_classifier.predict_proba(X_test_pca)

# calculate accuracy
accuracy = accuracy_score(y_test, svm_classifier.predict(X_test_pca))
print("Accuracy:", accuracy * 100)

# compute ROC curve and ROC area
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(lfw_people.target_names)):
    fpr[i], tpr[i], _ = roc_curve(np.array(y_test == i), y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(8, 6))
for i in range(len(lfw_people.target_names)):
    plt.plot(fpr[i], tpr[i],
             label='ROC curve for class{} (AUC = {:.2f})'.format(lfw_people.target_names[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positve rate')
plt.title('Receiver Operating characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()
