# import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc


df = pd.read_csv("diabetes.csv")
df.head(10)

X = df.iloc[:, 0:-1].values
y = df.iloc[:, -1].values


# Train test-split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)


# Svm classifier
model_SVC = SVC(kernel="rbf", random_state=42)
model_SVC.fit(X_train, y_train)
y_pred_svm = model_SVC.decision_function(X_test)


# Logisitic Classifier
model_logistic = LogisticRegression()
model_logistic.fit(X_train, y_train)
y_pred_logistic = model_logistic.decision_function(X_test)


# Plot ROC and AUC
logistic_fpr, logistic_tpr, thresold = roc_curve(y_test, y_pred_logistic)
auc_logistic = auc(logistic_fpr, logistic_tpr)


svm_fpr, svm_tpr, thresold = roc_curve(y_test, y_pred_svm)
auc_svm = auc(svm_fpr, svm_tpr)

plt.figure(figsize=(5, 5), dpi=100)
plt.plot(svm_fpr, svm_tpr, linestyle="-", label="SVM (auc=%0.3f)" % auc_svm)
plt.plot(
    logistic_fpr,
    logistic_tpr,
    linestyle="-",
    label="Logistic (auc=%0.3f)" % auc_logistic,
)

plt.xlabel("False positive rate --->")
plt.ylabel("True positive rate ---->")

plt.legend()

plt.show()


""" So the logistic regression has better auc than svm so we will use this instead of SVM)"
"""
