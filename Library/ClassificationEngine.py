from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from typing import List, Any

import numpy as np

class ClassificationEngine:
    def __init__(self):
        """
        Initializes the DataPreprocessor class.
        """
        pass    
    
    def fitAndPredictAll(self, classifiers: List[Any], X: np.ndarray, Y: np.ndarray) -> None:
        """
        # Define classification algorithms
        g = GaussianNB()
        b = BernoulliNB()
        k = KNeighborsClassifier()
        l = LogisticRegression()
        d = DecisionTreeClassifier()
        r = RandomForestClassifier()
        h = GradientBoostingClassifier()
    
        # Add the algorithms you define to the array
        myAlgorithmArray = [g, b, k, l, d, r, h] 
        
        Trains given classification algorithms with the dataset and calculates and prints accuracy score, precision score, 
        confusion matrix, and classification report for the prediction results.

        Parameters:
        - classifiers (List[Any]): List of instantiated classification algorithms.
        - X (np.ndarray): The feature matrix for training and testing.
        - Y (np.ndarray): The target vector for training and testing.
        """
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=13)

        results = []  # Store results for tabulation

        for classifier in classifiers:
            classifier_name = type(classifier).__name__
            classifier.fit(x_train, y_train)
            y_pred = classifier.predict(x_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            # Note: Directly printing confusion_matrix or classification_report can clutter the output
            # Consider storing these values if needed for further analysis

            results.append([classifier_name, accuracy, precision])

        # Print the results in a tabular format
        print(tabulate(results, headers=["Classifier", "Accuracy Score", "Precision Score"]))