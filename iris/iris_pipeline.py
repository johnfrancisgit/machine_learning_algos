from sklearn import (datasets, tree) 
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


iris = datasets.load_iris()

x = iris.data  # Features
y = iris.target  # Labels

# Parition data into two halves for train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

# Create classifier
my_classifier = tree.DecisionTreeClassifier()
# Train model
my_classifier.fit(x_train, y_train)
# Create predictons
predictions = my_classifier.predict(x_test)
print(predictions)
# Test accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy score is {}%".format(accuracy * 100))

