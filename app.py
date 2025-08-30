import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import sys
import os

def main():
    # Load MNIST training data
    data_path = 'mnist_test.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        sys.exit(1)
    data = pd.read_csv(data_path)
    print("Data loaded. Shape:", data.shape)
    print(data.head())

    # Visualize the 4th image
    img = data.iloc[3, 1:].values.reshape(28, 28).astype('uint8')
    plt.imshow(img, cmap='gray')
    plt.title(f"Label: {data.iloc[3, 0]}")
    plt.show()  # <-- Use blocking show for the first image

    # Prepare features and labels
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]

    # Split into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
    print(f"Training set size: {x_train.shape}")
    print(f"Test set size: {x_test.shape}")

    # Train Random Forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(x_train, y_train)
    print("Model training complete.")

    # Predict on test set
    pred = rf.predict(x_test)

    # Calculate accuracy
    accuracy = np.mean(pred == y_test.values)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")

    # Show classification report
    print("\nClassification Report:")
    print(classification_report(y_test, pred))

    # Show confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred))

    # Show a few predictions
    for i in range(5):
        img = x_test.iloc[i].values.reshape(28, 28).astype('uint8')
        plt.imshow(img, cmap='gray')
        plt.title(f"Predicted: {pred[i]}, Actual: {y_test.values[i]}")
        plt.pause(1)  # <-- Pause for 1 second
        plt.clf()     # <-- Clear the figure for the next image

if __name__ == "__main__":
    main()

