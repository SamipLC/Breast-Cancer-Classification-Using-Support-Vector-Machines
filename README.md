# Breast Cancer Classification Using Support Vector Machines

This project demonstrates the classification of breast cancer tumors (malignant vs. benign) using Support Vector Machines (SVM). It explores both Linear and Non-Linear (Radial Basis Function) kernels to find the optimal decision boundary for the data.

## Dataset
The project utilizes the Wisconsin Breast Cancer dataset provided by `scikit-learn` (`load_breast_cancer`). It contains 30 numerical features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass, with a total of 569 samples.

## Project Workflow
1. **Data Preprocessing:** The dataset is split into training (70%), validation (15%), and test (15%) sets. Features are scaled using `StandardScaler` to ensure unit variance and zero mean, which is critical for SVM distance calculations.
2. **Linear SVM:** Models are trained using a linear kernel to explore the effect of the regularization parameter `C` on validation accuracy and the trade-off between underfitting and overfitting.
3. **Non-Linear SVM (RBF Kernel):** A grid search approach is used to tune both `C` and `gamma` parameters, allowing the model to capture non-linear relationships without overfitting the training data.
4. **Final Evaluation:** The best hyperparameter combination is evaluated on the unseen test set using metrics like Accuracy, Precision, Recall, F1-Score, and a Confusion Matrix.
5. **Decision Boundary Visualization:** A 2D visualization (using mean radius and mean texture) compares the rigid geometric boundary of the Linear SVM against the flexible, contoured boundary of the RBF SVM.

## Best Model Results
The optimal model was identified as the **RBF SVM** with `C=10` and `gamma=0.001`. 

Performance metrics on the final test set:
* **Accuracy:** 0.9535
* **Precision:** 0.9310
* **Recall:** 1.0000
* **F1 Score:** 0.9643

## Key Concepts Explored
* **Feature Scaling:** Standardizing features prevents variables with larger ranges from dominating the Euclidean distance calculation of the SVM margin.
* **Regularization (C):** Controls the penalty for misclassified points. A low `C` increases the margin but allows more violations (underfitting), while a high `C` creates a strict margin (overfitting).
* **Gamma (γ):** Controls the influence radius of individual support vectors in the RBF kernel. High gamma limits influence (jagged boundaries, overfitting), while low gamma expands influence (smooth boundaries, underfitting).

## Requirements
* `numpy`
* `matplotlib`
* `scikit-learn`
