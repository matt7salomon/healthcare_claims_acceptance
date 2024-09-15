### **Overview of LightGBM (LGBMClassifier)**

**LightGBM** is a gradient boosting framework that uses tree-based learning algorithms. It is designed to be highly efficient and scalable, handling large datasets and high-dimensional features well. LightGBM builds trees in a leaf-wise manner rather than a level-wise manner, which often results in better performance and faster computation. 

### **Code Breakdown**

1. **Imports and Helper Functions:**

   - The code imports necessary libraries, including pandas, numpy, and matplotlib, along with machine learning libraries from scikit-learn and xgboost. Note that `LGBMClassifier` from LightGBM is used but not imported in the code snippet provided.

   - **`get_error_rate(pred, Y)`**: Calculates the error rate by comparing predictions with actual labels.
   - **`print_error_rate(err)`**: Prints error rates for training and test datasets.
   - **`generic_clf(Y_train, X_train, Y_test, X_test, clf)`**: A generic function to fit a classifier and return error rates for both training and test datasets.

2. **AdaBoost Implementation:**

   - **`adaboost_clf(Y_train, X_train, Y_test, X_test, M, clf)`**: Implements the AdaBoost algorithm using a given classifier. AdaBoost works by fitting multiple weak learners (e.g., decision trees) sequentially, with each learner focusing on the errors made by the previous ones. It uses weighted samples to adaptively change the focus of the model training.

3. **LightGBM Implementation:**

   - **`lgb_clf(Y_train, X_train, Y_test, X_test, M)`**: This function implements a boosting approach similar to AdaBoost but using LightGBM.

   **Process:**
   - **Initialize Weights**: Sets equal weights for all training samples initially.
   - **Train Classifier**: Fits the LightGBM model to the training data.
   - **Predict**: Makes predictions on both training and test datasets.
   - **Calculate Error and Update Weights**: Calculates the error rate, updates weights based on prediction errors, and combines predictions from each iteration.
   - **Return Error Rates**: Returns the final error rates for the training and test datasets.

4. **Plotting Function:**

   - **`plot_error_rate(er_train, er_test)`**: Plots the error rates for training and test datasets against the number of iterations, allowing for a visual comparison of performance.

5. **Main Script:**

   - **Data Preparation**:
     - Uses `make_hastie_10_2()` to generate a synthetic dataset for binary classification.
     - Splits the data into training and test sets.

   - **Model Training and Evaluation**:
     - Trains a simple decision tree (`DecisionTreeClassifier`) and evaluates it.
     - Runs the `lgb_clf` function with LightGBM across different iteration ranges and tracks error rates.
     - Plots the error rates to visualize performance.





## Implementation of AdaBoost classifier

### Description

This is an implementation of the AdaBoost algorithm for a two-class classification problem. The algorithm sequentially applies a weak classification to modified versions of the data. By increasing the weights of the missclassified observations, each weak learner focuses on the error of the previous one. The predictions are aggregated through a weighted majority vote. 

### Methods
Adaboost algorithm: <br />
<img src="https://github.com/jaimeps/adaboost-implementation/blob/master/images/adaboost_algo.png" width="600"> <br />

### Example
Using the Hastie (10.2) dataset, we can appreciate a significant reduction in the error rate as we increase the number of iterations. <br />
<img src="https://github.com/jaimeps/adaboost-implementation/blob/master/images/error_rate.png" width="500"> <br />

### References
- Trevor Hastie, Robert Tibshirani, Jerome Friedman - *The Elements of Statistical Learning*
- https://github.com/jaimeps/adaboost-implementation/
