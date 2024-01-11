This repository was created to publish the [Titanic challenge](https://www.kaggle.com/competitions/titanic)

The code was divided into 5 steps, in search of the best possible result, each with different alternatives, which will be explained below, what was done and why.

## [FIRST CODE](https://github.com/LucasNatalePires/kaggle_titanic/blob/main/titanic_version1.ipynb):

  - Handling null data using [mean()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.mean.html) and
[mode()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.mode.html)

  - Due to the high cardinality, detected by the [nunique()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.nunique.html)
function of some columns, at this stage, I chose to execute them since there was no pattern initially

  - I excluded the 'Embarked' column because it had string values. At first, I tested the model's accuracy without treating it

  - I created 3 different models using [KNC](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html),
[Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) and
[Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
I also tested the [accuracy](https://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score) and
[Matrix Confusion](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) of the respective models

Score: 0.66746


## [SECOND CODE](https://github.com/LucasNatalePires/kaggle_titanic/blob/main/titanic_version2.ipynb):

  - In addition to everything that was done in the first code, the only addition was:
    - I treated the 'Embarked' column, considering that the variables contained in it were of the string type, therefore,
the [One Hot Encoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
algorithm models would not work

Score: 0.76555


## [THIRD CODE](https://github.com/LucasNatalePires/kaggle_titanic/blob/main/titanic_version3.ipynb)
  
  - I used [Robust Scaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html) 
to scale the 'Age' and 'Fare' columns, very discrepant values.

  - These values ​​can be easily detected using [Mat Plot Lib](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html#matplotlib.pyplot.subplots)

  - Creation of columns from the 'SibSp' and 'Parch' columns seeking the best accuracy

  - Correlation of variables to understand what can be created/deleted

Score: 0.76555


## [FOURTH CODE](https://github.com/LucasNatalePires/kaggle_titanic/blob/main/titanic_version4.ipynb):

- In this stage, all treatments already carried out in the previous stage were applied.

- In addition to Random Forest, I applied [MLP Classifier (neural networks)](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier) to select the best parameters

- Despite the apparent improvement, there was **Overfitting**(basically when the algorithm works very well for training, but does not perform the same in testing)

Score: 0.69856


## [FINAL CODE](https://github.com/LucasNatalePires/kaggle_titanic/blob/main/titanic_version5.ipynb)

  - To solve the problem of **Overfitting**I used [Grid Search CV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
to find the best parameters

  - In the end, we used **Random Forest** to make the submission and had an improvement compared to the previous code

Score: 0.7799
