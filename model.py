import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


#creating a function to perform logistic regression models quicker
def logistic_regression(X_train, y_train):
    '''
    This function creates a regression algorithm on the train dataset
    '''
    #defining logistic regression function
    logit = LogisticRegression()
    #fitting the data into the model
    logit.fit(X_train, y_train)
    
    #creating a list comprehension for the column names
    names = [column for column in X_train.columns]
    #adding intercept to the end of the list
    names.append('intercept')
    #creating a dataframe from the regression coefficient values and intercept
    coeff = pd.DataFrame(np.append(logit.coef_, logit.intercept_)).T
    #renaming the column names with the list of names
    coeff.columns = names
    
    # 'logit.predict' predicts class labels for samples in the parenthesis
    y_pred = logit.predict(X_train)
    # 'predict_prob' predicts probability estimates
    # y_pred_proba = logit.predict_proba(X_train)
    
    #creates a confusion matrix to see how accurate the model is
    cm = pd.DataFrame(confusion_matrix(y_train, y_pred))
    
    #creating a classification report and saving it as a DataFrame
    class_report = pd.DataFrame(classification_report(y_train, y_pred, output_dict=True))
    return coeff, cm, class_report



## creating a function to perform decision trees quicker
def decision_tree(X_train, y_train, depth_number):
    '''
    This function uses classification algorithms on the train dataset
    '''
    #defining DecisionTreeClassifier and setting max depth number
    clf = DecisionTreeClassifier(max_depth= depth_number, random_state=123)
    #fitting the data to the model
    clf.fit(X_train, y_train)
    # 'logit.predict' predicts class labels for samples in the parenthesis
    y_pred = clf.predict(X_train)
    # 'predict_proba' predicts porbability estimates
    # y_pred_proba = clf.predict_proba(X_train)
    #creating a confusion matrix and storing it in a DataFrame
    cm = pd.DataFrame(confusion_matrix(y_train, y_pred))
    #creating a classification report and saving it as a DataFrame
    class_report = pd.DataFrame(classification_report(y_train, y_pred, output_dict=True))
    return cm, class_report




def random_forest(X_train, y_train, min_sample, maximum_depth):
    '''
    This function uses a machine learning algorithm called bagging on the test dataset
    '''
    #defining Random Forest function and setting min_sample and max_depth
    rf = RandomForestClassifier(min_samples_leaf= min_sample , max_depth = maximum_depth, random_state = 123)
    #fitting the function
    rf.fit(X_train,y_train)
    #making a prediction
    y_pred = rf.predict(X_train)
    #creating a confusion matrix and storing it in a DataFrame
    cm = pd.DataFrame(confusion_matrix(y_train, y_pred))
    #creating a classification report and saving it as a DataFrame
    class_report = pd.DataFrame(classification_report(y_train, y_pred, output_dict=True))
    return cm, class_report




def kneighbors(X_train, y_train, n_neighbor):
    '''
    This function uses a classification algorithm called k-nearest neighbor on the train dataset
    '''
    #defining the function and setting the neighbors
    knn = KNeighborsClassifier(n_neighbors=n_neighbor)
    #fitting the function to the model
    knn.fit(X_train, y_train)
    #making a prediction
    y_pred = knn.predict(X_train)
    #creating a confusion matrix and storing it in a dataframe
    cm = pd.DataFrame(confusion_matrix(y_train, y_pred))
    #creating a classification report and saving it as a DataFrame
    class_report = pd.DataFrame(classification_report(y_train, y_pred, output_dict=True))
    return cm, class_report


def logistic_regression_validate(X_train, y_train, X_validate, y_validate):
    '''
    This function creates a regression algorithm on the validate/test dataset
    '''
    #defining logistic regression function
    logit = LogisticRegression()
    #fitting the data into the model
    logit.fit(X_train, y_train)
    
    #creating a list comprehension for the column names
    names = [column for column in X_train.columns]
    #adding intercept to the end of the list
    names.append('intercept')
    #creating a dataframe from the regression coefficient values and intercept
    coeff = pd.DataFrame(np.append(logit.coef_, logit.intercept_)).T
    #renaming the column names with the list of names
    coeff.columns = names
    
    # 'logit.predict' predicts class labels for samples in the parenthesis
    y_pred = logit.predict(X_validate)
    # 'predict_prob' predicts probability estimates
    # y_pred_proba = logit.predict_proba(X_train)
    
    #creates a confusion matrix to see how accurate the model is
    cm = pd.DataFrame(confusion_matrix(y_validate, y_pred))
    
    #creating a classification report and saving it as a DataFrame
    class_report = pd.DataFrame(classification_report(y_validate, y_pred, output_dict=True))
    return coeff, cm, class_report

def decision_tree_validate(X_train, y_train, X_validate, y_validate, depth_number):
    '''
    This function uses classification algorithms on the validate/test dataset
    '''
    #defining DecisionTreeClassifier and setting max depth number
    clf = DecisionTreeClassifier(max_depth= depth_number, random_state=123)
    #fitting the data to the model
    clf.fit(X_train, y_train)
    # 'logit.predict' predicts class labels for samples in the parenthesis
    y_pred = clf.predict(X_validate)
    # 'predict_proba' predicts porbability estimates
    # y_pred_proba = clf.predict_proba(X_train)
    #creating a confusion matrix and storing it in a DataFrame
    cm = pd.DataFrame(confusion_matrix(y_validate, y_pred))
    #creating a classification report and saving it as a DataFrame
    class_report = pd.DataFrame(classification_report(y_validate, y_pred, output_dict=True))
    return cm, class_report


def random_forest_validate(X_train, y_train, X_validate, y_validate, min_sample, maximum_depth):
    '''
    This function uses a machine learning algorithm called bagging on the validate/test dataset
    '''
    #defining Random Forest function and setting min_sample and max_depth
    rf = RandomForestClassifier(min_samples_leaf= min_sample , max_depth = maximum_depth, random_state = 123)
    #fitting the function
    rf.fit(X_train,y_train)
    #making a prediction
    y_pred = rf.predict(X_validate)
    #creating a confusion matrix and storing it in a DataFrame
    cm = pd.DataFrame(confusion_matrix(y_validate, y_pred))
    #creating a classification report and saving it as a DataFrame
    class_report = pd.DataFrame(classification_report(y_validate, y_pred, output_dict=True))
    return cm, class_report

def kneighbors_validate(X_train, y_train, X_validate, y_validate, n_neighbor):
    '''
    This function uses a classification algorithm called k-nearest neighbor on the validate/test dataset
    '''
    #defining the function and setting the neighbors
    knn = KNeighborsClassifier(n_neighbors=n_neighbor)
    #fitting the function to the model
    knn.fit(X_train, y_train)
    #making a prediction
    y_pred = knn.predict(X_validate)
    #creating a confusion matrix and storing it in a dataframe
    cm = pd.DataFrame(confusion_matrix(y_validate, y_pred))
    #creating a classification report and saving it as a DataFrame
    class_report = pd.DataFrame(classification_report(y_validate, y_pred, output_dict=True))
    return cm, class_report