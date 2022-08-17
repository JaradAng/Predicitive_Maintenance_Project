import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression, LinearRegression, LassoLars, TweedieRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier




import warnings
warnings.filterwarnings("ignore")


def prep_maint():


    df = pd.read_csv('predictive_maintenance.csv')
   
   #Rename Cols to be more friendly
    df.rename(columns = {'Air temperature [K]':'air_temp', 'Process temperature [K]':'process_temp','Rotational speed [rpm]':'rpm', 'Torque [Nm]':'torque', 'Tool wear [min]': 'tool_wear', 'Target':  'machine_failure', 'Failure Type':  'failure_type' }, inplace = True)

    # Create dummy variables for type
    dummy_df = pd.get_dummies(df['Type'])

    #Concat dummy df to the regular df
    df = pd.concat([df, dummy_df], axis=1)
    
    # Dropping unnecassary columns
    df = df.drop(columns=['UDI', 'Product ID', 'Type'])

    return df


# Splitting the data set via train validate test
def split_maint(df):

    # split the data
    train_validate, test = train_test_split(df, test_size=.2, 
                                            random_state=123, 
                                            stratify=df.machine_failure)
    train, validate = train_test_split(train_validate, test_size=.2, 
                                       random_state=123, 
                                       stratify=train_validate.machine_failure)
    return train, validate, test

def split_maint_fail_type(df):

    # split the data
    train_validate, test = train_test_split(df, test_size=.2, 
                                            random_state=123, 
                                            stratify=df.failure_type)
    train, validate = train_test_split(train_validate, test_size=.2, 
                                       random_state=123, 
                                       stratify=train_validate.failure_type)
    return train, validate, test


def toque_bin(df):
    # create a categorical feature
    df['torque_bin'] = pd.qcut(df.air_temp, 3, labels=['low', 'medium', 'high'])

    # Plotting out internet service type against churn with overal churn mean as the black line
    sns.barplot('torque_bin', 'machine_failure', data=df, alpha=.5)
    plt.axhline(df.machine_failure.mean(), ls = '--', color = 'black')
    plt.xlabel('Torque bin')
    plt.ylabel('Machine Failure')
    plt.title('Torque bins against failure average')


def process_bin(df):
    # create a categorical feature
    df['process_temp_bin'] = pd.qcut(df.process_temp, 3, labels=['cool', 'mild', 'hot'])

    # Plotting out internet service type against churn with overal churn mean as the black line
    sns.barplot('process_temp_bin', 'machine_failure', data=df, alpha=.5)
    plt.axhline(df.machine_failure.mean(), ls = '--', color = 'black')
    plt.xlabel('Process temp bin')
    plt.ylabel('Machine failure avg')
    plt.title('Process temp bins against failure average')

def tool_bin(df):
    # create a categorical feature
    df['tool_wear_bin'] = pd.qcut(df.tool_wear, 3, labels=['new', 'med', 'old'])

    # Plotting out internet service type against churn with overal churn mean as the black line
    sns.barplot('tool_wear_bin', 'machine_failure', data=df, alpha=.5)
    plt.axhline(df.machine_failure.mean(), ls = '--', color = 'black')
    plt.xlabel('Tool Wear bin')
    plt.ylabel('Machine failure avg')
    plt.title('Tool wear bins against failure average')


def rpm_bin(df):
    # create a categorical feature
    df['rpm_bin'] = pd.qcut(df.rpm, 3, labels=['slow', 'medium', 'fast'])
    # Plotting out internet service type against churn with overal churn mean as the black line
    sns.barplot('rpm_bin', 'machine_failure', data=df, alpha=.5)
    plt.axhline(df.machine_failure.mean(), ls = '--', color = 'black')
    plt.xlabel('Service Type')
    plt.ylabel('Failure Rate')
    plt.title('RPM speed bins against failure average')


def air_bin(df):
    # create a categorical feature
    df['air_temp_bin'] = pd.qcut(df.process_temp, 3, labels=['cool', 'mild', 'hot'])

    # Plotting out internet service type against churn with overal churn mean as the black line
    sns.barplot('air_temp_bin', 'machine_failure', data=df, alpha=.5)
    plt.axhline(df.machine_failure.mean(), ls = '--', color = 'black')
    plt.xlabel('Air temp bin')
    plt.ylabel('Machine failure rate')
    plt.title('Air temp bins against failure average')

def drop_bins(df):

    df = df.drop(columns= ['air_temp_bin', 'rpm_bin', 'tool_wear_bin', 'process_temp_bin', 'torque_bin'])

    return df


def MinMax_scaler(train, validate, test):
 
    scaler = MinMaxScaler().fit(train)

    scaler.fit(train)

        #Columns to scale
    cols = ['air_temp', 'process_temp', 'rpm', 'torque', 'tool_wear']
    # Fit numerical features to scaler
    scaler.fit(train[cols])
    # Set the features to transformed value
    train[cols] = scaler.transform(train[cols])
    validate[cols] = scaler.transform(validate[cols])
    test[cols] = scaler.transform(test[cols])
    
    x_train_scaled = pd.DataFrame(scaler.transform(train), index=train.index, columns= train.columns)
    x_validate_scaled = pd.DataFrame(scaler.transform(validate), index=validate.index, columns=validate.columns)
    x_test_scaled = pd.DataFrame(scaler.transform(test), index=test.index, columns = test.columns)
    
    return x_train_scaled, x_validate_scaled, x_test_scaled



# def MinMax_scaler(x_train, x_validate, x_test):
 
#     scaler = MinMaxScaler().fit(x_train)

#     scaler.fit(x_train)
    
#     x_train_scaled = pd.DataFrame(scaler.transform(x_train), index=x_train.index, columns=x_train.columns)
#     x_validate_scaled = pd.DataFrame(scaler.transform(x_validate), index=x_validate.index, columns=x_validate.columns)
#     x_test_scaled = pd.DataFrame(scaler.transform(x_test), index=x_test.index, columns = x_test.columns)
    
#     return x_train_scaled, x_validate_scaled, x_test_scaled


def decision_tree(x_train, y_train, x_validate, y_validate):
    # Making the model for Random Forest using the max depth found in exploration loop and setting other hyperparameters
    tree2 = DecisionTreeClassifier(max_depth=4, random_state=100)

    tree2.fit(x_train, y_train)


    y_pred2 = tree2.predict(x_train)
   
    y_v_pred = tree2.predict(x_validate)

  
    confusion_matrix(y_train, y_pred2)

    print(classification_report(y_train, y_pred2))


    print(classification_report(y_validate, y_v_pred))





def random_forest(x_train, y_train, x_validate, y_validate):
    
    rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', min_samples_leaf=5, n_estimators=200, max_depth=4, random_state=123)

    
    rf.fit(x_train, y_train)

   
    y_pred = rf.predict(x_train)

    y_val_pre = rf.predict(x_validate)
    
    print(classification_report(y_train, y_pred))

    print(classification_report(y_validate, y_val_pre))

   



def k_neighbors(x_train, y_train, x_validate, y_validate):

    knn1 = KNeighborsClassifier(n_neighbors=2)

    knn1.fit(x_train, y_train)

    y_pred = knn1.predict(x_train)

    y_v_pred = knn1.predict(x_validate)

 

    print(classification_report(y_train, y_pred))

    print(classification_report(y_validate, y_v_pred))


def svm_model(x_train, y_train, x_validate, y_validate):
    svm_model_linear = SVC(kernel = 'linear', C = 10).fit(x_train, y_train)
    svm_predictions = svm_model_linear.predict(x_validate)
  
    # model accuracy for X_test  
    accuracy = svm_model_linear.score(x_validate, y_validate)
  
    # creating a confusion matrix
    cm = confusion_matrix(y_validate, svm_predictions)

    y_pred = svm_model_linear.predict(x_train)

    y_v_pred = svm_model_linear.predict(x_validate)

 
    print(classification_report(y_train, y_pred))

    print(classification_report(y_validate, y_v_pred))



def xg_model(x_train, y_train, x_validate, y_validate):

    # fit model no training data
    model = XGBClassifier()
    
    model.fit(x_train, y_train)

    y_pred = model.predict(x_train)

    y_v_pred = model.predict(x_validate)

 
    print(classification_report(y_train, y_pred))

    print(classification_report(y_validate, y_v_pred))