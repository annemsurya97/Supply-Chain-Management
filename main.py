from tkinter import *
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog

import numpy as np 
import pandas as pd 
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, RidgeClassifier, ElasticNet
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score
import os, joblib

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

global filename
global classifier
global X, y, X_train, X_test, y_train, y_test ,Predictions
global dataset, df1, df2, sc, train_or_load_dnn, dnn_model
global le, labels

def upload():
    global filename
    global dataset, df1
    filename = filedialog.askopenfilename(initialdir = "Datasets")
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded\n\n')
    df1= pd.read_csv(filename, encoding='latin1')
    text.insert(END,'\n\n IoT Network Dataset: \n', str(df1))
    text.insert(END,df1)

def preprocess():
    global dataset, df1, df2
    global X, y, X_train, X_test, y_train, y_test, sc, le, labels
    text.delete('1.0', END)

    # Display basic information about the dataset
    #text.insert(END, '\n\nInformation of the dataset: \n', str(df1.info()))
    print(df1.info())
    text.insert(END, '\n\nDescription of the dataset: \n' + str(df1.describe().T))
    text.insert(END, '\n\nChecking null values in the dataset: \n' + str(df1.isnull().sum()))
    text.insert(END, '\n\nUnique values in the dataset: \n' + str(df1.nunique()))
    
        
    # Function to fill NaNs with the most frequent value
    def fill_with_mode(df):
        for column in df.columns:
            most_frequent_value = df1[column].mode()[0]
            df1[column].fillna(most_frequent_value, inplace=True)
    
    # Apply the function
    fill_with_mode(df1)
    
    
    labels=df1['Order Status'].unique()

    # Create a count plot
    sns.set(style="darkgrid")  # Set the style of the plot
    plt.figure(figsize=(8, 6))  # Set the figure size
    ax = sns.countplot(x='Order Status', data=df1)
    plt.title("Count Plot")  # Add a title to the plot
    plt.xlabel("Categories")  # Add label to x-axis
    plt.ylabel("Count")  # Add label to y-axis
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')
    
    print(df1['Order Status'].unique())
    
    le = LabelEncoder()
    for col in df1.columns:
        if df1[col].dtype == 'object':
            df1[col] = le.fit_transform(df1[col])
            
    print(df1.info()) 
    
    y = df1['Order Status'].values
    X = df1.drop(columns=['Order Status'], axis=1).values
    
    print(df1['Order Status'].unique())
    
       
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    # Create a count plot
    sns.set(style="darkgrid")  # Set the style of the plot
    plt.figure(figsize=(8, 6))  # Set the figure size
    ax = sns.countplot(x=y_res, data=df1)
    plt.title("Count Plot")  # Add a title to the plot
    plt.xlabel("Categories")  # Add label to x-axis
    plt.ylabel("Count")  # Add label to y-axis
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')
        
    # Splitting training data and testing data
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=77)    
    text.insert(END, "\n\nTotal Records used for training: " + str(len(X_train)) + "\n")
    text.insert(END, "\n\nTotal Records used for testing: " + str(len(X_test)) + "\n\n")


    # Normalize feature columns except target columns
    for index in range(len(df1.columns) - 3):  # Exclude the last three columns (target columns)
        df1.iloc[:, index] = (df1.iloc[:, index] - df1.iloc[:, index].mean()) / df1.iloc[:, index].std()
        
    '''
    # Display histograms
    df1.hist(figsize=(14, 16))
    plt.title('histplot of all columns')
    #plt.show()

    # Distplot of 's'
    plt.figure(figsize=(10, 6))
    sns.histplot(df1['Order Status'], kde=True, bins=10)
    plt.title('Distplot of Order Status column')
    #plt.show()'''

    # Correlation heatmap
    plt.figure(figsize=(14, 14))
    sns.set(font_scale=1)
    sns.heatmap(df1.corr(), cmap='GnBu_r', annot=True, square=True, linewidths=.5)
    plt.title('Variable Correlation in Heatmap')
    plt.show()


   
precision = []
recall = []
fscore = []
accuracy = []

#function to calculate various metrics such as accuracy, precision etc
def calculateMetrics(algorithm, testY,predict):
    global labels
    
    testY = testY.astype('int')
    predict = predict.astype('int')
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100 
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    print(algorithm+' Accuracy    : '+str(a))
    print(algorithm+' Precision   : '+str(p))
    print(algorithm+' Recall      : '+str(r))
    print(algorithm+' F1-SCORE      : '+str(f))
    text.insert(END, "Performance Metrics of " + str(algorithm) + "\n")
    text.insert(END, "Accuracy: " + str(a) + "\n")
    text.insert(END, "Precision: " + str(p) + "\n")
    text.insert(END, "Recall: " + str(r) + "\n")
    text.insert(END, "F1-SCORE: " + str(f) + "\n\n")
    report=classification_report(predict, testY,target_names=labels)
    print('\n',algorithm+" classification report\n",report)
    text.insert(END, "classification report: \n" + str(report) + "\n\n")
    conf_matrix = confusion_matrix(testY, predict) 
    plt.figure(figsize =(5, 5)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="Blues" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()


def RidgeRegressor():
    global ridge_clf, X_train, X_test, y_train, y_test
    global predict

    Classifier = 'model/RidgeClassifier.pkl'
    if os.path.exists(Classifier):
        # Load the trained model from the file
        ridge_clf = joblib.load(Classifier)
        print("Model loaded successfully.")
        predict = ridge_clf.predict(X_test)
        calculateMetrics("RidgeClassifier", predict, y_test)
    else:
        # Initialize and train the Ridge Classifier model
        ridge_clf = RidgeClassifier()
        ridge_clf.fit(X_train, y_train)
        # Save the trained model to a file
        joblib.dump(ridge_clf, Classifier) 
        print("Model saved successfully.")
        predict = ridge_clf.predict(X_test)
        calculateMetrics("RidgeClassifier", predict, y_test)
    
def DTC():
    global dtc_clf, X_train, X_test, y_train, y_test
    global predict

    Classifier = 'model/DecisionTreeClassifier.pkl'
    if os.path.exists(Classifier):
        # Load the trained model from the file
        dtc_clf = joblib.load(Classifier)
        print("Model loaded successfully.")
        predict = dtc_clf.predict(X_test)
        calculateMetrics("Decision Tree Classifier", predict, y_test)
    else:
        # Initialize and train the Ridge Classifier model
        dtc_clf = DecisionTreeClassifier()
        dtc_clf.fit(X_train, y_train)
        # Save the trained model to a file
        joblib.dump(dtc_clf, Classifier) 
        print("Model saved successfully.")
        predict = dtc_clf.predict(X_test)
        calculateMetrics("Decision Tree Classifier", predict, y_test)    
    
def FFNN():
    global ann_model, rfc_clf, X_train, X_test, y_train, y_test
    global predict, sc

    # Standardize the input features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # One-hot encode the labels
    y_train = to_categorical(y_train, num_classes=9)  # Adjust num_classes to your actual class count
    y_test = to_categorical(y_test, num_classes=9)
    # ANN model file path
    Classifier = 'model/ANNModel.h5'
    RFC_Model = 'model/RFC.pkl'

    if os.path.exists(Classifier) and os.path.exists(RFC_Model):
        # Load both trained models from files
        ann_model = load_model(Classifier)
        print("ANN model loaded successfully.")

        # Get features from ANN for RFC training
        X_train_features = ann_model.predict(X_train)
        X_test_features = ann_model.predict(X_test)
        X_test_features1 = X_test_features.argmax(axis=1)
        #calculateMetrics("ANN", X_test_features1, y_test)

        # Load the trained RFC model
        rfc_clf = joblib.load(RFC_Model)
        print("RFC model loaded successfully.")

        # Make predictions
        predict = rfc_clf.predict(X_test_features)
        predict = predict.argmax(axis=1)
        print('RFC model predicted:', predict)
        print('y_test output:', y_test)
        calculateMetrics("RandomForestClassifier", predict, y_test)
    else:
        # Build and train the ANN model if not already trained
        ann_model = Sequential()
        ann_model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
        ann_model.add(Dense(64, activation='relu'))
        ann_model.add(Dense(64, activation='relu'))
        ann_model.add(Dense(32, activation='relu'))
        ann_model.add(Dense(9, activation='softmax'))  # Adjust output layer based on number of classes

        # Compile the ANN model
        ann_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the ANN model
        ann_model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

        # Save the trained ANN model to a file
        ann_model.save(Classifier)
        print("ANN model saved successfully.")

        # Get features from the ANN model for RFC
        X_train_features = ann_model.predict(X_train)
        X_test_features = ann_model.predict(X_test)
        X_test_features1 = X_test_features.argmax(axis=1)
        #calculateMetrics("ANN", X_test_features1, y_test)

        # Train the RFC model
        rfc_clf = RandomForestClassifier()
        rfc_clf.fit(X_train_features, y_train)

        # Save the trained RFC model to a file
        joblib.dump(rfc_clf, RFC_Model)
        print("RFC model saved successfully.")

        # Make predictions
        predict = rfc_clf.predict(X_test_features)
        predict = predict.argmax(axis=1)
        print('RFC model predicted:', predict)
        print('y_test output:', y_test)
        calculateMetrics("RandomForestClassifier", predict, y_test)
               
def predict():
    global sc, rfc_clf, ann_model, labels
    labels = [
        "normal",            # 0
        "wrong setup",       # 1
        "ddos",              # 2
        "Data type probing", # 3
        "scan attack",       # 4
        "man in the middle"  # 5
    ]
    
    # Load test file
    file = filedialog.askopenfilename(initialdir="Datasets")
    test = pd.read_csv(file)
    
    # Display loaded test data
    text.delete('1.0', END)
    text.insert(END, f'{file} Loaded\n')
    text.insert(END, "\n\nLoaded test data: \n" + str(test) + "\n")
    
    # Remove feature names to avoid StandardScaler issue
    test_values = test.values
    
    # Apply scaling
    test_scaled = sc.transform(test_values)
    
    # Make predictions using the ANN model
    ann_predictions = ann_model.predict(test_scaled)
    
    # RFC model uses the ANN predictions as input (verify this is intended)
    rfc_predictions = rfc_clf.predict(ann_predictions)
    
    # If RFC gives multi-class probabilities, use argmax to get class indices
    predicted_classes = rfc_predictions.argmax(axis=1)
    
    # Map predicted class indices to class labels
    predicted_labels = [labels[p] for p in predicted_classes]
    
    # Add the predicted values to the test data
    test['Predicted'] = predicted_labels
    
    # Display the predictions
    text.insert(END, "\n\nModel Predicted value in test data: \n" + str(test) + "\n")


  
def graph():
    columns = ["Algorithm Name", "Accuracy", "Precision", "Recall", "f1-score"]
    algorithm_names = ["DTC Classification", "FFNN+RF Classification"]
    
    # Combine metrics into a DataFrame
    values = []
    for i in range(len(algorithm_names)):
        values.append([algorithm_names[i], accuracy[i], precision[i], recall[i], fscore[i]])
    
    temp = pd.DataFrame(values, columns=columns)
    text.delete('1.0', END)
    # Insert the DataFrame in the text console
    text.insert(END, "All Model Performance metrics:\n")
    text.insert(END, str(temp) + "\n")
    
    # Plotting the performance metrics
    metrics = ["Accuracy", "Precision", "Recall", "f1-score"]
    index = np.arange(len(algorithm_names))  # Positions of the bars

    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.2  # Width of the bars
    opacity = 0.8

    # Plotting each metric with an offset
    plt.bar(index, accuracy, bar_width, alpha=opacity, color='b', label='Accuracy')
    plt.bar(index + bar_width, precision, bar_width, alpha=opacity, color='g', label='Precision')
    plt.bar(index + 2 * bar_width, recall, bar_width, alpha=opacity, color='r', label='Recall')
    plt.bar(index + 3 * bar_width, fscore, bar_width, alpha=opacity, color='y', label='f1-score')

    # Labeling the chart
    plt.xlabel('Algorithm')
    plt.ylabel('Scores')
    plt.title('Performance Comparison of All Models')
    plt.xticks(index + bar_width, algorithm_names)  # Setting the labels for x-axis (algorithms)
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()


def close():
  main.destroy()

# Main window setup
main = Tk()
main.title("IoT Network Performance Prediction")
main.geometry("1200x800")  # Spacious window size
main.config(bg='#2B3A67')  # Navy Blue background for a sleek look

# Title Label with a gradient-like dark-to-light theme
font = ('Verdana', 20, 'bold')
title = Label(main, text='Enhancing IoT Network Performance Through Predictive Modeling with Machine Learning Regression',
              bg='#282828', fg='#FFD700', font=font, height=2)  # Dark background with Gold text
title.pack(fill=X, pady=10)

# Frame to hold buttons and text console
main_frame = Frame(main, bg='#2B3A67')  # Navy Blue for consistency
main_frame.pack(fill=BOTH, expand=True, padx=20, pady=20)

# Frame to hold buttons (centered and in two rows)
button_frame = Frame(main_frame, bg='#2B3A67')
button_frame.pack(pady=20)

# Button Font and Style
font1 = ('Arial', 12, 'bold')

# Helper function to create buttons with fancy color tones
def create_button(text, command, row, column):
    btn = Button(button_frame, text=text, command=command, bg='#1E90FF', fg='white',  # Dodger Blue buttons
                 activebackground='#FFA07A', font=font1, width=25, relief=RAISED, bd=4)  # Light Salmon hover effect
    btn.grid(row=row, column=column, padx=20, pady=15)

# Adding buttons in two rows, three buttons per row
create_button("Upload IoT Network Dataset", upload, 0, 0)
create_button("Data Preprocessing and EDA", preprocess, 0, 1)
create_button("DTC Classifier", DTC, 0, 2)
create_button("FFNN+RF Classifier", FFNN, 1, 0)
create_button("Performance Metrics Graph", graph, 1, 1)
create_button("Prediction on Test Data", predict, 1, 2)

# Text console styling with scrollbar in fancy tones
text_frame = Frame(main_frame, bg='#2B3A67')  # Consistent background
text_frame.pack(fill=BOTH, expand=True, padx=20, pady=20)

# Text box styling with a centered and modern look
text = Text(text_frame, height=15, width=90, wrap=WORD, bg='#F5DEB3', fg='#483D8B', font=('Comic Sans MS', 14))  # Wheat background
scroll = Scrollbar(text_frame, command=text.yview)
text.configure(yscrollcommand=scroll.set)

text.pack(side=LEFT, fill=BOTH, expand=True)
scroll.pack(side=RIGHT, fill=Y)

# Adding the Close Application button with consistent style and size
close_button = Button(button_frame, text="Close Application", command=close, bg='#B22222', fg='white',  # Firebrick button
                      activebackground='#FF6347', font=font1, width=25, relief=RAISED, bd=4)

# Placing the Close button in the second row, third column (consistent layout)
close_button.grid(row=1, column=2, padx=20, pady=15)


main.mainloop()