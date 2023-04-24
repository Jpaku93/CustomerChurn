# streamlit application to insert csv data and predict the output using machine learning
# import libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score



# import warnings filter
from warnings import simplefilter
from sklearn.metrics import recall_score, precision_score, f1_score
# encode categorical data
from sklearn.preprocessing import LabelEncoder
# scale features with more than n unique values
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

def preprocess(dataset, uniquevaluesthresh = 10, test_size = 0.2, random_state = 0):
    mm = MinMaxScaler()
    le = LabelEncoder()

    # encode object type columns
    for col in dataset.columns:
        if dataset[col].dtype == 'object':
            dataset[col] = le.fit_transform(dataset[col])
            # SCALING
            if dataset[col].nunique() > uniquevaluesthresh:
                dataset[col] = mm.fit_transform(dataset[col].values.reshape(-1,1))
                    

    # confirm encoding by checking types
    if col in dataset.columns:
        if dataset[col].dtype != 'object':
            print('All columns are encoded')

    # split dataset into features and target
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # split dataset into training and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)

    return X_train, X_test, y_train, y_test

# create a function to calculate the scores
def scores(y_test, y_pred):
    # calculate the scores
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return precision, recall, f1

# streamlit title
st.title("Machine Learning Application")
# streamlit subheader
st.subheader("Insert the data to predict the output")

# side bar
st.sidebar.title("Machine Learning Application")
st.sidebar.subheader("Insert the data to predict the output")   
# streamlit file uploader
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
if uploaded_file is not None:
    # make an option to choose data description or data prediction
    option = st.sidebar.selectbox('Choose an option',('Data Description', 'Data Analysis', 'Data Prediction'))
    # if option is data description
    if option == 'Data Description':
        # display the uploaded file
        df = pd.read_csv(uploaded_file)
        st.write(df)
        # display the shape of the uploaded file
        st.write("Shape")
        st.write(df.shape)
        
        # display the summary of the uploaded file
        st.write("Summary")
        st.write(df.describe())

        # display the correlation of the uploaded file
        st.write("Correlation")
        st.write(df.corr())

        # display the covariance of the uploaded file
        st.write("Covariance")
        st.write(df.cov())

    elif option == 'Data Analysis':
        st.sidebar.subheader("Data Analysis")
        # preprocess the uploaded file
        df = pd.read_csv(uploaded_file) 
        # Pie chart for the last column
        st.sidebar.subheader("Pie Chart")
        # pie chart for the last column 
        fig, ax = plt.subplots()
        ax.pie(df[df.columns[-1]].value_counts(), labels = df[df.columns[-1]].value_counts().index, autopct = '%1.1f%%')
        ax.axis('equal')
        st.pyplot(fig)

        # plot histogram of continous columns
        st.sidebar.subheader("Histogram")
        # find the continous columns and has more than 5 unique values
        cont = []
        for col in df.columns:
            if df[col].dtype != 'object':
                if df[col].nunique() > 5:
                    cont.append(col)
        # plot histogram of continous columns
        fig, ax = plt.subplots()
        ax.hist(df[cont])
        ax.legend(cont)
        st.pyplot(fig)





    elif option == 'Data Prediction':
        models = [] # list of models
        accuracy = [] # accuracy score
        precision = [] # precision score
        recall = [] # recall score
        f1 = [] # f1 score
        cva = [] # cross validation accuracy

        # preprocess the uploaded file
        df = pd.read_csv(uploaded_file)
        # pop the last column
        target = df.pop(df.columns[-1])
        
        # multi select box to choose the model
        classifier = st.sidebar.multiselect('Choose the model',('Logistic Regression','Random Forest Regression','KNN Classifier','XGBoost','LightGBM'))

        # checkbox to add cross validation
        cross_validation = st.sidebar.checkbox('Cross Validation')

        # multi select box to choose the columns to display
        multi = st.sidebar.multiselect('Choose the columns to display', df.columns)

        if multi:
            df1 = df[multi]
            # concat the target column to the end of the dataframe
            df1 = pd.concat([df1, target], axis = 1)
            st.write(df1) 
            
        else:
            df1 = df
            # concat the target column to the end of the dataframe
            df1 = pd.concat([df1, target], axis = 1)
            st.write(df1)
            

        X_train, X_test, y_train, y_test = preprocess(df1)
        # if logistic regression is chosen 
        if 'Logistic Regression' in classifier:
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression()
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)
            models.append('Logistic Regression')
            accuracy.append(lr.score(X_test, y_test))
            prec, rec, f = scores(y_test, y_pred)
            precision.append(prec)
            recall.append(rec)
            f1.append(f)
            if cross_validation:
                # cross validation
                accuracies = cross_val_score(estimator = lr, X = X_train, y = y_train, cv = 10)
                cva.append(accuracies.mean())

        # if random forest regression is chosen
        if 'Random Forest Regression' in classifier:
            from sklearn.ensemble import RandomForestClassifier
            rfr = RandomForestClassifier(n_estimators = 10, random_state = 0)
            rfr.fit(X_train, y_train)
            y_pred = rfr.predict(X_test)
            models.append('Random Forest Regression')
            accuracy.append(rfr.score(X_test, y_test))
            rfprec, rfrec, rff = scores(y_test, y_pred)
            precision.append(rfprec)
            recall.append(rfrec)
            f1.append(rff)
            if cross_validation:
                # cross validation
                accuracies = cross_val_score(estimator = rfr, X = X_train, y = y_train, cv = 10)
                cva.append(accuracies.mean())

        # if KNN regression is chosen
        if 'KNN Classifier' in classifier:
            from sklearn.neighbors import KNeighborsClassifier
            KNN = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
            KNN.fit(X_train, y_train)
            y_pred = KNN.predict(X_test)
            models.append('KNN Classifier')
            accuracy.append(KNN.score(X_test, y_test))
            knnprec, knnrec, knnf = scores(y_test, y_pred)
            precision.append(knnprec)
            recall.append(knnrec)
            f1.append(knnf)
            if cross_validation:
                # cross validation
                accuracies = cross_val_score(estimator = KNN, X = X_train, y = y_train, cv = 10)
                cva.append(accuracies.mean())

        # if XGBoost is chosen
        if 'XGBoost' in classifier:
            from xgboost import XGBClassifier
            xgb = XGBClassifier()
            xgb.fit(X_train, y_train)
            y_pred = xgb.predict(X_test)
            models.append('XGBoost')
            accuracy.append(xgb.score(X_test, y_test))
            xgbprec, xgbrec, xgbf = scores(y_test, y_pred)
            precision.append(xgbprec)
            recall.append(xgbrec)
            f1.append(xgbf)
            if cross_validation:
                # cross validation
                accuracies = cross_val_score(estimator = xgb, X = X_train, y = y_train, cv = 10)
                cva.append(accuracies.mean())

        # if LightGBM is chosen
        if 'LightGBM' in classifier:
            from lightgbm import LGBMClassifier
            lgbm = LGBMClassifier()
            lgbm.fit(X_train, y_train)
            y_pred = lgbm.predict(X_test)
            models.append('LightGBM')
            accuracy.append(lgbm.score(X_test, y_test))
            lgbmprec, lgbmrec, lgbmf = scores(y_test, y_pred)
            precision.append(lgbmprec)
            recall.append(lgbmrec)
            f1.append(lgbmf)
            if cross_validation:
                # cross validation
                accuracies = cross_val_score(estimator = lgbm, X = X_train, y = y_train, cv = 10)
                cva.append(accuracies.mean())


        if cross_validation:
            st.write("Scores")
            st.write(pd.DataFrame({ 'model': models ,'Accuracy':accuracy,'Precision':precision,'Recall':recall,'F1':f1, 'Cross Validation Accuracy':cva}))
        else:
            # display the scores as a table
            st.write("Scores")
            st.write(pd.DataFrame({ 'model': models ,'Accuracy':accuracy,'Precision':precision,'Recall':recall,'F1':f1}))

        rfr = RandomForestClassifier(n_estimators = 10, random_state = 0)
        rfr.fit(X_train, y_train)
        importances = rfr.feature_importances_
        indices = np.argsort(importances)[::-1]
        features = df.columns

        import matplotlib.pyplot as plt
        # display the feature importance as a bar chart with feature names as x labels
        st.write("Feature Importance")
        fig, ax = plt.subplots()
        ax.bar(range(X_train.shape[1]), importances[indices])
        ax.set_xticks(range(X_train.shape[1]))
        ax.set_xticklabels(features[indices], rotation=90)
        st.pyplot(fig)




        

        
        


        

 




            
        # # Random Forest Regression
        # rfr = RandomForestRegressor(n_estimators = 10, random_state = 0)
        # rfr.fit(X_train, y_train)
        # y_pred = rfr.predict(X_test)
        # accuracy.append(rfr.score(X_test, y_test))
        # # precision.append(precision_score(y_test, y_pred))
        # # recall.append(recall_score(y_test, y_pred))
        # # f1.append(f1_score(y_test, y_pred))
        
        # #KNN Regression
        # knnr = KNeighborsRegressor(n_neighbors = 5, metric = 'minkowski', p = 2)
        # knnr.fit(X_train, y_train)
        # y_pred = knnr.predict(X_test)
        # accuracy.append(knnr.score(X_test, y_test))
        # # precision.append(precision_score(y_test, y_pred))
        # # recall.append(recall_score(y_test, y_pred))
        # # f1.append(f1_score(y_test, y_pred))

        # # #XGBoost
        # xgbr = XGBRegressor()
        # xgbr.fit(X_train, y_train)
        # y_pred = xgbr.predict(X_test)
        # accuracy.append(xgbr.score(X_test, y_test))

        # show the scores in a table
        # st.write("Scores")
        # scores = pd.DataFrame({'accuracy': accuracy}, index = ['Logistic Regression', 'Random Forest Regression', 'K-Nearest Neighbors Regression', 'XGBoost'])
        # st.write(scores)









        # #xgboost
        # xgbr = XGBRegressor()
        # xgbr.fit(X_train, y_train)
        # y_pred = xgbr.predict(X_test)
        # accuracy.append(xgbr.score(X_test, y_test))
        # precision.append(precision_score(y_test, y_pred))
        # recall.append(recall_score(y_test, y_pred))
        # f1.append(f1_score(y_test, y_pred))

        # # show the scores in a table
        # st.write("Scores")
        # scores = pd.DataFrame({'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}, index = ['Logistic Regression', 'Random Forest Regression', 'K-Nearest Neighbors Regression', 'XGBoost'])
        # st.write(scores)



        
   




  
        

    

