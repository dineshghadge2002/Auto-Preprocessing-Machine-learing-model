def data_pre_model():
    #importing models
    import pandas as pd
    import matplotlib as plt
    import seaborn as sns
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.linear_model import LinearRegression
    from sklearn import metrics
    from sklearn.linear_model import LinearRegression
    
    #load dataset
    data = input("Enter your dataset:")
    dataset=pd.read_csv(data)
    print(dataset.columns)
    
    #selecting target column
    dg=input("Enter your target column:")
    dataset[dg]
    y=dataset[[dg]]
    dataset.drop(dg,axis=1,inplace=True)
    
    #Working with catagorical variable:
    for i in range(len(dataset.columns)):
        if dataset.nunique()[i]<100:
            le = LabelEncoder()
            dataset[dataset.columns[i]]=le.fit_transform(dataset[dataset.columns[i]])  
            
            
    #column/feature selection
    for i in range(len(dataset.columns)-1):
        if type(dataset[dataset.columns[i]].iloc[2]) == str:
            dataset.drop(dataset.columns[i],axis=1,inplace=True)
            i-1
        if y[dg].nunique() <= (2*len(y)/100):
            if len(dataset) == dataset.nunique()[i]:
                dataset.drop(dataset.columns[i],axis=1,inplace=True)
                i-1
                
    #working with null values
    if dataset.isnull().any().any() == True:
        for i in range(0,len(dataset.columns)):
            if type(dataset.columns[i]) == str:
                    count=dataset[dataset.columns[i]].mean()
                    print(count)
                    dataset[dataset.columns[i]].fillna(value = count, inplace=True)
    
    #changing datatype of column name
    dataset.columns=dataset.columns.astype(str)
    
    #selecting dependent and indipendent variable
    x=dataset

    
    #fiting data prediction model
    
    if y[dg].nunique() <= (2*len(y)/100):
        le = LabelEncoder()
        y[dg]=le.fit_transform(y[dg])
        if y[dg].nunique() == 2:
            #Simple Logistics Regression
            x_train,x_test,y_train,y_test=train_test_split( x, y, test_size=0.20, random_state=42)
            model=LogisticRegression()
            model.fit(x_train,y_train)
            y_pred=model.predict(x_test)
            print(confusion_matrix(y_test,y_pred))
            accuracy = accuracy_score(y_test, y_pred)
            print("Accuracy:", accuracy)

            conf_matrix = confusion_matrix(y_test, y_pred)
            print("Confusion Matrix:\n", conf_matrix)

            class_report = classification_report(y_test, y_pred)
            print("Classification Report:\n", class_report)
        else:
            #Multilogistics Regression
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
            model=LogisticRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        print('----------------------------------------------------------------------------')            
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)
        print('----------------------------------------------------------------------------')
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:\n", conf_matrix)
        print('----------------------------------------------------------------------------')
        class_report = classification_report(y_test, y_pred)
        print("Classification Report:\n", class_report)
        print('----------------------------------------------------------------------------')        
    else:
        if len(x.columns) == 1 :
            #Simple linear Regression
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
            model=LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            #Multilinear Regression
            model=LinearRegression()
            x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)
            model.fit(x_train,y_train)
            y_pred = model.predict(x_test)
         
        print('----------------------------------------------------------------------------')
        #Co-efficient :
        print('\nCo-efficient :',model.coef_)
        print('----------------------------------------------------------------------------')
        
        #Intercept :
        print('\nIntercept :',model.intercept_)
        print('----------------------------------------------------------------------------')

        #Mean Absolute Error (MAE) :
        #Lower MAE indicate better performance.
        mae = mean_absolute_error(y_test, y_pred)
        print('\nMean Absolute Error (MAE) :',mae)
        print('----------------------------------------------------------------------------')

        #Mean Square Error :
        #Lower MSE values indicate better performance.
        mse = mean_squared_error(y_test, y_pred)
        print('\nMean Square Error :',mse)
        print('----------------------------------------------------------------------------')

        #Root Mean Squared Error (RMSE) :
        #Lower RMSE values indicate better performance.
        rmse = np.sqrt(mse)
        print('\nRoot Mean Squared Error (RMSE) :',rmse)
        print('----------------------------------------------------------------------------')

        #R-squared (R²) / Coefficient of Determination:
        #Higher R² values indicate better performance.
        r2 = r2_score(y_test, y_pred)
        print('\nR-squared (R²) / Coefficient of Determination:',r2)
        print('----------------------------------------------------------------------------')

data_pre_model()    