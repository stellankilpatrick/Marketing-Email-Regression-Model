from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
import pandas as pd
import math
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    confusion_matrix, log_loss
)

class UserPredictor:
    
    def __init__(self):
        # initialize pipeline and training variables
        self.pipeline = None
        self.xcols = None
        
    ### Merges user data frame and logs data frame into one, aggregating values across user ids
    def add_logs_as_features(self, train_users, train_logs):

        train_logs['date'] = pd.to_datetime(train_logs['date'], format='%m/%d/%Y', errors='coerce')

        # aggregate user data of time spent and web pages visited
        logs_agg = train_logs.groupby('user_id').agg(
            time_spent =('seconds', 'sum'),
            pages_visited =('url', 'count'),
            laptop_visits = ('url', lambda x: (x=='/laptop.html').sum()),
            first_click = ('date', 'min'),
            last_click = ('date', 'max'),
        ).reset_index()

        logs_agg['date_span'] = (logs_agg['last_click'] - logs_agg['first_click']).dt.days

        users_df = train_users.merge(logs_agg, on='user_id', how='left')
        users_df = users_df.drop(columns=['first_click', 'last_click'])
        users_df.fillna(0, inplace=True) # fill N/A values with 0

        return users_df
        
    ### Fits logistic regression pipeline to training data
    def fit(self, train_users, train_logs, train_y):
        train_users = self.add_logs_as_features(train_users, train_logs)
        
        # drops columns that don't affect the accuracy of the model
        X = train_users.drop(columns=['user_id', 'names', 'badge'])
        y = train_y['y']
        
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression())
        ])
        
        self.pipeline.fit(X, y)
        
    ## Predicts outcomes for test data and returns a boolean array
    def predict(self, test_users, test_logs):
        test_users = self.add_logs_as_features(test_users, test_logs)
    
        # drops columns that don't affect the accuracy of the model
        X = test_users.drop(columns=['user_id', 'names', 'badge'])

        predict_model = self.pipeline.predict(X)
        
        self.y_prob = self.pipeline.predict_proba(X)[:, 1]

        predict_array = np.array(predict_model, dtype=bool) # turn model into boolean array
        return predict_array
    
    ### returns the coefficients of all of the model's variable in a data frame
    def feature_coefficients(self, train_users, train_logs, train_y):
        train_users = self.add_logs_as_features(train_users, train_logs)

        # drops columns that don't affect the accuracy of the model
        X = train_users.drop(columns=['user_id', 'names', 'badge'])
        features = X.columns
        y = train_y['y']

        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression())
        ])

        self.pipeline.fit(X, y)

        model = self.pipeline.named_steps['model'] # accesses logistic regression model from pipeline
        coefs = model.coef_[0]

        # create Pandas dataframe to store coefficients and their variable names
        self.coefficients = pd.DataFrame({
            'Feature': features,
            'Coefficient': coefs
        }).sort_values(by='Coefficient', ascending=False).reset_index(drop=True)

        return self.coefficients
    
    ### Calculates the probability of a customer clicking the email based on age, past purchase amound ($), time spent on website (seconds), and webpages visited using log-odds
    def probability(self, age, past_purchase_amt, time_spent, pages_visited, laptop_visits, date_span):
        # Map the feature names to their coefficients
        coefficients_dict = self.coefficients.set_index('Feature')['Coefficient'].to_dict()
        
        # Calculate the weighted sum
        weighted_sum = (coefficients_dict['age'] * age +
                        coefficients_dict['past_purchase_amt'] * past_purchase_amt +
                        coefficients_dict['time_spent'] * time_spent +
                        coefficients_dict['pages_visited'] * pages_visited +
                        coefficients_dict['laptop_visits'] * laptop_visits +
                        coefficients_dict['date_span'] * date_span)
        
        # Compute the probability using the sigmoid function
        probability = round((1 / (1 + math.exp(-weighted_sum)))*100, 1)

        return (f'This customer, {age} years of age, has spent ${round(past_purchase_amt, 2)} on past purchases, visited {pages_visited} webpages, {laptop_visits} of them for the laptop, for a total of {round(time_spent/60, 1)} minutes, and had {date_span} days between their first and last visit. They have a {probability}% chance of clicking on the link in the email.')

    def performance(self, train_users, train_y, test_users, test_logs, test_y):
        y_pred = self.predict(test_users, test_logs)
        y_prob = self.y_prob # from predict()
        model = self.pipeline.named_steps['model']

        accuracy = accuracy_score(test_y['y'], y_pred)
        precision = precision_score(test_y['y'], y_pred)
        recall = recall_score(test_y['y'], y_pred)
        self.conf_matrix = confusion_matrix(test_y['y'], y_pred)

        tn, fp, fn, tp = self.conf_matrix.ravel()

        specificity = tn / (tn + fp)

        scores = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall (True Positive)', 'Specificity (True Negative)'],
            'Value': [accuracy, precision, recall, specificity]
        })

        plt.bar(x = scores['Metric'], height = scores['Value'], color = 'green')
        plt.title("Performance Metrics")
        plt.xticks(rotation=15)
        plt.ylim(0, 1)
        plt.show()

        return scores