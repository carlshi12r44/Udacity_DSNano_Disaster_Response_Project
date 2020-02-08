# import libraries
import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
import re
import sqlite3
import numpy as np
import pandas as pd
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

def load_data(database_filepath):
    '''
    Load_data
    Takes the database filepath to load the data, Runs a query to extract the values for 
    the X , Y & category names and return them

    @params:
        database_filepath: the database filepath
    @return:
        X: the training data for the project
        Y: the labels for the project
        category_names: the name of the categories
    '''

    # load data from the database
    engine = create_engine('sqlite:///{}'.format(database_filepath))

    # connect to the databse
    conn = sqlite3.connect(database_filepath)

    # run a query
    df = pd.read_sql('SELECT * FROM df', conn)

    # extract values from X and y
    X =  df['messages']
    Y = df.iloc[:, 4:]

    category_names = list(Y)

    return X, Y, category_names


def tokenize(text):
    '''
    Tokenize:
        tokenize the text, stemmatize and lemmatize the text to output a cleaned text
    @params:
        text: origin input of the text
    @return:

    '''
    # normalize the text
    text = re.sub(r'[^a-zA-Z0-9]',' ', text.lower())

    words = word_tokenize(text)

    tokens = [w for w in words if w not in stopwords.words("english")]

    # stemming and lemming
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []

    for t in tokens:
        clean_t = lemmatizer.lemmatize(t).strip()
        clean_tokens.append(clean_t)

    return clean_tokens

def build_model():
    '''
    Build_model: build a pipeline that can process the text and then performs muilti-output classification
    on the 36 categories in the database. GridSearchCV is used to find the best parameters for the model
    
    @params:
        None
    @returns:
        cv: pipeline used the GridSearchCV method 
    '''
    # build the pipeline
    pipeline = Pipeline([
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    # use grid search to find better parameters
    parameters = {
        'text_pipeline__tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [10, 50, 100],
        'clf__estimator__learning_rate': [0.1, 1, 5],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Uses the model to predict and evaluate the results on the 36 categories
    reports the F1 scores, and recall for each output category of the dataset
    
    @params:
        model: the pipeline we build in the previous function
        X_test: the testing X inputs
        Y_test: the ground true labels
        category_names: the names of the categories
    
    @return:
        return the score
    '''

    # predict on test data
    y_pred = model.predict(X_test)

    # report the f1 score, and recall for every category
    for i in range(36):
        category = category_names[i]
        f1 = f1_score(Y_test.iloc[:, i], y_pred.iloc[:, i])
        precision = precision_score(Y_test.iloc[:, i], y_pred[:, i])
        recall = recall_score(Y_test.iloc[:, i], y_pred[:, i])
        print(category)
        print('\tF1_score: %.4f\tPrecision: %.4f\t Recall: %.4f\n' % (f1, precision, recall))




def save_model(model, model_filepath):
    '''
    Save_model
        Saves the model to the model_filepath 
    '''

    #export the model using pickel
    pickle.dump(mdoel, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()