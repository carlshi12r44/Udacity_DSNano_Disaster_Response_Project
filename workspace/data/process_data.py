import sys
import sqlite3
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load Data Function
    
    @params:
        messages_filepath: path to the messages csv file
        categories_filepath: path to the categories csv file

    @output:
        df: the merged dataframes
    '''
    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)

    # merge the data frames
    df = pd.merge(messages_df, categories_df, on='id')
    return df

def clean_data(df):
    '''
    Create a clean data, combined dataframes of categories and 
    messages dummy variables
    
    @params:
        messages_df: DataFrame. 
        categories_df: DataFrame
    
    @output:
        df: cleaned df 
    '''
    categories = df.categories.str.split(pat=';', expand=True)
    firstrow = categories.iloc[0,:]

    categories_names = firstrow.apply(lambda x : x[:-2])

    categories.columns = categories_names
    
    for col in categories:
        # set the categories values to be numerical 0/1
        categories[col] = categories[col].str[-1].astype(int)

    df = df.drop('categories', axis=1)
    # concatenate the new categories with the origin dataframe
    df = pd.concat([df, categories], axis=1)

    df = df.drop_duplicates()

    return df

def save_data(df, database_filename):
    '''
    Save dataframe to database in
    '''  
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('df', engine, index=False)
    

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()