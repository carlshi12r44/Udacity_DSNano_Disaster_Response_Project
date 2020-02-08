# Disaster Response Classification Project :ocean: :ocean: :ocean:
# Table of Contents
  - [Description](#description)
  - [Getting Started](#getting-started)
  - [Dependencies](#dependencies)
  - [Installation](#installation)
  - [Execution](#execution)
  - [License](#license)
  - [Acknowledgement](#acknowledgement)
  - [Web Showcase](#web-showcase)

# Description
This project uses natural language processing to determine whether a text is a message for the disaster. This project is a part of the Udacity's nanodegree Data Scientist program. The initial dataset contains the pre-labelled tweets and messages from real-life disaster, provided by the FigureEight.

The project is divided in three sections:
 1. ETL(Extract, Transform, Load) pipeline: data preprocessing - to extract data from source, clean data and save them
 2. Machine Learning pipeline: utiilize the nltk packages and scikit-learn packages to analyze the text messages. 
 3. Flask Web Application: The web application application that will show the distributions of the categories results and allow users to input texts

# Getting Started
This project is running on Python 3.x under distribution of Anaconda
## Dependencies
  - NLTK
  - Scikit-Learn
  - Numpy
  - Flask
  - os
  - re
  - pandas
  - pickle
  - sqlalchemy
  - sqlite3
## Installation
Git this repo by 
```
git clone https://github.com/carlshi12r44/Udacity_DSNano_Disaster_Response_Project.git
```
## Execution
Run the following commands in the project's root directory to set up your database and model.
- To run ETL pipeline that cleans data and stores in database
```
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```
- To run ML pipeline that trains classifier and saves
```
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```
- Run the following command in the app's directory to run your web app.
```
python run.py
```
- Go to
```
http://127.0.0.1:3001/
```

# License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
# Acknowledgement
Udacity provides the skeleton codes
FigureEight provides the datasets
# Web Showcase
![Image of Classification](../Screenshots/Classify_Result.png)
![Image of Distributions](../Screenshots/distribution.png)
