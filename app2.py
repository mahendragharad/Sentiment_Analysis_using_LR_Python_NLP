from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import logging 
import pickle 
import re
from exception import CustomException
from logger import logging 
import os 

try : 
    logging.info("creating the pkl file of model and TFIDF matrix to predict the ouput and clean the comment")
    model = pickle.load(open('clf.pkl' , 'rb'))
    tfidf = pickle.load(open('tfidf.pkl' , 'rb'))
except Exception as e :
    logging.info("Error Accured while creating PKL file of model and TFIDF matrix")
    raise CustomException(e)

app = Flask(__name__)

def clean_text(text):
    try :
        logging.info("inside clean text function")

        logging.info("Tokenize the text")
        tokens = word_tokenize(text)
        
        logging.info("Convert text to lowercase")
        tokens = [word.lower() for word in tokens]

        logging.info("Remove punctuation and special characters from text")
        tokens = [word for word in tokens if word.isalpha()]
        
        logging.info("Remove stopwords from the text")
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        
        logging.info(" Applying Lemmatization to the text")
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        
        logging.info("Join tokens back into a string")
        clean_text = ' '.join(tokens)
        
        logging.info("Returning the clean_text")
        return clean_text
    except Exception as e :
        logging.info("error Occured in Clean_text Function")
        raise CustomException(e)

def predict_sentiment(review):
    try :
        logging.info("Inside Predict Sentiment function")

        logging.info("Transforming the review into TF-IDF features")
        review_tfidf = tfidf.transform([review])
        
    
        logging.info("Predict sentiment (positive or negative) by using out trained model")
        prediction = model.predict(review_tfidf)[0]

        logging.info("returnnig prediction")
        return prediction
    
    except Exception as e :
        logging.info("Error Accured in Predict Sentiment Function")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST' , 'GET'])
def predict():
    try :
        logging.info("Now Inside Predict Route")

        if request.method == 'POST':
            logging.info("Getting review From the user to predict")
            review = request.form['review']

            logging.info("Applying clean_text Function to the text")
            cleaned_comment = clean_text(review)

            logging.info("creating TFIDF Matrix using our PKL file of TFIDF")
            comment_vector = tfidf.transform([cleaned_comment])

            logging.info("Getting prediction from by predicting comment using our trained model pickle file")
            prediction = model.predict(comment_vector)[0]
        
            return render_template('index.html', review=review, prediction=prediction)
        else:
            return render_template('index.html')
    except Exception as e :
        logging.info("Error Occured in predict route during prediction of Review")
        raise CustomException(e)

if __name__ == '__main__':
    app.run(debug=True)
