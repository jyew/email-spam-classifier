from flask import Flask
from flask import Response
from flask import request
from flask_cors import CORS
from flask import jsonify
from flask_restful import Api, Resource, reqparse
from json import dumps
from json import loads
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import joblib
import re
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stemmer = WordNetLemmatizer()
porter = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))


app = Flask(__name__)
CORS(app)
api = Api(app)

parser = reqparse.RequestParser()
tfidfPath = './models/tfid_convertor.sav'
modelPath = './models/finalized_model.sav'

tfid = joblib.load(tfidfPath)
eclf1 = joblib.load(modelPath)


@app.route('/')
def index():
    return "Welcome to Jordan Email Scammer Identifier App"

def preprocessingText(corpus, lowercase=True, rmPunctuation=True, rpURL=True, rpNumber=True, stemming=True):
    """Input is assumed to be vector of documents"""

    documents = []
    for text in corpus:
        document = text

        # convert bytes into text
        if type(text) == bytes:    
            document = text.decode('utf-8', errors='ignore')

        # HYPERPARAMETER
        # Converting to Lowercase
        if lowercase:
            document = document.lower()

        # replace URL
        if rpURL:
            # replace URL
            document = re.sub(r'http\S+', 'url', document, flags=re.MULTILINE)

        # replace numbers
        if rpNumber:
            document = re.sub("\d+", "number", document)

        # remove all special characters including punctuation
        if rmPunctuation:
            # only keep word
            document = re.sub(r'\W', ' ', document)
            # remove all single characters
            document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
            # Remove single characters from the start
            document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # OTHER PREPROCESSING METHODS
        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)
        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)
        
        # removing stopwords
        document = document.split()
        document = [word for word in document if word not in STOPWORDS]

        if stemming:
            # Lemmatization
            document = [stemmer.lemmatize(word) for word in document]
            # stemming
            document = [porter.stem(word) for word in document]

        document = ' '.join(document)
        documents.append(document)
    return documents

def postprocessing(results, labels):
    return [labels[str(r)] for r in results]

class health(Resource):
    def get(self):
        return "Health_OK"

class identify_scammer(Resource):
    """Sample request: 
    curl --location --request POST 'http://localhost:8000/identify_scammer' \
    --header 'Content-type: application/json' \
    --data-raw '{"queries": ["Subject: unbelievable new homes made easy im wanting to show you this  
    dorcas pittman", "Subject: re : hello  gerry ,  let me review my calendar in the beginning of the 
    next year and i shall  e - mail you  with a suggested date . my assistant will update my schedule 
    for 2001 in the  first week  of january and i shall be able to select a date for ypur presentaton .
      vince kaminski    cc :  subject : re : hello  dear mr . kaminski  please excuse the cancellation 
      due to illness . the students do not care who  they infect near the end of the semester , 
      they just want to get done !  here is my available schedule for next year . i am now overloaded next week  
      with tasks to complete the semester . i do hope that we can reschedule  during the first quarter next year 
      .  i would note that my schedule is most free for thursday or friday . i could  fly out late wednesday night .  
      cordially ,  gerry  teaching schedule  m 11 - 12  t and r 10 - 12 and 2 - 4  t 12 - 2 ep & es seminar  m 6 - 8
        t 6 - 8  w 6 - 8  ( r = thursday )  workshops :  jan 12 - 13 des moines  jan 26 - 27 des moines  feb 9 - 10 
        des moines  ieee wpm conference  feb 28 - 31 columbus , ohio"]}'
    """
    def post(self):
        parser.add_argument('queries', type=str, action="append", location='json')
        args = parser.parse_args()
        queries = args['queries']
        print(queries)
        if queries is not None:
            results = []
            input_tf = tfid.transform(preprocessingText(queries))
            predicted_labels = eclf1.predict(input_tf.toarray())
            processed_results = postprocessing(predicted_labels, {"0": "not scam", "1": "scam"})
            response = {'queries': queries, 'results': processed_results} 
            return jsonify(response)
        return Response("Please submit a list of strings", status=400)

api.add_resource(health, '/health')
api.add_resource(identify_scammer, '/identify_scammer')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port='8000')
