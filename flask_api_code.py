
from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse
import pickle
import numpy as np
import json



# code to initiate the flask app and API.
app = Flask(__name__)
api = Api(app)

# create parser for payload data
parser = reqparse.RequestParser()
parser.add_argument('data')

# define how api will respond to post requests.
class IrisClassifier(Resource):
    def post(self):
        args = parser.parse_args()
        X = np.array(json.loads(args['data']))
        prediction = model.predict(X)
        return jsonify(prediction.tolist())

api.add_resource(IrisClassifier, '/iris')

if __name__ == '__main__':
    # load model
    with open('model.pickle', 'rb') as f:
        model = pickle.load(f)
    app.run(debug=True)