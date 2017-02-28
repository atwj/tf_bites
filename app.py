import os


from flask import Flask
from flask_cors import CORS, cross_origin
from flask_restful import Resource, Api, reqparse, abort
from oauth2client.client import GoogleCredentials
from oauth2client.service_account import ServiceAccountCredentials
from google.cloud import storage
import label_image as li
import machine_learning as ml

app = Flask(__name__)
api = Api(app)
CORS(app)

scopes = ['https://www.googleapis.com/auth/cloud-platform']
credentials = ServiceAccountCredentials.from_json_keyfile_name(
    'your-key', scopes)

# credentials = GoogleCredentials.get_application_default()
client = storage.Client(project='your-gc-app-endpoint', credentials=credentials)
bucket = client.get_bucket('your-gc-app-endpoint')

parser = reqparse.RequestParser()
parser.add_argument('image_name')
parser.add_argument('start_date')
parser.add_argument('end_date')
parser.add_argument('lat')
parser.add_argument('long')

class helloWorld(Resource):
    def get(self):
        return 'helloWorld'

class labelImage(Resource):
    def get(self, image_name):
        location = "images/%s" % (image_name)
        blob = bucket.get_blob(location)
        image_data = blob.download_as_string()
        output = li.label(image_data)
        return str((item for item in output if item['name'] == 'mosquito').next())

    def post(self):
        args = parser.parse_args()
        location = "images/%s" % (args['image_name'])
        blob = bucket.get_blob(location)
        image_data = blob.download_as_string()
        output = li.label(image_data)
        return (item for item in output if item['name'] == 'mosquito').next()

class getAllLabels(Resource):
    def get(self, image_name):
        location = "images/%s" % (image_name)
        blob = bucket.get_blob(location)
        image_data = blob.download_as_string()
        output = li.label(image_data)
        return output

class getReport(Resource):
    def post(self):
        args = parser.parse_args()
        start_date = args['start_date']
        end_date = args['end_date']
        report = ml.getReport(start_date, end_date)
        if(report == False):
            abort(404, message='Start date cannot be greater than End Date')
        return report

class getMosquitoActivity(Resource):
    def post(self):
        args = parser.parse_args()
        lat = args['lat']
        long = args['long']
        output = ml.getMosquitoActivity(lat,long)
        print output
        return output

api.add_resource(helloWorld, '/')
api.add_resource(labelImage, '/label/<string:image_name>', '/label')
api.add_resource(getAllLabels, '/all_labels/<string:image_name>')
api.add_resource(getReport, '/getReport')
api.add_resource(getMosquitoActivity, '/getMosquitoActivity')

if __name__ == "__main__":
    app.run(host='0.0.0.0')
