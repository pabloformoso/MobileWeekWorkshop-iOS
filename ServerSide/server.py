import io
import os
import sys
import uuid
import base64
import logging
import threading
import PIL.Image as Image
import turicreate as tc

from flask import Flask, request, jsonify
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask import make_response
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from queue import Queue
from flask import send_from_directory
from marshmallow import fields
from marshmallow import post_load

_format = {'JPG': 0, 'PNG': 1, 'RAW': 2, 'UNDEFINED': 3}

app = Flask(__name__)
if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host='0.0.0.0', debug=True)

#configure logging
logging.basicConfig( level=logging.DEBUG,
                     format='[%(levelname)s] - %(threadName)-10s : %(message)s')

#configure images destination folder
app.config['UPLOADED_IMAGES_DEST'] = './images'
images = UploadSet('images', IMAGES)
configure_uploads(app, images)

#configure sqlite database
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'facerecognition.sqlite')
db = SQLAlchemy(app)
ma = Marshmallow(app)

#model/users is a many to many relationship,  that means there's a third #table containing user id and model id
users_models = db.Table('users_models',
                        db.Column("user_id", db.Integer, db.ForeignKey('user.id')),
                        db.Column("model_id", db.Integer, db.ForeignKey('model.version'))
                        )
# model table
class Model(db.Model):
    version = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(100))
    users = db.relationship('User', secondary=users_models)

# user table
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(300))
    position = db.Column(db.String(300))

    def __init__(self, name, position):
        self.name = name
        self.position = position
 
#user schema
class UserSchema(ma.Schema):
    class Meta:
        fields = ('id', 'name', 'position')

#model schema
class ModelSchema(ma.Schema):
    version = fields.Int()
    url = fields.Method("add_host_to_url")
    users = ma.Nested(UserSchema, many=True)
    # this is necessary because we need to append the current host to the model url
    def add_host_to_url(self, obj):
        return request.host_url + obj.url

#initialize everything
user_schema = UserSchema()
users_schema = UserSchema(many=True)
model_schema = ModelSchema()
models_schema = ModelSchema(many=True)
db.create_all()



#error handlers
@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'not found'}), 404)

@app.errorhandler(400)
def not_found(error):
    return make_response(jsonify({'error': 'bad request'}), 400)


#recognize
@app.route("/api/v1.0/user/recognize", methods=['POST'])
def recognize_user():
    if 'image' in request.files:
        img_data = request.files.get('image')
        pil_img = Image.open(io.BytesIO(base64.b64decode(img_data)))        
        image = from_pil_image()
        model = tc.load_model(os.path.join(basedir, "models/Faces_v2.model"))
        predictions = model.predict(image)

        return make_response(jsonify({
            'status': 'find',
            'predictions': predictions
        }), 200)
    else:
        return make_response(jsonify({
            'status': 'failed', 
            'error': 'oppsss', 
            'message:' : 'Something went really wrong'}), 500)   

def from_pil_image(pil_img):
    """
    Returns a graphlab.Image constructed from the passed PIL Image
    Parameters
    ----------
        pil_img : PIL.Image
            A PIL Image that is to be converted to a graphlab.Image
    Returns
    --------
        out: graphlab.Image
            The input converted to a graphlab.Image
    """
    # Read in PIL image data and meta data
    height = pil_img.size[1]
    width = pil_img.size[0]
    if pil_img.mode == 'L':
        image_data = bytearray([z for z in pil_img.getdata()])
        channels = 1
    elif pil_img.mode == 'RGB':
        image_data = bytearray([z for l in pil_img.getdata() for z in l ])
        channels = 3
    else:
        image_data = bytearray([z for l in pil_img.getdata() for z in l])
        channels = 4
    format_enum = _format['RAW']
    image_data_size = len(image_data)

    # Construct a graphlab.Image

    img = turicreate.Image(_image_data=image_data, _width=width, _height=height, _channels=channels, _format_enum=format_enum, _image_data_size=image_data_size)
    
    return img

#register user
@app.route("/api/v1.0/user/register", methods=['POST'])
def register_user():
    if not request.form or not 'name' in request.form:
        return make_response(jsonify({
            'status': 'failed', 
            'error': 'bad request', 
            'message:' : 'Name is required'}), 
                             400)
    else:
        name = request.form['name']
        position = request.form.get('position')
        if position is None:
            position = ""
        newuser = User(name, position)
        db.session.add(newuser)
        db.session.commit()
        if 'photos' in request.files:
            uploaded_images = request.files.getlist('photos')
            save_images_to_folder(uploaded_images, newuser)
        return jsonify({
            'status' : 'success', 
            'user' :  user_schema.dump(newuser).data })

#function to save images to image directory
def save_images_to_folder(images_to_save, user):
    for a_file in images_to_save:
        # save images to images folder using user id as a subfolder name
        images.save(a_file, str(user.id), str(uuid.uuid4()) + '.')

    # get the last trained model
    model = Model.query.order_by(Model.version.desc()).first()
    if model is not None:
        # increment the version
        queue.put(model.version + 1)
    else:
        # create first version
        queue.put(1)
        
        
@app.route("/api/v1.0/model" , methods=['GET'])
def get_model_info():
    models_schema.context['request'] = request
    model = Model.query.order_by(Model.version.desc()).first()
    if model is None:
        return make_response(jsonify({'status': 'failed', 'error': 'model is not ready'}), 400)
    else:
        return jsonify({'status' : 'success', 'model' : model_schema.dump(model).data})
    
    
def train_model():
    while True:
        #get the next version
        version = queue.get()
        logging.debug('loading images')
        data = tc.image_analysis.load_images('images', with_path=True)

        # From the path-name, create a label column
        data['label'] = data['path'].apply(lambda path: path.split('/')[-2])
        print("Label: ", data['label'])
        
        # use the model version to construct a filename
        filename = 'Faces_v' + str(version)
        mlmodel_filename = filename + '.mlmodel'
        models_folder = 'models/'

        # Save the data for future use
        data.save(models_folder + filename + '.sframe')

        result_data = tc.SFrame( models_folder + filename +'.sframe')
        train_data, test_data = result_data.random_split(0.8)

        #the next line starts the training process
        model = tc.image_classifier.create(train_data, target='label', model='resnet-50', verbose=True)

        db.session.commit()
        logging.debug('saving model')
        model.save( models_folder + filename + '.model')
        logging.debug('saving coremlmodel')
        model.export_coreml(models_folder + mlmodel_filename)

        # save model data in database
        modelData = Model()
        modelData.url = models_folder + mlmodel_filename
        classes = model.classes
        for userId in classes:
            user = User.query.get(userId)
            if user is not None:
                modelData.users.append(user)
        db.session.add(modelData)
        db.session.commit()
        logging.debug('done creating model')
        # mark this task as done
        queue.task_done()

#configure queue for training models
queue = Queue(maxsize=0)
thread = threading.Thread(target=train_model, name='TrainingDaemon')
thread.setDaemon(False)
thread.start()
