from keras.models import model_from_json
from os.path import splitext,basename
import json

def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        # print("Loading model successfully...")
        return model
    except Exception as e:
        print(e)
        return None
