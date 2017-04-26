import cv2
import numpy as np
import base64
from boto.s3.connection import S3Connection
from boto.s3.key import Key
import os

def pull_file_from_s3(key):
    def get_bucket():
	    access= os.environ['SECRET_ENV_AWS_ACCESS_KEY_SHARED']
	    secret= os.environ['SECRET_ENV_AWS_SECRET_KEY_SHARED']
	    conn = S3Connection(access,secret)
	    b = conn.get_bucket('ds-cloud-demonstration-shared',validate=False)
	    return b

    s3_bucket = get_bucket()
    payload = s3_bucket.get_key(key)
    local_file = payload.get_contents_to_filename(key)
    return True

if not os.path.isfile('haarcascade_frontalface_default.xml'):
    pull_file_from_s3('haarcascade_frontalface_default.xml')
    pull_file_from_s3('haarcascade_mcs_nose.xml')

import mustache

def run(img):
    try:
        nparr = np.fromstring(img.decode('base64'), np.uint8)
    except:
        return {'error': 'unable to decode image'}
    try:
        frame = mustache.make_img(nparr)
    except:
        return {'error': 'unable to create mustache'}
    try:
        cnt = cv2.imencode('.png',frame)[1]
        return {'image': base64.b64encode(cnt)}
    except:
        return {'error': 'unable to convert mustache to a png'}

if __name__ == '__main__':
    with open('img') as f:
        print run(f.read())
