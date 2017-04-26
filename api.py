import cv2
import numpy as np
import base64
from boto.s3.connection import S3Connection
from boto.s3.key import Key
import os

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
