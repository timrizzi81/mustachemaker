### Mustache Model

Use data science to put a mustache on your face!

![Alt text](https://s3-us-west-1.amazonaws.com/data-mustache/eb929fae-42f5-cc45-41fd-ed3944797351.png)

#### The model
The model does face and nose recognition [OpenCV](http://opencv.org/), an open source computer vision library. In particular, the model uses a [Haar Cascades](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) object classifier.

The API expects a base64 encoded image:
```javascript
{"img": "...YW55IGNhcm5hbCBwbGVhcw=="}
```

To call the API from your own code, you'll need to encode your selfies as base64. Base64 encoding comes standard in most languages. Here's an example in Python:
```python
import base64

with open("my_selfie.png", "rb") as f:
    b64_string = base64.b64encode(f.read())
```

#### The web app
The app server is deployed on Heroku. It has three main jobs: proxy requests to the Model API, handle some logic for storing photos, serve the html/js client.

The web app and Model API can be changed independently, as long as the API contract between them is always in sync.

![App diagram](https://s3-us-west-1.amazonaws.com/data-mustache-test/app-diagram.png)
