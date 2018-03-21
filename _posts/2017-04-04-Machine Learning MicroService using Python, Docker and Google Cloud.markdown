---
layout: post
comments: true
title:  "ML MicroService"
---

# Machine Learning MicroService using Python, Docker and Google Cloud.

Machine Learning algorithms have a wide range of applications, we experience them everyday. But what is Machine Learning? [This course](https://www.coursera.org/learn/machine-learning) by [Andrew Ng](http://www.andrewng.org/) will give you some useful insights about what it is, why is it useful, how it works. 

This blog is not about Machine Learning but, to explain how Machine Learning models can be used as a Microservice. Here is the idea: You take a Machine Learning algorithm (of your choice) and create a simple RESTful service. 

* we will send training data via the API for the algorithm to train and generate a model, the API will return our cross-validation results as a response

* we will send the test data via the API, the api will return test results as a response.

Just creating the RESTful service may not be okay here because if you are running a machine learning on your server, you need a install lots of software and there are always dependecy nightmares. SO, we will use [docker](https://www.docker.com/) and containerize our service. Read more about docker in the website.

Let's get started....

`Note`: I used `Python` and `JSON` on `OSX` platform in this blog. It will be similar or very straight forward if you are using other plaforms or technologies.

### I just want to validate the results..

In this step, we'll just classify our results using the API. In the later steps, we will go into details about the server is built, how the service is containerized and so on..

* Go to terminal and type in the following commands:

```bash
$ cd Desktop
$ mkdir client
$ cd client
$ echo "print 'Hello World'" > json_generator.py
```
You have created a new directory called **client** in your desktop and inside the directory, you have written a simple `python` script called **json_generator.py**. Now open the file with your favorite text editor and let's written some code. If you don't know have a preference for a text editor, use [Sublime text](https://www.sublimetext.com/), it's cool!!

* If you are using MacOS, open your terminal and enter the following commands:

```bash
$ echo $'json_tricks==3.8.0\npandas==0.18.0\nFlask==0.12.1\nnumpy==1.12.1\nscikit_learn==0.18.1\nxgboost==0.6a2' > requirements.txt
$ pip install -r requirements.txt
```
You have just installed all the modules you needed.

* Now that you have opened the json_generator.py file in your text editor, enter the following lines of code:

```python
	import numpy as np
	import pandas as pd
	from json_tricks.np import dumps
	from sklearn.cross_validation import train_test_split

	'''
	The data-set used for this example was Congressional Voting Records from UCI machine learning repository.
	Please find the link to the web-site here https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records
	'''
	def pre_process():
		'''reading the data set using a pandas dataframe'''
		df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data', header=None)

		'''
		The length of a feature vector is 16 and all values are either `y` or `n`.
		Replacing all occurences of `n` with 0.
		Replacing all occurences of `y` with 1.
		Missing values are represented with `?`. Replacing all occurences of `?` with NaN.
		'''
		df = df.replace('n', 0)
		df = df.replace('y', 1)
		df = df.replace('?', -1)

		'''
		The first column represent the labels.The 2nd to 16th column indicates features.
		'''

		labels = df[0].as_matrix()
		features = df[[x for x in range(1,len(df.columns))]].as_matrix()
		return features, labels
```

The function `pre_process` is doing the following things:

	1. Reading the data from the url into a pandas dataframe

	2. Since the features in the data contains only three kinds of values, we are:

		1. Replacing all `n` with `0`

		2. Replacing all `y` with `1`

		3. Replcaing all missing values i.e `?` with `-1`

	3. The first column in the dataframe are our labels. 

	4. Finally, we are returning all our features and labels as `numpy` arrays.

* The data-set which I have used is a very simple data-set. If you want to use your own data-set, please write your own functionality inside thi functions and return features and labels. Remember to do [One hot encoding](https://www.quora.com/What-is-one-hot-encoding-and-when-is-it-used-in-data-science) for your features.

* Now that we have our base function ready, let's split the data into training and testing sets and convert them into JSON. The code is shown below:

```python
'''
@author: Teja Kanchinadam

The below code is used to make json files for train and test sets
'''

import numpy as np
import pandas as pd
from json_tricks.np import dumps
from sklearn.cross_validation import train_test_split

'''
Below method is responsible for pre-processing. This function needs to be changed for every data-set.
The data-set used for this example was Congressional Voting Records from UCI machine learning repository.
Please find the link to the web-site here https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records
'''

'''
Base class or perhaps the only class which is responsible for pre-processing the data, dividng the data 
into test and train sets and converting them into JSON objects.

Parameters
----------------
test_set_size: Float
	percentage in size of test set. default value is 25% or 0.25
cross_validation_folds: Int
	number of folds in the cross-validation set. default value is 5
'''


class MakeJSON(object):

	def __init__(self, test_set_size=0.25, cross_validation_folds=5):
		self.test_set_size = test_set_size
		self.cross_validation_folds = cross_validation_folds
		self.train_dict = {}
		self.test_dict = {}

	def pre_process(self):
		'''reading the data set using a pandas dataframe'''
		df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data', header=None)

		'''
		The length of a feature vector is 16 and all values are either `y` or `n`.
		Replacing all occurences of `n` with 0.
		Replacing all occurences of `y` with 1.
		Missing values are represented with `?`. Replacing all occurences of `?` with NaN.
		'''
		df = df.replace('n', 0)
		df = df.replace('y', 1)
		df = df.replace('?', -1)

		'''
		The first column represent the labels.The 2nd to 16th column indicates features.
		'''

		labels = df[0].as_matrix()
		features = df[[x for x in range(1,len(df.columns))]].as_matrix()
		return features, labels
	
	'''
	The wrapper function which call all other functions from this class.
	'''
	def run_wrapper(self):
		features, labels = self.pre_process()
		x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = self.test_set_size)
		self.make_train_json(x_train, y_train)
		self.make_test_json(x_test, y_test)

	'''
	The below functon makes the training set into train.json

	Format of JSON
	--------------------
	{
		'features': train_features,
		'labels' : train_labels,
		'folds': cross-calidation folds,
		'parameters': parameters for the classifier, default is None
	}
	'''
	def make_train_json(self, x, y):
		self.train_dict = {'features':x, 'labels':y, 'folds': self.cross_validation_folds,'parameters':None}
		tr_json = dumps(self.train_dict)
		target = open('train.json', 'w')
		target.write(tr_json)
		target.close()

	'''
	The below functon makes the training set into train.json

	Format of JSON
	--------------------
	{
		'features': test_features,
		'labels' : test_labels // can be None
	}
	'''

	def make_test_json(self, x, y):
		self.test_dict = {'features':x, 'labels':y}
		ts_json = dumps(self.test_dict)
		target = open('test.json', 'w')
		target.write(ts_json)
		target.close()

'''
Main method in this module, which initializes the MakeJson object and calls the run_wrapper method.
'''
def main():
	obj = MakeJSON()
	obj.run_wrapper()

if __name__ == "__main__":
	main()

```

* Go to terminal and type in the following commands:

``` bash
$ cd Desktop/restful
$ cd ls
requirements.txt json_generator.py
$ python json_generator.py
$ ls
requirements.txt json_generator.py train.json test.json
```

Awesome, we have created out `train.json` and `test.json` files.

* let's make the API calls now:

**Training**

```bash
$ curl -H "Content-Type:application/json" --data @train.json http://104.196.154.0:8080/
{
  "Accuracy": 0.96013986013986019, #Model checking or cross-validation scores
  "Metrics": [
    {
      "Class": "democrat", 
      "F1-Score": 0.97, 
      "Precision": 0.97, 
      "Recall": 0.96, 
      "Support": 196.0
    }, 
    {
      "Class": "republican", 
      "F1-Score": 0.95, 
      "Precision": 0.94, 
      "Recall": 0.96, 
      "Support": 130.0
    }, 
    {
      "Class": "Total", 
      "F1-Score": 0.96, 
      "Precision": 0.96, 
      "Recall": 0.96, 
      "Support": 326.0
    }
  ]
}
```
Since we had two classes, the classification metrics are shown for each individual class `democrat` and `republican` and also the average of those two classes.

**Testing**

```bash
$ curl -H "Content-Type:application/json" --data @test.json http://104.196.154.0:8080/test
{
  "Accuracy": 0.94495412844036697, #Testing score
  "Metrics": [
    {
      "Class": "democrat", 
      "F1-Score": 0.96, 
      "Precision": 1.0, 
      "Recall": 0.92, 
      "Support": 71.0
    }, 
    {
      "Class": "republican", 
      "F1-Score": 0.93, 
      "Precision": 0.86, 
      "Recall": 1.0, 
      "Support": 38.0
    }, 
    {
      "Class": "Total", 
      "F1-Score": 0.95, 
      "Precision": 0.95, 
      "Recall": 0.94, 
      "Support": 109.0
    }
  ], 
  "Predictions": [
    "republican", 
    "republican", 
    "democrat", 
    "republican", 
    "democrat", 
    ..................
```

The final test score response is shown above. Since, we have send the true labels we got the metrics as well. If we would have passes only the features, we would have obtained only **predicted labels**

The algorithm used to classify the data is [Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_boosting). I will talk about it more in the coming sections.

### I want to install it on my personal cloud

Awesome, yes you can install it on personal cloud of your choice. I will show on how to install it on [Google Cloud](--) and for other cloud platforms, it will be similar (or painful).

* Create your Google cloud account. It's free and google gives you free $300 credit for the first 60 days.

**Update** Google is now giving free trail for the first 12 months now.

* Create a new project called `ml-services`, google will generate a new project-id with the name lile `ml-services-some_number`

* Navigate to API manager section and the enable the API for **Google Compute Enginer API**

* Download [gcloud]() on your local machine. The file extension would be **tar.gz**

If you are on Mac/Linux, the code will look something like this.

```bash
$ cd google-cloud-sdk
$ ./install.sh
```

* Install `kubectl` 

```bash
$ gcloud components install kubectl
```
* Exporting your project id

```bash
$ export PROJECT_ID = `ml-services-some_number`
```

* Let's initialize the gcloud now..

```bash
$ gcloud init
```
You will be asked some questions. Be certain to select your account, zone and project.

* Creating a cluster

```bash
$ gcloud container clusters create ml-restful-cluster
```
The name of my cluster is `ml-restful-cluster`. You can name it whatever you want.

The process will take 3-5 minutes...

* The next step is to run the docker image on the cluster. Download and install [Docker]() on your machine.

* I have created an image for this tutorial already, let's pull that image from my **docker-hub**

```bash
$ docker pull tkanchin/ml-services-docker:v1
$ docker build -i -t tkanchin/ml-services-docker:v1 .
```

* Sweet, now that we have our image, let's run in our cluster.

Let's create a default authorization for our application.

```bash
$ gcloud auth application-default login
```

Deploy image to the cluster using **kubectl**

```bash
$ kubectl run ml-restful-cluster --image=tkanchin/ml-services-docker:v1 --port=8080
```

* Check the deployment status

```bash
$ kubectl get deployment
```
This may take a minute. 

* All set. Let's expose our cluster to the outside world.

```bash
$ kubectl expose deployment ml-restful-cluster --type="LoadBalancer" 
```

* FInd the external ip address

```bash
$ kubectl get services ml-restful-cluster
NAME                 CLUSTER-IP    EXTERNAL-IP     PORT(S)          AGE
ml-restful-cluster   10.3.243.46   104.196.154.0   8080:31555/TCP   1m

```
**For your application, the external-ip will be different**

* Repeat the above steps --

1. Create a train.json and test.json files

2. make the API call, replacing the ip address of the above with your new external ip address.

### I want to write my own image

Great, let's build things from the scratch now.. 

* Create a new folder and name it as 'server'. Create a new directory 'app' inside the server directory.

```bash
$ cd 
$ cd Desktop
$ mkdir server
$ cd server
$ mkdir app
```
* Copy the `requirements.txt` file created in the above steps to the 'app' folder.

* create a file `main.py` in the app directory. Copy the below code into the file

```python

'''
Some useful imports
'''

from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import json

'''
The main class, I am using an XGBOOST classifier in my example because it is robust and scalable. You can use which ever algorithm you want to use.
'''

class Algorithm(object):

	'''Want to change the algorithm change it here'''
	def __init__(self):
		self.clf = xgb.XGBClassifier()

	''' Doing a stratified k-fold cross-vailadation for model checking and training our classifier.
	'''
	def train(self, features, labels, folds):
		avg = []
		y_true = []
		y_pred = []
		kf = StratifiedKFold(labels, n_folds=folds)
		for train_index, test_index in kf:
			x_train, x_test = features[train_index], features[test_index]
			y_train, y_test = labels[train_index], labels[test_index]
			self.clf.fit(x_train,y_train)
			YP = self.clf.predict(x_test)
			avg.append(accuracy_score(y_test,YP))
			y_true.append(y_test.tolist())
			y_pred.append(YP.tolist())
		y_pred = [item for sublist in y_pred for item in sublist]
		y_true = [item for sublist in y_true for item in sublist]
		clf_metrics = self.classification_details(classification_report(y_true, y_pred))
		return_json = {'Metrics':clf_metrics, 'Accuracy':np.mean(avg)}
		self.clf = self.clf.fit(features, labels)
		return return_json
    
    '''
    Testing the classifier with the test set.
    '''
	def test(self, features, labels=None):
		y_pred = self.clf.predict(features)
		if labels is not None:
			acc = accuracy_score(labels, y_pred)
			clf_metrics = self.classification_details(classification_report(labels, y_pred))
			return {'Predictions':y_pred.tolist(), 'Metrics':clf_metrics, 'Accuracy':acc}
		return {'Predictions':y_pred.tolist()}
    
    '''
    Helper class. hacky!!!!
    '''
	def classification_details(self, string):
		string = [x for x in string.split() if x]
		last = string[-7:]
		last = last[3:]
		string = string[4:-7]
		listi = []
		dicti ={}
		for i in range(0, len(string), 5):
			dicti['Class'] = str(string[i])
			dicti['Precision'] = float(string[i + 1])
			dicti['Recall'] = float(string[i + 2])
			dicti['F1-Score'] = float(string[i +3])
			dicti['Support'] = float(string[i+4])
			listi.append(dicti)
			dicti = {}
		dicti['Class'] = 'Total'
		dicti['Precision'] = float(last[0])
		dicti['Recall'] = float(last[1])
		dicti['F1-Score'] = float(last[2])
		dicti['Support'] = float(last[3])
		listi.append(dicti)
		return listi

'''
Initialiing the flask app
'''
app = Flask(__name__)

algo = Algorithm()

'''
POST methods for train.json and test.json
'''
@app.route('/', methods=['POST'])
def train():
	resp = algo.train(np.array(request.json['features']["__ndarray__"]), np.array(request.json['labels']["__ndarray__"]), request.json['folds'])
	return jsonify(resp)

@app.route('/test', methods=['POST'])
def test():
	resp = algo.test(np.array(request.json['features']["__ndarray__"]), np.array(request.json['labels']["__ndarray__"]))
	return jsonify(resp)

'''
running the app in a development environment and exposing the port 8080
'''
if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8080, debug=True)

```

* Great, the app is now ready. Now first test this on a localhost

```bash
$ cd server/app
$ python main.py
```

App is now running on your localhost with port '8080' exposed. Try the curl operations using train.json and test.json and replace the ip address with `localhost` or `127.0.0.1`

* Next step is to write the dockerfile.

Navigate to the server directory and create a new file named `Dockerfile`

```bash
$ cd
$ cd Desktop/server
$ ls
app Dockerfile
```

Open your `Dockerfile` and type in the following code.

```
#We start from a know image from the docker-hub, in our case it is ubuntu:lastest 

#latest is called the TAG for that image

FROM ubuntu:latest

#update the package manager in your container

RUN apt-get update -y

#install the python dependencies in the container

RUN apt-get install -y python-pip python-dev build-essential

#copying the contents of the app folder 

COPY ./app /app

# Speciying the working directory as app

WORKDIR /app

# Now that we are in working directory, which is app. Install the modules in requiments.txt

RUN pip install -r requirements.txt

#CMD indicates command prompt or terminal. main.py is the starting point of our container.

CMD ["python", "main.py"]

#exposing the port 8080 for the container as well

EXPOSE 8080
```

* From the same directory, build the docker image

```bash
$ docker build -t dockerUserName/ImageName:version_number .
```

* check the images using the following command

```bash
$ docker images
```

Now repeat the above steps, tweak and play with the code and create a ML service for yourself.

**Note** The code is just for practice purposes. This type of code will **Never** be used in a production environment. Please optimize it according to your needs and have fun!!
