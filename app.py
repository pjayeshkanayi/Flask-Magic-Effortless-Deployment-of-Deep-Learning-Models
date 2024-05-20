from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

dic = {0 : 'Good_Tomato', 1 : 'Rotten_Tomato'}

model = load_model('tomato_condition_classification.h5')


def predict_label(img_path):
	i = image.load_img(img_path, target_size=(150,150))
	i = image.img_to_array(i)/255.0
	i = i.reshape(1, 150,150,3)
	p = model.predict(i)
	result=p[0][0]
	npresult=round(result)
	return dic[npresult]


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Welcome ......"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	
	app.run(debug = True)