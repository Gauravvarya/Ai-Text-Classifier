from flask import Flask, render_template, request
import pickle

# create flask app
app = Flask(__name__)

# load the trained model (vectorizer, model, category names)
with open('model.pkl', 'rb') as f:
   vectorizer, model, categories = pickle.load(f)


@app.route('/', methods=["GET", "POST"])
def home():
   prediction = None
   if request.method == "POST":
      # Get user input
      user_input = request.form.get('text', '')
      x = vectorizer.transform([user_input])
      # model.predict returns an array of predicted label indices
      pred_index = model.predict(x)[0]
      # Map index to category name
      try:
         prediction = categories[pred_index]
      except Exception:
         prediction = str(pred_index)

   return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
   app.run(debug=True, port=5000)