import pickle
from sklearn.datasets import fetch_20newsgroups
#fetch_20newsgroups internet se dataset load krne ke liye import kr rahe
# sk learn se dataset loader import kr rahe
from sklearn.feature_extraction.text import TfidfVectorizer
#tfidf vectorizer import kr rahe text data ko vector me convert krne ke liye
from sklearn.naive_bayes import MultinomialNB
#naive bayes import kr rahe classification ke liye
data = fetch_20newsgroups(subset='train')
#thousands of data load kr rahe
#store Documents
#Documents = data.data
# store Text and labels in x and y
x_text = data.data
y = data.target
#create tf-idf object
vectorizer = TfidfVectorizer()
#create text in to numbers
x = vectorizer.fit_transform(x_text)
#ceate model
model = MultinomialNB()
#Train the model
model.fit(x, y)
#print(x.shape)
print("model trained successfully")

#PREDICTION
my_text = ["The team won the cricket match"]
#convert text to vector
my_text_vector = vectorizer.transform(my_text)
#predict category
prediction =model.predict(my_text_vector)
#print prediction number
print(prediction)
#print category name
print(data.target_names[prediction[0]])
with open('model.pkl', 'wb')as f:
    pickle.dump((vectorizer,model,data.target_names),f)
    print("model saved successfully")
#print(data.target_names)