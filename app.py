from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import string
import re
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
df = pd.read_csv("mpr8.csv")

features = ['keywords','genres','nas']
##Step 3: Create a column in DF which combines all selected features
for feature in features:
    df[feature] = df[feature].fillna('')

def combine_features(row):
    try:
        return row['keywords'] +" "+row["genres"]+" "+row["nas"]
    except:
        print ("Error:", row)   

df["combined_features"] = df.apply(combine_features,axis=1)
df.originalTitle  = df.originalTitle.astype(str).apply(lambda x : x.replace("'", ''))
originalTitlelist = df.originalTitle.values.tolist()

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html',prediction = originalTitlelist)

@app.route('/predict',methods=['POST'])
def predict():
    
   
    if request.method == 'POST':

        message = request.form.get('message')
                
        ###### helper functions. Use them when needed #######
        def get_originalTitle_from_index(index):
            return df[df.index == index]["originalTitle"].values[0]

        def get_index_from_originalTitle(originalTitle):
            return df[df.originalTitle == originalTitle]["index"].values[0]

        def get_poster_from_index(index):
            return df[df.originalTitle == index]["poster"].values[0]
        def get_url_from_index(index):
            return df[df.originalTitle == index]["URL"].values[0]
        def get_ytb_from_index(index):
            return df[df.originalTitle == index]["youtube"].values[0]

        def get_kwd_from_index(index):
            return df[df.originalTitle == index]["keywords"].values[0]
        def get_gen_from_index(index):
            return df[df.originalTitle == index]["genres"].values[0]
        def get_ar_from_index(index):
            return df[df.originalTitle == index]["averageRating"].values[0]
        def get_nv_from_index(index):
            return df[df.originalTitle == index]["numVotes"].values[0]




        with open('count_matrix.pkl', 'rb') as f:
            count_matrix = pickle.load(f)

        with open('cosine_sim.pkl', 'rb') as f:
            cosine_sim = pickle.load(f)

        movie_user_likes = message

        ## Step 6: Get index of this movie from its originalTitle
        movie_index = get_index_from_originalTitle(movie_user_likes)

        similar_movies =  list(enumerate(cosine_sim[movie_index]))

        ## Step 7: Get a list of similar movies in descending order of similarity score
        sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)

        ## Step 8: Print originalTitles of first 50 movies
        i=0
        movie0 = []
        for element in sorted_similar_movies:
                movie0.append(str(get_originalTitle_from_index(element[0])))
                i=i+1
                if i>7:
                    break

        
        movie1 = []
        movie2 = []
        movie3 = []
        movie4 = []
        movie5 = []
        movie6 = []
        movie7 = []
        
        for element in movie0:
                        
            movie1.append(get_url_from_index(element))
            movie2.append(get_poster_from_index(element))
            movie3.append(get_ytb_from_index(element))
            movie4.append(get_kwd_from_index(element))
            movie5.append(get_gen_from_index(element))
            movie6.append(get_ar_from_index(element))
            movie7.append(get_nv_from_index(element))
        

    return render_template('result.html',movie0=movie0,movie1=movie1,movie2=movie2,movie3=movie3,movie4=movie4,movie5 = movie5,movie6=movie6,movie7=movie7)



if __name__ == '__main__':
    app.run(debug=True)
