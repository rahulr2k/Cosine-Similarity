from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import string
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
df = pd.read_csv("MPR.csv")
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
        ##################################################

       ##Step 2: Select Features
        
        features = ['keywords','genres']
        ##Step 3: Create a column in DF which combines all selected features
        for feature in features:
            df[feature] = df[feature].fillna('')

        def combine_features(row):
            try:
                return row['keywords'] +" "+row["genres"]
            except:
                print ("Error:", row)	

        df["combined_features"] = df.apply(combine_features,axis=1)

        #print "Combined Features:", df["combined_features"].head()

        ##Step 4: Create count matrix from this new combined column
        cv = CountVectorizer()

        count_matrix = cv.fit_transform(df["combined_features"])

        ##Step 5: Compute the Cosine Similarity based on the count_matrix
        cosine_sim = cosine_similarity(count_matrix) 
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
        
        for element in movie0:
            
            movie1.append(get_url_from_index(element))
            movie2.append(get_poster_from_index(element))
        

    return render_template('result.html',movie0=movie0,movie1=movie1,movie2=movie2)



if __name__ == '__main__':
    app.run(debug=True)
