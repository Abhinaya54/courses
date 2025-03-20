from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load pre-trained model
try:
    with open('recommendation_model.pkl', 'rb') as f:
        df, countvect, cosine_mat = pickle.load(f)
        print("Model Loaded Successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    df, countvect, cosine_mat = None, None, None


# Course Recommendation Function
def recommend_course(title, numrec=6):
    course_index = pd.Series(df.index, index=df['course_title']).drop_duplicates()
    if title not in course_index:
        return pd.DataFrame()
    index = course_index[title]
    scores = list(enumerate(cosine_mat[index]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    selected_course_index = [i[0] for i in sorted_scores[1:numrec + 1]]
    rec_df = df.iloc[selected_course_index]
    return rec_df[['course_title', 'url', 'price']]


# Search Term Function
def searchterm(term):
    result_df = df[df['course_title'].str.contains(term, case=False, na=False)]
    return result_df.sort_values(by='num_subscribers', ascending=False).head(6)


# Home Route
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        titlename = request.form.get('course')
        print(f"Received course name: {titlename}")

        if titlename:
            recdf = recommend_course(titlename)
            print(f"Recommendation DataFrame:\n{recdf}")

            if not recdf.empty:
                coursemap = dict(zip(recdf['course_title'], recdf['url']))
                return render_template('index.html', coursemap=coursemap, coursename=titlename, showtitle=True)
            else:
                resultdf = searchterm(titlename)
                print(f"Search Result DataFrame:\n{resultdf}")

                if not resultdf.empty:
                    coursemap = dict(zip(resultdf['course_title'], resultdf['url']))
                    return render_template('index.html', coursemap=coursemap, coursename=titlename, showtitle=True)
                else:
                    print("No results found!")
                    return render_template('index.html', showerror=True, coursename=titlename)

    return render_template('index.html')


# New Course Page Route
@app.route('/course.html')
def course_page():
    return render_template('course.html')


# Run Flask Application
if __name__ == '__main__':
    app.run(debug=True, port=5000)
