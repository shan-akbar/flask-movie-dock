from flask import render_template, url_for, flash, redirect, request, jsonify
from moviedock import app, db, bcrypt
from moviedock.forms import RegistrationForm, LoginForm, ReviewForm, SearchForm, CheckboxForm
from moviedock.models import User, Movie, Reviews, Recommendations
from flask_login import login_user, current_user, logout_user, login_required
import pandas as pd
import numpy as np
from moviedock.recommendation_models import content_based_recommendations, colaborative_filtering, colaborative_filtering1, get_sentiment_persentage
import datetime, random



"""
all_reviews=Reviews.query.all()
for r in all_reviews:
    sentiment_persentage=get_sentiment_persentage(r.review)
    r.negative = sentiment_persentage[0][0]
    r.neutral = sentiment_persentage[0][1]
    r.positive = sentiment_persentage[0][2]
    print(r.negative)
    print(r.neutral)
    print(r.positive)
db.session.commit()
print("HI i am exiting the program!")
exit()
"""
"""
for index, row in movies_data.iterrows():
    movie = Movie(movie_link=row['movie_imdb_link'], movie_title=row['movie_title'], genres=row['genres'],
                    language=row['language'], movie_year=row['movie_year'],actor_1_name=row['actor_1_name'], actor_2_name=row['actor_2_name'],
                    actor_3_name=row['actor_3_name'], director_name=row['director_name'], movie_duration=row['movie_duration'],
                    content_rating=row['content_rating'], rating=row['imdb_score'], plot_summary=row['plot_summary'],
                    poster_link=row['poster_link'])

    db.session.add(movie)

db.session.commit()
hashed_password = bcrypt.generate_password_hash("password").decode('utf-8')
for index, row in users_data.iterrows():

    user = User(username=row['user_name'], email=row['user_name']+"@gmail.com", password=hashed_password)

    db.session.add(user)

db.session.commit()
for index, row in reviews_data.iterrows():
    date_time_obj = datetime.datetime.strptime(row['time_stamp'], '%d-%b-%y')
    review = Reviews(review=row['review'], user_id=row['userId'], movie_id=row['movie_id'], date_posted=date_time_obj)
    db.session.add(review)
db.session.commit()
"""
@app.route("/" , methods=['POST', 'GET'])
@app.route("/home", methods=['POST', 'GET'])
def home():
    movies = Movie.query.filter(Movie.rating >= 8.0 ).all()
    message = "Top Rated Movies"
    if current_user.is_authenticated:
        message="You May Like"
        recommendations =  Recommendations.query.filter(Recommendations.user_id==current_user.id).all()
        recommended_movies=[]
        for reco_movie in recommendations:
            recommended_movies.append(Movie.query.filter(Movie.id ==reco_movie.movie_id ).all()[0])
        if(len(recommended_movies)!=0):
            movies = recommended_movies
            movies.reverse()

        else:
            if(len(current_user.genre_preference)!=0):
                message = "Movies suitable to your Intrests"
                tokens=current_user.genre_preference.split("|")
                movies = []
                for token in tokens:
                    search = "%{}%".format(token)
                    movies . extend( Movie.query.filter(Movie.genres.like(search)).filter(Movie.rating >= 8.0 ).all())
                random.shuffle(movies)

    form = SearchForm()
    if form.validate_on_submit():
        tag = form.search.data
        search = "%{}%".format(tag)
        movies = Movie.query.filter(Movie.movie_title.like(search)).all()
        movies . extend( Movie.query.filter(Movie.genres.like(search)).all())
        movies . extend( Movie.query.filter(Movie.actor_1_name.like(search)).all())
        movies . extend( Movie.query.filter(Movie.actor_2_name.like(search)).all())
        movies . extend( Movie.query.filter(Movie.actor_3_name.like(search)).all())
        movies . extend( Movie.query.filter(Movie.director_name.like(search)).all())
        message="Search Results"
    if(len(movies) > 50):
        movies = movies[0:50]
    return render_template('home.html', movies=movies, title="Movies", message=message, form=form)

@app.route('/newUser', methods=['post', 'get'])
@login_required
def newUser():
    form = CheckboxForm()
    if current_user.is_authenticated:
        if form.validate_on_submit():
            genre_preference = ""
            if(form.action.data):
                genre_preference+= "Action"
            if(form.adventure.data):
                if(len(genre_preference) != 0):
                    genre_preference+= "|"
                genre_preference+= "Adventure"
            if(form.animation.data):
                if(len(genre_preference) != 0):
                    genre_preference+= "|"
                genre_preference+= "Animation"
            if(form.comedy.data):
                if(len(genre_preference) != 0):
                    genre_preference+= "|"
                genre_preference+= "Comedy"
            if(form.crime.data):
                if(len(genre_preference) != 0):
                    genre_preference+= "|"
                genre_preference+= "Crime"
            if(form.drama.data):
                if(len(genre_preference) != 0):
                    genre_preference+= "|"
                genre_preference+= "Drama"
            if(form.fantasy.data):
                if(len(genre_preference) != 0):
                    genre_preference+= "|"
                genre_preference+= "Fantasy"
            if(form.horror.data):
                if(len(genre_preference) != 0):
                    genre_preference+= "|"
                genre_preference+= "Horror"
            if(form.mystery.data):
                if(len(genre_preference) != 0):
                    genre_preference+= "|"
                genre_preference+= "Mystery"
            if(form.romance.data):
                if(len(genre_preference) != 0):
                    genre_preference+= "|"
                genre_preference+= "Romance"
            if(form.scifi.data):
                if(len(genre_preference) != 0):
                    genre_preference+= "|"
                genre_preference+= "Sci-Fi"
            if(form.thriller.data):
                if(len(genre_preference) != 0):
                    genre_preference+= "|"
                genre_preference+= "Thriller"
            current_user.genre_preference = genre_preference
            db.session.commit()
            return redirect(url_for('home'))
        else:
            return render_template('newUser.html', form=form)
    else:
        return redirect(url_for('home'))

@app.route("/about")
def about():

    return render_template('about.html', title='About')


@app.route("/register", methods=['GET', 'POST'])
def register():
    global current_user
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash(f'Account created for {form.username.data}!', 'success')
        return redirect(url_for('newUser'))
    return render_template('register.html', title='Register', form=form)


@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        if current_user.username == "admin":
            return redirect(url_for('admin'))
        else:
            return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            if(current_user.username == "admin"):
                return redirect(next_page) if next_page else redirect(url_for('admin'))
            else:
                return redirect(next_page) if next_page else redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', title='Login', form=form)

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route("/account")
@login_required
def account():
    return render_template('account.html', title='Account')

@app.route("/review/<int:movie_id>", methods=['GET', 'POST'])
@login_required
def review(movie_id):
    review = Reviews()
    movie_row = Movie.query.filter_by(id=movie_id).first()
    form = ReviewForm()
    if form.validate_on_submit():
        sentiment_persentage=get_sentiment_persentage(form.review.data)
        print(sentiment_persentage)
        previous_review=Reviews.query.filter(Reviews.user_id==current_user.id).filter(Reviews.movie_id==movie_id).all()
        if(len(previous_review)!=0):
            previous_review[0].review = form.review.data
            previous_review[0].negative = sentiment_persentage[0][0]
            previous_review[0].neutral = sentiment_persentage[0][1]
            previous_review[0].positive = sentiment_persentage[0][2]
            review = previous_review[0]
            temp=Recommendations.query.filter(Recommendations.user_id==current_user.id).filter(Recommendations.reviewed_movie==movie_id).all()
            for r in temp:
                db.session.delete(r)
        else:
            review = Reviews(review=form.review.data, user_id=current_user.id, movie_id=movie_id,
            negative = sentiment_persentage[0][0], neutral = sentiment_persentage[0][1], positive = sentiment_persentage[0][2])
            db.session.add(review)
        db.session.commit()
        recommended_movies=colaborative_filtering1(current_user, movie_row, review)
        for r_movie_id in recommended_movies:
            recommendation = Recommendations(user_id=current_user.id, movie_id=r_movie_id, reviewed_movie=movie_id)
            db.session.add(recommendation)
        db.session.commit()
        flash('Review has been submitted!', 'success')
        return redirect(url_for('home'))
    return render_template('review.html', movie_row=movie_row, title='review', form=form)


@app.route('/reviews/<int:movie_id>')
def reviews(movie_id):
    movie_row = Movie.query.filter_by(id=movie_id).first()
    user_reviews = movie_row.reviews

    current_user_reviews = {}
    users_reviews_dict = {}

    for user_review in user_reviews:
        check = True
        user = User.query.filter_by(id=user_review.user_id).first()
        if current_user.is_authenticated:
            if current_user.id == user.id:
                current_user_reviews[user.username] = user_review
                check = False
        if(check):
            users_reviews_dict[user.username] = user_review
    return render_template('reviews.html', movie_row=movie_row, title="User Reviews", users_reviews_dict=users_reviews_dict,
    current_user_reviews = current_user_reviews)

@app.route('/movie/<int:movie_id>')
def movie(movie_id):
    #need to update model based on id change.
    movie_row = Movie.query.filter_by(id=movie_id).first()
    total_reviews = len(movie_row.reviews)
    top_10_indexes, _, _ = content_based_recommendations(movie_id)
    recommended_movies = Movie.query.filter(Movie.id.in_(top_10_indexes)).all()

    return render_template('movie.html', movie_row=movie_row, recommended_movies = recommended_movies,
    movie_id=movie_id, total_reviews=total_reviews)

@app.route("/admin")
@login_required
def admin():
    if current_user.username == "admin":
        users = User.query.all()

        return render_template('admin.html', title='Admin', users=users)


@app.route('/search', methods=['POST'])
def search():

    return render_template('home.html', )
