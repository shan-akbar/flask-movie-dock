from datetime import datetime
from moviedock import db, login_manager
from flask_login import UserMixin
#from flask_whooshalchemy import whoosh_index

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    genre_preference = db.Column(db.String(100))
    reviews = db.relationship('Reviews', backref='author', lazy=True)
    recommended = db.relationship('Recommendations', backref='user', lazy=True)

class Movie(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    movie_link = db.Column(db.String(200), unique=True, nullable=False)
    movie_title = db.Column(db.String(30), nullable=False)
    genres = db.Column(db.String(100))
    language = db.Column(db.String(20))
    movie_year = db.Column(db.Integer)
    actor_1_name = db.Column(db.String(30))
    actor_2_name = db.Column(db.String(30))
    actor_3_name = db.Column(db.String(30))
    director_name = db.Column(db.String(30))
    movie_duration = db.Column(db.String(20))
    content_rating = db.Column(db.String(20))
    rating = db.Column(db.Float)
    plot_summary = db.Column(db.String(1000))
    poster_link = db.Column(db.String(1000))
    reviews = db.relationship('Reviews', backref='movie', lazy=True)
    recommended = db.relationship('Recommendations', backref='movie', lazy=True)

class Reviews(db.Model):
    review_id = db.Column(db.Integer, primary_key=True)
    review = db.Column(db.Text, nullable=False)
    like = db.Column(db.Boolean)
    dislike = db.Column(db.Boolean)
    rating = db.Column(db.Float)
    positive = db.Column(db.Float)
    neutral = db.Column(db.Float)
    negative = db.Column(db.Float)
    date_posted = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    movie_id = db.Column(db.Integer, db.ForeignKey('movie.id'), nullable=False)

class Recommendations(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    movie_id = db.Column(db.Integer, db.ForeignKey('movie.id'), nullable=False)
    reviewed_movie = db.Column(db.Integer)
