from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, PowerTransformer
import numpy as np
import pandas as pd
import kagglehub

def download_data():
    path = kagglehub.dataset_download("gauravkumar2525/top-rated-movies-from-tmdb")
    df = pd.read_csv(path + '\\top_rated_movies.csv')
    df.to_csv("movies.csv", index=False)
    return df

def preprocessing_data_frame(path):
    df = pd.read_csv(path)
    cat_columns = ['original_title']
    numeric_features = ['vote_average', 'vote_count',
                        'release_year', 'release_month', 'release_day', 'release_weekday',
                        'is_weekend_release', 'years_since_release', 'overview_length',
                        'title_length', 'vote_power', 'vote_engagement', 'rating_to_years', 'votes_per_year',
                        'rating_power']
    df['release_date'] = pd.to_datetime(df['release_date'])
    df = df[df['release_date'] > '1900-01-01']
    df = df[df['popularity'] <= 100.0]

    df['release_year'] = df['release_date'].dt.year
    df['release_month'] = df['release_date'].dt.month
    df['release_day'] = df['release_date'].dt.day
    df['release_weekday'] = df['release_date'].dt.weekday
    df['is_weekend_release'] = df['release_weekday'].isin([5, 6]).astype(int)
    df['years_since_release'] = 2023 - df['release_year']
    df['overview_length'] = df['overview'].str.len()
    df['title_length'] = df['original_title'].str.len()
    df['vote_power'] = df['vote_average'] * np.log1p(df['vote_count'])
    df['vote_engagement'] = df['vote_average'] * np.log1p(df['vote_count'])
    df['rating_to_years'] = df['vote_average'] / (df['years_since_release'] + 1)
    df['votes_per_year'] = df['vote_count'] / (df['years_since_release'] + 1)

    # df['log_popularity'] = np.log1p(df['popularity'])
    df['rating_power'] = df['vote_average'] * np.log1p(df['vote_count'])
    df = df.drop(columns=['overview', 'release_date', 'id'])

    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    ordinal = OrdinalEncoder()
    ordinal.fit(df[cat_columns])
    Ordinal_encoded = ordinal.transform(df[cat_columns])
    df_ordinal = pd.DataFrame(Ordinal_encoded, columns=cat_columns)
    df[cat_columns] = df_ordinal[cat_columns]

    df.to_csv('df_clear.csv')
    return True

download_data()
preprocessing_data_frame("movies.csv")
