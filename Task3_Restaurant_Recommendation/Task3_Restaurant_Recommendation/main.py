import pandas as pd

# Sample restaurant dataset
data = {
    'Restaurant': ['Shabree', 'Vaishali', 'Barbeque Nation', 'Maratha Food', 'Pizza Hut', 'Dosa Plaza'],
    'Location': ['Pune', 'Pune', 'Mumbai', 'Pune', 'Mumbai', 'Pune'],
    'Cuisine': ['Maharashtrian', 'South Indian', 'BBQ', 'Maharashtrian', 'Italian', 'South Indian'],
    'Rating': [4.5, 4.3, 4.6, 4.2, 4.0, 4.4],
    'Price': ['Medium', 'Low', 'High', 'Low', 'Medium', 'Low']
}

df = pd.DataFrame(data)

def recommend_restaurants(location, cuisine, min_rating=4.0):
    filtered = df[
        (df['Location'].str.lower() == location.lower()) &
        (df['Cuisine'].str.lower() == cuisine.lower()) &
        (df['Rating'] >= min_rating)
    ]
    return filtered.sort_values(by='Rating', ascending=False)

print("=== Restaurant Recommendation System ===")
user_location = input("तुझं location टाक: ")
user_cuisine = input("
