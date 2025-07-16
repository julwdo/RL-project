import os
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
import pickle

kaggle_username = ''
kaggle_key = ''

#os.environ['KAGGLE_USERNAME'] = kaggle_username
#os.environ['KAGGLE_KEY'] = kaggle_key

#os.system('kaggle datasets download sarthak71/food-recipes -p data --unzip')

recipes_df = pd.read_csv("data/food_recipes.csv")
recipes_df.columns

recipes_df['category'].unique()
recipes_df = recipes_df[recipes_df['category'] == 'Pasta Recipes']

recipes_df = recipes_df[['recipe_title', 'ingredients', 'rating']].copy()

recipes_df.isna().sum()

lemmatizer = WordNetLemmatizer()
split_pattern = re.compile(r'\s*\|\s*|\s+and\s+')
english_only_pattern = re.compile(r'^[A-za-z\s]+$')

def split_ingredients(text):
    """Split ingredient string on '|' or 'and' (with spaces)."""
    return split_pattern.split(text)

def is_english_only(text):
    """Check if text contains only English letters and spaces."""
    return bool(english_only_pattern.fullmatch(text.strip()))

def process_ingredient(ingredient):
    """Lowercase, lemmatize each word as noun, and rejoin."""
    words = ingredient.lower().split()
    lemmatized_words = [lemmatizer.lemmatize(w, pos='n') for w in words]
    return ' '.join(lemmatized_words).strip()

def process_ingredient_string(text):
    """Split ingredient string, filter English-only, lemmatize."""
    ingredients = split_ingredients(text)
    cleaned = [
        process_ingredient(ing) 
        for ing in ingredients 
        if is_english_only(ing)
    ]
    return cleaned if cleaned else None

recipes_df['ingredients'] = recipes_df['ingredients'].apply(process_ingredient_string)

recipes_df.isna().sum()
recipes_df.dropna(subset=['ingredients'], inplace=True)

selected_recipes = []
unique_ingredients = set()

for row in recipes_df.itertuples(index=False):
    current_ingredients = set(row.ingredients)
    
    if len(unique_ingredients | current_ingredients) <= 25:
        selected_recipes.append((
            row.recipe_title,
            frozenset(current_ingredients),
            row.rating
        ))
        unique_ingredients.update(current_ingredients)
        
        if len(unique_ingredients) == 25:
            break

print(f"Selected {len(selected_recipes)} recipes with {len(unique_ingredients)} unique ingredients.")

index_to_ingredient = {i: ing for i, ing in enumerate(unique_ingredients)}

ingredient_to_cost = {
    'tomato': 2,
    'red chilli powder': 2,
    'salt': 1,
    'chicken breast': 7,
    'rosemary': 2,
    'conchiglie pasta': 4,
    'onion': 1,
    'extra virgin olive oil': 5,
    'broccoli': 3,
    'sugar': 1,
    'white vinegar': 2,
    'tagliatelle pasta': 4,
    'red chilli flake': 2,
    'parmesan cheese': 6,
    'dried oregano': 2,
    'basil leaf': 2,
    'black pepper powder': 2,
    'fresh thyme leaf': 3,
    'ginger': 2,
    'dry red chilli': 2,
    'lemon juice': 2,
    'whole black pepper corn': 3,
    'walnut': 6,
    'spaghetti pasta': 3,
    'garlic': 1
}

data_to_save = {
    'recipes': selected_recipes,
    'index_to_ingredient': index_to_ingredient,
    'ingredient_to_cost': ingredient_to_cost
}

with open('data/processed_data.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)