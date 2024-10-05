# wine-data
# Import necessary libraries
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re

# Download necessary NLTK resources (uncomment the next two lines if you haven't downloaded them yet)
# nltk.download('punkt')
# nltk.download('wordnet')

# Correct file path
file_path = r'C:\Users\Dell\Desktop\wine-ratings.csv'

# Read the CSV file
wine_data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(wine_data.head())

# Step 1: Identify and remove duplicate rows
# Check for duplicates
print("Number of duplicate rows:", wine_data.duplicated().sum())

# Remove duplicates
wine_data_no_duplicates = wine_data.drop_duplicates()

# Step 2: Identify missing values
# Check for missing values
print("Missing values in each column:\n", wine_data_no_duplicates.isnull().sum())

# Step 3: Remove rows with missing values (you can also fill them if needed)
# Remove missing values
non_dm_data = wine_data_no_duplicates.dropna()

# Display the cleaned dataset
print(non_dm_data.head())

# Check if any duplicates or missing values remain
print("Remaining duplicates:", non_dm_data.duplicated().sum())
print("Remaining missing values:\n", non_dm_data.isnull().sum())

# Step 4: Calculate percentage distribution of wine varieties
variety_counts = non_dm_data['variety'].value_counts(normalize=True) * 100
percentage_distribution = variety_counts.round(1).reset_index()
percentage_distribution.columns = ['variety', 'percentage']
print("\nPercentage distribution of wine varieties:\n", percentage_distribution)

# Step 5: Classify wines based on rating
def classify_quality(rating):
    if 85 <= rating < 90:
        return 'Good'
    elif 90 <= rating < 95:
        return 'Outstanding'
    elif 95 <= rating <= 100:
        return 'Exceptional'
    else:
        return 'Not Rated'

# Create a new column 'wine_quality'
non_dm_data['wine_quality'] = non_dm_data['rating'].apply(classify_quality)

# Display the updated DataFrame
print("\nUpdated DataFrame with wine quality:\n", non_dm_data[['name', 'rating', 'wine_quality']])

# Step 6: Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Function to process notes
def process_notes(note):
    # Tokenize
    words = nltk.word_tokenize(note)
    # Remove non-alphabetic characters and convert to lowercase
    words = [re.sub(r'[^a-zA-Z]', '', word).lower() for word in words if word.isalpha()]
    # Apply stemming and lemmatization
    stemmed_words = [stemmer.stem(word) for word in words]
    lemmatized_words = [lemmatizer.lemmatize(word) for word in stemmed_words]
    return ' '.join(lemmatized_words)

# Apply the process_notes function to the notes column
non_dm_data['processed_notes'] = non_dm_data['notes'].apply(process_notes)

# Display the updated DataFrame with processed notes
print("\nUpdated DataFrame with processed notes:\n", non_dm_data[['notes', 'processed_notes']])

# Step 7: Save the updated DataFrame to a new CSV file
non_dm_data.to_csv(r'C:\Users\Dell\Desktop\updated_wine_data.csv', index=False)
print("Updated data saved to 'updated_wine_data.csv'.")

# Section B: Text Preprocessing

# Step 1: Convert all text to lowercase for uniformity
wine_data['notes'] = wine_data['notes'].str.lower()

# Step 2: Remove numbers
wine_data['notes'] = wine_data['notes'].str.replace(r'\d+', '', regex=True)

# Step 3: Eliminate punctuation marks
wine_data['notes'] = wine_data['notes'].str.translate(str.maketrans('', '', string.punctuation))

# Step 4: Lemmatization with POS tagging
lemmatizer = WordNetLemmatizer()

def lemmatize_with_pos(text):
    tokens = word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)  # Get POS tags
    lemmatized_words = []
    for word, tag in pos_tags:
        # Convert NLTK POS tags to WordNet POS tags
        if tag.startswith('J'):
            pos = 'a'  # Adjective
        elif tag.startswith('V'):
            pos = 'v'  # Verb
        elif tag.startswith('N'):
            pos = 'n'  # Noun
        elif tag.startswith('R'):
            pos = 'r'  # Adverb
        else:
            pos = None  # Not a known tag
        lemmatized_word = lemmatizer.lemmatize(word, pos) if pos else lemmatizer.lemmatize(word)
        lemmatized_words.append(lemmatized_word)
    return ' '.join(lemmatized_words)

wine_data['notes'] = wine_data['notes'].apply(lemmatize_with_pos)

# Step 5: Remove stopwords
stop_words = set(stopwords.words('english'))
wine_data['notes'] = wine_data['notes'].apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))

# Create a new column "description_token" to store the cleaned text
wine_data['description_token'] = wine_data['notes']

# Display the updated DataFrame with cleaned notes
print("\nUpdated DataFrame with cleaned notes:\n", wine_data[['notes', 'description_token']].head())

# Section C: Analysis

# 1. Identify top 5 words associated with wine quality
def get_top_words_by_quality(df, quality):
    words = df[df['wine_quality'] == quality]['description_token'].str.cat(sep=' ').split()
    word_counts = Counter(words)
    return word_counts.most_common(5)

# Group by wine quality and get top words
top_words = {
    'Good': get_top_words_by_quality(wine_data, 'Good'),
    'Outstanding': get_top_words_by_quality(wine_data, 'Outstanding'),
    'Exceptional': get_top_words_by_quality(wine_data, 'Exceptional')
}

print("\nTop 5 words associated with each wine quality:")
for quality, words in top_words.items():
    print(f"{quality}: {words}")

# 2. Display the top 5 regions producing the highest number of Exceptional quality wines
exceptional_wines = wine_data[wine_data['wine_quality'] == 'Exceptional']
top_regions_exceptional = exceptional_wines['region'].value_counts().head(5)
print("\nTop 5 regions producing the highest number of Exceptional quality wines:")
print(top_regions_exceptional)

# 3. Identify regions with the highest diversity of wine varieties
diversity = wine_data.groupby('region')['variety'].nunique().reset_index()
diversity.columns = ['region', 'diversity_count']
highest_diversity_regions = diversity.nlargest(5, 'diversity_count')
print("\nRegions with the highest diversity of wine varieties:")
print(highest_diversity_regions)
