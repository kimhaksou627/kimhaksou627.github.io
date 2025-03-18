import pandas as pd
import ast
import statsmodels.api as sm

# Load the data with the correct encoding
try:
    df = pd.read_csv('Spotify Dataset.csv', encoding='utf-8')
except UnicodeDecodeError:
    try:
        df = pd.read_csv('Spotify Dataset.csv', encoding='latin1')
    except UnicodeDecodeError:
        df = pd.read_csv('Spotify Dataset.csv', encoding='gbk')

# Display the first few rows to understand the structure
print(df.head())

# Remove duplicates
df = df.drop_duplicates()

# Check if 'id' and 'release_date' columns exist before trying to drop them
columns_to_drop = ['id', 'release_date']
existing_columns = df.columns.tolist()
columns_to_drop = [col for col in columns_to_drop if col in existing_columns]
df = df.drop(columns=columns_to_drop)

# Handle missing values (example: fill missing values in 'popularity' with the mean)
df['popularity'] = df['popularity'].fillna(df['popularity'].mean())

# Standardize artist names (example: convert to lowercase and remove brackets and apostrophes)
def clean_artists(artists_str):
    try:
        # Safely evaluate the string as a list
        artists_list = ast.literal_eval(artists_str)
        # Join the artists' names into a single string separated by commas
        return ', '.join(artists_list)
    except (ValueError, SyntaxError):
        # Handle cases where the string is not a valid list
        return artists_str

df['artists'] = df['artists'].apply(clean_artists)

# Normalize numerical data (example: normalize 'energy')
df['energy'] = (df['energy'] - df['energy'].min()) / (df['energy'].max() - df['energy'].min())

# Correct inconsistent data (example: replace typos in 'name')
df['name'] = df['name'].replace({'old_name': 'new_name'})

# Add missing information (example: fill missing 'explicit' with 0)
df['explicit'] = df['explicit'].fillna(0)

# Handle outliers (example: cap 'loudness' at a certain threshold)
df['loudness'] = df['loudness'].clip(lower=-20, upper=0)

# Consolidate categories (example: group rare artists into 'Other')
artist_counts = df['artists'].value_counts()
df['artists'] = df['artists'].apply(lambda x: 'Other' if artist_counts[x] < 5 else x)

# Add missing information (example: fill missing 'explicit' with 0)
df['explicit'] = df['explicit'].fillna(0)

# Save the cleaned data to a new CSV file with the correct encoding
#df.to_csv('cleaned_data.csv', index=False, encoding='utf-8')

#Load the cleaned data
df_cleaned = pd.read_csv('cleaned_data.csv')

#Select the relevant variables for the regression model
independent_vars = ['acousticness', 'danceability', 'energy', 'explicit', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'year']
dependent_var = 'popularity'

#Prepare the data for the regression model
X = df_cleaned[independent_vars]
y = df_cleaned[dependent_var]

#Add a constant to the independent variables for the intercept
X = sm.add_constant(X)

#Fit the linear regression model
model = sm.OLS(y, X).fit()

#Extract the coefficients, standard errors, and p-values from the model summary
summary = model.summary()
coefficients = model.params
std_errors = model.bse
p_values = model.pvalues

#Create a DataFrame to store the results
results_df = pd.DataFrame({
    'Variable': coefficients.index,
    'Coefficient': coefficients.values,
    'Std_Error': std_errors.values,
    'P_Value': p_values.values
})

#Add significance levels
results_df['Significance'] = results_df['P_Value'].apply(lambda p: '*' if p < 0.01 else '' if p < 0.05 else '')

#Print the formatted results
print(results_df[['Variable', 'Coefficient', 'Std_Error', 'P_Value', 'Significance']])

# Calculate summary statistics
summary_stats = df_cleaned[independent_vars + [dependent_var]].describe()

# Select only the rows for mean, min, max, and standard deviation (std)
summary_stats_filtered = summary_stats.loc[['mean', 'min', 'max', 'std']]

# Transpose the filtered summary statistics for better readability
summary_stats_filtered_transposed = summary_stats_filtered.transpose()

# Print the filtered and transposed summary statistics
print(summary_stats_filtered_transposed)