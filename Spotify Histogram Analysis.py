import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



# Load the data
df = pd.read_csv('cleaned_data.csv')

# Define the attributes list
attributes = ['acousticness', 'danceability', 'energy', 'loudness', 'speechiness', 'instrumentalness', 'liveness', 'valence', 'tempo']

most_popular_song = df.loc[df['popularity'].idxmax()]
least_popular_song = df.loc[df['popularity'].idxmin()]
avg_popularity = df['popularity'].mean()
closest_to_avg = (df['popularity'] - avg_popularity).abs().idxmin()
avg_song = df.loc[closest_to_avg]



# Define custom colors
background_color = '#212121'
all_songs_color = '#1db954'
most_popular_color = 'red'
least_popular_color = 'green'
average_color = 'orange'

# Create a figure with subplots
fig, axes = plt.subplots(nrows=len(attributes), ncols=1, figsize=(10, 20))

# Set the background color
fig.patch.set_facecolor(background_color)
for ax in axes:
    ax.set_facecolor(background_color)

# Loop through each attribute and create a histogram
for i, attr in enumerate(attributes):
    sns.histplot(df[attr], ax=axes[i], kde=True, color=all_songs_color, label='All Songs')
    axes[i].axvline(most_popular_song[attr], color=most_popular_color, linestyle='dashed', linewidth=2, label='Most Popular')
    axes[i].axvline(least_popular_song[attr], color=least_popular_color, linestyle='dashed', linewidth=2, label='Least Popular')
    axes[i].axvline(avg_song[attr], color=average_color, linestyle='dashed', linewidth=2, label='Average')
    axes[i].set_title(f'Histogram of {attr}', color='white')
    axes[i].legend(loc='upper right', framealpha=0.5)
    axes[i].tick_params(axis='x', colors='white')
    axes[i].tick_params(axis='y', colors='white')
    axes[i].spines['bottom'].set_color('white')
    axes[i].spines['top'].set_color('white')
    axes[i].spines['right'].set_color('white')
    axes[i].spines['left'].set_color('white')
    axes[i].grid(False)

plt.tight_layout()
plt.show()
