IMDB Movie Reviews
Overview
This Jupyter notebook demonstrates the application of t-SNE (t-Distributed Stochastic Neighbor Embedding) for visualizing high-dimensional text data from the IMDB movie review dataset. The dataset contains 50,000 movie reviews labeled as "positive" or "negative." The notebook walks through the entire process, from data loading and preprocessing to dimensionality reduction using t-SNE and visualization.

Dataset
The dataset used is the IMDB Dataset of 50K Movie Reviews, which includes:

Review Text: The textual content of movie reviews.

Sentiment Labels: Binary labels indicating whether the review is "positive" or "negative."

Dependencies
To run this notebook, ensure you have the following Python libraries installed:

numpy

pandas

matplotlib

seaborn

re

nltk

sklearn.feature_extraction.text.TfidfVectorizer

sklearn.manifold.TSNE

You can install these dependencies using pip:

bash
Copy
pip install numpy pandas matplotlib seaborn nltk scikit-learn
Notebook Structure
Import Libraries: Load necessary Python libraries for data manipulation, text processing, and visualization.

Load the Dataset: Read the IMDB dataset from a CSV file and perform initial exploration.

Basic Exploration: Check for missing values and analyze the distribution of sentiments.

Subset the Data: Sample a balanced subset of the data (1000 positive and 1000 negative reviews) for computational efficiency.

Clean & Preprocess Text:

Convert text to lowercase.

Remove non-alphabetic characters and stopwords.

Tokenize the text.

Convert Text to Numeric Features (TF-IDF): Use TF-IDF vectorization to transform text into a numerical format suitable for t-SNE.

Apply t-SNE: Reduce the high-dimensional TF-IDF vectors to 2D for visualization.

Visualization: Plot the t-SNE results to explore the separation between positive and negative reviews.

Key Steps
Text Cleaning: The notebook includes a function to clean and preprocess the review text, ensuring consistency and removing noise.

Dimensionality Reduction: t-SNE is applied to the TF-IDF vectors to project the data into a 2D space, making it easier to visualize patterns.

Perplexity Analysis: The notebook includes a section to analyze the effect of perplexity on t-SNE's performance, helping to choose an optimal value.

Usage
Run the Notebook: Execute each cell sequentially to follow the workflow.

Modify Parameters: Adjust the perplexity value in the t-SNE step or the sample size to experiment with different settings.

Visualize Results: The final visualization helps identify clusters of similar reviews and potential separations between sentiments.

Results
The t-SNE visualization provides insights into the structure of the dataset:

Clusters of reviews with similar sentiment.

Overlaps or separations between positive and negative reviews.

The impact of perplexity on the visualization quality.

Applications
This notebook can be adapted for:

Sentiment analysis tasks.

Exploratory data analysis (EDA) for text data.

Understanding the effectiveness of t-SNE for text visualization.

License
This project is open-source and available under the MIT License. Feel free to use, modify, and distribute the code as needed.

Acknowledgments
Dataset sourced from Kaggle.

Libraries and tools used: scikit-learn, nltk, pandas, matplotlib.

For questions or contributions, please open an issue or submit a pull request. Happy analyzing!

