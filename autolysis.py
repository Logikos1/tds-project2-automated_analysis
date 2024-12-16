# /// script
# requires-python = ">=3.11"  # Ensure Python version is 3.11 or higher
# dependencies = [
#   "matplotlib",     # For plotting graphs
#   "seaborn",        # For statistical data visualization
#   "pandas",         # For data manipulation and analysis
#   "scipy",          # For scientific computing (stats, statistical tests, etc.)
#   "scikit-learn",   # For machine learning tasks (SimpleImputer, LabelEncoder)
#   "Pillow",         # For image processing (PIL module)
#   "requests",       # For making HTTP requests
#   "numpy",          # For numerical operations
#   "tenacity",       # For retry logic
# ]
# ///

import os
import sys
import pandas as pd
import requests
import json
from scipy import stats
from scipy.stats import f_oneway
from PIL import Image
from io import BytesIO
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('Agg')  # Set the backend to Agg for non-interactive environments
import seaborn as sns
import matplotlib.patches as mpatches
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import numpy as np
import ast
from tenacity import retry, wait_exponential, stop_after_attempt


# Setup code: Retrieve the AIPROXY token and construct the OpenAI API endpoint URL
BASE_URL =  "https://aiproxy.sanand.workers.dev/openai/"

# The OpenAI API endpoint for chat-based completions
URL = BASE_URL + "v1/chat/completions"

# Get the AIPROXY token
AIPROXY_TOKEN = os.getenv('AIPROXY_TOKEN')

# Ensure the token is stripped of any leading/trailing whitespace or newline characters
AIPROXY_TOKEN = AIPROXY_TOKEN.strip() if AIPROXY_TOKEN else None


if not AIPROXY_TOKEN:
    raise ValueError("AIPROXY_TOKEN is not set.")

#End of set-up code



# Retry decorator to handle transient errors using tenacity
@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(5))
def generate_readme(story, charts):
    """
    Generates a README file content by combining a given story and a list of charts.

    This function takes a descriptive story and a list of charts (each chart represented 
    as a tuple containing the chart title and the chart file path or URL) and returns a 
    formatted string suitable for a README file. The story is included as the main content, 
    followed by a section for data visualizations, where each chart is presented with a title 
    and an embedded image.

    Parameters:
    - story (str): A narrative or explanation that serves as the main content of the README.
    - charts (list of tuples): A list of charts where each chart is represented as a tuple.
      The first element of the tuple is the chart title (str), and the second element is the
      chart's file path or URL (str) to the chart image.

    Returns:
    - str: A string formatted as a README file, containing the story and visualizations.
    
    Example:
    --------
    story = "This project involves data analysis of XYZ."
    charts = [("Chart 1", "charts/chart1.png"), ("Chart 2", "charts/chart2.png")]
    
    readme = generate_readme(story, charts)
    print(readme)
    """
    readme = ""
    readme += story

    #Include Visuals
    readme += "\n# Data Visualizations\n"
    for chart in charts:
        readme += f"### **{chart[0]}**\n\n"
        readme += f"![{chart[1]}]({chart[1]})\n\n"

    return readme


def generate_story(AIPROXY_TOKEN,
                    URL,
                    dataset_filename,
                    dataset_description,
                    key_column_exploration_result,
                    dataset_analysis_result) :

    """
    Generates a markdown story from a dataset analysis, leveraging a language model (LLM).

    This function interacts with an AI API (such as OpenAI's GPT) to generate a cohesive and structured story 
    about a dataset, based on the dataset's description, key column exploration results, and analysis results. 
    The generated story is formatted as a markdown file suitable for inclusion in a README. It narrates the dataset 
    analysis in a story-like format with sections dedicated to the dataset overview, analysis methods, key insights, 
    and actionable recommendations.

    Parameters:
    - AIPROXY_TOKEN (str): The authentication token for API access.
    - URL (str): The endpoint URL for the API request.
    - dataset_filename (str): The name of the dataset file being analyzed.
    - dataset_description (str): A description of the dataset that provides context.
    - key_column_exploration_result (str): Results from the exploration of the key column in the dataset.
    - dataset_analysis_result (str): Results from any additional dataset analysis, including statistical tests.

    Returns:
    - str: A markdown formatted string that tells a story about the dataset, its analysis, and insights.

    Example:
    --------
    AIPROXY_TOKEN = "your-token-here"
    URL = "https://api.example.com"
    dataset_filename = "data.csv"
    dataset_description = "This dataset contains information about sales and customer data."
    key_column_exploration_result = "The key column, 'Product Category', shows significant correlation with sales."
    dataset_analysis_result = "The 1-way ANOVA results show that 'Product Category' significantly affects sales."

    markdown_story = generate_story(AIPROXY_TOKEN, URL, dataset_filename, dataset_description, 
                                    key_column_exploration_result, dataset_analysis_result)

    print(markdown_story)
    
    Notes:
    ------
    - The AI model will generate a story in a structured, narrative format divided into chapters. 
      These chapters will cover the dataset's overview, analysis methods, key insights, and actionable recommendations.
    - The function sends a POST request to the API with the necessary parameters, including the dataset information 
      and analysis results, to generate the markdown content.
    - Charts and figures (Figure 1, Figure 2) are referenced in the generated story but will not be included 
      in the markdown content directly; they will be inserted later by an automated script.
    - The generated markdown content will be approximately 1500 tokens long and should be concise, actionable, 
      and easy to understand.
    """
        
    # Set the headers with the token
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }

    # Set up the system behavior and user prompt for the AI interaction.
    # 'behaviour' defines the role of the assistant as a data analyst.
    # 'prompt' contains the task and rules for selecting the most impactful column in the dataset.
    # The 'messages' list is then constructed with the system's instructions and the user input prompt.

    behaviour = """You are a data analyst who produces intriguing story-like narratives about datasets whereby the narratives have sequential flow and cohesion.
    Your task is to produce a story that highlights key insights and patterns discovered from the dataset summaries, analysis results and charts provided to you."""

    prompt = f"""A dataset has been analyzed to uncover patterns, key insights, and actionable recommendations. Based on this analysis, generate a **markdown** file that narrates a **cohesive and engaging story** about the dataset while maintaining a professional and subtle tone. Adapt your narrative based on the dataset's complexity and the insights discovered. Your task involves:

    ---

    ### **1. Dataset Overview**
    - **Section Title (Centered):** "Chapter One: The Beginning"
    - **Section Sub-Title (Centered):** "Mysterious Mr. Dataset"
    - Describe the dataset's structure, size, and unique characteristics. 
    - Summarize key features using **bullet points** for clarity.
    - Focus on making the dataset’s description intriguing but concise.

    ### **2. Analysis Methods**
    - **Section Title (Centered):** "Chapter Two: The Plot Thickens"
    - **Section Sub-Title (Centered):** "Detective Mr. Analyst"
    - Outline the steps taken to explore and analyze the dataset. 
    - Clearly explain the choice of methods (e.g., correlation, ANOVA) for both **numerical** and **categorical** data.
    - Highlight preprocessing steps and their importance for accurate analysis.

    ### **3. Key Insights and Patterns**
    - **Section Title (Centered):** "Chapter Three: The Revelation"
    - **Section Sub-Title (Centered):** "Omnipotent Patterns"
    - Narrate the most significant patterns and trends revealed by the analysis.
    - Present **bullet points** of the key findings and reference visualizations as necessary (e.g., "See Figure 1").
    - Make logical connections between insights and dataset features.

    ### **4. Implications and Actions**
    - **Section Title (Centered):** "Chapter Four: The Deed That Must Be Done"
    - **Section Sub-Title (Centered):** "The Act"
    - Propose **actionable recommendations** based on the insights derived.
    - Where insights are limited, suggest further analysis or questions for exploration.
    - Ensure recommendations are concise, impactful, and relevant.

    ---

    ### **Tone and Style Guidelines**
    1. Use a **serious and analytical tone** with a subtle sense of intrigue.
    2. Maintain **logical flow and cohesion** throughout the story.
    3. The narrative should adapt to the dataset’s complexity:
    - For rich datasets: Provide detailed insights and specific recommendations.
    - For sparse datasets: Focus on general observations and suggest follow-up analyses.
    4. Include clear headers, subheaders, and **bullet points** for clarity and organization.
    5. Avoid over-reliance on predefined storytelling templates and focus on tailoring the story to the dataset provided.

    ---

    ### **Dataset Information**
    - **Dataset filename:** `{dataset_filename}`
    - **Dataset description:** `{dataset_description}`
    - **Key column exploration result:** `{key_column_exploration_result}`
    - **Dataset analysis result:** `{dataset_analysis_result}`
    - **Key column exploration chart**: \\ Figure 1\\
    - **Dataset analysis chart**: \\ Figure 2 \\
    ---

    ### **Critical Final Instructions**
    1. Your response must be **markdown-formatted content only**; no additional explanations or placeholders.
    2. The markdown must be well-structured and concise (~1500 tokens).
    3. Charts will be referenced as **Figure 1** or **Figure 2** but will not include links. The automated script will add these.
    4. Ensure the narrative is professional and subtle, avoiding dramatic language.

    ---

    **Remember**: Your task is to narrate an insightful and engaging story about the dataset, seamlessly linking its description, analysis, and actionable recommendations.
    """

    messages = [
        {"role": "system", "content": behaviour},
        {"role": "user", "content": prompt }
    ]


    # Data for the request, including the 'messages' parameter
    data = {
        "model": "gpt-4o-mini",  # Can also use gpt-3.5-turbo
        "messages": messages,
        "max_tokens": 3000,  # Optional parameter to limit token usage
        "temperature" : 1.0
    }

    # Send the POST request to the OpenAI API via AIPROXY
    response = requests.post(URL, headers=headers, data=json.dumps(data))

    # Check if the request was successful and return the result
    if response.status_code == 200:
        result = response.json()
        return result['choices'][0]['message']['content']
    else:
        return None




def resize_chart_for_llm(fig, new_size=(512, 512)):
    """
    Resize the saved chart figure and return it as a BytesIO object for sending to the LLM for saving tokens.

    Parameters:
    fig (matplotlib.figure.Figure): The matplotlib figure object to resize.
    new_size (tuple): Desired size of the chart (default is 512x512).

    Returns:
    BytesIO object containing the resized image.
    """
    try:
        # Save the figure to a BytesIO buffer
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=200)  # Save with higher dpi for better quality
        buf.seek(0)  # Reset buffer to the start

        # Open the image from the buffer using Pillow
        img = Image.open(buf)

        # Resize the image to the desired size in memory (use LANCZOS for high-quality downsampling)
        img_resized = img.resize(new_size, Image.Resampling.LANCZOS)

        # Save the resized image to a new BytesIO buffer
        resized_buf = BytesIO()
        img_resized.save(resized_buf, format="PNG")
        resized_buf.seek(0)  # Reset the buffer to the beginning

        return resized_buf
    except Exception as e:
        print(f"Error while resizing the image: {e}")
        raise


def plot_anova_result(dataset_analysis_result, key_column):
    """
    This function takes the ANOVA analysis result and plots a horizontal bar plot of the F-statistics for factors
    categorized as significant and non-significant. It also adjusts the plot title to reflect the key variable used
    in the analysis.

    Parameters:
    - dataset_analysis_result: A dictionary containing 'Significant Factors' and 'Non-significant Factors',
      with F-statistic and p-value for each factor.
    - key_column: The categorical variable used for performing ANOVA.
    """
    # Flatten the ANOVA results for plotting
    plot_data = []
    for column, stats in {**dataset_analysis_result['1-Way ANOVA Analysis Result']['Significant Columns'], **dataset_analysis_result['1-Way ANOVA Analysis Result']['Non-significant Columns']}.items():
        plot_data.append({
            'Column': column,
            'F-statistic': round(stats[0], 2),  # F-statistic
            'p-value': round(stats[1], 2),      # p-value
            'Significance': 'Significant' if stats[1] <= 0.05 else 'Non-significant'
        })

    # Create a DataFrame for plotting
    df = pd.DataFrame(plot_data)

    # Dynamically adjust figure size based on the number of factors (categories) and significance categories
    num_significance = len(df['Significance'].unique())  # Number of unique significance categories
    height = min(2 + len(df['Column'].unique()) * 0.5, 12)  # Adjust height for more factors
    figsize = (12, max(3, height))  # Set the figure size

    # Create the plot
    plt.figure(figsize=figsize)
    
    # Improved color palette for better contrast
    palette = {'Significant': '#1f77b4', 'Non-significant': '#ff7f0e'}

    # Create the bar plot
    ax = sns.barplot(x='F-statistic', y='Column', hue='Significance', data=df, palette=palette, orient='h')

    # Title and labels
    plt.title(f"F-statistics for 1-way ANOVA with '{key_column}' selected as Grouping Column", fontsize=16)
    plt.xlabel(f"F-statistic Values for Columns (w.r.t. '{key_column}' column)", fontsize=12)
    plt.ylabel('Columns', fontsize=12)

    # Ensure the legend always shows both 'Significant' and 'Non-significant'
    handles, labels = ax.get_legend_handles_labels()

    # Manually add a legend handle for 'Non-significant' if it's missing
    if 'Non-significant' not in labels:
        # Create a uniform Patch for Non-significant with the same color as the bars
        non_significant_patch = mpatches.Patch(color='#ff7f0e', label='Non-significant')
        handles.append(non_significant_patch)
        labels.append('Non-significant')

    # Set the legend with both 'Significant' and 'Non-significant'
    ax.legend(handles, labels, title='Significance', loc='upper right', fontsize=12)

    # Return the current figure (to be displayed or saved later)
    return plt.gcf()  # Return the figure object


def plot_correlation_result(dataset_analysis_result, key_column):
    """
    This function takes the correlation data (dataset_analysis_result) and plots a horizontal bar plot of correlations
    with variables categorized by different categories. It displays the correlation values at the bar tips.

    Parameters:
    - dataset_analysis_result: A dictionary containing categories and their corresponding correlations.
    """
    # Flatten the dictionary for plotting
    plot_data = []
    for category, correlations in dataset_analysis_result["Correlation Analysis Result"].items():
        for column, correlation in correlations.items():
            plot_data.append({'Category': category, 'Column': column, 'Correlation': correlation})

    # Create a DataFrame for plotting
    df = pd.DataFrame(plot_data)

    # Dynamically adjust figure size based on the number of variables and categories
    num_variables = len(df['Column'].unique())
    num_categories = len(df['Category'].unique())

    # Dynamically adjust figure size for horizontal bar chart
    height = min(2 + num_variables * 0.5, 12)  # Cap the height at 12
    width = max(12, num_categories * 1.2)  # Adjust width based on number of categories

    figsize = (width, max(3, height))

    # Create the bar plot
    plt.figure(figsize=figsize)

    # Improved color palette for better contrast
    palette = sns.color_palette("coolwarm", n_colors=num_categories)

    # Create the bar plot
    ax = sns.barplot(x='Correlation', y='Column', hue='Category', data=df, palette=palette, orient='h')

    # Title and labels
    plt.title(f"Density Plot (KDE) - Correlation Trends w.r.t. '{key_column}' column", fontsize=16)
    plt.xlabel('Correlation Coefficient Value', fontsize=12)
    plt.ylabel('Columns', fontsize=12)

    # Annotate each bar with its corresponding correlation value
    for p in ax.patches:
        width = p.get_width()
        ax.text(width + 0.05, p.get_y() + p.get_height() / 2, f'{width:.2f}', ha='left', va='center', color='black', fontsize=12)

    # Move the legend outside to avoid overlap
    plt.legend(loc='lower left', bbox_to_anchor=(1, 1), title='Correlation Categories', fontsize=12)

    # Move figtext outside the plot area
    plt.figtext(0.05, 0.95, f"Mean: {dataset_analysis_result['Additional Statistics']['Mean Correlation']} | "
                             f"Std Dev: {dataset_analysis_result['Additional Statistics']['Standard Deviation']} | "
                             f"Skewness: {dataset_analysis_result['Additional Statistics']['Skewness']} | "
                             f"Kurtosis: {dataset_analysis_result['Additional Statistics']['Kurtosis']}",
                fontsize=10, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7))

    # Automatically adjust the layout to avoid clipping
    plt.tight_layout()

    # Return the current figure object (to be displayed or saved later)
    return plt.gcf()





def perform_anova(data, key_column, p_value_threshold=0.05):
    """
    Perform ANOVA on the given dataset for all numeric columns, grouped by a key categorical variable.

    Parameters:
    - data: Pandas DataFrame containing the dataset.
    - key_column: Categorical column to group data by.
    - p_value_threshold: Threshold to classify factors as significant or non-significant.

    Returns:
    - Dictionary containing significant and non-significant factors.
    """
    anova_results = {}

    # Check that the key_column is in the data
    if key_column not in data.columns:
        raise ValueError(f"'{key_column}' is not a column in the DataFrame.")

    # Automatically get numeric columns
    numeric_columns = data.select_dtypes(include=['number']).columns.tolist()

    if len(numeric_columns) == 0:
        raise ValueError("No numeric columns found in the dataset.")

    # Perform ANOVA for each numeric column
    for col in numeric_columns:
        # Group data by the key variable
        groups = [group[col].dropna() for _, group in data.groupby(key_column)]

        # Filter out groups with fewer than two values
        groups = [group for group in groups if len(group) > 1]

        # Apply ANOVA only if we have at least two groups
        if len(groups) > 1:
            f_stat, p_value = f_oneway(*groups)
            anova_results[col] = [round(f_stat, 2), round(p_value, 6)]  # Round for readability
        else:
            anova_results[col] = ["Error", "Not enough data"]

    # Process the ANOVA results to classify as significant or non-significant
    dataset_analysis_result = {
        'Dataset Analysis Technique' : '1-Way ANOVA',
        'Correlation w.r.t. Key Column' : key_column,
        '1-Way ANOVA Analysis Result' : { 'Significant Columns': {},
        'Non-significant Columns': {} },
        'Dataset Analysis Chart Figure' : f"Figure 2 : Horizontal Bar Plot -  1-Way ANOVA Analysis w.r.t. '{key_column}' column",
        'Dataset Analysis Chart Title' : f"1-Way ANOVA Analysis w.r.t. '{key_column}' column",
        'Dataset Analysis Chart Filename' : "dataset_analysis_chart.png",
        'Plot Type of Dataset Analyis Chart': 'Horizontal Bar Plot'
    }


    for col, result in anova_results.items():
        if result == ['Error', 'Not enough data']:
            dataset_analysis_result ['1-Way ANOVA Analysis Result']['Non-significant Columns'][col] = result
        else:
            f_stat, p_value = result
            if p_value <= p_value_threshold:
                dataset_analysis_result ['1-Way ANOVA Analysis Result']['Significant Columns'][col] = [f_stat, p_value]
            else:
                dataset_analysis_result ['1-Way ANOVA Analysis Result']['Non-significant Columns'][col] = [f_stat, p_value]

    return dataset_analysis_result


def perform_correlation(df, key_column):
    """
    Performs correlation analysis on a DataFrame with respect to a specified key column.

    This function calculates the correlation between the specified `key_column` and all other numeric columns in the dataset. It classifies the correlations into five categories based on predefined thresholds:
    - Strong Positive
    - Moderate Positive
    - Weak Correlations
    - Moderate Negative
    - Strong Negative

    Additionally, the function computes and returns a set of descriptive statistics for the correlations (excluding the key column itself), such as mean, standard deviation, median, range, skewness, and kurtosis.

    Parameters:
    ----------
    df : pandas.DataFrame
        The input DataFrame containing numeric columns for correlation analysis.

    key_column : str
        The name of the column to analyze the correlation with. This column must be numeric.

    Returns:
    -------
    dict
        A dictionary containing the following keys:
        - 'Dataset Analysis Technique': The analysis technique used (i.e., "Correlation Analysis").
        - 'Correlation w.r.t. Key Column': The name of the key column.
        - 'Correlation Analysis Result': A dictionary containing the categorized correlations for each variable.
        - 'Dataset Analysis Chart Figure': A placeholder figure name (e.g., "Figure 2").
        - 'Dataset Analysis Chart Title': The title for the plot generated for correlation analysis.
        - 'Dataset Analysis Chart Filename': The filename for saving the plot (e.g., 'dataset_analysis_chart.png').
        - 'Plot Type of Dataset Analysis Chart': The type of plot generated (i.e., 'Horizontal Bar Plot').
        - 'Additional Statistics': A dictionary containing descriptive statistics for the correlations, including:
            - Mean Correlation
            - Standard Deviation
            - Median Correlation
            - Range of Correlations
            - Skewness
            - Kurtosis

    Notes:
    ------
    - The correlation is computed using Pearson’s method.
    - The correlations are classified into categories based on the following thresholds:
        - Strong Positive: ≥ 0.75
        - Moderate Positive: ≥ 0.50
        - Moderate Negative: ≤ -0.50
        - Strong Negative: ≤ -0.75
        - Weak Correlations: Between -0.50 and 0.50
    - If no correlations are available (i.e., no other numeric columns are present), an error message will be returned.

    Example:
    --------
    result = perform_correlation(df, 'target_column')
    print(result['Correlation Analysis Result'])
    """
    # Ensure the key_column and other columns are numeric
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    if key_column not in numeric_columns:
        raise ValueError(f"The key_column '{key_column}' must be a numeric column.")

    # Compute correlation of the key variable with all numeric variables
    correlations = df[numeric_columns].corr()[key_column].sort_values(ascending=False)

    # Initialize the structure for significant correlations
    significant_correlations = {
        "Strong Positive": {}, "Moderate Positive": {}, "Weak Correlations": {},
        "Moderate Negative": {}, "Strong Negative": {}
    }

    # Define thresholds for classification
    thresholds = {
        "strong_positive": 0.75, "moderate_positive": 0.50,
        "moderate_negative": -0.50, "strong_negative": -0.75
    }

    # Classify correlations
    for variable, correlation in correlations.items():
        if variable == key_column:
            continue
        if correlation >= thresholds["strong_positive"]:
            significant_correlations["Strong Positive"][variable] = round(correlation, 2)
        elif correlation >= thresholds["moderate_positive"]:
            significant_correlations["Moderate Positive"][variable] = round(correlation, 2)
        elif correlation <= thresholds["strong_negative"]:
            significant_correlations["Strong Negative"][variable] = round(correlation, 2)
        elif correlation <= thresholds["moderate_negative"]:
            significant_correlations["Moderate Negative"][variable] = round(correlation, 2)
        else:
            significant_correlations["Weak Correlations"][variable] = round(correlation, 2)

    # Exclude the key variable itself for descriptive statistics
    correlations_without_key = correlations.drop(key_column, errors='ignore')
    if correlations_without_key.empty:
        return {"Error": "No correlations available for analysis."}

    # Compute descriptive statistics
    mean_corr = correlations_without_key.mean()
    std_corr = correlations_without_key.std()
    median_corr = correlations_without_key.median()
    range_corr = correlations_without_key.max() - correlations_without_key.min()
    skew_corr = ((correlations_without_key - mean_corr) ** 3).mean() / (std_corr**3)
    kurt_corr = ((correlations_without_key - mean_corr) ** 4).mean() / (std_corr**4)

    # Add the plot type and statistics to the result
    dataset_analysis_result = {
        'Dataset Analysis Technique': 'Correlation Analysis',
        'Correlation w.r.t. Key Column': key_column,
        'Correlation Analysis Result': significant_correlations,
        'Dataset Analysis Chart Figure' : f"Figure 2 : Horizontal Bar Plot - Correlation Analysis w.r.t. '{key_column}' column",
        'Dataset Analysis Chart Title': f"Correlation Analysis w.r.t. '{key_column}' column",
        'Dataset Analysis Chart Filename': "dataset_analysis_chart.png",
        'Plot Type of Dataset Analysis Chart': 'Horizontal Bar Plot',
        'Additional Statistics': {
            'Mean Correlation': round(mean_corr, 2),
            'Standard Deviation': round(std_corr, 2),
            'Median Correlation': round(median_corr, 2),
            'Range of Correlations': round(range_corr, 2),
            'Skewness': round(skew_corr, 2),
            'Kurtosis': round(kurt_corr, 2)
        }
    }

    return dataset_analysis_result


def clean_data_for_analysis(df, key_column_type, key_column):
    """
    Clean and preprocess the dataset based on the key variable type.
    Includes filtering useful categorical columns for correlation analysis.

    Parameters:
    - df (pd.DataFrame): Input dataset.
    - key_column (str): Column to treat as the key variable.
    - key_column_type (str): Type of key variable ('numerical' or 'categorical').

    Returns:
    - df_preprocessed (pd.DataFrame): Cleaned DataFrame ready for analysis.
    """
    df_preprocessed = df.copy()

    # 1. Handle Missing Values
    df_preprocessed = handle_missing_values(df_preprocessed)

    # 2. Handle Outliers (optional)
    df_preprocessed = handle_outliers(df_preprocessed)


    return df_preprocessed


def handle_missing_values(df):
    """
    Impute or remove missing values in the dataset.
    Numerical columns are imputed with mean, categorical columns with 'Unknown' or left as NaN.

    Parameters:
    - df (pd.DataFrame): Input dataframe with potential missing values.

    Returns:
    - df (pd.DataFrame): Dataframe with missing values handled.
    """
    # Impute missing numerical values with the mean
    numerical_cols = df.select_dtypes(include=['number']).columns
    imputer_num = SimpleImputer(strategy='mean')
    df[numerical_cols] = imputer_num.fit_transform(df[numerical_cols])

    # Handle categorical columns with missing values
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown')  # Or use NaN if preferred

    return df


def handle_outliers(df, z_threshold=3):
    """
    Remove or cap outliers in the numerical columns using Z-score method.

    Parameters:
    - df (pd.DataFrame): Input dataframe.
    - z_threshold (float): Z-score threshold above which values are considered outliers.

    Returns:
    - df (pd.DataFrame): Dataframe with outliers removed or capped.
    """

    numerical_cols = df.select_dtypes(include=['number']).columns

    # Calculate Z-scores for numerical columns
    z_scores = np.abs(stats.zscore(df[numerical_cols]))

    # Remove rows with Z-scores greater than threshold
    df_cleaned = df[(z_scores < z_threshold).all(axis=1)]

    return df_cleaned


def key_column_exploration_result_and_plot(df, col_type, col_name):
    """
    Analyzes and visualizes a specified column from the dataframe based on its type (numerical or categorical).

    This function generates a plot and statistical summary for a given column:
    - If the column is **numerical**, it generates a **Density Plot (KDE)** to visualize the distribution of values.
    - If the column is **categorical**, it generates a **Frequency Count Bar Chart** to visualize the distribution of categories.

    The function also returns a dictionary containing statistical insights and the plot. The statistics differ based on the column type.

    Parameters:
    ----------
    df (pd.DataFrame): The input dataframe containing the data.
    col_type (str): The type of the column to analyze ('numerical' or 'categorical').
    col_name (str): The name of the column to analyze.

    Returns:
    -------
    dict: A dictionary with the following keys:
        - 'Key Column Name': The name of the analyzed column.
        - 'Key Column Type': The type of the column ('Numerical' or 'Categorical').
        - 'Key Column Exploration Chart Figure': A placeholder figure name (e.g., "Figure 1").
        - 'Key Column Exploration Chart Title': The title for the plot.
        - 'Key Column Exploration Chart Filename': The filename for saving the plot (e.g., 'key_column_exploration_chart.png').
        - 'plot_type': The type of plot generated ('Density Plot (KDE)' or 'Frequency Count Bar Chart').
        - Statistics:
            - For numerical columns: 'mean', 'median', 'std_dev', 'skewness', 'kurtosis', 'min_value', 'max_value', 'quantiles', 'normality_test'.
            - For categorical columns: 'unique_values', 'value_counts', 'mode', 'missing_values', 'missing_percentage'.
    plt.Figure: The generated plot figure object (from `plt.gcf()`), which can be saved or displayed.

    Notes:
    ------
    - If the column type is unknown, the function prints an error message and returns `None`.
    - The function assumes that the provided `col_name` exists in the dataframe.
    - The plot is either a **Density Plot** for numerical columns or a **Bar Chart** for categorical columns.
    - Missing values are handled in categorical columns by providing the count and percentage of missing data.

    """
    # Check if the column exists in the dataframe
    if col_name not in df.columns:
        print(f"Column '{col_name}' not found in the dataframe.")
        return

    # Extract the column data
    column_data = df[col_name]
    result = {}

    # Check if the column is numerical
    if col_type == 'numerical':
        # Plot density plot for numerical data (KDE)
        plt.figure(figsize=(10, 6))
        sns.kdeplot(column_data, fill=True, color='blue', alpha=0.8)  # Updated to use 'fill' instead of 'shade'
        plt.title(f"Density Plot - Distribution of '{col_name}' Column Values")
        plt.xlabel(f"Values in '{col_name}' column")
        plt.ylabel("Density")
        plt.grid(True)

        # Generate result for numerical column
        result['Key Column Name'] = col_name
        result['Key Column Type'] = 'Numerical'
        # Key Column Exploration Chart
        result['Key Column Exploration Chart Figure'] = f"Figure 1 : Density Plot - Distribution of '{col_name}' Column Values"
        result['Key Column Exploration Chart Title'] = f"Density Plot (KDE) - Distribution of '{col_name}' Column Values"
        result['Key Column Exploration Chart Filename'] = 'key_column_exploration_chart.png'
        result['plot_type'] = 'Density Plot (KDE)'
        # Statistics
        result['mean'] = round(column_data.mean(), 2)
        result['median'] = round(column_data.median(), 2)
        result['std_dev'] = round(column_data.std(), 3)
        result['skewness'] = round(column_data.skew(), 3)
        result['kurtosis'] = round(column_data.kurtosis(), 3)
        result['min_value'] = round(column_data.min(), 3)
        result['max_value'] = round(column_data.max(), 3)
        result['quantiles'] = column_data.quantile([0.25, 0.5, 0.75]).to_dict()
        # Additional statistical test (if needed)
        result['normality_test'] = round(stats.normaltest(column_data.dropna()).pvalue, 5)

        return result, plt.gcf()

    # Check if the column is categorical
    elif col_type == 'categorical':
        # Plot frequency count bar chart for categorical data (Fixing the warning)
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=col_name, hue=col_name, palette='Set2', legend=False)
        plt.title(f"Frequency Count of '{col_name}' Column")
        plt.xlabel(f"'{col_name}' Column Values")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45, ha='right')  # Rotate labels if necessary
        plt.grid(True)

        # Generate result for categorical column
        result['Key Column Name'] = col_name
        result['Key Column Type'] = 'Categorical'
        # Key Column Exploration Chart
        result['Key Column Exploration Chart Figure'] = f"Figure 1 : Frequency Count Bar Chart - Frequency Count of '{col_name}' Column"
        result['Key Column Exploration Chart Title'] = f"Frequency Count of '{col_name}' Column"
        result['Key Column Exploration Chart Filename'] = 'key_column_exploration_chart.png'
        result['plot_type'] = 'Frequency Count Bar Chart'
        # Statistics
        result['unique_values'] = column_data.nunique()
        result['value_counts'] = column_data.value_counts().to_dict()
        result['mode'] = column_data.mode().iloc[0]
        result['missing_values'] = column_data.isnull().sum()
        result['missing_percentage'] = column_data.isnull().mean() * 100

        return result, plt.gcf()

    else:
        print(f"Unknown column type '{col_type}' for '{col_name}'.")
        return None, None

# Retry decorator to handle transient errors using tenacity
@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(5))
def select_key_column(AIPROXY_TOKEN, URL, dataset_filename, dataset_description) :
    """
    Selects the most impactful key column from the dataset to analyze, leveraging the LLM (Large Language Model) for insight generation.

    This function sends a request to an LLM-based service, providing the dataset's description, to ask the model to identify
    the column that is most likely to have the greatest impact on the analysis. The selected column will guide further analysis
    and help uncover important patterns or relationships in the data.

    Parameters:
    ----------
    AIPROXY_TOKEN (str): The token used for authentication with the AIPROXY service.
    URL (str): The endpoint URL for sending the request to the API.
    dataset_filename (str): The name of the dataset file (used for reference in the request).
    dataset_description (str): A string containing a summary or description of the dataset, including column types, missing values, etc.

    Returns:
    -------
    str: The name of the most impactful key column selected by the LLM, based on the dataset description.
    None: If the LLM fails to select a key column or there is an error with the request/response.

    Notes:
    ------
    - The function relies on an LLM (Large Language Model) to understand the dataset description and make an informed decision
      about the most important column for analysis.
    - The accuracy of the selection depends on the quality of the description and the LLM's ability to interpret it.
    """

    # Set the headers with the token
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }

    # Set up the system behavior and user prompt for the AI interaction.
    # 'behaviour' defines the role of the assistant as a data analyst.
    # 'prompt' contains the task and rules for selecting the most impactful column in the dataset.
    # The 'messages' list is then constructed with the system's instructions and the user input prompt.

    behaviour = """You are a data analyst.
    Your task is to select the most impactful column in the dataset to analyze, discover patterns, and gain key insights."""

    prompt = f"""
    The dataset filename is {dataset_filename}.
    Here is the dataset summary delimited by ``: `{dataset_description}`.

    Your task : -- Select the most impactful column to discover patterns and gain key insights from the dataset. --

    **Forget the context from any previous prompts before trying to respond to this prompt**
    **Respond ONLY with a dictionary in this format: {{'col_type': 'categorical/numerical', 'col_name': 'name_of_column'}}**
    """

    messages = [
        {"role": "system", "content": behaviour},
        {"role": "user", "content": prompt }
    ]


    # Data for the request, including the 'messages' parameter
    data = {
        "model": "gpt-4o-mini",  # Can also use gpt-3.5-turbo
        "messages": messages,
        "max_tokens": 200,  # Optional parameter to limit token usage
        "temperature" : 0.0
    }

    # Send the POST request to the OpenAI API via AIPROXY
    response = requests.post(URL, headers=headers, data=json.dumps(data))

    # Check if the request was successful and return the result
    if response.status_code == 200:
        result = response.json()
        return result['choices'][0]['message']['content']
    else:
        return None

def fallback_select_key_column(df):
    """
    Selects the most impactful categorical column in a DataFrame based on two criteria:
    - The column with the least unique values.
    - The column with the highest number of missing (null) values.

    The function evaluates each categorical column by combining the number of unique values and missing values. 
    It selects the column that minimizes the sum of these two factors, assuming that the most impactful column
    will have the least complexity (fewer unique values) and will be missing data that could influence analysis.

    Parameters:
    ----------
    df : pandas.DataFrame
        The input DataFrame from which the most impactful categorical column will be selected.

    Returns:
    -------
    dict
        A dictionary containing the type and name of the most impactful categorical column.
        The format is:
        {'col_type': 'categorical', 'col_name': 'name_of_column'}

    Notes:
    ------
    - The function focuses on categorical columns (columns with dtype 'object').
    - The column selection process uses the sum of unique values and missing values as a metric for "impact".
    - If there are no categorical columns in the DataFrame, the function will return `None`.

    Example:
    --------
    df = pd.DataFrame({
        'type': ['A', 'B', 'A', None, 'C'],
        'category': ['X', 'Y', 'Y', 'X', 'Z'],
        'value': [10, 20, 30, 40, 50]
    })
    
    result = fallback_select_key_column(df)
    print(result)
    # Output: {'col_type': 'categorical', 'col_name': 'type'}
    """
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    best_column = None
    best_score = float('inf')  # To track the column with least unique values + missing values

    for col in categorical_columns:
        unique_values = df[col].nunique()
        missing_values = df[col].isnull().sum()
        score = unique_values + missing_values  # Combine both factors to select the most impactful column

        if best_column is None or score < best_score:
            best_column = col
            best_score = score

    return {'col_type': 'categorical', 'col_name': best_column}


def generate_dataset_description(df):
    """
    Generate a description of the dataset, including details about its columns, missing values, and columns with few unique values.

    This function analyzes the provided Pandas DataFrame and generates a textual description summarizing:
    - The number of rows and columns.
    - The numeric and categorical columns.
    - The percentage of missing values for each column.
    - Columns with fewer than a defined threshold of unique values (default 20), which may be useful for categorization or ANOVA analysis.

    Parameters:
    ----------
    df : pandas.DataFrame
        The input dataset for which the description will be generated.

    Returns:
    -------
    str
        A string containing a summary of the dataset's structure, including information about column types,
        missing values, and columns with few unique values.

    Notes:
    ------
    - The threshold for "few unique values" is set to 20 by default, but can be adjusted in the code if necessary.
    - The function assumes that the dataset contains columns with either numeric or object data types.
    """

    # Get columns and their data types
    column_info = df.dtypes
    categorical_cols = column_info[column_info == 'object'].index.tolist()
    numeric_cols = column_info[column_info != 'object'].index.tolist()

    # Get missing values information
    missing_values = df.isna().sum()

    # Create a description string
    dataset_description = f"Dataset has {len(df)} rows and {len(df.columns)} columns."
    dataset_description += f"\nNumeric columns: '{', '.join(numeric_cols)}' "
    dataset_description += f"\nCategorical columns : '{', '.join(categorical_cols)}'"


    # Add missing values information
    missing_info = {}
    for col in df.columns:
        missing_percentage = round(missing_values[col] / len(df) * 100, 0)
        if missing_percentage > 0:
            missing_info[col] = f"{missing_percentage}%"

    # Adding to dataset_description if there is missing info
    if missing_info:
        dataset_description += f"\nPercentage of missing values in columns : {missing_info}."


    # Identify and add columns with few unique values to dataset_description
    # Few unique values are indicative of categorical columns which helps in grouping for ANOVA.
    columns_with_few_unique_values = {}
    threshold = 20  #Cut-off for number of unique values considered as `few` is set at 20

    for column in df.columns:
        unique_values = df[column].nunique()  # Get number of unique values

        if unique_values < threshold:
            columns_with_few_unique_values[column] = df[column].unique().tolist()  # Store column name and unique values

    if columns_with_few_unique_values :
        dataset_description += "\nColumns with less than 20 unique values :" + str(columns_with_few_unique_values)


    return dataset_description


def save_and_resize_charts(key_column_exploration_result, dataset_analysis_result, key_column_exploration_chart, dataset_analysis_chart):
    """Function to save charts and resize them for LLM."""
    # Save charts
    key_column_exploration_chart.savefig("key_column_exploration_chart")
    dataset_analysis_chart.savefig("dataset_analysis_chart", bbox_inches='tight')

    # Resize charts for LLM
    key_column_exploration_chart_for_LLM = resize_chart_for_llm(key_column_exploration_chart)
    dataset_analysis_chart_for_LLM = resize_chart_for_llm(dataset_analysis_chart)

    return key_column_exploration_chart_for_LLM, dataset_analysis_chart_for_LLM


def load_and_validate_dataset():
    """Loads the dataset from a file, validates the filename, and handles encodings."""

    # Check if dataset filename is provided in command-line and load the dataset file
    if len(sys.argv) < 2:
        print("Provide CSV filename")
        return None, None  # Return None if filename is not provided

    dataset_filename = sys.argv[1]

    # Try loading the dataset using different encodings
    encodings_to_try = ['utf-8', 'ISO-8859-1', 'latin1', 'cp1252']
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(dataset_filename, encoding=encoding)
            print(f"Successfully read the file with encoding: {encoding}")
            
            # Check if the dataset is empty
            if df.empty:
                print(f"Warning: The dataset '{dataset_filename}' is empty.")
                return None, dataset_filename  # Return None if dataset is empty

            return df, dataset_filename  # Return the dataframe and filename upon success
        except Exception as e:
            print(f"Error reading {dataset_filename} with encoding {encoding}: {e}")

    # If all attempts fail, print an error message
    print(f"Failed to read {dataset_filename} with multiple encodings.")
    return None, dataset_filename  # Return None if all encodings fail


def main():

    # Load and validate dataset
    df, dataset_filename = load_and_validate_dataset()
    if df is None:
        return


    dataset_description = generate_dataset_description(df)
    print("In main, data descp" , dataset_description)

    key_column_string = str(select_key_column(AIPROXY_TOKEN, URL, dataset_filename, dataset_description))
    print("key_column", key_column_string)

    # Check if the result is None (represented as string "None") or empty
    if key_column_string == "None" or not key_column_string.strip():
        print("No key column selected, falling back to default column.")
        # Use fallback if no valid key column was returned
        key_column = fallback_select_key_column(df)
    else:
        print("key_column_string", key_column_string)
    
        # Convert the string to a dictionary using ast.literal_eval
        try:
            key_column = ast.literal_eval(key_column_string)
        except ValueError as e:
            print(f"Error while parsing key column: {e}")
            key_column = fallback_select_key_column(df)  # fallback if parsing fails
    
    # Print the selected key column
    print("Selected key column:", key_column)


    if key_column['col_type'] == 'numerical' :
        key_column_exploration_result, key_column_exploration_chart = key_column_exploration_result_and_plot(df, key_column['col_type'],key_column['col_name'])
        df_preprocessed = clean_data_for_analysis(df, key_column['col_type'], key_column['col_name'])
        dataset_analysis_result = perform_correlation(df_preprocessed,key_column['col_name'])
        dataset_analysis_chart = plot_correlation_result(dataset_analysis_result,key_column['col_name'])
    else :
        key_column_exploration_result, key_column_exploration_chart = key_column_exploration_result_and_plot(df, key_column['col_type'],key_column['col_name'])
        df_preprocessed = clean_data_for_analysis(df, key_column['col_type'], key_column['col_name'])
        dataset_analysis_result = perform_anova(df,key_column['col_name'])
        dataset_analysis_chart = plot_anova_result(dataset_analysis_result,key_column['col_name'])

    # Generate and save charts for LLM
    key_column_exploration_chart_for_LLM, dataset_analysis_chart_for_LLM = save_and_resize_charts(key_column_exploration_result, dataset_analysis_result, key_column_exploration_chart, dataset_analysis_chart)


    #Call LLM for to get story
    story = generate_story(AIPROXY_TOKEN,
                           URL,
                           dataset_filename,
                           dataset_description,
                           key_column_exploration_result,
                           dataset_analysis_result)


    # Collect Chart Filenames for embedding in README
    charts = []
    chart_fig = key_column_exploration_result['Key Column Exploration Chart Figure']
    chart_name = key_column_exploration_result['Key Column Exploration Chart Filename']
    charts.append([chart_fig, chart_name])
    chart_fig = dataset_analysis_result['Dataset Analysis Chart Figure']
    chart_name = dataset_analysis_result['Dataset Analysis Chart Filename']
    charts.append([chart_fig, chart_name])


    # Combine the story and charts for embedding in README
    readme = generate_readme(story, charts)

    #Create README.md File
    with open("README.md", "w") as f :
        f.write(readme)
    print("README.md created")



    return (dataset_filename,df, dataset_description, key_column,
            dataset_analysis_result, dataset_analysis_chart,
            key_column_exploration_result, key_column_exploration_chart,
            key_column_exploration_chart_for_LLM, dataset_analysis_chart_for_LLM,
            readme, df_preprocessed)

if __name__ == "__main__":
    (dataset_filename,df, dataset_description, key_column,
    dataset_analysis_result, dataset_analysis_chart,
    key_column_exploration_result, key_column_exploration_chart,
    key_column_exploration_chart_for_LLM, dataset_analysis_chart_for_LLM,
    readme, df_preprocessed
    ) = main()
