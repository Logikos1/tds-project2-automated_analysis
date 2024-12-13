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


# Setup code: Retrieve the AIPROXY token and construct the OpenAI API endpoint URL
BASE_URL =  "https://aiproxy.sanand.workers.dev/openai/"

# The OpenAI API endpoint for chat-based completions
URL = BASE_URL + "v1/chat/completions"

# Get the AIPROXY token
AIPROXY_TOKEN = os.getenv('AIPROXY_TOKEN')
#AIPROXY_TOKEN = 'eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIxZjEwMDY5NTRAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.uowN6hWuO8EJ4H37FC_luSuUnDAOJsKtM0GPa4im2yE'

# Ensure the token is stripped of any leading/trailing whitespace or newline characters
AIPROXY_TOKEN = AIPROXY_TOKEN.strip() if AIPROXY_TOKEN else None



if not AIPROXY_TOKEN:
    raise ValueError("AIPROXY_TOKEN is not set.")

#End of set-up code



# Function to generate the README file
def generate_readme(story, charts):
    readme = story

    #Include Visuals
    readme += "\n## Data Visualizations\n"
    for chart in charts:
        readme += f"### **{chart[0]}**\n\n"
        readme += f"![{chart[0]}]({chart[1]})\n\n"

    return readme

# This function asks the LLM to
def generate_narrative(AIPROXY_TOKEN,
                       URL,
                       dataset_filename ) :

    # Set the headers with the token
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }

    # Set up the system behavior and user prompt for the AI interaction.
    # 'behaviour' defines the role of the assistant as a data analyst.
    # 'prompt' contains the task and rules for selecting the most impactful column in the dataset.
    # The 'messages' list is then constructed with the system's instructions and the user input prompt.

    behaviour = """You are a data analyst who produces story-like reports. Your task is to return a README.md"""

    prompt = f"""

    Your Task : --- Generate a **story** from the data and respond with a well-structured **README.md** ---

    - **Dataset filename**: `{dataset_filename}`
    - Key Column : Overall(Categorical)
    - Correlation detected.

    **Your response should only be the content of the README.md itself.**
    """

    messages = [
        {"role": "system", "content": behaviour},
        {"role": "user", "content": prompt }
    ]


    # Data for the request, including the 'messages' parameter
    data = {
        "model": "gpt-4o-mini",  # Can also use gpt-3.5-turbo
        "messages": messages,
        "max_tokens": 75,  # Optional parameter to limit token usage
        "temperature" : 1.0
    }

    # Send the POST request to the OpenAI API via AIPROXY
    #response = requests.post(URL, headers=headers, data=json.dumps(data))

    # Check if the request was successful and return the result
    #if response.status_code == 200:
    #    result = response.json()
    #    return result['choices'][0]['message']['content']
    #else:
    #    return None

    return {'col_type': 'numerical', 'col_name': 'overall'}


#Function to resize the chart figure for saving tokens, for sending to LLM
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


def plot_anova_result(analysis_result, key_variable):
    """
    This function takes the ANOVA analysis result and plots a horizontal bar plot of the F-statistics for factors
    categorized as significant and non-significant. It also adjusts the plot title to reflect the key variable used
    in the analysis.

    Parameters:
    - analysis_result: A dictionary containing 'Significant Factors' and 'Non-significant Factors',
      with F-statistic and p-value for each factor.
    - key_variable: The categorical variable used for performing ANOVA.
    """

    # Flatten the ANOVA results for plotting
    plot_data = []
    for column, stats in {**analysis_result['Significant Columns'], **analysis_result['Non-significant Columns']}.items():
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
    ax = sns.barplot(x='F-statistic', y='Column', hue='Significance', data=df, palette={'Significant': 'green', 'Non-significant': 'gray'}, orient='h')

    # Title and labels
    plt.title(f"F-statistics for 1-way ANOVA with '{key_variable}' selected as Grouping Column")
    plt.xlabel(f"F-statistic Values for Columns (w.r.t. '{key_variable}' column)")
    plt.ylabel('Columns')

    # Ensure the legend always shows both 'Significant' and 'Non-significant'
    handles, labels = ax.get_legend_handles_labels()

    # Manually add a legend handle for 'Non-significant' if it's missing
    if 'Non-significant' not in labels:
        # Create a uniform Patch for Non-significant with the same color as the bars
        non_significant_patch = mpatches.Patch(color='gray', label='Non-significant')
        handles.append(non_significant_patch)
        labels.append('Non-significant')

    # Set the legend with both 'Significant' and 'Non-significant'
    ax.legend(handles, labels, title='Significance', loc='upper right')

    # Return the current figure (to be displayed or saved later)
    return plt.gcf()  # Return the figure object


# Function to create plot of Correlation analysis result
def plot_correlation_result(analysis_result, key_variable):
    """
    This function takes the correlation data (analysis_result) and plots a horizontal bar plot of correlations
    with variables categorized by different categories. It displays the correlation values at the bar tips.

    Parameters:
    - analysis_result: A dictionary containing categories and their corresponding correlations.
    """

    # Flatten the dictionary for plotting
    plot_data = []
    for category, correlations in analysis_result["Correlation Analysis Result"].items():
        for variable, correlation in correlations.items():
            plot_data.append({'Category': category, 'Variable': variable, 'Correlation': correlation})

    # Create a DataFrame for plotting
    df = pd.DataFrame(plot_data)

    # Dynamically adjust figure size based on the number of variables and categories
    num_variables = len(df['Variable'].unique())
    num_categories = len(df['Category'].unique())

    # Dynamically adjust figure size for horizontal bar chart
    height = min(2 + num_variables * 0.5, 12)  # Cap the height at 12
    width = max(10, num_categories * 1.2)  # Adjust width based on number of categories

    figsize = (width, max(3, height))

    # Create the bar plot
    plt.figure(figsize=figsize)
    ax = sns.barplot(x='Correlation', y='Variable', hue='Category', data=df, palette='coolwarm', orient='h')

    # Title and labels
    plt.title(f"Correlation Trends w.r.t. '{key_variable}' column", fontsize=16)
    plt.xlabel('Correlation Coefficient Value', fontsize=12)
    plt.ylabel('Columns', fontsize=12)

    # Annotate each bar with its corresponding correlation value
    for p in ax.patches:
        width = p.get_width()
        ax.text(width + 0.05, p.get_y() + p.get_height() / 2, f'{width:.2f}', ha='left', va='center', color='black', fontsize=12)

    # Move the legend outside to avoid overlap
    plt.legend(loc='lower left', bbox_to_anchor=(1, 1), title='Correlation Categories', fontsize=10)

    # Move figtext outside the plot area
    plt.figtext(0.05, 0.95, f"Mean: {analysis_result['Additional Statistics']['Mean Correlation']} | "
                             f"Std Dev: {analysis_result['Additional Statistics']['Standard Deviation']} | "
                             f"Skewness: {analysis_result['Additional Statistics']['Skewness']} | "
                             f"Kurtosis: {analysis_result['Additional Statistics']['Kurtosis']}",
                fontsize=10, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7))


    # Automatically adjust the layout to avoid clipping
    plt.tight_layout()

    # Return the current figure object (to be displayed or saved later)
    return plt.gcf()



#Function to perform ANOVA if a categorical column is selected as most impactful column by LLM
def perform_anova(data, key_variable, p_value_threshold=0.05):
    """
    Perform ANOVA on the given dataset for all numeric columns, grouped by a key categorical variable.

    Parameters:
    - data: Pandas DataFrame containing the dataset.
    - key_variable: Categorical column to group data by.
    - p_value_threshold: Threshold to classify factors as significant or non-significant.

    Returns:
    - Dictionary containing significant and non-significant factors.
    """
    anova_results = {}

    # Check that the key_variable is in the data
    if key_variable not in data.columns:
        raise ValueError(f"'{key_variable}' is not a column in the DataFrame.")

    # Automatically get numeric columns
    numeric_columns = data.select_dtypes(include=['number']).columns.tolist()

    if len(numeric_columns) == 0:
        raise ValueError("No numeric columns found in the dataset.")

    # Perform ANOVA for each numeric column
    for col in numeric_columns:
        # Group data by the key variable
        groups = [group[col].dropna() for _, group in data.groupby(key_variable)]

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
        'Correlation w.r.t. Key Column' : key_variable,
        '1-Way ANOVA Analysis Result' : { 'Significant Columns': {},
        'Non-significant Columns': {} },
        'Dataset Analysis Chart Figure' : "Figure 2",
        'Dataset Analysis Chart Title' : f"1-Way ANOVA Analysis w.r.t. '{key_variable}' column",
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

def perform_correlation(data, key_variable):
    # Ensure the key_variable and other columns are numeric
    numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
    if key_variable not in numeric_columns:
        raise ValueError(f"The key_variable '{key_variable}' must be a numeric column.")

    # Compute correlation of the key variable with all numeric variables
    correlations = data[numeric_columns].corr()[key_variable].sort_values(ascending=False)

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
        if variable == key_variable:
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
    correlations_without_key = correlations.drop(key_variable, errors='ignore')
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
        'Correlation w.r.t. Key Column': key_variable,
        'Correlation Analysis Result': significant_correlations,
        'Dataset Analysis Chart Figure' : "Figure 2",
        'Dataset Analysis Chart Title': f"Correlation Analysis w.r.t. '{key_variable}' column",
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



def clean_data_for_analysis(df, key_variable_type, key_variable):
    """
    Clean and preprocess the dataset based on the key variable type.
    Includes filtering useful categorical columns for correlation analysis.

    Parameters:
    - df (pd.DataFrame): Input dataset.
    - key_variable (str): Column to treat as the key variable.
    - key_variable_type (str): Type of key variable ('numerical' or 'categorical').

    Returns:
    - df_preprocessed (pd.DataFrame): Cleaned DataFrame ready for analysis.
    """
    df_preprocessed = df.copy()

    # 1. Handle Missing Values
    df_preprocessed = handle_missing_values(df_preprocessed)

    # 2. Handle Outliers (optional)
    df_preprocessed = handle_outliers(df_preprocessed)

    '''
    # 3. Filter useful categorical columns
    useful_cats = filter_categorical_columns(df_preprocessed)

    # 4. Encode useful categorical columns
    df_preprocessed = encode_categorical_columns(df_preprocessed, exclude_column=key_variable, one_hot=False)

    # 5. Ensure key variable handling
    if key_variable_type == "categorical" and key_variable in useful_cats:
        df_preprocessed[key_variable] = LabelEncoder().fit_transform(df_preprocessed[key_variable].astype(str))
    '''

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
    from scipy import stats
    numerical_cols = df.select_dtypes(include=['number']).columns

    # Calculate Z-scores for numerical columns
    z_scores = np.abs(stats.zscore(df[numerical_cols]))

    # Remove rows with Z-scores greater than threshold
    df_cleaned = df[(z_scores < z_threshold).all(axis=1)]

    return df_cleaned


def plot_key_column_exploration_chart(df, col_type, col_name):
    """
    Function to analyze and plot a column from a dataframe based on the type of the column.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    col_type (str): Type of the column ('numerical' or 'categorical').
    col_name (str): The name of the column to analyze.

    Generates:
    - If the column is numerical: A Density Plot (KDE).
    - If the column is categorical: A Frequency Count Bar Chart.
    - Returns a dictionary with relevant statistics and description.
    """
    # Check if the column exists in the dataframe
    if col_name not in df.columns:
        print(f"Column '{col_name}' not found in the dataframe.")
        return

    # Extract the column data
    column_data = df[col_name]
    description = {}

    # Check if the column is numerical
    if col_type == 'numerical':
        # Plot density plot for numerical data (KDE)
        plt.figure(figsize=(10, 6))
        sns.kdeplot(column_data, fill=True, color='blue', alpha=0.8)  # Updated to use 'fill' instead of 'shade'
        plt.title(f"Density Plot - Distribution of '{col_name}' Column Values")
        plt.xlabel(f"Values in '{col_name}' column")
        plt.ylabel("Density")
        plt.grid(True)

        # Generate description for numerical column
        description['Key Column Name'] = col_name
        description['Key Column Type'] = 'Numerical'
        # Key Column Exploration Chart
        description['Key Column Exploration Chart Figure'] = "Figure 1"
        description['Key Column Exploration Chart Title'] = f"Density Plot - Distribution of '{col_name}' Column Values"
        description['Key Column Exploration Chart Filename'] = 'key_column_exploration_chart.png'
        description['plot_type'] = 'Density Plot (KDE)'
        # Statistics
        description['mean'] = round(column_data.mean(),2)
        description['median'] = round(column_data.median(),2)
        description['std_dev'] = round(column_data.std(),3)
        description['skewness'] = round(column_data.skew(),3)
        description['kurtosis'] = round(column_data.kurtosis(),3)
        description['min_value'] = round(column_data.min(),3)
        description['max_value'] = round(column_data.max(),3)
        description['quantiles'] = column_data.quantile([0.25, 0.5, 0.75]).to_dict()
        # Additional statistical test (if needed)
        description['normality_test'] = round(stats.normaltest(column_data.dropna()).pvalue,5)



        return plt.gcf(), description

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

        # Generate description for categorical column
        description['Key Column Name'] = col_name
        description['Key Column Type'] = 'Categorical'
        # Key Column Exploration Chart
        description['Key Column Exploration Chart Figure'] = "Figure 1"
        description['Key Column Exploration Chart Title'] = f"Frequency Count of '{col_name}' Column"
        description['Key Column Exploration Chart Filename'] = 'key_column_exploration_chart.png'
        description['plot_type'] = 'Frequency Count Bar Chart'
        # Statistics
        description['unique_values'] = column_data.nunique()
        description['value_counts'] = column_data.value_counts().to_dict()
        description['mode'] = column_data.mode().iloc[0]
        description['missing_values'] = column_data.isnull().sum()
        description['missing_percentage'] = column_data.isnull().mean() * 100

        return plt.gcf(), description

    else:
        print(f"Unknown column type '{col_type}' for '{col_name}'.")
        return None, None

# This function asks the LLM to select the most impactful column from the dataset to guide the analysis.
def select_key_column(AIPROXY_TOKEN, URL, dataset_filename, dataset_description) :
    """
    Selects the most impactful key column from the dataset to analyze, discover patterns, and gain insights.

    Parameters:
    - AIPROXY_TOKEN (str): The token used for authentication with the AIPROXY service.
    - URL (str): The endpoint URL for sending the request to the API.
    - dataset_description (str): A string containing the summary or description of the dataset.

    Returns:
    - str: The name of the most impactful key column selected from the dataset.
    - None: If the selection fails or there is an error.
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
        "max_tokens": 25  # Optional parameter to limit token usage
    }

    # Send the POST request to the OpenAI API via AIPROXY
    #response = requests.post(URL, headers=headers, data=json.dumps(data))

    # Check if the request was successful and return the result
    #if response.status_code == 200:
    #    result = response.json()
    #    return result['choices'][0]['message']['content']
    #else:
    #    return None

    return str({'col_type': 'numerical', 'col_name': 'overall'})


# Function to generate the dataset_description with missing values and unique values
def generate_dataset_description(df):

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


#Load the dataset as a Pandas dataframe
def load_dataset(filename):
    encodings_to_try = ['utf-8', 'ISO-8859-1', 'latin1', 'cp1252']

    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(filename, encoding=encoding)
            print(f"Successfully read the file with encoding: {encoding}")
            return df
        except Exception as e:
            print(f"Error reading {filename} with encoding {encoding}: {e}")

    print(f"Failed to read {filename} with multiple encodings.")
    return None


def main():

    #Check if dataset filename is provided in command-line and load the dataset file
    if len(sys.argv) < 2:
        print("Provide CSV filename")
    else:
        dataset_filename = sys.argv[1]
        df = load_dataset(dataset_filename)
        if df is not None:
            print(f'{dataset_filename} df created.')


    dataset_description = generate_dataset_description(df)
    print("In main, data descp" , dataset_description)

    key_column_string = select_key_column(AIPROXY_TOKEN, URL, dataset_filename, dataset_description)
    print("key_column", key_column_string)

    # Convert the string to a dictionary
    key_column = ast.literal_eval(key_column_string)


    if key_column['col_type'] == 'numerical' :
        #%matplotlib agg
        key_column_exploration_chart, key_column_exploration = plot_key_column_exploration_chart(df, key_column['col_type'],key_column['col_name'])
        df_preprocessed = clean_data_for_analysis(df, key_column['col_type'], key_column['col_name'])
        dataset_analysis_result = perform_correlation(df_preprocessed,key_column['col_name'])
        #%matplotlib agg
        dataset_analysis_chart = plot_correlation_result(dataset_analysis_result,key_column['col_name'])
    else :
        #%matplotlib agg
        key_column_exploration_chart,key_column_exploration = plot_key_column_exploration_chart(df, key_column['col_type'],key_column['col_name'])
        df_preprocessed = clean_data_for_analysis(df_preprocessed, key_column['col_type'], key_column['col_name'])
        dataset_analysis_result = perform_anova(df,key_column['col_name'])
        #%matplotlib agg
        dataset_analysis_chart = plot_anova_result(dataset_analysis_result,key_column['col_name'])

    key_column_exploration_chart.savefig("key_column_exploration_chart")
    dataset_analysis_chart.savefig("dataset_analysis_chart",bbox_inches = 'tight' )

    key_column_exploration_chart_for_LLM = resize_chart_for_llm(key_column_exploration_chart)
    dataset_analysis_chart_for_LLM = resize_chart_for_llm(dataset_analysis_chart)

    #Call LLM for to get story
    story = f"""# Media Analysis Report

## Introduction

In this report, we explore a dataset containing various media-related variables that allow us to analyze how different factors correlate with the overall ratings of media items. The primary goal is to uncover relationships that could enhance our understanding and strategy within media production, marketing, and consumption.

## Dataset Overview

- **Filename**: '{dataset_filename}'
"""


    # Collect Chart Filenames for embedding in README
    charts = []
    chart_fig = key_column_exploration['Key Column Exploration Chart Figure']
    chart_name = key_column_exploration['Key Column Exploration Chart Filename']
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
            key_column_exploration_chart, key_column_exploration,
            key_column_exploration_chart_for_LLM, dataset_analysis_chart_for_LLM,
            readme, df_preprocessed)

if __name__ == "__main__":
    (dataset_filename,df, dataset_description, key_column,
    dataset_analysis_result, dataset_analysis_chart,
    key_column_exploration_chart, key_column_exploration,
    key_column_exploration_chart_for_LLM, dataset_analysis_chart_for_LLM,
    readme, df_preprocessed
    ) = main()
