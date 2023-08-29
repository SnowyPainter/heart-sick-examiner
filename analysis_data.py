import form_data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

x, y = form_data.get_df()
x = form_data.get_normalized(x)

def plot_histogram_pairs(dataframe):
    sns.set(style='whitegrid')
    combinations = list(itertools.combinations(dataframe.columns, 2))
    
    for feature1, feature2 in combinations:
        plt.figure(figsize=(12, 6))
        sns.histplot(data=dataframe, x=feature1, kde=True, label=feature1, color='blue')
        sns.histplot(data=dataframe, x=feature2, kde=True, label=feature2, color='orange')
        plt.legend()
        plt.title(f'Histograms of {feature1} and {feature2}')
        plt.xlabel('Feature Value')
        plt.ylabel('Frequency')
        plt.show()
def corr():
    correlation_matrix = x.corr()
    combinations = list(itertools.combinations(correlation_matrix.columns, 2))
    top_corr_combinations = sorted(combinations, key=lambda pair: abs(correlation_matrix.loc[pair[0], pair[1]]), reverse=True)[:5]

    print("Top 5 Correlation Combinations:")
    for feature1, feature2 in top_corr_combinations:
        print(f"{feature1} and {feature2}")

def andrews_curve():
    plt.figure(figsize=(10, 6))
    pd.plotting.andrews_curves(x[['oldpeak', 'slope', 'thalach', 'age', 'exang']], 'slope', colormap='tab10')
    plt.title("Andrews Curves")
    plt.show()
