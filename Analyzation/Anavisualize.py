import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = sns.load_dataset("iris")

df.to_csv("iris.csv", index=False)

print("iris.csv has been saved!")

print("First 5 rows:")
print(df.head())

print("Dataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

df_clean = df.dropna() 

print("\nAfter cleaning:")
print(df_clean.info())

print(df.describe())

grouped = df.groupby('species').agg({'sepal_length': 'mean', 'sepal_width': 'mean' , 'petal_width':'mean' , 'petal_length':'mean'})
print(grouped)

# # Create a line plot
# sns.lineplot(x="species", y="sepal_length", data=df)
# plt.title("Average sepal Length by Species")
# plt.xlabel("Species")
# plt.ylabel("sepal Length")
# plt.show()

# # Bar Chart
# sns.barplot(x="species", y="petal_length", data=df)
# plt.title("Average Petal Length by Species")
# plt.xlabel("Species")
# plt.ylabel("Petal Length")
# plt.show()

# # Histogram
# sns.histplot('species', bins=1, color='purple')
# plt.title("sepal-width distribution")
# plt.xlabel("species")
# plt.ylabel("sepal_width Frequency")
# plt.show()

sns.scatterplot(x="species", y="petal_width", data=df)
plt.title("Species vs Petal Width")
plt.show()