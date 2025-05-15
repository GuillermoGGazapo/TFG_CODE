#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%%

df = pd.read_excel("BBDD_FINAL.xlsx")  
df.replace(",", ".", regex=True, inplace=True) 
df.iloc[:, 1:] = df.iloc[:, 1:].astype(float) 
#%%
print("Estadísticas descriptivas:\n", df.describe())
#%%

plt.figure(figsize=(8, 5))
sns.histplot(df["Close"], bins=20, kde=True)
plt.title("Distribución de los Precios de Cierre")
plt.xlabel("Precio de Cierre")
plt.ylabel("Frecuencia")
plt.show()
#%%

plt.figure(figsize=(10, 6))
sns.boxplot(data=df[["Close", "Open", "High", "Low"]])
plt.title("Distribución de Precios (Boxplot)")
plt.ylabel("Valor")
plt.show()
#%%

plt.figure(figsize=(8, 5))
sns.scatterplot(x=df["Spot Price"], y=df["Close"])
plt.title("Relación entre Precio Spot y Precio de Cierre")
plt.xlabel("Spot Price")
plt.ylabel("Close Price")
plt.show()
#%%

df["Exchange Date Prime"] = pd.to_datetime(df["Exchange Date Prime"], format="%d-%b-%Y", errors='coerce')  
df = df.sort_values("Exchange Date Prime")

plt.figure(figsize=(10, 5))
sns.lineplot(x=df["Exchange Date Prime"], y=df["Close"], marker="o")
plt.title("Evolución del Precio de Cierre")
plt.xlabel("Fecha")
plt.ylabel("Precio de Cierre")
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlaciones entre Variables")
plt.show()
# %%
