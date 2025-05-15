#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
#%%
df = pd.read_excel(r"BBDD_FINAL_ESTACIONARIA3.xlsx", parse_dates=True, index_col=0)
df = df.sort_index(ascending=True).copy()

#%%

train_end_date = '2024-08-31'
pred_start_date = '2024-09-01'
pred_end_date   = '2024-12-20'

# Variable objetivo (en log)
target_col = 'Close_log'

# Variables exógenas
feature_cols = [
    'Open','Volume','OI','Bid','Ask',
    'Spot Price','DXY','Inflation','US10-y note Price', 'Open 10y','High 10y',
    'Low 10y', 'Change %', 'OVX Price', 'Open OVX', 'High OVX',
    'Low OVX', 'Change % OVX' 
    
]


df_train = df.loc[:train_end_date].copy()
df_pred  = df.loc[pred_start_date:pred_end_date].copy()

X_train = df_train[feature_cols]
y_train = df_train[target_col]

X_pred = df_pred[feature_cols]



# %%
# Verif Estacionariedad + SARIMAX

result_adf = adfuller(df_train['Close_log_diff'].dropna())
print("ADF Statistic:", result_adf[0])
print("p-value:", result_adf[1])

best_aic = np.inf
best_order = None
best_model = None

p_values = [0, 1, 2]
d_values = [0, 1]
q_values = [0, 1, 2]

for p in p_values:
    for d in d_values:
        for q in q_values:
            try:
                model = SARIMAX(
                    endog=y_train,
                    exog=X_train,
                    order=(p, d, q),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                results = model.fit(disp=False)
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = (p, d, q)
                    best_model = results
            except:
                continue

print("Mejor (p,d,q):", best_order)
print("AIC:", best_aic)

sarimax_model = best_model
print(sarimax_model.summary())

#%% Predecir

start = len(df_train) 
end   = start + len(df_pred) - 1

sarimax_forecast_log = sarimax_model.predict(
    start=start, 
    end=end, 
    exog=X_pred
)
sarimax_forecast_log.index = df_pred.index

sarimax_forecast_close = np.exp(sarimax_forecast_log)

mask_2024 = (df.index >= '2024-01-01') & (df.index <= '2024-12-31')
df_2024 = df.loc[mask_2024]

df_sept_dic = df.loc['2024-09-01':'2024-12-20']

plt.figure(figsize=(10,6))

plt.plot(
    df_2024.index, 
    np.exp(df_2024['Close_log']), 
    color='blue', 
    label='Histórico 2024'
)

plt.plot(
    df_sept_dic.index, 
    np.exp(df_sept_dic['Close_log']), 
    color='orange', 
    linewidth=2,      # un poco más grueso
    label='Real (1-Sep a 20-Dic)'
)

plt.plot(
    sarimax_forecast_close.index, 
    sarimax_forecast_close, 
    color='green', 
    marker='o',
    label='Predicción (SARIMAX)'
)

plt.xlim(pd.to_datetime("2024-01-01"), pd.to_datetime("2024-12-31"))
plt.title("Histórico 2024 y Predicción SARIMAX (Sep-Dic)")
plt.xlabel("Fecha")
plt.ylabel("Precio (escala original)")
plt.legend()
plt.show()
if 'Close_log' in df_pred.columns:
    y_true_close = np.exp(df_pred[target_col])
    y_pred_close = sarimax_forecast_close.reindex(y_true_close.index)

    rmse = np.sqrt(mean_squared_error(y_true_close, y_pred_close))
    mae  = mean_absolute_error(y_true_close, y_pred_close)
    r2   = r2_score(y_true_close, y_pred_close)

    print("RMSE:", rmse)
    print("MAE: ", mae)
    print("R2:  ", r2)


# %%
# XGBOOST
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1]
}

tscv = TimeSeriesSplit(n_splits=3)

xgb_reg = XGBRegressor(random_state=42)

grid_search = GridSearchCV(
    estimator=xgb_reg, 
    param_grid=param_grid,
    cv=tscv,
    scoring='r2',  
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print("Mejores hiperparámetros XGBoost:", grid_search.best_params_)
print("Mejor puntuación en validación:", grid_search.best_score_)

best_xgb_model = grid_search.best_estimator_

# %%
# Predicción en las fechas futuras
xgb_forecast_log = best_xgb_model.predict(X_pred)

xgb_forecast_close = np.exp(xgb_forecast_log)

df_train2 = df.loc[:'2024-08-31'].copy()
df_pred   = df.loc['2024-09-01':'2024-12-20'].copy()

X_train2 = df_train2[feature_cols]
y_train2 = df_train2[target_col]

X_pred = df_pred[feature_cols]
y_pred = df_pred[target_col] 
model_val = SARIMAX(
    endog=y_train2, 
    exog=X_train2, 
    order=best_order
)
results_val = model_val.fit(disp=False)


start_index = len(y_train2) 
end_index   = start_index + len(X_pred) - 1

pred_log = results_val.predict(
    start=start_index, 
    end=end_index,
    exog=X_pred
)
#%%
# Reasignamos las fechas reales a la serie pronosticada
pred_log.index = df_pred.index
pred_close = np.exp(pred_log) 

real_close = np.exp(y_pred)
# Métricas SARIMAX
rmse = np.sqrt(mean_squared_error(real_close, pred_close))
mae  = mean_absolute_error(real_close, pred_close)
r2   = r2_score(real_close, pred_close)

print("SARIMAX - RMSE:", rmse)
print("SARIMAX - MAE: ", mae)
print("SARIMAX - R2:  ", r2)

# Métricas XGBoost
best_xgb_model.fit(X_train2, y_train2)
pred_log_xgb   = best_xgb_model.predict(X_pred)
pred_close_xgb = np.exp(pred_log_xgb)

rmse_xgb = np.sqrt(mean_squared_error(real_close, pred_close_xgb))
mae_xgb  = mean_absolute_error(real_close, pred_close_xgb)
r2_xgb   = r2_score(real_close, pred_close_xgb)

print("XGBoost - RMSE:", rmse_xgb)
print("XGBoost - MAE: ", mae_xgb)
print("XGBoost - R2:  ", r2_xgb)
#%%
# Gráfico pred v Real
plt.figure(figsize=(10,5))
plt.plot(df_pred.index, real_close, label='Real (1-Sep a 20-Dic)', marker='o')
plt.plot(df_pred.index, pred_close, label='SARIMAX Pred', marker='x')
plt.plot(df_pred.index, pred_close_xgb, label='XGBoost Pred', marker='s')
plt.title("Comparación de predicción vs. real (1-Sep a 20-Dic-2024)")
plt.xlabel("Fecha")
plt.ylabel("Precio Futuros (escala original)")
plt.legend()
plt.show()
# Gráficos Residuos SARIMAX prelim
resid_sarimax_log = y_pred - pred_log  
sns.histplot(resid_sarimax_log, kde=True)
plt.title("Histograma de residuos (log) - SARIMAX (Sept-Dic)")
plt.show()

resid_sarimax_original = real_close - pred_close
sns.histplot(resid_sarimax_original, kde=True)
plt.title("Histograma de residuos (original) - SARIMAX (Sept-Dic)")
plt.show()

# %%
resid_sarimax = real_close - pred_close
resid_xgb     = real_close - pred_close_xgb

# RESIDUOS SARIMAX
plt.figure(figsize=(12,6))

plt.subplot(2,2,1)
sns.histplot(resid_sarimax, kde=True, color='blue')
plt.title("Histograma de residuos SARIMAX")

plt.subplot(2,2,2)
plt.plot(resid_sarimax.index, resid_sarimax, marker='o', linestyle='-')
plt.axhline(0, color='red', linestyle='--')
plt.title("Residuos SARIMAX vs. tiempo")

plt.subplot(2,2,3)
plot_acf(resid_sarimax, ax=plt.gca(), lags=20)
plt.title("ACF de residuos SARIMAX")

plt.subplot(2,2,4)
plot_pacf(resid_sarimax, ax=plt.gca(), lags=20, method='ywm')
plt.title("PACF de residuos SARIMAX")

plt.tight_layout()
plt.show()


# RESIDUOS XGBoost
plt.figure(figsize=(12,6))

plt.subplot(2,2,1)
sns.histplot(resid_xgb, kde=True, color='green')
plt.title("Histograma de residuos XGBoost")

plt.subplot(2,2,2)
plt.plot(resid_xgb.index, resid_xgb, marker='o', linestyle='-')
plt.axhline(0, color='red', linestyle='--')
plt.title("Residuos XGBoost vs. tiempo")

plt.subplot(2,2,3)
plot_acf(resid_xgb, ax=plt.gca(), lags=20)
plt.title("ACF de residuos XGBoost")

plt.subplot(2,2,4)
plot_pacf(resid_xgb, ax=plt.gca(), lags=20, method='ywm')
plt.title("PACF de residuos XGBoost")

plt.tight_layout()
plt.show()
#%%

# Test ARCH
arch_test_stat, arch_test_pval, _, _ = het_arch(resid_sarimax, nlags=12)
print("SARIMAX: ARCH Test")
print(f"Statistic: {arch_test_stat}, p-value: {arch_test_pval}")
arch_test_stat_xgb, arch_test_pval_xgb, _, _ = het_arch(resid_xgb, nlags=12)
print("XGBoost: ARCH Test")
print(f"Statistic: {arch_test_stat_xgb}, p-value: {arch_test_pval_xgb}")



# %%
# INTERFAZ DEMO 
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


df_forecast = pd.DataFrame({
    'Exchange Date Prime': pd.date_range(start='2024-09-02', periods=80),
    'Price': np.random.uniform(68, 78, 80)
})


def calculate_premium(spot_price, strike_price, volatility=0.2, risk_free_rate=0.01, time_to_maturity=0.25):
    from scipy.stats import norm
    import math

    d1 = (np.log(spot_price / strike_price) + (risk_free_rate + 0.5 * volatility**2) * time_to_maturity) / (volatility * math.sqrt(time_to_maturity))
    d2 = d1 - volatility * math.sqrt(time_to_maturity)
    call_price = spot_price * norm.cdf(d1) - strike_price * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(d2)
    return call_price


def analyze_portfolio(portfolio, current_price, available_options):
    strategy = []
    for strike, amount in portfolio.items():
        if current_price < strike:
            
            strategy.append(f"VENDER: {amount} opciones al strike price {strike}€")
        else:
          
            better_options = [opt for opt in available_options if opt < strike]
            if better_options:
                best_option = max(better_options)
                strategy.append(f"COMPRAR: {amount} opciones al strike price {best_option}€")
            else:
                strategy.append(f"MANTENER: {amount} opciones al strike price {strike}€")
    return strategy


root = tk.Tk()
root.title("FarSight")
plot_frame = ttk.Frame(root)
plot_frame.pack()



def month_selected(event):
    selected_month = month_var.get()
    month_mapping = {'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12}
    filtered_df = df_forecast[df_forecast['Exchange Date Prime'].dt.month == month_mapping[selected_month]]
    month_display.config(text=f"Showing data for {selected_month}\n{filtered_df[['Exchange Date Prime', 'Price']].head()}")


def calculate_premiums():
    selected_month = month_var.get()
    month_mapping = {'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12}
    filtered_df = df_forecast[df_forecast['Exchange Date Prime'].dt.month == month_mapping[selected_month]]
    premiums = {strike: calculate_premium(filtered_df['Price'].mean(), strike) for strike in range(68, 76)}
    premium_display.config(text=f"Premiums: {premiums}")
    
    
    for widget in plot_frame.winfo_children():
        widget.destroy()

    
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(list(premiums.keys()), list(premiums.values()), marker='o', linestyle='--')
    ax.set_title(f"Primas de Opciones del Brent Crude para {selected_month}")
    ax.set_xlabel("Strike Price")
    ax.set_ylabel("Premium (EUR)")
    ax.grid(True)
    
    
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

def get_strategy():
    try:
        portfolio = eval(portfolio_entry.get())
        selected_date = date_entry.get()
        current_price = df_forecast.loc[df_forecast['Exchange Date Prime'] == pd.to_datetime(selected_date), 'Price'].values[0]
        available_options = range(68, 76)
        strategy = analyze_portfolio(portfolio, current_price, available_options)
        strategy_display.config(text="\n".join(strategy))
    except Exception as e:
        messagebox.showerror("Error", f"Fecha o input inválidos: {e}")


month_var = tk.StringVar()
month_label = ttk.Label(root, text="Selecciona el mes:")
month_label.pack()
month_combo = ttk.Combobox(root, textvariable=month_var, values=["Septiembre", "Octubre", "Noviembre", "Diciembre"])
month_combo.bind("<<ComboboxSelected>>", month_selected)
month_combo.pack()

month_display = ttk.Label(root, text="")
month_display.pack()

premium_button = ttk.Button(root, text="Calcular Primas", command=calculate_premiums)
premium_button.pack()

premium_display = ttk.Label(root, text="")
premium_display.pack()

portfolio_label = ttk.Label(root, text="Introduzca las opciones disponibles (formato: {(precio): (nºopciones)}:")
portfolio_label.pack()
portfolio_entry = ttk.Entry(root)
portfolio_entry.pack()

date_label = ttk.Label(root, text="Escribe la fecha (YYYY-MM-DD):")
date_label.pack()
date_entry = ttk.Entry(root)
date_entry.pack()

strategy_button = ttk.Button(root, text="Modelar estrategia de cobertura", command=get_strategy)
strategy_button.pack()

strategy_display = ttk.Label(root, text="")
strategy_display.pack()

root.mainloop()
# %%
