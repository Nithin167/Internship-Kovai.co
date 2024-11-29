import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from google.colab import drive
drive.mount('/content/drive')
df= pd.read_csv('/content/drive/MyDrive/Daily_Public_Transport_Passenger_Journeys_by_Service_Type_20241129.csv')
print(df)


#TASK 1
#Insight : Analyse the dataset and provide 4 to 5 key insights
# 1.Percentage on each route
print("1) Percentage Contribution of Each Route Type:")
total_local_route = df['Local Route'].sum()
total_light_rail = df['Light Rail'].sum()
total_peak_service = df['Peak Service'].sum()
total_rapid_route = df['Rapid Route'].sum()
total_school = df['School'].sum()
total_other = df['Other'].sum()
total_usage = total_local_route + total_light_rail + total_peak_service + total_rapid_route + total_school + total_other
percentage_contribution = {
    'Local Route': (total_local_route / total_usage) * 100,
    'Light Rail': (total_light_rail / total_usage) * 100,
    'Peak Service': (total_peak_service / total_usage) * 100,
    'Rapid Route': (total_rapid_route / total_usage) * 100,
    'School': (total_school / total_usage) * 100,
    'Other': (total_other / total_usage) * 100
}
for route, percentage in percentage_contribution.items():
    print(f"{route}: {percentage:.2f}%")

# 2. Total Usage by Route Type
route_usage = df[['Local Route', 'Light Rail', 'Rapid Route']].sum()
print("\n2) Total Usage by Route Type:")
print(route_usage)

# 3. School-related Insights
school_routes = df['School'].sum()
print(f"\n3) Total School Route Usage: {school_routes}")

# 4. Visualizing Trends
route_usage.plot(kind='bar', title='4) Total Usage by Route Type', ylabel='Number of Passengers', xlabel='Route Type', figsize=(10, 5))
plt.show()

# 5. Date-wise Analysis for Peak Service
plt.figure(figsize=(10, 5))
plt.plot(df['Peak Service'], marker='o', label='Peak Service Usage')
plt.title('5) Peak Service Usage Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.grid()
plt.legend()
plt.show()


#TASK 2
#Forecast : Forecast the Local Route, Light Rail , Peak Service, Rapid Route, School (details about this field are in the dataset) for the next 7 days
forecasts = {}
for column in ['Local Route', 'Light Rail', 'Peak Service', 'Rapid Route', 'School']: 
    print(f"Forecasting for: {column}")
        
    model = ExponentialSmoothing(df[column], trend="add", seasonal=None, seasonal_periods=None)
    fitted_model = model.fit()
    forecast = fitted_model.forecast(7)
    forecasts[column] = forecast 

    plt.figure(figsize=(10, 6))
    plt.plot(df[column], label='Historical Data', marker='o')
    plt.plot(forecast, label='7-Day Forecast', linestyle='--', color='red', marker='x')
    plt.title(f"{column} Forecast")
    plt.xlabel('Date')
    plt.ylabel('Passengers')
    plt.legend()
    plt.grid()
    plt.show()

print("\n7-Day Forecasts:")
for column, forecast in forecasts.items():
    print(f"{column}:\n{forecast}\n")