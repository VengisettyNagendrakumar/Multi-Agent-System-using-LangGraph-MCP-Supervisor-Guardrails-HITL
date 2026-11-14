from mcp.server.fastmcp import FastMCP 
import os 
import requests 
from dotenv import load_dotenv
load_dotenv()

mcp=FastMCP("Weather MCP Server")
OPENWEATHER_API_KEY=os.getenv("OPENWEATHER_API_KEY")

@mcp.tool()
def get_current_weather(city:str):
    response=requests.get(
        "https://api.openweathermap.org/data/2.5/weather", #we are creating custom mcp os if weather api chnaged you are responsible to chnage the api here
        params={
            "q":city,
            "appid":OPENWEATHER_API_KEY,
            "units":"metric"
        }
    )
    data=response.json()
    if response.status_code!=200:
        return data
    return {
        "city": data["name"],
        "temperature_c": data["main"]["temp"],
        "feels_like_c": data["main"]["feels_like"],
        "humidity": data["main"]["humidity"],
        "condition": data["weather"][0]["description"],
        "wind_speed": data["wind"]["speed"]
    }
    #@mcp.tool() is different: it registers a function as a tool the AI can call. Your weather function does an ordinary blocking requests.get(), so it can be a regular def. It could be async def if you used an async HTTP client, such as httpx.AsyncClient.
@mcp.tool()
def get_forecast(city: str):

    url = (
        "https://api.openweathermap.org/data/2.5/forecast"
    )

    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }

    response = requests.get(
        url,
        params=params
    )

    data = response.json()

    forecast = []

    # Return first 5 forecast entries
    for item in data["list"][:5]:

        forecast.append(
            {
                "datetime": item["dt_txt"],
                "temperature": item["main"]["temp"],
                "weather": item["weather"][0]["description"]
            }
        )

    return {
        "city": city,
        "forecast": forecast
    }




if __name__ == "__main__":
    mcp.run()
# That block runs mcp.run() only when custom_weather_mcp_server.py is run directly as the program.

# In your MCP client config, Python launches that file, so it counts as running directly and mcp.run() starts the server. If another Python file imports it, the block does not run automatically.

