import os 
import asyncio
import certifi 
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

from langchain_groq import ChatGroq

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXE = os.path.join(BASE_DIR, "myenv", "Scripts", "python.exe")
WEATHER_SERVER = os.path.join(BASE_DIR, "custom_weather_mcp_server.py")

load_dotenv()
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

llm=ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
    max_tokens=1024
)

os.environ["SSL_CERT_FILE"]=certifi.where()#certifi.where() returns the path to the certificate bundle installed with the certifi package. That bundle contains trusted Certificate Authority (CA) certificates.
os.environ["REQUEST_CA_BUNDLE"]=certifi.where()#SSL_CERT_FILE and REQUEST_CA_BUNDLE tell SSL/networking tools where to find trusted certificates when checking a website’s HTTPS certificate.
#Those lines force Python’s HTTPS libraries to use certifi’s trusted-certificate file when connecting to the flight API:
TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")
AVIATION_STACK_API_KEY=os.getenv("AVIATIONSTACK_API_KEY")
OPENWEATHER_API_KEY=os.getenv("OPENWEATHER_API_KEY")
client=MultiServerMCPClient(
    {
        "tavily":{ #tavily mcp server for remote srevres it uses streamble http and for local it uses stdio
            "transport":"streamable_http",
            "url":f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}"
        },
        
        "Aviationstack":{
            "transport":"stdio", #this is local mcp server we took this from someone github  called avaitionstack-mcp
            "command":"uvx", #just using their configuration
            "args":[
                "aviationstack-mcp"
            ],
            "env":{
                "AVIATION_STACK_API_KEY":AVIATION_STACK_API_KEY
            }
            
        },
        "weather": {
            "transport": "stdio",
            "command": PYTHON_EXE,
            "args": [
                WEATHER_SERVER
            ],
            "env": {
                "OPENWEATHER_API_KEY": OPENWEATHER_API_KEY
            }
    }
        
    }
)
async def get_all_tools():
    tools=await client.get_tools()
    print("\nAvailable MCP Tools: \n")
    for tool in tools:
        print(tool.name)
    return tools

search_tool=None 
aviation_tools={}
async def initialize_mcp():
    global search_tool  
    global  aviation_tools
    if search_tool is not None and aviation_tools:
        return 
    tools=await get_all_tools()
    print("\nAvailable MCP Tools: \n")
    for tool in tools:
        print(tool.name)
    search_tool=next(
        tool 
        for tool in tools if tool.name=="tavily_search"
    )
    aviation_tools={
        tool.name:tool 
        for tool in tools
        if tool.name!="tavily_search"
    }
async def tavily_mcp_search(query:str):
    await initialize_mcp()
    result=await search_tool.ainvoke(
        {
            "query":query
        }
    )
    return result

#initialize_mcp() is async because it calls await get_all_tools(), which communicates with MCP servers and may take time. await lets Python wait for that work without blocking everything else.
async def aviation_mcp_call(tool_name:str,tool_args:dict=None):
    tools=await client.get_tools() #Get a list of all tools from all connected servers
    tool=next(
        t for t in tools
        if t.name==tool_name
    )
    result=await tool.ainvoke(
        tool_args or {}
    )
    return result

#weather tools 
weather_tool=None
forecast_tool=None
async def intialize_weather_tool():
    global weather_tool, forecast_tool
    if weather_tool is not None and forecast_tool:
        return 
    tools=await client.get_tools()
    weather_tool= next(
        tool 
        for tool in tools if tool.name=="get_current_weather"
    )
    forecast_tool=next(
         tool 
                for tool in tools if tool.name=="get_forecast"
    )
    
async def weather_mcp_seacrh(city:str):
    await intialize_weather_tool()
    return await weather_tool.ainvoke(
        {
            "city":city
        }
    )
async def forecast_mcp_search(city: str):
    await intialize_weather_tool()
    return await forecast_tool.ainvoke(
        {
            "city": city,
        }
    )
#destination extractor in the prompt  like there will be prompt user is giving to get weather we need to extract the city name
def extract_destination(query:str):
    prompt=f"""
    Extract only the destination city or country.
    Query:
    {query}
    Return only destination name
    """
    response=llm.invoke(prompt)
    return response.content.strip()
    
    