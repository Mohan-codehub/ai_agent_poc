import openai
from phi.agent import Agent
import phi.api
from phi.model.openai import OpenAIChat
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv

import os
import phi 
from phi.playground import Playground, serve_playground_app

load_dotenv()

phi.api=os.getenv("PHI_API_KEY")

##web search agent 
web_search_agent=Agent(
    name ="browser agent",
    role="search something in the internet",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools =[DuckDuckGo()],
    instructions=["Always include the sources"],
    show_tool_calls=True,
    markdown=True,

)

#####financial agent ###########

finance_agent=Agent(
    name="Finance Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)],
    instructions=["use Tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

app=Playground(agents=[finance_agent,web_search_agent]).get_app()

if __name__=="__main__":
    serve_playground_app("playground:app",reload =True)