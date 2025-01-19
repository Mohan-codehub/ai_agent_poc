from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

from dotenv import load_dotenv

import openai
import os

load_dotenv()

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


###mutlimoel agent ###
multi_ai_agent=Agent(
    team=[web_search_agent,finance_agent],
    instructions=["Always include the sources","use Tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent.print_response("share the latest news for TCS just give me headline only",stream=True)