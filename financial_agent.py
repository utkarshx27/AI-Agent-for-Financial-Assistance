from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai
import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key=os.getenv("OPENAI_API_KEY")

web_search_agent = Agent(
    name='web_search_agent',
    role='search the web for the information',
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Always include the source of the information in the response."],
    show_tool_calls=True,
    markdown=True,
)


## Funancial Agent

financial_agent = Agent(
    name = "Finance AI Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,
                      company_news=True, company_info=True, technical_indicators=True, historical_prices=True,),
    ],
    instructions=["Use tables to display the data."],
    show_tool_calls=True,
    markdown=True,
    
)

multi_ai_agent = Agent(
    team=[web_search_agent, financial_agent],
    instructions=["Always include source of the information in the response.", "Use table to display the data."],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent.print_response("list top performing stock with their future growth analysis.", stream=True)