import requests
import pandas as pd
import yfinance as yf
import numpy as np
from bs4 import BeautifulSoup
import time
import re
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from io import StringIO
import warnings
import json
import traceback
from functools import wraps
import logging
from scipy import stats
from statistics import mean, median
import threading
from queue import Queue
import cvxpy as cp
from sklearn.linear_model import LinearRegression
from numpy.linalg import inv, pinv
import math

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stock_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("IndianStockAnalyzer")

warnings.filterwarnings('ignore')

def retry(max_retries=3, delay=2, backoff=2, exceptions=(Exception,)):
    """Decorator for retrying function calls with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries, current_delay = 0, delay
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    retries += 1
                    if retries >= max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries: {str(e)}")
                        raise
                    logger.warning(f"Function {func.__name__} failed, retrying in {current_delay} seconds: {str(e)}")
                    time.sleep(current_delay)
                    current_delay *= backoff
            return func(*args, **kwargs)
        return wrapper
    return decorator

class IndianStockAnalyzer:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Updated Screener URL format
        self.screener_base_url = "https://www.screener.in/company/{ticker}/consolidated/"
        self.cache = {}
        self.cache_expiry = {}
        self.cache_timeout = 3600  # 1 hour cache
        
        # API keys storage (user can provide these)
        self.api_keys = {
            'newsapi': None,
            'alphavantage': None,
            'finnhub': None
        }
        
        # Sector classification database
        self.sector_db = self.load_sector_database()
        
        # Transaction cost parameters (as percentage of trade value)
        self.transaction_costs = {
            'brokerage': 0.0005,  # 0.05% brokerage
            'stt': 0.00025,       # 0.025% Securities Transaction Tax
            'gst': 0.0018,        # 0.18% GST on brokerage
            'stamp_duty': 0.00015, # 0.015% stamp duty
            'exchange_charge': 0.000035 # 0.0035% exchange charge
        }
        
        # Tax rates (for Indian investors)
        self.tax_rates = {
            'stcg': 0.15,      # Short-term capital gains (holding < 1 year)
            'ltcg': 0.10,      # Long-term capital gains (holding > 1 year) above ₹1L
            'dividend': 0.10,   # Dividend distribution tax
            'stcg_debt': 0.30,  # Short-term capital gains for debt funds
            'ltcg_debt': 0.10   # Long-term capital gains for debt funds (with indexation)
        }
        
        # Request timeout configuration
        self.request_timeout = 15
        self.max_retries = 3

    def set_api_keys(self, newsapi=None, alphavantage=None, finnhub=None):
        """Set API keys for various services"""
        self.api_keys['newsapi'] = newsapi
        self.api_keys['alphavantage'] = alphavantage
        self.api_keys['finnhub'] = finnhub
        logger.info("API keys updated")

    def load_sector_database(self):
        """Load sector classification database with expanded classifications"""
        # Expanded sector classification database
        sector_mapping = {
            'RELIANCE': {'sector': 'Energy Minerals', 'industry': 'Oil & Gas Refining & Marketing'},
            'INFY': {'sector': 'Technology Services', 'industry': 'IT Services & Consulting'},
            'TCS': {'sector': 'Technology Services', 'industry': 'IT Services & Consulting'},
            'HDFCBANK': {'sector': 'Finance', 'industry': 'Private Banks'},
            'ICICIBANK': {'sector': 'Finance', 'industry': 'Private Banks'},
            'HINDUNILVR': {'sector': 'Consumer Non-Durables', 'industry': 'Personal Care'},
            'ITC': {'sector': 'Consumer Non-Durables', 'industry': 'Cigarettes & Tobacco Products'},
            'SBIN': {'sector': 'Finance', 'industry': 'Public Banks'},
            'BHARTIARTL': {'sector': 'Communications', 'industry': 'Telecom Services'},
            'KOTAKBANK': {'sector': 'Finance', 'industry': 'Private Banks'},
            'BAJFINANCE': {'sector': 'Finance', 'industry': 'NBFC'},
            'HCLTECH': {'sector': 'Technology Services', 'industry': 'IT Services & Consulting'},
            'AXISBANK': {'sector': 'Finance', 'industry': 'Private Banks'},
            'LT': {'sector': 'Industrial Services', 'industry': 'Engineering & Construction'},
            'MARUTI': {'sector': 'Consumer Durables', 'industry': 'Passenger Cars & Utility Vehicles'},
            'ASIANPAINT': {'sector': 'Process Industries', 'industry': 'Paints'},
            'HINDALCO': {'sector': 'Non-Energy Minerals', 'industry': 'Aluminium'},
            'WIPRO': {'sector': 'Technology Services', 'industry': 'IT Services & Consulting'},
            'SUNPHARMA': {'sector': 'Health Technology', 'industry': 'Pharmaceuticals'},
            'ONGC': {'sector': 'Energy Minerals', 'industry': 'Oil & Gas Exploration & Production'},
            'TITAN': {'sector': 'Consumer Durables', 'industry': 'Watches & Accessories'},
            'NESTLEIND': {'sector': 'Consumer Non-Durables', 'industry': 'Food Processing'},
            'ULTRACEMCO': {'sector': 'Non-Energy Minerals', 'industry': 'Cement'},
            'JSWSTEEL': {'sector': 'Non-Energy Minerals', 'industry': 'Steel'},
            'TATAMOTORS': {'sector': 'Consumer Durables', 'industry': 'Automobiles'},
            'ADANIPORTS': {'sector': 'Transportation', 'industry': 'Port Services'},
            'POWERGRID': {'sector': 'Utilities', 'industry': 'Power Transmission'},
            'NTPC': {'sector': 'Utilities', 'industry': 'Power Generation'},
            'INDUSINDBK': {'sector': 'Finance', 'industry': 'Private Banks'},
            'TECHM': {'sector': 'Technology Services', 'industry': 'IT Services & Consulting'},
            'BAJAJFINSV': {'sector': 'Finance', 'industry': 'Financial Services'},
            'DRREDDY': {'sector': 'Health Technology', 'industry': 'Pharmaceuticals'},
            'HDFC': {'sector': 'Finance', 'industry': 'Housing Finance'},
            'HDFCLIFE': {'sector': 'Finance', 'industry': 'Insurance'},
            'SBILIFE': {'sector': 'Finance', 'industry': 'Insurance'},
            'DIVISLAB': {'sector': 'Health Technology', 'industry': 'Pharmaceuticals'},
            'BRITANNIA': {'sector': 'Consumer Non-Durables', 'industry': 'Food Processing'},
            'CIPLA': {'sector': 'Health Technology', 'industry': 'Pharmaceuticals'},
            'GRASIM': {'sector': 'Process Industries', 'industry': 'Cement & Chemicals'},
            'UPL': {'sector': 'Process Industries', 'industry': 'Agrochemicals'},
            'TATASTEEL': {'sector': 'Non-Energy Minerals', 'industry': 'Steel'},
            'BPCL': {'sector': 'Energy Minerals', 'industry': 'Oil Refining'},
            'HEROMOTOCO': {'sector': 'Consumer Durables', 'industry': 'Two Wheelers'},
            'EICHERMOT': {'sector': 'Consumer Durables', 'industry': 'Automobiles'},
            'COALINDIA': {'sector': 'Energy Minerals', 'industry': 'Coal Mining'},
            'SHREECEM': {'sector': 'Non-Energy Minerals', 'industry': 'Cement'},
            'HINDZINC': {'sector': 'Non-Energy Minerals', 'industry': 'Zinc Mining'},
            'DABUR': {'sector': 'Consumer Non-Durables', 'industry': 'Personal Care'},
            'ABBOTINDIA': {'sector': 'Health Technology', 'industry': 'Pharmaceuticals'},
            'BERGEPAINT': {'sector': 'Process Industries', 'industry': 'Paints'},
            'PIDILITIND': {'sector': 'Process Industries', 'industry': 'Adhesives'},
            'HAVELLS': {'sector': 'Consumer Durables', 'industry': 'Electrical Equipment'},
            'AMBUJACEM': {'sector': 'Non-Energy Minerals', 'industry': 'Cement'},
            'ACC': {'sector': 'Non-Energy Minerals', 'industry': 'Cement'},
            'GODREJCP': {'sector': 'Consumer Non-Durarables', 'industry': 'Personal Care'},
            'BIOCON': {'sector': 'Health Technology', 'industry': 'Biotechnology'},
            'MOTHERSUMI': {'sector': 'Consumer Durables', 'industry': 'Auto Components'},
            'BOSCHLTD': {'sector': 'Consumer Durables', 'industry': 'Auto Components'},
            'M&M': {'sector': 'Consumer Durables', 'industry': 'Automobiles'},
            'ASHOKA': {'sector': 'Industrial Services', 'industry': 'Engineering & Construction'},
        }
        return sector_mapping

    def get_sector_info(self, ticker):
        """Get sector and industry information for a ticker"""
        if ticker in self.sector_db:
            return self.sector_db[ticker]
        
        # If not in our database, try to fetch from Yahoo Finance
        try:
            yf_ticker = f"{ticker}.NS"
            stock = yf.Ticker(yf_ticker)
            info = stock.info
            return {
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown')
            }
        except:
            return {'sector': 'Unknown', 'industry': 'Unknown'}

    def cached_request(self, key, func, *args, **kwargs):
        """Cache results of function calls to avoid repeated requests"""
        current_time = time.time()
        if key in self.cache and current_time - self.cache_expiry.get(key, 0) < self.cache_timeout:
            return self.cache[key]
        
        result = func(*args, **kwargs)
        self.cache[key] = result
        self.cache_expiry[key] = current_time
        return result

    def get_tickers_from_user(self):
        print("Indian Stock Analysis Tool")
        print("--------------------------")
        print("1. Enter comma-separated stock tickers (e.g., INFY,RELIANCE,TCS)")
        print("2. Type 'SCAN' to analyze top stocks from news sources")
        choice = input("Enter your choice: ").strip().upper()
        
        if choice == 'SCAN':
            return self.scan_stocks_from_news()
        else:
            tickers = choice.split(',')
            return [ticker.strip() for ticker in tickers if ticker.strip()]
    
    def scan_stocks_from_news(self):
        """Scan news sources to identify frequently mentioned stocks"""
        logger.info("Scanning news sources for stock mentions...")
        
        # Sources to scan for stock mentions
        news_sources = [
            "https://www.livemint.com/market/stock-market-news",
            "https://economictimes.indiatimes.com/markets/stocks/news",
            "https://www.moneycontrol.com/news/business/stocks/",
            "https://www.business-standard.com/markets/news",
            "https://www.cnbctv18.com/market/",
            "https://www.etnownews.com/markets",
            "https://www.ndtvprofit.com/markets",
            "https://www.zeebiz.com/markets",
            "https://www.financialexpress.com/market/stock-market-news/"
        ]
        
        mentioned_stocks = {}
        
        for url in news_sources:
            try:
                response = self.session.get(url, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    text = soup.get_text().upper()
                    
                    # Look for stock tickers in the text
                    for ticker in self.sector_db.keys():
                        if ticker in text:
                            mentioned_stocks[ticker] = mentioned_stocks.get(ticker, 0) + 1
                            
                    # Also look for company names
                    for ticker, info in self.sector_db.items():
                        company_name = info.get('industry', '').upper()
                        if company_name and company_name in text:
                            mentioned_stocks[ticker] = mentioned_stocks.get(ticker, 0) + 1
                            
            except Exception as e:
                logger.warning(f"Error scanning {url}: {str(e)}")
                continue
        
        # Get top 10 mentioned stocks
        top_stocks = sorted(mentioned_stocks.items(), key=lambda x: x[1], reverse=True)[:10]
        top_tickers = [stock[0] for stock in top_stocks]
        
        print(f"Found {len(top_tickers)} frequently mentioned stocks: {', '.join(top_tickers)}")
        return top_tickers
    
    @retry(max_retries=3, delay=2, backoff=2)
    def get_nse_data(self, ticker):
        """Get data from NSE website with enhanced error handling"""
        try:
            # Use the correct NSE URL format
            nse_url = f"https://www.nseindia.com/get-quotes/equity?symbol={ticker}"
            response = self.session.get(nse_url, headers=self.headers, timeout=self.request_timeout)
            response.raise_for_status()
            
            # Check if response contains valid content
            if response.text.strip():
                # Parse the HTML for data
                soup = BeautifulSoup(response.content, 'html.parser')
                data = {}
                
                # Extract key data points
                # Current price
                price_div = soup.find('div', id='quoteLtp')
                if price_div:
                    try:
                        data['current_price'] = float(price_div.get('data-price', '0').replace(',', ''))
                    except:
                        price_text = price_div.text.strip().replace(',', '')
                        data['current_price'] = float(price_text) if price_text else 0
                
                # Previous close
                prev_close_div = soup.find('div', id='prevClose')
                if prev_close_div:
                    try:
                        data['previous_close'] = float(prev_close_div.text.strip().replace(',', ''))
                    except:
                        data['previous_close'] = 0
                
                # Other key metrics
                metrics = soup.find_all('div', class_='cell')
                for i in range(0, len(metrics)-1, 2):
                    key = metrics[i].text.strip().lower().replace(' ', '_')
                    value = metrics[i+1].text.strip().replace(',', '')
                    try:
                        data[key] = float(value)
                    except:
                        data[key] = value
                
                return data
            else:
                logger.warning(f"NSE returned empty response for {ticker}")
                return self.get_nse_fallback_data(ticker)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching NSE data for {ticker}: {str(e)}")
            return self.get_nse_fallback_data(ticker)
        except Exception as e:
            logger.error(f"Unexpected error fetching NSE data for {ticker}: {str(e)}")
            return self.get_nse_fallback_data(ticker)
    
    def get_nse_fallback_data(self, ticker):
        """Fallback method to get basic NSE data when API fails"""
        try:
            # Try to get data from Yahoo Finance as fallback
            yf_ticker = f"{ticker}.NS"
            stock = yf.Ticker(yf_ticker)
            info = stock.info
            
            data = {
                'current_price': info.get('currentPrice', 0),
                'previous_close': info.get('previousClose', 0),
                'open': info.get('open', 0),
                'day_low': info.get('dayLow', 0),
                'day_high': info.get('dayHigh', 0),
                'volume': info.get('volume', 0),
                'average_volume': info.get('averageVolume', 0),
                'market_cap': info.get('marketCap', 0),
                'source': 'yahoo_fallback'
            }
            
            return data
            
        except Exception as e:
            logger.warning(f"Fallback NSE data also failed for {ticker}: {str(e)}")
            return None
    
    @retry(max_retries=3, delay=2, backoff=2)
    def get_screener_data(self, ticker):
        """Get financial data from Screener.in with error handling"""
        try:
            # Use the updated URL format with consolidated data
            url = self.screener_base_url.format(ticker=ticker)
            response = self.session.get(url, headers=self.headers, timeout=self.request_timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract key financial data
            data = {}
            
            # Company name
            name_el = soup.find('h1', class_='margin-0')
            if name_el:
                data['company_name'] = name_el.text.strip()
            
            # Extract financial ratios
            ratio_sections = soup.find_all('li', class_='flex flex-space-between')
            for section in ratio_sections:
                if section.find('span', class_='name'):
                    name = section.find('span', class_='name').text.strip()
                    value = section.find('span', class_='number').text.strip()
                    data[name] = value
            
            # Extract financial tables
            try:
                tables = pd.read_html(StringIO(response.text))
                for i, table in enumerate(tables):
                    if 'Particulars' in table.columns:
                        data['financials'] = table
                    elif 'Quarterly Results' in table.columns:
                        data['quarterly_results'] = table
            except Exception as e:
                logger.warning(f"Could not parse tables for {ticker}: {str(e)}")
            
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Screener data for {ticker}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching Screener data for {ticker}: {str(e)}")
            return None
    
    @retry(max_retries=3, delay=2, backoff=2)
    def get_yahoo_data(self, ticker):
        """Get data from Yahoo Finance with error handling"""
        try:
            # For Indian stocks, we need to add .NS for NSE
            yf_ticker = f"{ticker}.NS"
            stock = yf.Ticker(yf_ticker)
            info = stock.info
            hist = stock.history(period="2y")
            financials = stock.financials
            balance_sheet = stock.balance_sheet
            cashflow = stock.cashflow
            
            data = {
                'current_price': info.get('regularMarketPrice', info.get('currentPrice', 0)),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'pb_ratio': info.get('priceToBook', 0),
                'dividend_yield': info.get('dividendYield', 0),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0),
                'volume': info.get('volume', 0),
                'avg_volume': info.get('averageVolume', 0),
                'beta': info.get('beta', 0),
                'eps': info.get('trailingEps', 0),
                'book_value': info.get('bookValue', 0),
                'profit_margins': info.get('profitMargins', 0),
                'operating_margins': info.get('operatingMargins', 0),
                'return_on_equity': info.get('returnOnEquity', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'dividend_rate': info.get('dividendRate', 0),
                'dividend_payout_ratio': info.get('payoutRatio', 0),
                'history': hist,
                'financials': financials,
                'balance_sheet': balance_sheet,
                'cashflow': cashflow
            }
            return data
        except Exception as e:
            logger.error(f"Error fetching Yahoo data for {ticker}: {str(e)}")
            return None
    
    @retry(max_retries=2, delay=1, backoff=2)
    def get_moneycontrol_data(self, ticker):
        """Get data from MoneyControl"""
        try:
            url = f"https://www.moneycontrol.com/india/stockpricequote/{ticker.lower()}"
            response = self.session.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            data = {}
            
            # Extract key metrics
            technicals = {}
            tech_table = soup.find('table', class_='mctable1')
            if tech_table:
                rows = tech_table.find_all('tr')
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) == 2:
                        key = cols[0].text.strip()
                        value = cols[1].text.strip()
                        technicals[key] = value
            
            data['technicals'] = technicals
            return data
        except Exception as e:
            logger.warning(f"Could not fetch MoneyControl data for {ticker}: {str(e)}")
            return None
    
    @retry(max_retries=2, delay=1, backoff=2)
    def get_news_data(self, ticker):
        """Get recent news for the stock from multiple sources with improved extraction"""
        news_sources = [
            {
                'url': f"https://www.moneycontrol.com/news/tags/{ticker.lower()}.html",
                'source': 'MoneyControl',
                'headline_selector': 'h2 a, h3 a, .title a'
            },
            {
                'url': f"https://economictimes.indiatimes.com/markets/stocks/stock-quotes?ticker={ticker}",
                'source': 'Economic Times',
                'headline_selector': '.newsList a, .eachStory a, .title a'
            },
            {
                'url': f"https://www.livemint.com/market/stock-market-news/{ticker}-news",
                'source': 'Live Mint',
                'headline_selector': '.headline a, .title a, h2 a, h3 a'
            },
            {
                'url': f"https://www.business-standard.com/topic/{ticker.lower()}",
                'source': 'Business Standard',
                'headline_selector': '.headline a, h2 a, h3 a'
            },
            {
                'url': f"https://www.cnbctv18.com/search/?q={ticker}",
                'source': 'CNBC TV18',
                'headline_selector': '.headline a, .title a, h2 a, h3 a'
            },
            {
                'url': f"https://www.etnownews.com/search/?q={ticker}",
                'source': 'ET Now',
                'headline_selector': '.headline a, .title a, h2 a, h3 a'
            },
            {
                'url': f"https://www.ndtvprofit.com/search?query={ticker}",
                'source': 'NDTV Profit',
                'headline_selector': '.headline a, .title a, h2 a, h3 a'
            },
            {
                'url': f"https://www.zeebiz.com/search?text={ticker}",
                'source': 'Zee Business',
                'headline_selector': '.headline a, .title a, h2 a, h3 a'
            },
            {
                'url': f"https://hindi.cnbctv18.com/search/?q={ticker}",
                'source': 'CNBC Awaaz',
                'headline_selector': '.headline a, .title a, h2 a, h3 a'
            },
            {
                'url': f"https://www.financialexpress.com/?s={ticker}",
                'source': 'Financial Express',
                'headline_selector': '.title a, h2 a, h3 a'
            }
        ]
        
        # Add API-based news sources if keys are available
        if self.api_keys['newsapi']:
            news_sources.append({
                'url': f"https://newsapi.org/v2/everything?q={ticker}&apiKey={self.api_keys['newsapi']}",
                'source': 'NewsAPI',
                'type': 'api'
            })
        
        news_items = []
        for source in news_sources:
            try:
                if source.get('type') == 'api':
                    # Handle API-based news sources
                    response = self.session.get(source['url'], timeout=10)
                    if response.status_code == 200:
                        api_data = response.json()
                        for article in api_data.get('articles', [])[:5]:
                            news_items.append({
                                'headline': article.get('title', ''),
                                'link': article.get('url', ''),
                                'source': source['source'],
                                'date': article.get('publishedAt', datetime.now().strftime('%Y-%m-%d'))
                            })
                else:
                    # Handle HTML-based news sources
                    response = self.session.get(source['url'], headers=self.headers, timeout=10)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        # Extract news headlines using multiple selectors
                        headlines = []
                        selectors = source['headline_selector'].split(', ')
                        for selector in selectors:
                            elements = soup.select(selector)
                            for element in elements:
                                headline_text = element.text.strip()
                                if headline_text and len(headline_text) > 10:  # Minimum length
                                    headlines.append({
                                        'headline': headline_text,
                                        'element': element
                                    })
                        
                        # Get unique headlines
                        seen_headlines = set()
                        for headline in headlines[:8]:  # Get first 8 headlines
                            text = headline['headline']
                            if text not in seen_headlines:
                                element = headline['element']
                                link = element.get('href')
                                if link:
                                    if not link.startswith('http'):
                                        # Handle relative URLs
                                        if source['source'] == 'MoneyControl' and link.startswith('/'):
                                            link = f"https://www.moneycontrol.com{link}"
                                        elif source['source'] == 'Economic Times' and link.startswith('/'):
                                            link = f"https://economictimes.indiatimes.com{link}"
                                        elif source['source'] == 'Live Mint' and link.startswith('/'):
                                            link = f"https://www.livemint.com{link}"
                                        elif source['source'] == 'Business Standard' and link.startswith('/'):
                                            link = f"https://www.business-standard.com{link}"
                                        elif source['source'] == 'CNBC TV18' and link.startswith('/'):
                                            link = f"https://www.cnbctv18.com{link}"
                                        elif source['source'] == 'ET Now' and link.startswith('/'):
                                            link = f"https://www.etnownews.com{link}"
                                        elif source['source'] == 'NDTV Profit' and link.startswith('/'):
                                            link = f"https://www.ndtvprofit.com{link}"
                                        elif source['source'] == 'Zee Business' and link.startswith('/'):
                                            link = f"https://www.zeebiz.com{link}"
                                        elif source['source'] == 'CNBC Awaaz' and link.startswith('/'):
                                            link = f"https://hindi.cnbctv18.com{link}"
                                        elif source['source'] == 'Financial Express' and link.startswith('/'):
                                            link = f"https://www.financialexpress.com{link}"
                                    
                                    news_items.append({
                                        'headline': text,
                                        'link': link,
                                        'source': source['source'],
                                        'date': datetime.now().strftime('%Y-%m-%d')
                                    })
                                    seen_headlines.add(text)
            except Exception as e:
                logger.warning(f"Error fetching news from {source['source']}: {str(e)}")
                continue
        
        # If no news found, try additional sources
        if not news_items:
            additional_news = self.get_additional_news_sources(ticker)
            news_items.extend(additional_news)
                
        return news_items
    
    def get_additional_news_sources(self, ticker):
        """Get news from additional sources when primary sources fail"""
        additional_news = []
        additional_sources = [
            {
                'url': f"https://www.thehindubusinessline.com/search/?q={ticker}",
                'source': 'Hindu Business Line',
                'headline_selector': '.title a, h2 a, h3 a'
            },
            {
                'url': f"https://finshots.in/search/?q={ticker}",
                'source': 'Finshots',
                'headline_selector': '.title a, h2 a, h3 a'
            }
        ]
        
        for source in additional_sources:
            try:
                response = self.session.get(source['url'], headers=self.headers, timeout=8)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Extract news headlines
                    headlines = soup.select(source['headline_selector'])
                    for headline in headlines[:3]:  # Get first 3 headlines
                        text = headline.text.strip()
                        if text and len(text) > 10:
                            link = headline.get('href')
                            if link and not link.startswith('http'):
                                # Handle relative URLs
                                if source['source'] == 'Hindu Business Line':
                                    link = f"https://www.thehindubusinessline.com{link}"
                                elif source['source'] == 'Finshots':
                                    link = f"https://finshots.in{link}"
                            
                            additional_news.append({
                                'headline': text,
                                'link': link,
                                'source': source['source'],
                                'date': datetime.now().strftime('%Y-%m-%d')
                            })
            except Exception as e:
                logger.warning(f"Error fetching additional news from {source['source']}: {str(e)}")
                continue
        
        return additional_news
    
    def analyze_news_sentiment(self, news_items, sector):
        """Enhanced news sentiment analysis using keyword scoring with sector context"""
        if not news_items:
            return "neutral", 0
        
        # Define keywords with weights (expanded list)
        positive_keywords = {
            'profit': 2, 'growth': 2, 'expansion': 2, 'acquire': 2, 'win': 2, 
            'positive': 3, 'raise': 2, 'upgrade': 3, 'beat': 2, 'surge': 2,
            'record': 2, 'strong': 2, 'success': 2, 'opportunity': 1, 'innovate': 1,
            'approve': 2, 'award': 2, 'contract': 2, 'deal': 2, 'partner': 1,
            'launch': 1, 'expand': 2, 'invest': 1, 'develop': 1, 'modernize': 1,
            'digital': 1, 'technology': 1, 'efficient': 1, 'sustainable': 1, 'green': 1,
            'buy': 2, 'outperform': 2, 'bullish': 2, 'recommend': 2, 'target': 1
        }
        
        negative_keywords = {
            'loss': 3, 'decline': 2, 'fall': 2, 'drop': 2, 'cut': 2, 
            'negative': 3, 'downgrade': 3, 'miss': 2, 'investigation': 3,
            'warn': 2, 'weak': 2, 'failure': 3, 'risk': 2, 'concern': 2,
            'delay': 2, 'default': 3, 'bankrupt': 3, 'crisis': 3, 'litigation': 2,
            'probe': 2, 'fraud': 3, 'scam': 3, 'corruption': 3, 'violation': 2,
            'penalty': 2, 'fine': 2, 'sue': 2, 'lawsuit': 2, 'recall': 2,
            'sell': 2, 'underperform': 2, 'bearish': 2, 'avoid': 2, 'reduce': 2
        }
        
        # Sector-specific keywords
        construction_positive = {'project': 2, 'contract': 2, 'bid': 1, 'infrastructure': 1, 
                               'highway': 1, 'bridge': 1, 'construction': 1, 'development': 1}
        construction_negative = {'delay': 2, 'cost overrun': 3, 'deadline': 1, 'accident': 2, 
                               'safety': 1, 'labor': 1, 'strike': 2}
        
        finance_positive = {'loan': 1, 'credit': 1, 'lending': 1, 'interest': 1, 'banking': 1,
                          'finance': 1, 'investment': 2, 'capital': 1, 'growth': 2}
        finance_negative = {'default': 3, 'npa': 3, 'bad debt': 3, 'provision': 2, 'write-off': 3,
                          'rbi': 2, 'regulation': 1, 'compliance': 1}
        
        tech_positive = {'digital': 2, 'technology': 2, 'innovation': 2, 'software': 1, 'cloud': 2,
                       'ai': 3, 'artificial intelligence': 3, 'machine learning': 3, 'data': 2}
        tech_negative = {'cyber': 2, 'security': 2, 'breach': 3, 'hack': 3, 'outage': 2,
                       'downtime': 2, 'bug': 1, 'glitch': 1}
        
        positive_score = 0
        negative_score = 0
        
        for news in news_items:
            headline = news['headline'].lower()
            
            # Check for positive keywords
            for word, weight in positive_keywords.items():
                if word in headline:
                    positive_score += weight
            
            # Check for negative keywords
            for word, weight in negative_keywords.items():
                if word in headline:
                    negative_score += weight
            
            # Check for sector-specific keywords
            if 'construction' in sector.lower() or 'infrastructure' in sector.lower():
                for word, weight in construction_positive.items():
                    if word in headline:
                        positive_score += weight
                for word, weight in construction_negative.items():
                    if word in headline:
                        negative_score += weight
            elif 'finance' in sector.lower() or 'bank' in sector.lower():
                for word, weight in finance_positive.items():
                    if word in headline:
                        positive_score += weight
                for word, weight in finance_negative.items():
                    if word in headline:
                        negative_score += weight
            elif 'technology' in sector.lower() or 'software' in sector.lower():
                for word, weight in tech_positive.items():
                    if word in headline:
                        positive_score += weight
                for word, weight in tech_negative.items():
                    if word in headline:
                        negative_score += weight
        
        # Calculate sentiment score (-1 to +1)
        total_score = positive_score + negative_score
        if total_score == 0:
            return "neutral", 0
        
        sentiment_score = (positive_score - negative_score) / total_score
        
        # Adjust thresholds for more sensitive detection
        if sentiment_score > 0.1:
            return "positive", sentiment_score
        elif sentiment_score < -0.1:
            return "negative", sentiment_score
        else:
            return "neutral", sentiment_score
    
    def analyze_stock(self, ticker):
        """Main analysis function for a single stock with enhanced error handling"""
        logger.info(f"Analyzing {ticker}...")
        
        try:
            # Get data from multiple sources with caching and fallbacks
            nse_data = self.cached_request(f"nse_{ticker}", self.get_nse_data, ticker)
            
            # If NSE data fails, try fallback
            if nse_data is None:
                nse_data = self.get_nse_fallback_data(ticker)
                
            screener_data = self.cached_request(f"screener_{ticker}", self.get_screener_data, ticker)
            yahoo_data = self.cached_request(f"yahoo_{ticker}", self.get_yahoo_data, ticker)
            moneycontrol_data = self.cached_request(f"moneycontrol_{ticker}", self.get_moneycontrol_data, ticker)
            news_data = self.cached_request(f"news_{ticker}", self.get_news_data, ticker)
            
            # Get sector information
            sector_info = self.get_sector_info(ticker)
            
            # If news data is empty, try to get some basic news
            if not news_data:
                logger.warning(f"No news found for {ticker}, trying alternative sources")
                # Try additional news sources
                additional_news = self.get_additional_news_sources(ticker)
                news_data.extend(additional_news)
            
            # Combine all data
            all_data = {
                'ticker': ticker,
                'nse_data': nse_data,
                'screener_data': screener_data,
                'yahoo_data': yahoo_data,
                'moneycontrol_data': moneycontrol_data,
                'news_data': news_data,
                'sector_info': sector_info
            }
            
            # Perform analysis and generate recommendation
            recommendation = self.generate_recommendation(all_data)
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'Ticker': ticker,
                'Recommendation': "ERROR",
                'Current Price': "N/A",
                'Target Price': "N/A",
                'Upside/Downside': "N/A",
                'Time Horizon': "N/A",
                'Remarks': f"Error in analysis: {str(e)}",
                'P/E Ratio': "N/A",
                'Market Cap': "N/A",
                'Sector': "N/A",
                'Industry': "N/A",
                'Valuation Models': {}
            }
    
    def calculate_dcf_valuation(self, data):
        """Discounted Cash Flow valuation model"""
        try:
            yahoo_data = data.get('yahoo_data', {})
            if not yahoo_data or 'cashflow' not in yahoo_data or yahoo_data['cashflow'].empty:
                return None
                
            # Get free cash flow data
            cashflow = yahoo_data['cashflow']
            if 'Free Cash Flow' not in cashflow.index:
                return None
                
            fcf_series = cashflow.loc['Free Cash Flow']
            
            # Use the most recent FCF value
            if fcf_series.empty:
                return None
                
            current_fcf = fcf_series.iloc[0]
            
            # If we don't have enough historical data, use a simple approach
            if len(fcf_series) < 3:
                # Assume 10% growth rate for 5 years, then 3% terminal growth
                growth_rate = 0.10
                terminal_growth = 0.03
                discount_rate = 0.10  # WACC
                
                # Project FCF for 5 years
                fcf_projections = []
                for i in range(1, 6):
                    fcf_projections.append(current_fcf * (1 + growth_rate) ** i)
                
                # Terminal value
                terminal_value = fcf_projections[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
                
                # Discount all cash flows
                enterprise_value = 0
                for i, fcf in enumerate(fcf_projections):
                    enterprise_value += fcf / (1 + discount_rate) ** (i + 1)
                
                enterprise_value += terminal_value / (1 + discount_rate) ** 5
                
                # Adjust for cash and debt (simplified)
                equity_value = enterprise_value
                
                # Get shares outstanding
                shares_outstanding = yahoo_data.get('market_cap', 0) / yahoo_data.get('current_price', 1) if yahoo_data.get('current_price', 0) > 0 else 0
                
                if shares_outstanding > 0:
                    dcf_value = equity_value / shares_outstanding
                    return dcf_value
                
            else:
                # More sophisticated DCF with historical growth rate calculation
                # Calculate historical FCF growth rate
                fcf_values = [fcf_series.iloc[i] for i in range(min(3, len(fcf_series)))]
                growth_rates = []
                
                for i in range(1, len(fcf_values)):
                    if fcf_values[i-1] != 0:
                        growth_rates.append((fcf_values[i] - fcf_values[i-1]) / abs(fcf_values[i-1]))
                
                if growth_rates:
                    historical_growth = mean(growth_rates)
                    # Use a more conservative estimate for future growth
                    growth_rate = max(min(historical_growth, 0.15), 0.05)
                    
                    # Rest of DCF calculation
                    terminal_growth = 0.03
                    discount_rate = 0.10  # WACC
                    
                    # Project FCF for 5 years
                    fcf_projections = []
                    for i in range(1, 6):
                        fcf_projections.append(current_fcf * (1 + growth_rate) ** i)
                    
                    # Terminal value
                    terminal_value = fcf_projections[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
                    
                    # Discount all cash flows
                    enterprise_value = 0
                    for i, fcf in enumerate(fcf_projections):
                        enterprise_value += fcf / (1 + discount_rate) ** (i + 1)
                    
                    enterprise_value += terminal_value / (1 + discount_rate) ** 5
                    
                    # Adjust for cash and debt (simplified)
                    equity_value = enterprise_value
                    
                    # Get shares outstanding
                    shares_outstanding = yahoo_data.get('market_cap', 0) / yahoo_data.get('current_price', 1) if yahoo_data.get('current_price', 0) > 0 else 0
                    
                    if shares_outstanding > 0:
                        dcf_value = equity_value / shares_outstanding
                        return dcf_value
                        
        except Exception as e:
            logger.warning(f"Error in DCF calculation: {str(e)}")
            return None
        
        return None
    
    def calculate_ddm_valuation(self, data):
        """Dividend Discount Model valuation"""
        try:
            yahoo_data = data.get('yahoo_data', {})
            if not yahoo_data:
                return None
                
            current_price = yahoo_data.get('current_price', 0)
            dividend_rate = yahoo_data.get('dividend_rate', 0)
            dividend_yield = yahoo_data.get('dividend_yield', 0)
            
            # If we don't have dividend rate, calculate from yield and price
            if dividend_rate == 0 and dividend_yield > 0 and current_price > 0:
                dividend_rate = dividend_yield * current_price
                
            if dividend_rate <= 0:
                return None  # Company doesn't pay dividends
                
            # Estimate growth rate (simplified: use historical EPS growth or ROE * retention ratio)
            eps = yahoo_data.get('eps', 0)
            roe = yahoo_data.get('return_on_equity', 0)
            payout_ratio = yahoo_data.get('dividend_payout_ratio', 0)
            
            if roe > 0 and payout_ratio > 0:
                retention_ratio = 1 - payout_ratio
                growth_rate = roe * retention_ratio
            else:
                # Assume conservative growth rate
                growth_rate = 0.05
                
            # Required rate of return (simplified)
            required_return = 0.10  # 10% required return
            
            # Gordon Growth Model: Value = D1 / (r - g)
            if required_return <= growth_rate:
                # Use two-stage DDM instead
                # Stage 1: high growth for 5 years
                # Stage 2: perpetual growth at lower rate
                stage1_growth = growth_rate
                stage2_growth = 0.03  # Terminal growth
                
                # Project dividends for 5 years
                dividends = []
                for i in range(1, 6):
                    dividends.append(dividend_rate * (1 + stage1_growth) ** i)
                
                # Terminal value at year 5
                terminal_value = dividends[-1] * (1 + stage2_growth) / (required_return - stage2_growth)
                
                # Discount all cash flows
                value = 0
                for i, div in enumerate(dividends):
                    value += div / (1 + required_return) ** (i + 1)
                
                value += terminal_value / (1 + required_return) ** 5
                
                return value
            else:
                # Standard Gordon Growth Model
                ddm_value = dividend_rate * (1 + growth_rate) / (required_return - growth_rate)
                return ddm_value
                
        except Exception as e:
            logger.warning(f"Error in DDM calculation: {str(e)}")
            return None
    
    def calculate_eva_valuation(self, data):
        """Economic Value Added valuation model"""
        try:
            yahoo_data = data.get('yahoo_data', {})
            if not yahoo_data:
                return None
                
            # Get financial data
            financials = yahoo_data.get('financials', pd.DataFrame())
            balance_sheet = yahoo_data.get('balance_sheet', pd.DataFrame())
            
            if financials.empty or balance_sheet.empty:
                return None
                
            # Extract NOPAT (Net Operating Profit After Tax)
            if 'Operating Income' in financials.index:
                operating_income = financials.loc['Operating Income'].iloc[0]
            elif 'EBIT' in financials.index:
                operating_income = financials.loc['EBIT'].iloc[0]
            else:
                return None
                
            # Get tax rate (simplified)
            if 'Income Tax Expense' in financials.index and 'Pretax Income' in financials.index:
                tax_expense = financials.loc['Income Tax Expense'].iloc[0]
                pretax_income = financials.loc['Pretax Income'].iloc[0]
                if pretax_income != 0:
                    tax_rate = tax_expense / pretax_income
                else:
                    tax_rate = 0.25  # Assume 25% tax rate
            else:
                tax_rate = 0.25  # Assume 25% tax rate
                
            nopat = operating_income * (1 - tax_rate)
            
            # Get invested capital
            if 'Total Assets' in balance_sheet.index and 'Total Current Liabilities' in balance_sheet.index:
                total_assets = balance_sheet.loc['Total Assets'].iloc[0]
                current_liabilities = balance_sheet.loc['Total Current Liabilities'].iloc[0]
                
                # Adjust for non-interest-bearing current liabilities (simplified)
                # Assume 50% of current liabilities are non-interest-bearing
                invested_capital = total_assets - (0.5 * current_liabilities)
            else:
                return None
                
            # Calculate WACC (Weighted Average Cost of Capital)
            # Cost of equity
            risk_free_rate = 0.07  # Assume 7% risk-free rate for India
            market_risk_premium = 0.05  # Assume 5% market risk premium
            beta = yahoo_data.get('beta', 1.0)
            cost_of_equity = risk_free_rate + beta * market_risk_premium
            
            # Cost of debt (simplified)
            cost_of_debt = 0.10  # Assume 10% cost of debt
            
            # Debt-to-equity ratio
            debt_to_equity = yahoo_data.get('debt_to_equity', 0.5)
            
            # WACC calculation
            equity_ratio = 1 / (1 + debt_to_equity)
            debt_ratio = debt_to_equity / (1 + debt_to_equity)
            wacc = (equity_ratio * cost_of_equity) + (debt_ratio * cost_of_debt * (1 - tax_rate))
            
            # Calculate EVA
            eva = nopat - (wacc * invested_capital)
            
            # Market value added (simplified)
            # MVA = Market Cap - Invested Capital
            market_cap = yahoo_data.get('market_cap', 0)
            if market_cap > 0:
                mva = market_cap - invested_capital
                
                # Estimate intrinsic value based on EVA
                # Value = Invested Capital + PV of future EVA
                # Simplified: Assume EVA grows at GDP rate (5%)
                growth_rate = 0.05
                perpetuity_value = eva * (1 + growth_rate) / (wacc - growth_rate)
                
                intrinsic_value = invested_capital + perpetuity_value
                
                # Convert to per share value
                shares_outstanding = market_cap / yahoo_data.get('current_price', 1) if yahoo_data.get('current_price', 0) > 0 else 0
                
                if shares_outstanding > 0:
                    eva_value = intrinsic_value / shares_outstanding
                    return eva_value
                    
        except Exception as e:
            logger.warning(f"Error in EVA calculation: {str(e)}")
            return None
        
        return None
    
    def calculate_relative_valuation(self, data):
        """Relative valuation based on industry multiples"""
        try:
            yahoo_data = data.get('yahoo_data', {})
            if not yahoo_data:
                return None
                
            current_price = yahoo_data.get('current_price', 0)
            eps = yahoo_data.get('eps', 0)
            book_value = yahoo_data.get('book_value', 0)
            
            if not eps or not book_value:
                return None
                
            # Get industry average multiples based on sector
            sector_info = data.get('sector_info', {})
            sector = sector_info.get('sector', 'Unknown')
            
            industry_pe = self.get_industry_pe(sector)
            industry_pb = self.get_industry_pb(sector)
            
            if industry_pe and industry_pb:
                # Calculate fair value based on both P/E and P/B
                pe_value = eps * industry_pe
                pb_value = book_value * industry_pb
                
                # Weighted average (more weight to P/E for most companies)
                fair_value = (pe_value * 0.7) + (pb_value * 0.3)
                return fair_value
            elif industry_pe:
                # Use only P/E
                return eps * industry_pe
            elif industry_pb:
                # Use only P/B
                return book_value * industry_pb
            else:
                return None
                
        except Exception as e:
            logger.warning(f"Error in relative valuation: {str(e)}")
            return None
    
    def calculate_graham_number(self, data):
        """Benjamin Graham's valuation formula"""
        try:
            yahoo_data = data.get('yahoo_data', {})
            if not yahoo_data:
                return None
                
            eps = yahoo_data.get('eps', 0)
            book_value = yahoo_data.get('book_value', 0)
            
            if not eps or not book_value:
                return None
                
            # Graham's formula: sqrt(22.5 * EPS * BVPS)
            graham_number = np.sqrt(22.5 * eps * book_value)
            return graham_number
            
        except Exception as e:
            logger.warning(f"Error in Graham number calculation: {str(e)}")
            return None
    
    def get_industry_pe(self, sector):
        """Get industry average P/E ratio based on sector"""
        # In a real implementation, this would fetch from a reliable source or database
        industry_pe_map = {
            'Finance': 20.0,
            'Process Industries': 18.0,
            'Technology Services': 25.0,
            'Producer Manufacturing': 16.0,
            'Distribution Services': 15.0,
            'Industrial Services': 17.0,
            'Consumer Durables': 22.0,
            'Energy Minerals': 15.0,
            'Health Technology': 28.0,
            'Commercial Services': 19.0,
            'Health Services': 24.0,
            'Retail Trade': 20.0,
            'Consumer Non-Durables': 30.0,
            'Transportation': 16.0,
            'Non-Energy Minerals': 12.0,
            'Consumer Services': 21.0,
            'Utilities': 14.0,
            'Electronic Technology': 26.0,
            'Miscellaneous': 18.0,
            'Communications': 17.0,
            'Government': 12.0,
            'Unknown': 20.0  # Default for unknown sectors
        }
        
        return industry_pe_map.get(sector, 20.0)
    
    def get_industry_pb(self, sector):
        """Get industry average P/B ratio based on sector"""
        # In a real implementation, this would fetch from a reliable source or database
        industry_pb_map = {
            'Finance': 2.5,
            'Process Industries': 2.0,
            'Technology Services': 5.0,
            'Producer Manufacturing': 1.8,
            'Distribution Services': 1.7,
            'Industrial Services': 1.9,
            'Consumer Durables': 3.0,
            'Energy Minerals': 1.5,
            'Health Technology': 4.0,
            'Commercial Services': 2.2,
            'Health Services': 3.5,
            'Retail Trade': 2.8,
            'Consumer Non-Durables': 6.0,
            'Transportation': 1.8,
            'Non-Energy Minerals': 1.2,
            'Consumer Services': 2.5,
            'Utilities': 1.4,
            'Electronic Technology': 4.5,
            'Miscellaneous': 2.0,
            'Communications': 2.2,
            'Government': 1.1,
            'Unknown': 2.5  # Default for unknown sectors
        }
        
        return industry_pb_map.get(sector, 2.5)
    
    def generate_recommendation(self, data):
        """Generate investment recommendation based on collected data"""
        ticker = data['ticker']
        yahoo_data = data.get('yahoo_data', {})
        screener_data = data.get('screener_data', {})
        news_data = data.get('news_data', [])
        sector_info = data.get('sector_info', {})
        
        # Default values
        current_price = yahoo_data.get('current_price', 0)
        pe_ratio = yahoo_data.get('pe_ratio', 0)
        market_cap = yahoo_data.get('market_cap', 0)
        
        # Calculate valuations using multiple models
        valuation_models = {}
        
        # DCF Valuation
        dcf_value = self.calculate_dcf_valuation(data)
        if dcf_value:
            valuation_models['DCF'] = dcf_value
        
        # DDM Valuation
        ddm_value = self.calculate_ddm_valuation(data)
        if ddm_value:
            valuation_models['DDM'] = ddm_value
        
        # EVA Valuation
        eva_value = self.calculate_eva_valuation(data)
        if eva_value:
            valuation_models['EVA'] = eva_value
        
        # Relative Valuation
        relative_value = self.calculate_relative_valuation(data)
        if relative_value:
            valuation_models['Relative'] = relative_value
        
        # Graham Number
        graham_value = self.calculate_graham_number(data)
        if graham_value:
            valuation_models['Graham'] = graham_value
        
        # Calculate target price as weighted average of valuation models
        if valuation_models:
            # Weight models based on their reliability for this company
            weights = {
                'DCF': 0.3 if 'DCF' in valuation_models else 0,
                'DDM': 0.2 if 'DDM' in valuation_models else 0,
                'EVA': 0.2 if 'EVA' in valuation_models else 0,
                'Relative': 0.2 if 'Relative' in valuation_models else 0,
                'Graham': 0.1 if 'Graham' in valuation_models else 0
            }
            
            # Normalize weights if some models are missing
            total_weight = sum(weights.values())
            if total_weight > 0:
                for key in weights:
                    weights[key] /= total_weight
                
                target_price = sum(valuation_models[model] * weights[model] for model in valuation_models if model in weights)
            else:
                # Fallback: 15% upside if no valuation models work
                target_price = current_price * 1.15
        else:
            # Fallback: 15% upside if no valuation models work
            target_price = current_price * 1.15
        
        # Determine recommendation based on multiple factors
        recommendation_factors = []
        
        # 1. Valuation factor (P/E ratio)
        if pe_ratio > 0:
            sector = sector_info.get('sector', 'Unknown')
            industry_pe = self.get_industry_pe(sector)
            if industry_pe:
                pe_ratio_percent = pe_ratio / industry_pe
                if pe_ratio_percent < 0.8:
                    recommendation_factors.append(2)  # Strong buy signal
                elif pe_ratio_percent < 1.0:
                    recommendation_factors.append(1)  # Buy signal
                elif pe_ratio_percent < 1.2:
                    recommendation_factors.append(0)  # Neutral
                else:
                    recommendation_factors.append(-1)  # Sell signal
            else:
                if pe_ratio < 15:
                    recommendation_factors.append(1)  # Buy signal
                elif pe_ratio < 25:
                    recommendation_factors.append(0)  # Neutral
                else:
                    recommendation_factors.append(-1)  # Sell signal
        
        # 2. Price to target ratio
        if current_price > 0 and target_price > 0:
            price_ratio = target_price / current_price
            if price_ratio > 1.3:
                recommendation_factors.append(2)  # Strong buy signal
            elif price_ratio > 1.1:
                recommendation_factors.append(1)  # Buy signal
            elif price_ratio > 0.9:
                recommendation_factors.append(0)  # Neutral
            else:
                recommendation_factors.append(-1)  # Sell signal
        
        # 3. News sentiment
        sentiment, sentiment_score = self.analyze_news_sentiment(news_data, sector_info.get('sector', 'Unknown'))
        if sentiment == "positive":
            recommendation_factors.append(1)
        elif sentiment == "negative":
            recommendation_factors.append(-1)
        else:
            recommendation_factors.append(0)
        
        # Calculate overall recommendation score
        if recommendation_factors:
            avg_score = sum(recommendation_factors) / len(recommendation_factors)
            
            if avg_score > 0.5:
                recommendation = "STRONG BUY"
                horizon = "3-6 months"
                remarks = "Strong fundamentals and positive catalysts"
            elif avg_score > 0:
                recommendation = "BUY"
                horizon = "6-12 months"
                remarks = "Good fundamentals with growth potential"
            elif avg_score > -0.5:
                recommendation = "HOLD"
                horizon = "12-18 months"
                remarks = "Fairly valued with moderate growth prospects"
            else:
                recommendation = "SELL"
                horizon = "3-6 months"
                remarks = "Overvalued or facing headwinds"
        else:
            recommendation = "HOLD"
            horizon = "12-18 months"
            remarks = "Insufficient data for detailed analysis"
        
        # Add sentiment to remarks
        remarks += f" | News sentiment: {sentiment} ({sentiment_score:.2f})"
        
        # Add valuation details
        if valuation_models:
            valuation_details = " | Valuation: " + ", ".join([f"{k}: ₹{v:.2f}" for k, v in valuation_models.items()])
            remarks += valuation_details
        
        return {
            'Ticker': ticker,
            'Recommendation': recommendation,
            'Current Price': f"₹{current_price:.2f}" if current_price else "N/A",
            'Target Price': f"₹{target_price:.2f}" if target_price else "N/A",
            'Upside/Downside': f"{((target_price/current_price)-1)*100:.2f}%" if current_price and target_price and current_price > 0 else "N/A",
            'Time Horizon': horizon,
            'Remarks': remarks,
            'P/E Ratio': f"{pe_ratio:.2f}" if pe_ratio else "N/A",
            'Market Cap': f"₹{market_cap/10000000:.2f} Cr" if market_cap else "N/A",
            'Sector': sector_info.get('sector', 'Unknown'),
            'Industry': sector_info.get('industry', 'Unknown'),
            'Valuation Models': valuation_models
        }
    
    def generate_report(self, recommendations):
        """Generate a comprehensive report for all analyzed stocks"""
        print("\n" + "="*120)
        print("COMPREHENSIVE STOCK ANALYSIS REPORT")
        print("="*120)
        
        # Current date
        print(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n")
        
        # Summary table
        df = pd.DataFrame(recommendations)
        
        # Remove the Valuation Models column for display
        display_df = df.drop('Valuation Models', axis=1, errors='ignore')
        print(display_df.to_string(index=False))
        
        # Detailed analysis for each stock
        print("\n\nDETAILED ANALYSIS:")
        print("="*120)
        
        for rec in recommendations:
            print(f"\n{rec['Ticker']} Analysis:")
            print(f"- Recommendation: {rec['Recommendation']}")
            print(f"- Current Price: {rec['Current Price']}")
            print(f"- Target Price: {rec['Target Price']}")
            print(f"- Upside/Downside: {rec['Upside/Downside']}")
            print(f"- Time Horizon: {rec['Time Horizon']}")
            print(f"- P/E Ratio: {rec['P/E Ratio']}")
            print(f"- Market Cap: {rec['Market Cap']}")
            print(f"- Sector: {rec['Sector']}")
            print(f"- Industry: {rec['Industry']}")
            
            # Show valuation model details if available
            if 'Valuation Models' in rec and rec['Valuation Models']:
                print("- Valuation Models:")
                for model, value in rec['Valuation Models'].items():
                    print(f"  {model}: ₹{value:.2f}")
            
            print(f"- Remarks: {rec['Remarks']}")
            print("-" * 60)
        
        print("\nDISCLAIMER: This analysis is for informational purposes only and should not be considered as investment advice.")
        print("Please conduct your own research and consult with a financial advisor before making investment decisions.")
        
        # Save to CSV
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stock_analysis_report_{timestamp}.csv"
            display_df.to_csv(filename, index=False)
            print(f"\nReport saved to {filename}")
        except Exception as e:
            logger.error(f"Could not save report to CSV: {str(e)}")

def main():
    analyzer = IndianStockAnalyzer()
    
    # Ask for API keys
    print("API Keys (press Enter to skip any):")
    newsapi_key = input("NewsAPI Key: ").strip()
    alphavantage_key = input("AlphaVantage Key: ").strip()
    finnhub_key = input("Finnhub Key: ").strip()
    
    if newsapi_key or alphavantage_key or finnhub_key:
        analyzer.set_api_keys(newsapi_key, alphavantage_key, finnhub_key)
    
    # Get tickers from user
    tickers = analyzer.get_tickers_from_user()
    
    if not tickers:
        print("No valid tickers provided. Exiting.")
        return
    
    # Analyze each ticker
    recommendations = []
    for ticker in tickers:
        try:
            recommendation = analyzer.analyze_stock(ticker)
            recommendations.append(recommendation)
            # Delay to avoid being blocked
            time.sleep(1)
        except Exception as e:
            logger.error(f"Fatal error analyzing {ticker}: {str(e)}")
            continue
    
    # Generate final report
    if recommendations:
        analyzer.generate_report(recommendations)
    else:
        print("No analysis could be completed for the provided tickers.")

if __name__ == "__main__":
    main()
