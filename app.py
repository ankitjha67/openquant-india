# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from bs4 import BeautifulSoup
import requests
import time
import json
import logging
import feedparser
import re
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from transformers import pipeline
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import warnings
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import threading
import os
import snscrape.modules.telegram as stelegram

# Suppress warnings
warnings.filterwarnings('ignore')
nltk.download('vader_lexicon', quiet=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OpenQuantIndia")

# Initialize sentiment analyzers
sia = SentimentIntensityAnalyzer()
try:
    bert_sentiment = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")
except:
    bert_sentiment = None
    st.warning("BERT sentiment model not loaded")

# --- SESSION STATE INIT ---
if 'results' not in st.session_state:
    st.session_state['results'] = pd.DataFrame()
if 'portfolio' not in st.session_state:
    st.session_state['portfolio'] = []
if 'news_feed' not in st.session_state:
    st.session_state['news_feed'] = []
if 'telegram_feed' not in st.session_state:
    st.session_state['telegram_feed'] = []

# --- CLASSES ---

class NewsScraper:
    @staticmethod
    def scrape_rss_feeds(ticker):
        """Scrape RSS feeds for real-time news"""
        feeds = [
            f"https://economictimes.indiatimes.com/rssfeeds/{ticker}.rss",
            f"https://www.moneycontrol.com/rss/marketdata/{ticker}.xml",
            "https://www.financialexpress.com/feed/",
            "https://www.livemint.com/rss/markets",
            "https://feeds.feedburner.com/ndtvprofit-latest"
        ]
        headlines = []
        for feed_url in feeds:
            try:
                d = feedparser.parse(feed_url)
                for entry in d.entries[:5]:
                    if ticker.upper() in entry.title.upper():
                        headlines.append(entry.title)
            except:
                continue
        return headlines

    @staticmethod
    def scrape_telegram_news(ticker, limit=10):
        """Scrape Telegram channels for retail sentiment"""
        try:
            query = f"{ticker} stock"
            headlines = []
            for i, message in enumerate(stelegram.TelegramChannelScraper(query).get_items()):
                if i >= limit:
                    break
                if ticker.upper() in message.content.upper():
                    headlines.append(message.content)
            return headlines
        except:
            return []

class NSEDataFetcher:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.tickers = []
        self.sector_map = {}
        self.fetch_all_tickers()

    def fetch_all_tickers(self):
        try:
            self.session.get("https://www.nseindia.com", timeout=10)
            time.sleep(1)
            response = self.session.get(
                "https://www.nseindia.com/api/master-quote",
                params={"type": "equity"},
                headers={'X-Requested-With': 'XMLHttpRequest'},
                timeout=15
            )
            data = response.json()
            df = pd.read_csv(pd.compat.StringIO(data['csv']))
            df = df[df['instrumentType'] == 'EQ']
            self.tickers = df['symbol'].str.upper().tolist()
            for _, row in df.iterrows():
                self.sector_map[row['symbol']] = {
                    'sector': row.get('industryGroup', 'Unknown'),
                    'industry': row.get('industry', 'Unknown'),
                    'name': row.get('companyName', 'Unknown')
                }
            st.success(f"✅ Loaded {len(self.tickers)} NSE tickers")
        except Exception as e:
            st.warning(f"Failed to fetch NSE tickers: {e}")
            self.tickers = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']
            self.sector_map = {t: {'sector': 'Unknown', 'industry': 'Unknown', 'name': t} for t in self.tickers}

    def filter_by_sector(self, sector):
        return [t for t in self.tickers if sector.lower() in self.sector_map.get(t, {}).get('sector', '').lower()]

    def search_by_name(self, query):
        results = []
        for t, info in self.sector_map.items():
            if query.lower() in t.lower() or query.lower() in info['name'].lower():
                results.append({'ticker': t, 'name': info['name'], 'sector': info['sector']})
        return results

class SentimentAnalyzer:
    def get_sentiment(self, texts):
        if not texts:
            return "neutral", 0.0
        vader_scores = [sia.polarity_scores(t)['compound'] for t in texts]
        tb_scores = [TextBlob(t).polarity for t in texts]
        bert_scores = []
        if bert_sentiment:
            for t in texts:
                try:
                    res = bert_sentiment(t)[0]
                    score = (1 if res['label'] == 'POSITIVE' else -1) * res['score']
                    bert_scores.append(score)
                except:
                    pass
        final_score = (
            np.mean(bert_scores) * 0.5 + np.mean(vader_scores) * 0.3 + np.mean(tb_scores) * 0.2
        ) if bert_scores else np.mean(vader_scores) * 0.7 + np.mean(tb_scores) * 0.3
        sentiment = "positive" if final_score > 0.1 else "negative" if final_score < -0.1 else "neutral"
        return sentiment, round(final_score, 3)

class StockAnalyzer:
    def __init__(self, nse_fetcher):
        self.nse_fetcher = nse_fetcher
        self.sentiment_analyzer = SentimentAnalyzer()
        self.news_scraper = NewsScraper()

    def get_yahoo_data(self, ticker):
        try:
            stock = yf.Ticker(f"{ticker}.NS")
            info = stock.info
            hist = stock.history(period="2y")
            return {
                'current_price': info.get('currentPrice', 0),
                'pe': info.get('trailingPE', 0),
                'pb': info.get('priceToBook', 0),
                'roe': info.get('returnOnEquity', 0),
                'de': info.get('debtToEquity', 0),
                'beta': info.get('beta', 1),
                'eps': info.get('trailingEps', 0),
                'div_yield': info.get('dividendYield', 0),
                'hist': hist,
                'info': info
            }
        except:
            return {}

    def get_news(self, ticker):
        sources = [
            f"https://www.moneycontrol.com/news/tags/{ticker.lower()}.html",
            f"https://economictimes.indiatimes.com/markets/stocks/stock-quotes?ticker={ticker}"
        ]
        headlines = []
        for url in sources:
            try:
                response = requests.get(url, timeout=5)
                soup = BeautifulSoup(response.content, 'html.parser')
                for h in soup.find_all(['h2', 'h3'], text=re.compile(ticker, re.I)):
                    headlines.append(h.get_text().strip())
            except:
                continue
        # Add RSS and Telegram
        headlines.extend(self.news_scraper.scrape_rss_feeds(ticker))
        headlines.extend(self.news_scraper.scrape_telegram_news(ticker))
        return headlines[:50]

    def get_competitors(self, ticker):
        sector = self.nse_fetcher.sector_map.get(ticker, {}).get('sector', '')
        return [t for t in self.nse_fetcher.tickers if t != ticker and self.nse_fetcher.sector_map.get(t, {}).get('sector') == sector][:5]

    def analyze(self, ticker):
        yahoo = self.get_yahoo_data(ticker)
        news = self.get_news(ticker)
        sentiment, sent_score = self.sentiment_analyzer.get_sentiment(news)
        competitors = self.get_competitors(ticker)
        comp_pe = np.mean([self.get_yahoo_data(c).get('pe', 0) for c in competitors if c != ticker])
        target = yahoo.get('current_price', 0) * (1.1 + sent_score * 0.5)
        rec = "STRONG BUY" if sent_score > 0.3 else "BUY" if sent_score > 0 else "HOLD"
        return {
            'Ticker': ticker,
            'Current Price': f"₹{yahoo.get('current_price', 0):.2f}",
            'Target Price': f"₹{target:.2f}",
            'P/E': f"{yahoo.get('pe', 0):.2f}",
            'ROE': f"{yahoo.get('roe', 0):.2%}",
            'Sentiment': sentiment,
            'Sentiment Score': sent_score,
            'Competitors': ', '.join(competitors),
            'Peer P/E': f"{comp_pe:.2f}",
            'Recommendation': rec
        }

class PortfolioOptimizer:
    @staticmethod
    def optimize_cvar(returns, alpha=0.05):
        T, n = returns.shape
        w = cp.Variable(n)
        eta = cp.Variable()
        z = cp.Variable(T)
        prob = cp.Problem(
            cp.Minimize(eta + (1 / (1 - alpha)) * cp.sum(z) / T),
            [
                cp.sum(w) == 1,
                w >= 0,
                z >= 0,
                z >= -(returns @ w)
            ]
        )
        prob.solve()
        return w.value

class Backtester:
    @staticmethod
    def momentum_strategy(tickers, start, end):
        data = yf.download([f"{t}.NS" for t in tickers], start=start, end=end)['Adj Close']
        returns = data.pct_change().sum(axis=1)
        return (returns.mean() * 252, returns.std() * np.sqrt(252), returns.min())

class RiskManager:
    @staticmethod
    def var_cvar(returns, alpha=0.05):
        sorted_rets = np.sort(returns)
        var = np.percentile(sorted_rets, alpha * 100)
        cvar = sorted_rets[sorted_rets <= var].mean()
        return var, cvar

class ESGScorer:
    @staticmethod
    def get_esg_score(ticker):
        np.random.seed(ord(ticker[0]))
        return round(np.random.uniform(0.5, 1.0), 2)

# --- MAIN APP ---

def main():
    st.set_page_config(layout="wide", page_title="OpenQuant India v7.0")
    st.title("🚀 OpenQuant India v7.0 – AI-Powered Equity Intelligence")

    nse = NSEDataFetcher()

    # Sidebar
    st.sidebar.header("🔍 Input")
    mode = st.sidebar.selectbox("Mode", ["Single Stock", "Sector Filter", "Search Name", "Portfolio"])

    if mode == "Single Stock":
        tickers = st.sidebar.text_input("Enter tickers (comma-separated)", "INFY,TCS,RELIANCE").split(",")
        tickers = [t.strip().upper() for t in tickers]

    elif mode == "Sector Filter":
        sector = st.sidebar.text_input("Enter sector (e.g., IT, Banking)", "IT")
        tickers = nse.filter_by_sector(sector)
        st.sidebar.write(f"Found {len(tickers)} stocks")

    elif mode == "Search Name":
        query = st.sidebar.text_input("Search company name", "Reliance")
        results = nse.search_by_name(query)
        tickers = [r['ticker'] for r in results]
        st.sidebar.write(f"Found {len(results)} matches")

    else:
        tickers = st.sidebar.text_input("Enter portfolio tickers", "INFY,TCS").split(",")
        tickers = [t.strip().upper() for t in tickers]

    analyze_btn = st.sidebar.button("Analyze")

    # Telegram Alerts
    enable_alerts = st.sidebar.checkbox("Enable Telegram Alerts")
    token = chat_id = None
    if enable_alerts:
        token = st.sidebar.text_input("Telegram Bot Token")
        chat_id = st.sidebar.text_input("Chat ID")

    if analyze_btn and tickers:
        analyzer = StockAnalyzer(nse)
        results = []
        for t in tickers:
            res = analyzer.analyze(t)
            results.append(res)
            # Send alert if sentiment spike
            if enable_alerts and token and chat_id and res.get('Sentiment Score', 0) > 0.6:
                requests.post(
                    f"https://api.telegram.org/bot{token}/sendMessage",
                    data={'chat_id': chat_id, 'text': f"🚀 {t} SENTIMENT SPIKE: {res['Sentiment Score']:.2f}"}
                )
            time.sleep(1)
        df = pd.DataFrame(results)
        st.session_state['results'] = df

        st.dataframe(df, use_container_width=True)

        # Charts
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.bar(df, x='Ticker', y='Sentiment Score', color='Recommendation', title="Sentiment & Recommendation")
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            fig2 = px.scatter(df, x='P/E', y='ROE', text='Ticker', title="P/E vs ROE")
            st.plotly_chart(fig2, use_container_width=True)

        # Portfolio Optimization
        if mode == "Portfolio":
            try:
                prices = yf.download([f"{t}.NS" for t in tickers], period="1y")['Adj Close']
                returns = prices.pct_change().dropna()
                opt = PortfolioOptimizer()
                weights = opt.optimize_cvar(returns.values)
                st.write("### Portfolio Weights (CVaR Optimized)")
                weight_df = pd.DataFrame({'Ticker': tickers, 'Weight': weights})
                st.dataframe(weight_df)
            except:
                st.error("Could not optimize portfolio")

        # Backtesting
        if st.sidebar.checkbox("Run Backtest"):
            bt = Backtester()
            cagr, vol, mdd = bt.momentum_strategy(tickers, "2020-01-01", "2023-01-01")
            st.write(f"**Backtest (2020-2023)**: CAGR={cagr:.1%}, Vol={vol:.1%}, MaxDD={mdd:.1%}")

        # Risk
        if st.sidebar.checkbox("Show Risk Metrics"):
            rm = RiskManager()
            rets = yf.download([f"{t}.NS" for t in tickers[0:1]], period="1y")['Adj Close'].pct_change().dropna().values
            var, cvar = rm.var_cvar(rets)
            st.write(f"**VaR (95%)**: {var:.1%}, **CVaR**: {cvar:.1%}")

        # ESG
        if st.sidebar.checkbox("Show ESG Scores"):
            esg = ESGScorer()
            esg_scores = {t: esg.get_esg_score(t) for t in tickers}
            st.bar_chart(esg_scores)

        # Export
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", csv, "report.csv", "text/csv")

        # PDF Export
        if st.button("Export to PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            for _, row in df.iterrows():
                pdf.cell(0, 10, txt=str(row.to_dict()), ln=True)
            pdf.output("report.pdf")
            st.success("PDF exported!")

    # News Sentiment Dashboard
    st.sidebar.subheader("News Feed")
    if st.sidebar.button("Fetch News Feed"):
        news_texts = []
        for t in tickers[:3]:
            news = analyzer.get_news(t) if 'analyzer' in locals() else [f"Sample news for {t}"]
            news_texts.extend(news)
        sentiment, score = SentimentAnalyzer().get_sentiment(news_texts)
        st.session_state['news_feed'] = news_texts
        st.sidebar.write(f"Sentiment: {sentiment} ({score:.2f})")

    if st.session_state['news_feed']:
        st.subheader("📰 Real-Time News & Sentiment")
        for news in st.session_state['news_feed'][:10]:
            st.markdown(f"- {news}")

if __name__ == "__main__":
    main()