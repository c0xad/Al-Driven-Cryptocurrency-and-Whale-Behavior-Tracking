import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import logging
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import ccxt
import plotly.graph_objects as go
from typing import Dict, List, Tuple
import warnings
import nest_asyncio
import asyncio
import aiohttp
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, Concatenate
from xgboost import XGBClassifier
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import statsmodels.api as sm
from scipy.stats import norm
from tpot import TPOTClassifier, TPOTRegressor
import h2o
from h2o.automl import H2OAutoML
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import gym
from gym import spaces
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TradingEnvironment(gym.Env):
    def __init__(self, df, initial_balance=10000):
        super().__init__()
        self.df = df
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0
        
        # Action space: 0 (sell), 1 (hold), 2 (buy)
        self.action_space = spaces.Discrete(3)
        
        # Observation space: price, volume, technical indicators
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )
        
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        return self._get_observation()
    
    def _get_observation(self):
        obs = np.array([
            self.df.iloc[self.current_step]['close'],
            self.df.iloc[self.current_step]['volume'],
            self.df.iloc[self.current_step]['rsi'],
            self.df.iloc[self.current_step]['macd'],
            self.df.iloc[self.current_step]['macd_signal'],
            self.df.iloc[self.current_step]['macd_hist'],
            self.df.iloc[self.current_step]['bb_upper'],
            self.df.iloc[self.current_step]['bb_middle'],
            self.df.iloc[self.current_step]['bb_lower'],
            self.df.iloc[self.current_step]['ema_9'],
            self.df.iloc[self.current_step]['ema_21'],
            self.df.iloc[self.current_step]['atr'],
            self.df.iloc[self.current_step]['adx'],
            self.balance,
            self.position
        ], dtype=np.float32)
        return obs
    
    def step(self, action):
        done = self.current_step >= len(self.df) - 1
        if done:
            return self._get_observation(), 0, done, {}
            
        current_price = self.df.iloc[self.current_step]['close']
        next_price = self.df.iloc[self.current_step + 1]['close']
        
        # Execute trading action
        reward = 0
        if action == 2:  # Buy
            if self.position == 0:
                self.position = self.balance / current_price
                self.balance = 0
        elif action == 0:  # Sell
            if self.position > 0:
                self.balance = self.position * current_price
                self.position = 0
                
        # Calculate reward
        portfolio_value = self.balance + (self.position * next_price)
        reward = (portfolio_value - self.initial_balance) / self.initial_balance
        
        self.current_step += 1
        return self._get_observation(), reward, done, {}

class AdvancedTradingModel:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.h2o_automl = None
        self.tpot_model = None
        self.dqn_model = None
        self.ppo_model = None
        self.ensemble_model = None
        
    def prepare_features(self):
        # Advanced feature engineering
        self.data['price_momentum'] = self.data['close'].pct_change(5)
        self.data['volume_momentum'] = self.data['volume'].pct_change(5)
        self.data['volatility'] = self.data['close'].rolling(window=20).std()
        self.data['log_return'] = np.log(self.data['close']/self.data['close'].shift(1))
        
        # Custom technical indicators
        self.data['triple_ema'] = self.data['close'].ewm(span=8).mean() * 3 - \
                                 self.data['close'].ewm(span=16).mean() * 2 + \
                                 self.data['close'].ewm(span=24).mean()
        
        self.data['keltner_upper'] = self.data['ema_21'] + (self.data['atr'] * 2)
        self.data['keltner_lower'] = self.data['ema_21'] - (self.data['atr'] * 2)
        
    def train_automl_models(self):
        # Initialize H2O
        h2o.init()
        
        # Convert data to H2O frame
        train_data = h2o.H2OFrame(self.data.dropna())
        
        # Configure H2O AutoML
        self.h2o_automl = H2OAutoML(
            max_models=20,
            seed=42,
            balance_classes=True,
            max_runtime_secs=3600
        )
        
        # Train TPOT
        self.tpot_model = TPOTClassifier(
            generations=50,
            population_size=50,
            cv=5,
            random_state=42,
            verbosity=2,
            n_jobs=-1
        )
        
        X = self.data.drop(['target'], axis=1)
        y = self.data['target']
        self.tpot_model.fit(X, y)
        
    def train_rl_models(self):
        # Create custom environment
        env = make_vec_env(
            lambda: TradingEnvironment(self.data),
            n_envs=4,
            seed=42
        )
        
        # Initialize and train DQN
        self.dqn_model = DQN(
            "MlpPolicy",
            env,
            learning_rate=1e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=64,
            tau=0.001,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            verbose=1
        )
        
        self.dqn_model.learn(total_timesteps=100000)
        
        # Initialize and train PPO
        self.ppo_model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=False,
            sde_sample_freq=-1,
            target_kl=None,
            verbose=1
        )
        
        self.ppo_model.learn(total_timesteps=100000)
        
    def create_ensemble(self):
        # Create base models
        estimators = [
            ('rf', RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )),
            ('gb', GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )),
            ('xgb', XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ))
        ]
        
        # Create stacking ensemble
        self.ensemble_model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(),
            cv=5
        )
        
    def predict(self, X):
        # Combine predictions from all models
        automl_pred = self.h2o_automl.predict(h2o.H2OFrame(X))
        tpot_pred = self.tpot_model.predict_proba(X)
        dqn_pred = self.dqn_model.predict(X)
        ppo_pred = self.ppo_model.predict(X)
        ensemble_pred = self.ensemble_model.predict_proba(X)
        
        # Weighted ensemble of predictions
        final_pred = (
            0.2 * automl_pred +
            0.2 * tpot_pred +
            0.2 * dqn_pred +
            0.2 * ppo_pred +
            0.2 * ensemble_pred
        )
        
        return final_pred

class WhaleTracker:
    def __init__(self, api_key: str, solscan_api_key: str, min_whale_amount: float = 100000, initial_portfolio_value: float = 100000):
        self.api_key = api_key
        self.solscan_api_key = solscan_api_key
        self.min_whale_amount = min_whale_amount
        self.initial_portfolio_value = initial_portfolio_value
        self.exchange = ccxt.binance({
            'apiKey': '5941798b6a5c4fb17c4e1f5cef2c34007a312bd6fbdf82d6a5247566daad3c67',
            'secret': '2079cb861ae654ecd5ee16687e5d3bdc0fc0bfac4d9f3f0070ca0dcec96cb8b2',
            'options': {
                'defaultType': 'future',
                'testnet': True,
                'adjustForTimeDifference': True
            }
        })

        # Initialize FinGPT
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-2')
        self.model = AutoModelForSequenceClassification.from_pretrained(
            'microsoft/phi-2'
        ).to(self.device)

        self.historical_data = {}
        self.sentiment_scores = {}

        self.dynamic_threshold = min_whale_amount
        self.whale_addresses = set()
        self.transaction_cache = {}
        self.wallet_clusters = {}

        # Add rolling window parameters
        self.window_size = timedelta(hours=24)
        self.update_interval = timedelta(minutes=15)
        self.last_threshold_update = datetime.now()

        # Add technical indicators parameters
        self.price_impact_window = timedelta(minutes=5)
        self.mfi_period = 14
        self.vpt_period = 14
        self.volatility_window = 24

        # Historical data storage
        self.price_cache = {}
        self.volume_cache = {}
        self.technical_indicators = {}

        # Risk management parameters
        self.max_position_size = 0.2  # Maximum 20% of portfolio per position
        self.min_position_size = 0.05  # Minimum 5% of portfolio per position
        self.stop_loss_threshold = 0.05  # 5% stop loss
        self.take_profit_threshold = 0.15  # 15% take profit
        self.max_drawdown_limit = 0.25  # 25% maximum drawdown
        self.volatility_adjustment = 0.5  # Volatility scaling factor

        # Portfolio management
        self.portfolio = {}
        self.portfolio_history = []
        self.rebalance_threshold = 0.1  # 10% deviation triggers rebalancing
        self.last_rebalance = datetime.now()
        self.rebalance_interval = timedelta(days=7)

        # Risk metrics
        self.var_confidence = 0.95
        self.var_window = 30
        self.position_correlations = {}

        # Add caching and threading
        self.cache_lock = Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.cache_timeout = 60  # 1 minute cache timeout

        # Add memory management
        self.max_cache_size = 1000
        self.cleanup_threshold = 0.8  # Clean when 80% full

        # Initialize LSTM model for time-series prediction
        self.lstm_model = self.build_lstm_model()
        # Initialize XGBoost model for classification
        self.xgb_model = XGBClassifier()
        # Initialize reinforcement learning agent
        self.rl_agent = self.build_rl_agent()

        # Initialize variables for multi-factor model
        self.economic_factors = {}
        self.historical_returns = {}
        self.volatility_series = {}
        self.liquidity_series = {}
        # Initialize Bayesian prior probabilities
        self.prior_probabilities = {'BUY': 0.5, 'SELL': 0.5}

    def build_lstm_model(self):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(self.time_steps, self.num_features)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model

    def build_rl_agent(self):
        env = make_vec_env(self.create_trading_env, n_envs=1)
        agent = PPO('MlpPolicy', env, verbose=0)
        return agent

    def train_models(self, data: pd.DataFrame):
        # Prepare data for LSTM
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data[['close']])
        X_lstm, y_lstm = self.create_lstm_dataset(scaled_data)
        self.lstm_model.fit(X_lstm, y_lstm, epochs=10, batch_size=32)

        # Prepare data for XGBoost
        X_xgb = data.drop(['target'], axis=1)
        y_xgb = data['target']
        self.xgb_model.fit(X_xgb, y_xgb)

        # Train reinforcement learning agent
        self.rl_agent.learn(total_timesteps=10000)

    def create_lstm_dataset(self, data):
        X, y = [], []
        for i in range(self.time_steps, len(data)):
            X.append(data[i - self.time_steps:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    def cleanup_cache(self):
        """Clean up old cached data"""
        if len(self.transaction_cache) > self.max_cache_size * self.cleanup_threshold:
            oldest_first = sorted(
                self.transaction_cache.items(),
                key=lambda x: x[1]['timestamp']
            )
            to_remove = len(oldest_first) - int(self.max_cache_size * 0.5)
            for i in range(to_remove):
                del self.transaction_cache[oldest_first[i][0]]

    @lru_cache(maxsize=100)
    async def fetch_whale_transactions(self, symbol: str) -> pd.DataFrame:
        """Cached and async version of whale transaction fetching"""
        async with self.session.get(
            f"https://api.birdeye.so/v1/token/{symbol}/large_transactions",
            headers={'X-API-KEY': self.api_key}
        ) as response:
            data = await response.json()
            df = pd.DataFrame(data['data'])
            return df[df['amount'] >= self.min_whale_amount]

    def analyze_sentiment(self, texts: List[str]) -> float:
        """Analyze sentiment using FinGPT"""
        sentiments = []

        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                sentiment = torch.softmax(outputs.logits, dim=1)
                sentiments.append(sentiment[0][1].item())  # Positive sentiment score

        return np.mean(sentiments)

    def calculate_volume_anomaly(self, volumes: np.array) -> float:
        """Detect volume anomalies using Z-score"""
        if len(volumes) < 2:
            return 0
        z_score = (volumes[-1] - np.mean(volumes)) / np.std(volumes)
        return z_score

    def update_dynamic_threshold(self, symbol: str) -> float:
        """Dynamically adjust whale transaction threshold based on recent activity"""
        if symbol not in self.transaction_cache:
            return self.dynamic_threshold

        recent_txs = self.transaction_cache[symbol]
        if not recent_txs:
            return self.dynamic_threshold

        # Calculate 75th percentile of recent transaction volumes
        recent_volumes = [tx['amount'] for tx in recent_txs]
        new_threshold = np.percentile(recent_volumes, 75)

        # Smooth threshold changes
        self.dynamic_threshold = 0.7 * self.dynamic_threshold + 0.3 * new_threshold
        logging.info(f"Updated dynamic threshold for {symbol}: {self.dynamic_threshold}")
        return self.dynamic_threshold

    def fetch_solscan_data(self, symbol: str) -> pd.DataFrame:
        """Fetch additional on-chain data from Solscan"""
        endpoint = f"https://public-api.solscan.io/token/{symbol}/holders"
        headers = {'token': self.solscan_api_key}

        try:
            response = requests.get(endpoint, headers=headers)
            data = response.json()
            df = pd.DataFrame(data['data'])

            # Extract significant holders
            significant_holders = df[df['amount'] > self.dynamic_threshold]

            # Update whale addresses set
            self.whale_addresses.update(set(significant_holders['address']))

            return significant_holders
        except Exception as e:
            logging.error(f"Error fetching Solscan data: {e}")
            return pd.DataFrame()

    def analyze_wallet_clusters(self, transactions_df: pd.DataFrame) -> Dict:
        """Analyze wallet clustering and transaction patterns"""
        if transactions_df.empty:
            return {}

        clusters = {}
        graph = defaultdict(list)

        # Build transaction graph
        for _, tx in transactions_df.iterrows():
            graph[tx['from_address']].append(tx['to_address'])

        # Identify connected components (clusters)
        visited = set()

        def dfs(address, cluster_id):
            visited.add(address)
            clusters[address] = cluster_id
            for neighbor in graph[address]:
                if neighbor not in visited:
                    dfs(neighbor, cluster_id)

        cluster_id = 0
        for address in graph:
            if address not in visited:
                dfs(address, cluster_id)
                cluster_id += 1

        return clusters

    def fetch_aggregated_transactions(self, symbol: str, interval: str = '1h') -> pd.DataFrame:
        """Fetch and aggregate whale transactions with smoothing"""
        # Fetch raw transactions
        raw_df = self.fetch_whale_transactions(symbol)
        solscan_df = self.fetch_solscan_data(symbol)

        if raw_df.empty:
            return pd.DataFrame()

        # Update dynamic threshold
        if datetime.now() - self.last_threshold_update > self.update_interval:
            self.update_dynamic_threshold(symbol)
            self.last_threshold_update = datetime.now()

        # Merge transaction data
        merged_df = pd.merge(
            raw_df,
            solscan_df,
            left_on='address',
            right_on='address',
            how='left'
        )

        # Apply time-based grouping
        merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'])
        grouped_df = merged_df.set_index('timestamp').groupby(pd.Grouper(freq=interval))

        # Calculate aggregated metrics
        aggregated = pd.DataFrame({
            'volume': grouped_df['amount'].sum(),
            'tx_count': grouped_df.size(),
            'unique_addresses': grouped_df['address'].nunique(),
            'avg_amount': grouped_df['amount'].mean()
        }).reset_index()

        # Apply exponential smoothing
        aggregated['smooth_volume'] = aggregated['volume'].ewm(span=6).mean()

        # Cache transactions for threshold updates
        self.transaction_cache[symbol] = merged_df.to_dict('records')

        # Analyze wallet clusters
        self.wallet_clusters[symbol] = self.analyze_wallet_clusters(merged_df)

        return aggregated

    def get_transaction_metrics(self, symbol: str) -> Dict:
        """Calculate comprehensive transaction metrics"""
        if symbol not in self.transaction_cache:
            return {}

        txs = self.transaction_cache[symbol]
        clusters = self.wallet_clusters.get(symbol, {})

        metrics = {
            'total_volume': sum(tx['amount'] for tx in txs),
            'unique_wallets': len(set(tx['address'] for tx in txs)),
            'cluster_count': len(set(clusters.values())),
            'avg_cluster_size': len(clusters) / (len(set(clusters.values())) or 1),
            'whale_concentration': len(self.whale_addresses) / (len(set(tx['address'] for tx in txs)) or 1)
        }

        return metrics

    def generate_signals(self, symbol: str) -> Dict:
        """Enhanced signal generation with additional metrics"""
        # Fetch aggregated data
        agg_df = self.fetch_aggregated_transactions(symbol)

        if agg_df.empty:
            return {'signal': 'NEUTRAL', 'confidence': 0.0}

        # Calculate metrics
        metrics = self.get_transaction_metrics(symbol)

        # Analyze volume anomalies with smoothed data
        volume_anomaly = self.calculate_volume_anomaly(agg_df['smooth_volume'].values)

        # Get sentiment score with enhanced context
        sentiment_score = self.analyze_sentiment([
            f"Latest news and social media sentiment for {symbol}",
            f"Whale wallet activity and clustering for {symbol}",
            f"Market momentum and volume trends for {symbol}"
        ])

        # Enhanced signal generation
        signal_strength = (
            0.3 * volume_anomaly +
            0.3 * sentiment_score +
            0.2 * metrics['whale_concentration'] +
            0.2 * (metrics['cluster_count'] / (metrics['unique_wallets'] or 1))
        )

        # Generate signal with confidence
        if signal_strength > 0.7:
            return {
                'signal': 'BUY',
                'confidence': signal_strength,
                'metrics': metrics,
                'volume_anomaly': volume_anomaly,
                'sentiment': sentiment_score
            }
        elif signal_strength < -0.7:
            return {
                'signal': 'SELL',
                'confidence': abs(signal_strength),
                'metrics': metrics,
                'volume_anomaly': volume_anomaly,
                'sentiment': sentiment_score
            }
        else:
            return {
                'signal': 'NEUTRAL',
                'confidence': abs(signal_strength),
                'metrics': metrics,
                'volume_anomaly': volume_anomaly,
                'sentiment': sentiment_score
            }

    def visualize_metrics(self, symbol: str):
        """Visualize trading metrics and whale movements"""
        whale_df = self.fetch_whale_transactions(symbol)

        if whale_df.empty:
            logging.warning(f"No whale transactions found for {symbol}")
            return

        fig = go.Figure()

        # Plot whale transaction volumes
        fig.add_trace(go.Scatter(
            x=whale_df['timestamp'],
            y=whale_df['amount'],
            mode='markers',
            name='Whale Transactions',
            marker=dict(size=10)
        ))

        fig.update_layout(
            title=f'Whale Transactions for {symbol}',
            xaxis_title='Time',
            yaxis_title='Transaction Amount',
            template='plotly_dark'
        )

        fig.show()

    def backtest(self, symbol: str, days: int = 30) -> Dict:
        """Backtest trading strategy"""
        start_date = datetime.now() - timedelta(days=days)

        signals = []
        returns = []

        while (start_date < datetime.now()):
            signal = self.generate_signals(symbol)
            signals.append(signal)

            # Simulate returns (replace with actual historical price data)
            returns.append(np.random.normal(0, 1))

            start_date += timedelta(days=1)

        return {
            'total_signals': len(signals),
            'buy_signals': len([s for s in signals if s['signal'] == 'BUY']),
            'sell_signals': len([s for s in signals if s['signal'] == 'SELL']),
            'avg_confidence': np.mean([s['confidence'] for s in signals]),
            'simulated_returns': np.mean(returns)
        }

    def calculate_price_impact(self, symbol: str, transaction_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price impact of whale transactions"""
        # Fetch minute-level price data from Binance
        timeframes = {
            '1m': self.price_impact_window.total_seconds() / 60,
            '5m': self.price_impact_window.total_seconds() / 300
        }

        price_impacts = []

        for _, tx in transaction_df.iterrows():
            tx_time = pd.to_datetime(tx['timestamp'])

            # Fetch price data around transaction
            ohlcv = self.exchange.fetch_ohlcv(
                f"{symbol}/USDT",
                '1m',
                int(tx_time.timestamp() * 1000),
                limit=int(timeframes['1m'])
            )

            if not ohlcv:
                continue

            df_price = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )

            # Calculate price change after transaction
            pre_price = df_price['close'].iloc[0]
            post_price = df_price['close'].iloc[-1]
            price_change = (post_price - pre_price) / pre_price

            price_impacts.append({
                'timestamp': tx_time,
                'amount': tx['amount'],
                'price_impact': price_change,
                'volume_ratio': tx['amount'] / df_price['volume'].mean()
            })

        return pd.DataFrame(price_impacts)

    @lru_cache(maxsize=100)
    def calculate_technical_indicators(self, symbol: str) -> Dict:
        """Cached version of technical indicators"""
        with self.cache_lock:
            if symbol in self.technical_indicators:
                last_update, indicators = self.technical_indicators[symbol]
                if datetime.now() - last_update < timedelta(seconds=self.cache_timeout):
                    return indicators

            indicators = super().calculate_technical_indicators(symbol)
            self.technical_indicators[symbol] = (datetime.now(), indicators)
            return indicators

    def calculate_onchain_metrics(self, symbol: str) -> Dict:
        """Calculate on-chain metrics using Solscan data"""
        try:
            # Fetch token info
            token_info = requests.get(
                f"https://public-api.solscan.io/token/meta?tokenAddress={symbol}",
                headers={'token': self.solscan_api_key}
            ).json()

            # Fetch holder statistics
            holder_stats = requests.get(
                f"https://public-api.solscan.io/token/holders?tokenAddress={symbol}",
                headers={'token': self.solscan_api_key}
            ).json()

            # Calculate metrics
            total_holders = holder_stats['total']
            active_addresses = len([h for h in holder_stats['data'] if h['amount'] > 0])
            concentration_ratio = sum(
                h['amount'] for h in holder_stats['data'][:10]
            ) / token_info['supply']

            return {
                'total_holders': total_holders,
                'active_addresses': active_addresses,
                'concentration_ratio': concentration_ratio,
                'holder_change_24h': holder_stats.get('change24h', 0),
                'transfer_volume_24h': token_info.get('volume24h', 0)
            }
        except Exception as e:
            logging.error(f"Error calculating on-chain metrics: {e}")
            return {}

    def generate_enhanced_signals(self, symbol: str) -> Dict:
        """Generate trading signals using multiple factors"""
        # Fetch all necessary data
        agg_df = self.fetch_aggregated_transactions(symbol)
        if agg_df.empty:
            return {'signal': 'NEUTRAL', 'confidence': 0.0}

        price_impacts = self.calculate_price_impact(symbol, agg_df)
        technical_indicators = self.calculate_technical_indicators(symbol)
        onchain_metrics = self.calculate_onchain_metrics(symbol)

        # Calculate base metrics
        volume_anomaly = self.calculate_volume_anomaly(agg_df['smooth_volume'].values)
        sentiment_score = self.analyze_sentiment([
            f"Latest news and social media sentiment for {symbol}",
            f"Whale wallet activity and clustering for {symbol}",
            f"Market momentum and volume trends for {symbol}"
        ])

        # Calculate factor scores
        factors = {
            'volume_anomaly': {
                'score': volume_anomaly,
                'weight': 0.2
            },
            'sentiment': {
                'score': sentiment_score,
                'weight': 0.15
            },
            'price_impact': {
                'score': np.mean(price_impacts['price_impact']) if not price_impacts.empty else 0,
                'weight': 0.15
            },
            'technical': {
                'score': (
                    (technical_indicators['mfi'] > 70) * 1 +
                    (technical_indicators['vpt'] > 0) * 1 +
                    (technical_indicators['adl'] > 0) * 1
                ) / 3,
                'weight': 0.25
            },
            'onchain': {
                'score': (
                    (onchain_metrics['holder_change_24h'] > 0) * 1 +
                    (onchain_metrics['concentration_ratio'] < 0.5) * 1
                ) / 2,
                'weight': 0.25
            }
        }

        # LSTM prediction
        lstm_prediction = self.predict_with_lstm(symbol)
        # XGBoost prediction
        xgb_prediction = self.predict_with_xgboost(symbol)
        # Reinforcement learning action
        rl_action = self.rl_agent.predict(self.get_environment_state(symbol))[0]

        # Combine signals
        signal_strength = sum(f['score'] * f['weight'] for f in factors.values())
        signal_strength += (
            0.2 * lstm_prediction +
            0.2 * xgb_prediction +
            0.2 * (1 if rl_action == 'BUY' else -1)
        )

        # Update economic factors
        self.update_economic_factors(symbol)
        factors['economic'] = {
            'score': (
                self.economic_factors[symbol]['momentum'] * 0.5 +
                self.economic_factors[symbol]['mean_reversion'] * 0.5
            ),
            'weight': 0.2
        }

        # Adjust signal sensitivity
        self.adjust_signal_sensitivity(symbol)
        volatility = self.volatility_series[symbol]
        liquidity = self.liquidity_series[symbol]
        adaptive_weight = np.exp(-volatility) * np.log1p(liquidity)

        # Calculate weighted signal strength
        signal_strength = sum(f['score'] * f['weight'] for f in factors.values())
        signal_strength *= adaptive_weight

        # Bayesian inference for signal confirmation
        posterior_probabilities = self.perform_bayesian_inference(symbol, signal_strength)

        # Determine final signal
        if posterior_probabilities['BUY'] > 0.6:
            final_signal = 'BUY'
            confidence = posterior_probabilities['BUY']
        elif posterior_probabilities['SELL'] > 0.6:
            final_signal = 'SELL'
            confidence = posterior_probabilities['SELL']
        else:
            final_signal = 'NEUTRAL'
            confidence = max(posterior_probabilities.values())

        # Generate detailed signal
        signal_data = {
            'signal': final_signal,
            'confidence': confidence,
            'factors': factors,
            'adaptive_weight': adaptive_weight,
            'posterior_probabilities': posterior_probabilities,
            'metrics': {
                'technical_indicators': technical_indicators,
                'onchain_metrics': onchain_metrics,
                'price_impacts': price_impacts.to_dict() if not price_impacts.empty else {},
                'volume_metrics': agg_df.to_dict()
            }
        }

        return signal_data

    def predict_with_lstm(self, symbol: str) -> float:
        # Prepare data
        data = self.get_recent_price_data(symbol)
        scaled_data = MinMaxScaler().fit_transform(data[['close']])
        X_input = scaled_data[-self.time_steps:].reshape(1, self.time_steps, 1)
        # Predict
        prediction = self.lstm_model.predict(X_input)
        return prediction[0][0]

    def predict_with_xgboost(self, symbol: str) -> float:
        # Prepare data
        features = self.extract_features(symbol)
        # Predict
        prediction = self.xgb_model.predict_proba([features])
        return prediction[0][1]

    def create_trading_env(self):
        # Implement custom trading environment
        pass

    def get_environment_state(self, symbol: str):
        # Prepare state for RL agent
        pass

    def extract_features(self, symbol: str) -> List[float]:
        # Extract features for XGBoost
        data = self.get_feature_data(symbol)
        return data.values.tolist()

    def backtest_with_price_data(self, symbol: str, days: int = 30) -> Dict:
        """Backtest strategy using historical price data"""
        end_timestamp = int(datetime.now().timestamp() * 1000)
        start_timestamp = end_timestamp - (days * 24 * 60 * 60 * 1000)

        # Fetch historical price data
        ohlcv = self.exchange.fetch_ohlcv(
            f"{symbol}/USDT",
            '1h',
            start_timestamp,
            limit=days * 24
        )

        df_price = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )

        positions = []
        returns = []
        current_position = None

        for i in range(len(df_price)):
            timestamp = df_price['timestamp'].iloc[i]
            price = df_price['close'].iloc[i]

            # Generate signal for this timepoint
            signal = self.generate_enhanced_signals(symbol)

            # Trading logic
            if signal['signal'] == 'BUY' and current_position is None:
                current_position = {
                    'entry_price': price,
                    'entry_time': timestamp,
                    'confidence': signal['confidence']
                }
                positions.append(current_position)
            elif signal['signal'] == 'SELL' and current_position is not None:
                returns.append({
                    'entry_price': current_position['entry_price'],
                    'exit_price': price,
                    'return': (price - current_position['entry_price']) / current_position['entry_price'],
                    'confidence': current_position['confidence'],
                    'hold_time': timestamp - current_position['entry_time']
                })
                current_position = None

        # Calculate performance metrics
        if returns:
            performance = {
                'total_trades': len(returns),
                'win_rate': len([r for r in returns if r['return'] > 0]) / len(returns),
                'avg_return': np.mean([r['return'] for r in returns]),
                'sharpe_ratio': (
                    np.mean([r['return'] for r in returns]) /
                    np.std([r['return'] for r in returns])
                    if len(returns) > 1 else 0
                ),
                'avg_hold_time': np.mean([r['hold_time'] for r in returns]) / (1000 * 60 * 60),  # Convert to hours
                'returns': returns
            }
        else:
            performance = {
                'total_trades': 0,
                'win_rate': 0,
                'avg_return': 0,
                'sharpe_ratio': 0,
                'avg_hold_time': 0,
                'returns': []
            }

        return performance

    def calculate_position_size(self, symbol: str, signal_strength: float, portfolio_value: float) -> float:
        """Calculate optimal position size based on multiple factors"""
        # Get volatility adjustment
        volatility = self.calculate_technical_indicators(symbol)['volatility']
        volatility_factor = np.exp(-self.volatility_adjustment * volatility)

        # Calculate base position size from signal strength
        base_size = self.min_position_size + (
            (self.max_position_size - self.min_position_size) * abs(signal_strength)
        )

        # Adjust for volatility
        adjusted_size = base_size * volatility_factor

        # Apply portfolio constraints
        max_allowed = min(
            self.max_position_size,
            (portfolio_value * self.max_position_size) / self.calculate_var(symbol)
        )

        final_size = min(adjusted_size, max_allowed)
        final_size = max(final_size, self.min_position_size)

        return final_size

    def calculate_var(self, symbol: str, confidence: float = None) -> float:
        """Calculate Value at Risk for a symbol"""
        if confidence is None:
            confidence = self.var_confidence

        # Fetch historical prices
        ohlcv = self.exchange.fetch_ohlcv(
            f"{symbol}/USDT",
            '1d',
            limit=self.var_window
        )

        prices = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        returns = prices['close'].pct_change().dropna()

        # Calculate VaR using historical simulation
        var = np.percentile(returns, (1 - confidence) * 100)
        return abs(var)

    def update_stop_loss(self, position: Dict, current_price: float) -> Tuple[bool, float]:
        """Dynamic stop-loss adjustment based on price movement"""
        entry_price = position['entry_price']
        highest_price = position.get('highest_price', entry_price)

        # Update highest price
        if current_price > highest_price:
            position['highest_price'] = current_price

            # Trail the stop-loss
            if (current_price - entry_price) / entry_price > 0.1:  # 10% profit
                new_stop_loss = current_price * (1 - self.stop_loss_threshold * 0.5)  # Tighter stop-loss
                position['stop_loss'] = max(position.get('stop_loss', 0), new_stop_loss)

        # Check if stop-loss is triggered
        if current_price < position.get('stop_loss', entry_price * (1 - self.stop_loss_threshold)):
            return True, current_price

        return False, position.get('stop_loss', entry_price * (1 - self.stop_loss_threshold))

    def check_take_profit(self, position: Dict, current_price: float) -> bool:
        """Check if take-profit conditions are met"""
        entry_price = position['entry_price']
        profit_ratio = (current_price - entry_price) / entry_price

        # Dynamic take-profit based on volatility
        symbol = position['symbol']
        volatility = self.calculate_technical_indicators(symbol)['volatility']
        adjusted_tp = self.take_profit_threshold * (1 + volatility)

        return profit_ratio >= adjusted_tp

    def manage_portfolio(self, portfolio_value: float) -> List[Dict]:
        """Manage portfolio positions and generate trading actions"""
        actions = []
        total_exposure = sum(pos['size'] * pos['current_price'] for pos in self.portfolio.values())

        for symbol, position in self.portfolio.items():
            current_price = float(self.exchange.fetch_ticker(f"{symbol}/USDT")['last'])
            position['current_price'] = current_price

            # Check stop-loss and take-profit
            stop_loss_triggered, new_stop_loss = self.update_stop_loss(position, current_price)
            take_profit_triggered = self.check_take_profit(position, current_price)

            if stop_loss_triggered or take_profit_triggered:
                actions.append({
                    'symbol': symbol,
                    'action': 'SELL',
                    'reason': 'stop_loss' if stop_loss_triggered else 'take_profit',
                    'price': current_price,
                    'size': position['size']
                })
                continue

            # Check for rebalancing
            if datetime.now() - self.last_rebalance > self.rebalance_interval:
                target_size = self.calculate_position_size(
                    symbol,
                    position['signal_strength'],
                    portfolio_value
                )

                current_size = position['size']
                size_diff = target_size - current_size

                if abs(size_diff) / current_size > self.rebalance_threshold:
                    actions.append({
                        'symbol': symbol,
                        'action': 'REBALANCE',
                        'size_diff': size_diff,
                        'price': current_price
                    })

        return actions

    async def async_execute_trades(self, actions: List[Dict]) -> None:
        """Async version of trade execution"""
        for action in actions:
            try:
                if action['action'] == 'SELL':
                    await self.executor.submit(
                        self.exchange.create_market_sell_order,
                        f"{action['symbol']}/USDT",
                        action['size']
                    )
                    del self.portfolio[action['symbol']]
                    logging.info(f"Closed position in {action['symbol']}")

                elif action['action'] == 'REBALANCE':
                    if action['size_diff'] > 0:
                        await self.executor.submit(
                            self.exchange.create_market_buy_order,
                            f"{action['symbol']}/USDT",
                            abs(action['size_diff'])
                        )
                    else:
                        await self.executor.submit(
                            self.exchange.create_market_sell_order,
                            f"{action['symbol']}/USDT",
                            abs(action['size_diff'])
                        )

                    self.portfolio[action['symbol']]['size'] += action['size_diff']

            except Exception as e:
                logging.error(f"Error executing trade: {e}")

    def calculate_portfolio_metrics(self) -> Dict:
        """Calculate comprehensive portfolio metrics"""
        if not self.portfolio:
            return {}

        # Calculate returns and volatilities
        returns = []
        volatilities = []
        weights = []

        total_value = sum(pos['size'] * pos['current_price'] for pos in self.portfolio.values())

        for symbol, position in self.portfolio.items():
            # Calculate position returns
            price_data = pd.DataFrame(
                self.exchange.fetch_ohlcv(f"{symbol}/USDT", '1d', limit=30),
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )

            position_returns = price_data['close'].pct_change().dropna()
            position_volatility = position_returns.std() * np.sqrt(252)

            returns.append(position_returns)
            volatilities.append(position_volatility)
            weights.append(position['size'] * position['current_price'] / total_value)

        # Portfolio calculations
        portfolio_return = np.sum([r.mean() * w for r, w in zip(returns, weights)]) * 252
        portfolio_volatility = np.sqrt(
            np.sum([v**2 * w**2 for v, w in zip(volatilities, weights)])
        )

        # Calculate correlations
        correlation_matrix = pd.concat(returns, axis=1).corr()

        return {
            'total_value': total_value,
            'return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': portfolio_return / portfolio_volatility if portfolio_volatility != 0 else 0,
            'correlations': correlation_matrix.to_dict(),
            'var_95': self.calculate_portfolio_var(returns, weights),
            'positions': len(self.portfolio),
            'max_position_size': max(weights),
            'min_position_size': min(weights)
        }

    def calculate_portfolio_var(self, returns: List[pd.Series], weights: List[float]) -> float:
        """Calculate portfolio Value at Risk"""
        portfolio_returns = sum([r * w for r, w in zip(returns, weights)])
        var_95 = np.percentile(portfolio_returns, 5)
        return abs(var_95)

    def rebalance_portfolio(self, portfolio_value: float) -> None:
        """Perform portfolio rebalancing"""
        if datetime.now() - self.last_rebalance < self.rebalance_interval:
            return

        # Generate new signals for all tracked symbols
        new_positions = {}
        for symbol in self.portfolio.keys():
            signal = self.generate_enhanced_signals(symbol)
            if signal['signal'] != 'NEUTRAL':
                new_positions[symbol] = {
                    'size': self.calculate_position_size(
                        symbol,
                        signal['confidence'],
                        portfolio_value
                    ),
                    'signal_strength': signal['confidence']
                }

        # Calculate rebalancing actions
        actions = []
        for symbol, new_pos in new_positions.items():
            if symbol in self.portfolio:
                current_size = self.portfolio[symbol]['size']
                size_diff = new_pos['size'] - current_size

                if abs(size_diff) / current_size > self.rebalance_threshold:
                    actions.append({
                        'symbol': symbol,
                        'action': 'REBALANCE',
                        'size_diff': size_diff
                    })
            else:
                actions.append({
                    'symbol': symbol,
                    'action': 'BUY',
                    'size': new_pos['size']
                })

        # Execute rebalancing trades
        self.execute_trades(actions)
        self.last_rebalance = datetime.now()

        # Update portfolio metrics
        metrics = self.calculate_portfolio_metrics()
        self.portfolio_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics
        })

    async def batch_fetch_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Fetch multiple prices in parallel"""
        async def fetch_price(symbol):
            ticker = await self.executor.submit(
                self.exchange.fetch_ticker,
                f"{symbol}/USDT"
            )
            return symbol, float(ticker['last'])

        tasks = [fetch_price(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        return dict(results)

    async def async_manage_portfolio(self, portfolio_value: float) -> List[Dict]:
        """Async version of portfolio management"""
        if not self.portfolio:
            return []

        # Batch fetch all prices
        prices = await self.batch_fetch_prices(list(self.portfolio.keys()))

        actions = []
        for symbol, position in self.portfolio.items():
            current_price = prices[symbol]
            position['current_price'] = current_price

            # Rest of the portfolio management logic...
            # (Keep existing logic but use cached/batched data)

    async def update_portfolio_value(self) -> float:
        """Update and return current portfolio value"""
        if not self.portfolio:
            return self.initial_portfolio_value

        try:
            # Fetch current prices for all positions
            prices = await self.batch_fetch_prices(list(self.portfolio.keys()))

            # Calculate total portfolio value
            total_value = sum(
                position['size'] * prices[symbol]
                for symbol, position in self.portfolio.items()
            )

            self.portfolio_value = total_value
            return total_value

        except Exception as e:
            logging.error(f"Error updating portfolio value: {e}")
            return self.initial_portfolio_value

    async def async_rebalance_portfolio(self, portfolio_value: float) -> None:
        """Async version of portfolio rebalancing"""
        if datetime.now() - self.last_rebalance < self.rebalance_interval:
            return

        # Generate new signals for all tracked symbols
        new_positions = {}
        for symbol in self.portfolio.keys():
            signal = await self.generate_enhanced_signals(symbol)
            if signal['signal'] != 'NEUTRAL':
                new_positions[symbol] = {
                    'size': self.calculate_position_size(
                        symbol,
                        signal['confidence'],
                        portfolio_value
                    ),
                    'signal_strength': signal['confidence']
                }

        # Calculate rebalancing actions
        actions = []
        for symbol, new_pos in new_positions.items():
            if symbol in self.portfolio:
                current_size = self.portfolio[symbol]['size']
                size_diff = new_pos['size'] - current_size

                if abs(size_diff) / current_size > self.rebalance_threshold:
                    actions.append({
                        'symbol': symbol,
                        'action': 'REBALANCE',
                        'size_diff': size_diff
                    })
            else:
                actions.append({
                    'symbol': symbol,
                    'action': 'BUY',
                    'size': new_pos['size']
                })

        # Execute rebalancing trades
        await self.async_execute_trades(actions)
        self.last_rebalance = datetime.now()

        # Update portfolio metrics
        metrics = await self.async_calculate_metrics()
        self.portfolio_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics
        })

    async def async_calculate_metrics(self) -> Dict:
        """Async version of portfolio metrics calculation"""
        if not self.portfolio:
            return {}

        # Calculate returns and volatilities
        returns = []
        volatilities = []
        weights = []

        # Fetch all prices in parallel
        prices = await self.batch_fetch_prices(list(self.portfolio.keys()))
        total_value = sum(
            pos['size'] * prices[symbol]
            for symbol, pos in self.portfolio.items()
        )

        for symbol, position in self.portfolio.items():
            # Calculate position returns
            ohlcv = await self.executor.submit(
                self.exchange.fetch_ohlcv,
                f"{symbol}/USDT",
                '1d',
                limit=30
            )

            price_data = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )

            position_returns = price_data['close'].pct_change().dropna()
            position_volatility = position_returns.std() * np.sqrt(252)

            returns.append(position_returns)
            volatilities.append(position_volatility)
            weights.append(position['size'] * prices[symbol] / total_value)

        # Portfolio calculations
        if returns and weights:
            portfolio_return = np.sum([r.mean() * w for r, w in zip(returns, weights)]) * 252
            portfolio_volatility = np.sqrt(
                np.sum([v**2 * w**2 for v, w in zip(volatilities, weights)])
            )

            # Calculate correlations
            correlation_matrix = pd.concat(returns, axis=1).corr()

            return {
                'total_value': total_value,
                'return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': portfolio_return / portfolio_volatility if portfolio_volatility != 0 else 0,
                'correlations': correlation_matrix.to_dict(),
                'var_95': await self.async_calculate_portfolio_var(returns, weights),
                'positions': len(self.portfolio),
                'max_position_size': max(weights),
                'min_position_size': min(weights)
            }

        return {
            'total_value': total_value,
            'positions': len(self.portfolio)
        }

    async def async_calculate_portfolio_var(self, returns: List[pd.Series], weights: List[float]) -> float:
        """Async version of portfolio VaR calculation"""
        portfolio_returns = sum([r * w for r, w in zip(returns, weights)])
        var_95 = np.percentile(portfolio_returns, 5)
        return abs(var_95)

    def update_economic_factors(self, symbol: str):
        """Update economic factors like momentum and mean reversion"""
        price_data = self.get_recent_price_data(symbol)
        returns = price_data['close'].pct_change().dropna()
        self.historical_returns[symbol] = returns

        # Calculate momentum
        momentum = returns[-1] - returns.mean()
        # Calculate mean reversion
        mean_reversion = returns.mean() - returns[-1]

        self.economic_factors[symbol] = {
            'momentum': momentum,
            'mean_reversion': mean_reversion
        }

    def adjust_signal_sensitivity(self, symbol: str):
        """Adjust signal sensitivity based on volatility and liquidity"""
        price_data = self.get_recent_price_data(symbol)
        volumes = price_data['volume']
        volatility = price_data['close'].pct_change().rolling(window=10).std().iloc[-1]
        liquidity = volumes.rolling(window=10).mean().iloc[-1]

        self.volatility_series[symbol] = volatility
        self.liquidity_series[symbol] = liquidity

    def perform_bayesian_inference(self, symbol: str, signal_strength: float) -> Dict[str, float]:
        """Refine trade signals using Bayesian inference"""
        # Likelihood functions based on new data
        likelihood_buy = norm.pdf(signal_strength, loc=0.8, scale=0.1)
        likelihood_sell = norm.pdf(signal_strength, loc=-0.8, scale=0.1)

        # Prior probabilities
        prior_buy = self.prior_probabilities['BUY']
        prior_sell = self.prior_probabilities['SELL']

        # Posterior probabilities
        posterior_buy = likelihood_buy * prior_buy
        posterior_sell = likelihood_sell * prior_sell

        # Normalize
        total = posterior_buy + posterior_sell
        posterior_buy /= total
        posterior_sell /= total

        # Update priors for next iteration
        self.prior_probabilities['BUY'] = posterior_buy
        self.prior_probabilities['SELL'] = posterior_sell

        return {'BUY': posterior_buy, 'SELL': posterior_sell}

    def get_recent_price_data(self, symbol: str) -> pd.DataFrame:
        """Fetch recent price data for a symbol"""
        ohlcv = self.exchange.fetch_ohlcv(f"{symbol}/USDT", '1h', limit=500)
        price_data = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        price_data['timestamp'] = pd.to_datetime(price_data['timestamp'], unit='ms')
        return price_data

class OrderBookAnalyzer:
    def __init__(self, max_depth: int = 50, decay_factor: float = 0.95):
        self.max_depth = max_depth
        self.decay_factor = decay_factor
        self.book_cache = {}
        self.lock = Lock()
        self.scaler = MinMaxScaler()
        
    async def fetch_order_book(self, exchange: ccxt.Exchange, symbol: str) -> Dict:
        try:
            order_book = await exchange.fetch_order_book(symbol, self.max_depth)
            return {
                'bids': np.array(order_book['bids']),
                'asks': np.array(order_book['asks']),
                'timestamp': order_book['timestamp']
            }
        except Exception as e:
            logging.error(f"Error fetching order book: {e}")
            return None

    def calculate_liquidity_score(self, book_data: Dict) -> float:
        if book_data is None or 'bids' not in book_data or 'asks' not in book_data:
            return 0.0
            
        bids, asks = book_data['bids'], book_data['asks']
        
        # Calculate weighted depth
        bid_weights = np.exp(-np.arange(len(bids)) * self.decay_factor)
        ask_weights = np.exp(-np.arange(len(asks)) * self.decay_factor)
        
        bid_liquidity = np.sum(bids[:, 1] * bid_weights * bids[:, 0])
        ask_liquidity = np.sum(asks[:, 1] * ask_weights * asks[:, 0])
        
        return (bid_liquidity + ask_liquidity) / 2

    def calculate_price_impact(self, book_data: Dict, order_size: float) -> Tuple[float, float]:
        if book_data is None:
            return 0.0, 0.0
            
        bids, asks = book_data['bids'], book_data['asks']
        
        def calculate_impact(orders, size, side='buy'):
            remaining_size = size
            total_cost = 0
            for price, volume in orders:
                if remaining_size <= 0:
                    break
                filled = min(remaining_size, volume)
                total_cost += filled * price
                remaining_size -= filled
            
            return (total_cost / size - orders[0][0]) / orders[0][0] if size > 0 else 0
            
        buy_impact = calculate_impact(asks, order_size, 'buy')
        sell_impact = calculate_impact(bids, order_size, 'sell')
        
        return buy_impact, sell_impact

    def calculate_spread_metrics(self, book_data: Dict) -> Dict:
        if book_data is None:
            return {'relative_spread': 0.0, 'effective_spread': 0.0}
            
        bids, asks = book_data['bids'], book_data['asks']
        
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid_price = (best_bid + best_ask) / 2
        
        relative_spread = (best_ask - best_bid) / mid_price
        
        # Calculate effective spread using volume-weighted prices
        volume_window = 5
        vw_bid = np.average(bids[:volume_window, 0], weights=bids[:volume_window, 1])
        vw_ask = np.average(asks[:volume_window, 0], weights=asks[:volume_window, 1])
        effective_spread = (vw_ask - vw_bid) / mid_price
        
        return {
            'relative_spread': relative_spread,
            'effective_spread': effective_spread
        }

    @lru_cache(maxsize=100)
    def calculate_order_book_imbalance(self, book_data: Dict, depth: int = 10) -> float:
        if book_data is None:
            return 0.0
            
        bids, asks = book_data['bids'][:depth], book_data['asks'][:depth]
        
        bid_volume = np.sum(bids[:, 1])
        ask_volume = np.sum(asks[:, 1])
        
        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        return imbalance

    def get_liquidity_signals(self, book_data: Dict, trade_size: float) -> Dict:
        """Compute comprehensive liquidity signals for trading decisions"""
        liquidity_score = self.calculate_liquidity_score(book_data)
        buy_impact, sell_impact = self.calculate_price_impact(book_data, trade_size)
        spread_metrics = self.calculate_spread_metrics(book_data)
        imbalance = self.calculate_order_book_imbalance(book_data)
        
        # Normalize metrics for consistent scaling
        metrics = np.array([[liquidity_score, buy_impact, sell_impact, 
                           spread_metrics['relative_spread'], imbalance]])
        normalized_metrics = self.scaler.fit_transform(metrics)
        
        return {
            'liquidity_score': normalized_metrics[0][0],
            'buy_impact': normalized_metrics[0][1],
            'sell_impact': normalized_metrics[0][2],
            'relative_spread': normalized_metrics[0][3],
            'order_imbalance': normalized_metrics[0][4],
            'raw_liquidity': liquidity_score,
            'effective_spread': spread_metrics['effective_spread']
        }

    async def monitor_liquidity(self, exchange: ccxt.Exchange, symbol: str, 
                              interval: int = 60, trade_size: float = 1.0) -> None:
        """Continuous liquidity monitoring with async implementation"""
        while True:
            try:
                book_data = await self.fetch_order_book(exchange, symbol)
                signals = self.get_liquidity_signals(book_data, trade_size)
                
                with self.lock:
                    self.book_cache[symbol] = {
                        'timestamp': datetime.now(),
                        'signals': signals,
                        'raw_data': book_data
                    }
                
                logging.info(f"Liquidity signals for {symbol}: {signals}")
                await asyncio.sleep(interval)
                
            except Exception as e:
                logging.error(f"Error in liquidity monitoring: {e}")
                await asyncio.sleep(interval)

async def main():
    tracker = WhaleTracker(
        api_key='c62a9f5c66814a2fb9a93bad4c2bc63a',
        solscan_api_key='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjcmVhdGVkQXQiOjE3MzAzMTIzMTM4OTAsImVtYWlsIjoic2V5aGdhbGliaW5iZXlpdGlAZ21haWwuY29tIiwiYWN0aW9uIjoidG9rZW4tYXBpIiwiYXBpVmVyc2lvbiI6InYyIiwiaWF0IjoxNzMwMzEyMzEzfQ.bNjMGEm_wgOPvA1cycy8jtf1xU6KAVj5DEkrBZW8JWk',
        min_whale_amount=100000,
        initial_portfolio_value=100000  # Set initial portfolio value
    )

    async with aiohttp.ClientSession() as session:
        tracker.session = session
        while True:
            try:
                # Update portfolio value first
                portfolio_value = await tracker.update_portfolio_value()

                # Parallel processing of portfolio management tasks
                position_task = asyncio.create_task(
                    tracker.async_manage_portfolio(portfolio_value)
                )
                rebalance_task = asyncio.create_task(
                    tracker.async_rebalance_portfolio(portfolio_value)
                )
                metrics_task = asyncio.create_task(
                    tracker.async_calculate_metrics()
                )

                # Wait for all tasks to complete
                results = await asyncio.gather(
                    position_task,
                    rebalance_task,
                    metrics_task,
                    return_exceptions=True
                )

                # Handle results
                actions = results[0]  # Position task result
                if isinstance(actions, Exception):
                    logging.error(f"Position management error: {actions}")
                    continue

                await tracker.async_execute_trades(actions)

                metrics = results[2]  # Metrics task result
                if not isinstance(metrics, Exception):
                    logging.info(f"Portfolio metrics: {metrics}")

                # Log current portfolio value
                logging.info(f"Current portfolio value: {portfolio_value}")

                await asyncio.sleep(1)  # Reduced from 300 to 1 second

            except Exception as e:
                logging.error(f"Error in main loop: {e}")
                await asyncio.sleep(1)

if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.run(main())