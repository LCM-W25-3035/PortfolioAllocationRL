import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from stable_baselines3 import DDPG


class DDPGPortfolioRecommender:
    """
    A portfolio recommender using DDPG model with future sell signal simulation
    """

    def __init__(
        self, model_path, data_path, max_stocks=100, lookback=30, feature_dimension=7
    ):
        print(f"Loading model from: {model_path}")
        self.model = DDPG.load(model_path)
        self.max_stocks = max_stocks
        self.lookback = lookback
        self.feature_dimension = feature_dimension

        print(f"Loading data from: {data_path}")
        # Load the most recent data for prediction
        data = pd.read_csv(data_path)
        data["date"] = pd.to_datetime(data["date"])

        # Get the most recent date
        self.latest_date = data["date"].max()
        print(f"Latest data date: {self.latest_date}")

        # Select top stocks by volume
        recent_data = data[data["date"] >= (self.latest_date - pd.DateOffset(days=30))]
        avg_volumes = recent_data.groupby("tic")["volume"].mean()
        top_tickers = avg_volumes.nlargest(max_stocks).index.tolist()

        # Get the list of tickers
        self.tickers = sorted(top_tickers)
        print(f"Number of tickers selected: {len(self.tickers)}")

        # Store the latest prices
        latest_data = data[data["date"] == self.latest_date]
        self.latest_prices = {
            row["tic"]: row["close"]
            for _, row in latest_data.iterrows()
            if row["tic"] in self.tickers
        }

        # Track previous recommendations and purchase history
        self.previous_allocations = {}
        self.purchase_history = {}

        # Add technical indicators for prediction
        print("Preparing features...")
        self._prepare_features(data)
        print("Initialization complete!")

    def _prepare_features(self, data):
        """Prepare features for the model prediction"""
        # Filter for only the tickers we're using
        filtered_data = data[data["tic"].isin(self.tickers)]

        # Get the most recent dates for feature calculation
        recent_dates = sorted(filtered_data["date"].unique())[-self.lookback :]
        recent_data = filtered_data[filtered_data["date"].isin(recent_dates)]

        # Calculate features (matching the training features)
        features = np.zeros(
            (self.max_stocks, self.lookback, self.feature_dimension), dtype=np.float16
        )

        print(
            f"Calculating features for {len(self.tickers)} stocks with {self.feature_dimension} features..."
        )
        for i, ticker in enumerate(self.tickers):
            if i >= self.max_stocks:
                break

            ticker_data = recent_data[recent_data["tic"] == ticker].sort_values("date")
            if len(ticker_data) == self.lookback:
                # Calculate basic features
                returns = ticker_data["close"].pct_change().fillna(0).astype(np.float16).values
                volume_ma = (
                    ticker_data["volume"]
                    .rolling(window=10, min_periods=1)
                    .mean()
                    .values
                )
                price_ma = (
                    ticker_data["close"].rolling(window=20, min_periods=1).mean().values
                )

                # RSI calculation
                delta = ticker_data["close"].diff().fillna(0)
                gain = (delta.clip(lower=0)).rolling(window=14, min_periods=1).mean()
                loss = (-delta.clip(upper=0)).rolling(window=14, min_periods=1).mean()
                rs = gain / (loss + 1e-8)
                rsi = 100 - (100 / (1 + rs)).values

                # MACD
                exp1 = ticker_data["close"].ewm(span=12, adjust=False).mean()
                exp2 = ticker_data["close"].ewm(span=26, adjust=False).mean()
                macd = (exp1 - exp2).values

                # Volatility
                volatility = (
                    ticker_data["close"]
                    .pct_change()
                    .rolling(window=30, min_periods=1)
                    .std()
                    .fillna(0)
                    .values
                )

                # Momentum
                momentum = ticker_data["close"].pct_change(periods=10).fillna(0).values

                # Combine the features (using only feature_dimension number of features)
                for j in range(self.lookback):
                    if self.feature_dimension >= 1:
                        features[i, j, 0] = returns[j]
                    if self.feature_dimension >= 2:
                        features[i, j, 1] = volume_ma[j]
                    if self.feature_dimension >= 3:
                        features[i, j, 2] = price_ma[j]
                    if self.feature_dimension >= 4:
                        features[i, j, 3] = rsi[j]
                    if self.feature_dimension >= 5:
                        features[i, j, 4] = macd[j]
                    if self.feature_dimension >= 6:
                        features[i, j, 5] = volatility[j]
                    if self.feature_dimension >= 7:
                        features[i, j, 6] = momentum[j]

        self.recent_features = features
        print("Feature calculation complete!")

    def recommend_portfolio(self, amount_cad, profit_target_percentage=10):
        """Generate portfolio recommendations with profit tracking."""
        print(f"Generating recommendations for ${amount_cad} investment...")

        # Current date and future recommendation date
        current_date = pd.Timestamp.now()
        execution_date_str = current_date.strftime("%Y-%m-%d")
        recommendation_date = current_date + pd.DateOffset(days=7)
        recommendation_date_str = recommendation_date.strftime("%Y-%m-%d")

        # Use the model to predict allocations
        action, _ = self.model.predict(self.recent_features, deterministic=True)

        # Normalize allocations
        action = np.clip(action, 0, 1)
        action_sum = action.sum()
        if action_sum > 0:
            action /= action_sum
        else:
            action[0] = 1.0  # Set cash allocation to 100%

        # Allocate cash based on model recommendation
        allocations = {}
        cash_allocation = action[0] * amount_cad

        # Process allocations for each stock
        new_allocation_tickers = set()

        print("Processing stock allocations...")
        for i, ticker in enumerate(self.tickers):
            if i >= len(self.tickers) or i + 1 >= len(action):
                continue

            allocation = action[i + 1] * amount_cad

            # Only include stocks with positive allocations
            if allocation > 0 and ticker in self.latest_prices:
                price = self.latest_prices[ticker]
                shares = allocation / price if price > 0 else 0

                # Determine action type and profit
                if ticker in self.previous_allocations:
                    prev_shares = self.previous_allocations[ticker].get("shares", 0)

                    if shares > prev_shares:
                        action_type = "buy"
                        profit = allocation * (profit_target_percentage / 100)
                        profit_percentage = profit_target_percentage
                    elif shares < prev_shares:
                        action_type = "sell"
                        if ticker in self.purchase_history:
                            purchase_price = self.purchase_history[ticker].get(
                                "avg_price", price
                            )
                            shares_sold = prev_shares - shares
                            profit = (price - purchase_price) * shares_sold
                            profit_percentage = (
                                ((price - purchase_price) / purchase_price * 100)
                                if purchase_price > 0
                                else 0
                            )
                        else:
                            profit = 0
                            profit_percentage = 0
                    else:
                        action_type = "hold"
                        if ticker in self.purchase_history:
                            purchase_price = self.purchase_history[ticker].get(
                                "avg_price", price
                            )
                            profit = (price - purchase_price) * shares
                            profit_percentage = (
                                ((price - purchase_price) / purchase_price * 100)
                                if purchase_price > 0
                                else 0
                            )
                        else:
                            profit = 0
                            profit_percentage = 0
                else:
                    action_type = "buy"
                    profit = allocation * (profit_target_percentage / 100)
                    profit_percentage = profit_target_percentage

                # Add to new allocations
                new_allocation_tickers.add(ticker)

                # Update purchase history for buys
                if action_type == "buy":
                    if ticker not in self.purchase_history:
                        self.purchase_history[ticker] = {
                            "avg_price": price,
                            "total_shares": shares,
                            "initial_purchase_date": execution_date_str,
                        }
                    else:
                        # Calculate new average purchase price
                        current_shares = self.purchase_history[ticker]["total_shares"]
                        current_avg_price = self.purchase_history[ticker]["avg_price"]

                        new_total_shares = current_shares + shares
                        if new_total_shares > 0:
                            new_avg_price = (
                                (current_shares * current_avg_price) + (shares * price)
                            ) / new_total_shares
                            self.purchase_history[ticker]["avg_price"] = new_avg_price
                            self.purchase_history[ticker][
                                "total_shares"
                            ] = new_total_shares

                # Store allocation details
                allocations[ticker] = {
                    "allocation_percent": float(action[i + 1] * 100),
                    "allocation_cad": float(allocation),
                    "price_per_share": float(price),
                    "shares": float(shares),
                    "action": action_type,
                    "action_date": recommendation_date_str,
                    "profit": float(profit),
                    "profit_percentage": float(profit_percentage),
                }

        # Check for stocks that are being fully sold
        for ticker in self.previous_allocations:
            if ticker not in new_allocation_tickers:
                price = self.latest_prices.get(ticker, 0.0)
                prev_shares = self.previous_allocations[ticker].get("shares", 0)

                # Calculate profit from sale
                profit = 0.0
                profit_percentage = 0.0
                if ticker in self.purchase_history and prev_shares > 0:
                    purchase_price = self.purchase_history[ticker].get(
                        "avg_price", price
                    )
                    profit = (price - purchase_price) * prev_shares
                    profit_percentage = (
                        ((price - purchase_price) / purchase_price * 100)
                        if purchase_price > 0
                        else 0
                    )

                allocations[ticker] = {
                    "allocation_percent": 0.0,
                    "allocation_cad": 0.0,
                    "price_per_share": price,
                    "shares": 0.0,
                    "action": "sell",
                    "action_date": recommendation_date_str,
                    "profit": float(profit),
                    "profit_percentage": float(profit_percentage),
                }

        # Create action table
        action_table = []
        for ticker, details in allocations.items():
            action_table.append(
                {
                    "stock": ticker,
                    "date": details["action_date"],
                    "action": details["action"],
                    "shares": details["shares"],
                    "price": details["price_per_share"],
                    "profit": details.get("profit", 0.0),
                    "profit_percentage": details.get("profit_percentage", 0.0),
                }
            )

        # Update previous allocations for next time
        self.previous_allocations = {
            ticker: details
            for ticker, details in allocations.items()
            if details["allocation_percent"] > 0
        }

        # Calculate total profit
        total_profit = sum(
            details.get("profit", 0.0) for details in allocations.values()
        )
        total_profit_percentage = (
            (total_profit / amount_cad * 100) if amount_cad > 0 else 0.0
        )

        print(
            f"Recommendations generated! Expected profit: ${total_profit:.2f} ({total_profit_percentage:.2f}%)"
        )

        # Return the portfolio recommendations
        return {
            "cash_percent": float(action[0] * 100),
            "cash_amount": float(cash_allocation),
            "stock_allocations": allocations,
            "total_amount": float(amount_cad),
            "total_profit": float(total_profit),
            "total_profit_percentage": float(total_profit_percentage),
            "model_date": str(self.latest_date),
            "execution_date": execution_date_str,
            "recommendation_date": recommendation_date_str,
            "action_table": action_table,
        }

    def display_action_table(self, recommendation_result):
        """Display a formatted table of stock actions"""
        action_table = recommendation_result.get("action_table", [])

        if not action_table:
            print("No actions to display.")
            return

        # Print header
        print(
            f"\nStock Actions (Recommendations for {recommendation_result['recommendation_date']}):"
        )
        print(
            f"{'Stock':<8} {'Date':<12} {'Action':<6} {'Shares':<10} {'Price':<10} {'Profit':<12} {'Profit %':<10}"
        )
        print("-" * 80)

        # Print each action
        for item in action_table:
            print(
                f"{item['stock']:<8} {item['date']:<12} {item['action']:<6} {item['shares']:<10.2f} ${item['price']:<9.2f} ${item['profit']:<11.2f} {item['profit_percentage']:<9.2f}%"
            )

        # Print summary
        total_profit = recommendation_result.get("total_profit", 0.0)
        total_profit_percentage = recommendation_result.get(
            "total_profit_percentage", 0.0
        )
        print("-" * 80)
        print(
            f"Total Portfolio Profit: ${total_profit:.2f} ({total_profit_percentage:.2f}%)"
        )

    def export_action_table(self, recommendation_result, filename=None):
        """Export action table to CSV and print the data"""
        action_table = recommendation_result.get("action_table", [])

        if not action_table:
            print("No actions to export.")
            return

        if filename is None:
            filename = (
                f"portfolio_actions_{recommendation_result['recommendation_date']}.csv"
            )

        # Convert to DataFrame
        df = pd.DataFrame(action_table)

        # Export to CSV
        df.to_csv(filename, index=False)

        # Print the DataFrame
        print(f"\nAction Table Results:")
        print("-" * 100)
        print(df.to_string(index=False))
        print("-" * 100)

        # Print summary statistics
        total_profit = recommendation_result.get("total_profit", 0.0)
        total_profit_percentage = recommendation_result.get(
            "total_profit_percentage", 0.0
        )
        print(
            f"Total Portfolio Profit: ${total_profit:.2f} ({total_profit_percentage:.2f}%)"
        )
        print(f"Action table exported to {filename}")

    def simulate_future_recommendations(
        self, amount_cad, future_dates=30, price_change_range=(-0.15, 0.25)
    ):
        """
        Simulate portfolio recommendations over multiple future dates to show buy and sell actions

        Args:
            amount_cad: Initial investment amount
            future_dates: Number of days to simulate into the future
            price_change_range: Range of random price changes (min, max) as percentages

        Returns:
            List of portfolio recommendation results for each date
        """
        print(f"Simulating portfolio evolution over {future_dates} days...")

        # Store results for each date
        all_recommendations = []

        # Initial portfolio recommendation
        initial_portfolio = self.recommend_portfolio(amount_cad=amount_cad)
        all_recommendations.append(initial_portfolio)

        # Clone the latest prices for simulation
        simulated_prices = self.latest_prices.copy()
        original_prices = self.latest_prices.copy()

        # Track cumulative profit
        cumulative_profit = initial_portfolio["total_profit"]

        # Simulate for future dates
        for day in range(1, future_dates + 1):
            # Simulate price changes
            for ticker in simulated_prices:
                # Generate random price change within range
                pct_change = random.uniform(
                    price_change_range[0], price_change_range[1]
                )

                # Apply the change
                simulated_prices[ticker] = simulated_prices[ticker] * (1 + pct_change)

                # Introduce some trend based on previous recommendations
                # Stocks that performed well tend to continue doing well
                if ticker in self.purchase_history:
                    purchase_price = self.purchase_history[ticker].get("avg_price", 0)
                    current_price = simulated_prices[ticker]

                    # If the stock is doing well, slightly bias toward continued growth
                    if current_price > purchase_price * 1.1:  # More than 10% gain
                        simulated_prices[ticker] *= 1.01  # Small positive bias

                    # If the stock is doing poorly, slightly increase chance of recovery or further decline
                    elif current_price < purchase_price * 0.9:  # More than 10% loss
                        # Randomly decide if it will recover or decline more
                        if random.random() > 0.5:
                            simulated_prices[ticker] *= 1.02  # Small recovery
                        else:
                            simulated_prices[ticker] *= 0.98  # Further decline

            # Save original prices
            original_latest_prices = self.latest_prices

            # Temporarily replace prices with simulated ones
            self.latest_prices = simulated_prices

            # Create a date for this simulation
            simulation_date = pd.Timestamp.now() + pd.DateOffset(days=day)

            # Generate recommendation for this date
            print(
                f"\nDay {day} - Simulating for {simulation_date.strftime('%Y-%m-%d')}..."
            )
            try:
                portfolio = self.recommend_portfolio(
                    amount_cad=amount_cad, profit_target_percentage=10
                )

                # Add to results
                all_recommendations.append(portfolio)

                # Update cumulative profit
                day_profit = portfolio["total_profit"]
                cumulative_profit += day_profit

                print(
                    f"Day {day} profit: ${day_profit:.2f}, Cumulative: ${cumulative_profit:.2f}"
                )

            except Exception as e:
                print(f"Error on day {day}: {e}")

            # Restore original prices
            self.latest_prices = original_latest_prices

        print(
            f"\nSimulation complete. Generated {len(all_recommendations)} recommendations."
        )
        return all_recommendations

    def display_sell_signals(self, recommendation_results):
        """Display all sell signals from the recommendations"""
        print("\n=== SELL SIGNALS DETECTED ===")
        print(
            f"{'Day':<5} {'Date':<12} {'Stock':<8} {'Price':<10} {'Profit':<12} {'Profit %':<10}"
        )
        print("-" * 70)

        day = 0
        sell_signals_found = False

        for result in recommendation_results:
            day += 1
            date = result["recommendation_date"]

            for ticker, details in result["stock_allocations"].items():
                if details["action"] == "sell":
                    sell_signals_found = True
                    print(
                        f"{day:<5} {date:<12} {ticker:<8} ${details['price_per_share']:<9.2f} ${details['profit']:<11.2f} {details['profit_percentage']:<9.2f}%"
                    )

        if not sell_signals_found:
            print("No sell signals detected in the simulation period.")

    def generate_buy_sell_report(
        self, recommendation_results, output_file="buy_sell_report.csv"
    ):
        """Generate a comprehensive buy/sell report from simulation results"""
        # Prepare data
        report_data = []

        day = 0
        for result in recommendation_results:
            day += 1
            date = result["recommendation_date"]

            for ticker, details in result["stock_allocations"].items():
                if details["action"] in [
                    "buy",
                    "sell",
                ]:  # Include only buy and sell actions
                    report_data.append(
                        {
                            "day": day,
                            "date": date,
                            "stock": ticker,
                            "action": details["action"],
                            "shares": details["shares"],
                            "price": details["price_per_share"],
                            "allocation_cad": details["allocation_cad"],
                            "profit": details["profit"],
                            "profit_percentage": details["profit_percentage"],
                        }
                    )

        # Convert to DataFrame and save
        df = pd.DataFrame(report_data)

        # Save to CSV
        df.to_csv(output_file, index=False)

        # Create summary
        buy_actions = df[df["action"] == "buy"]
        sell_actions = df[df["action"] == "sell"]

        print(f"\n=== BUY/SELL REPORT SUMMARY ===")
        print(f"Total Buy Actions: {len(buy_actions)}")
        print(f"Total Sell Actions: {len(sell_actions)}")

        if len(sell_actions) > 0:
            print(f"Total Profit from Sells: ${sell_actions['profit'].sum():.2f}")

        print(f"Report saved to: {output_file}")

        return df

    def plot_sell_recommendations(self, recommendation_results):
        """Plot sell recommendations by date"""
        # Collect sell actions
        sell_data = []

        day = 0
        for result in recommendation_results:
            day += 1
            date = result["recommendation_date"]

            day_sells = 0
            day_profit = 0

            for ticker, details in result["stock_allocations"].items():
                if details["action"] == "sell":
                    day_sells += 1
                    day_profit += details["profit"]

            if day_sells > 0:
                sell_data.append(
                    {"day": day, "date": date, "count": day_sells, "profit": day_profit}
                )

        # Convert to DataFrame
        df = pd.DataFrame(sell_data)

        if len(df) == 0:
            print("No sell recommendations to plot.")
            return

        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Plot sell counts
        ax1.bar(df["day"], df["count"], color="red", alpha=0.7)
        ax1.set_title("Number of Sell Recommendations by Day")
        ax1.set_ylabel("Count")
        ax1.grid(True, linestyle="--", alpha=0.7)

        # Plot profit
        ax2.bar(df["day"], df["profit"], color="green", alpha=0.7)
        ax2.set_title("Profit from Sell Recommendations by Day")
        ax2.set_xlabel("Day")
        ax2.set_ylabel("Profit ($)")
        ax2.grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.show()

        return df
