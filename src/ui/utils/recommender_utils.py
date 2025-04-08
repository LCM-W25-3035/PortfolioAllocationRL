# utils/stock_utils.py
import random
import pandas as pd
import matplotlib.pyplot as plt


def simulate_future_recommendations_with_realistic_profits(
    self, amount_cad, future_days=7, price_change_range=(-0.15, 0.25)
):
    """
    Simulate portfolio recommendations with realistic dates and profit calculations

    Args:
        amount_cad: Initial investment amount
        future_days: Number of days to simulate into the future
        price_change_range: Range of random price changes (min, max) as percentages

    Returns:
        Dictionary containing portfolio recommendations with varying dates and realistic profits
    """
    print(
        f"Simulating portfolio evolution over {future_days} days with realistic profit calculations..."
    )

    # Initial portfolio recommendation
    initial_portfolio = self.recommend_portfolio(amount_cad=amount_cad)

    # Clone the latest prices for simulation
    simulated_prices = {
        ticker: self.latest_prices[ticker] for ticker in self.latest_prices
    }
    price_history = {
        ticker: [self.latest_prices[ticker]] for ticker in self.latest_prices
    }

    # Store action recommendations with dates
    date_based_recommendations = {"buys": [], "sells": [], "holds": []}

    # Simulation start date
    current_date = pd.Timestamp.now()
    date_range = [
        current_date + pd.DateOffset(days=i) for i in range(1, future_days + 1)
    ]

    # Track price trends for each stock to identify optimal points
    ticker_trends = {
        ticker: {
            "trend": "neutral",  # 'up', 'down', 'neutral'
            "days_in_trend": 0,
            "price_change": 0.0,
            "last_action_day": 0,
            "buy_opportunity": False,
            "sell_opportunity": False,
            "avg_price": self.latest_prices[ticker],
            "stop_loss": self.latest_prices[ticker] * 0.85,  # 15% stop loss
            "take_profit": self.latest_prices[ticker] * 1.2,  # 20% take profit
        }
        for ticker in self.latest_prices
    }

    # Establish initial positions based on initial recommendation
    positions = {}
    remaining_cash = amount_cad

    for ticker, details in initial_portfolio["stock_allocations"].items():
        if details["action"] == "buy" and details["shares"] > 0:
            # Calculate the actual cost including transaction fees
            shares = details["shares"]
            price = details["price_per_share"]
            transaction_amount = shares * price

            # Apply transaction fee (typically 0.1% to 0.5%)
            transaction_fee = transaction_amount * 0.001  # 0.1% fee
            total_cost = transaction_amount + transaction_fee

            # Update remaining cash
            if total_cost <= remaining_cash:
                remaining_cash -= total_cost

                positions[ticker] = {
                    "shares": shares,
                    "avg_price": price,
                    "cost_basis": total_cost,
                    "transaction_fee": transaction_fee,
                    "buy_date": current_date,
                    "stop_loss": price * 0.85,
                    "take_profit": price * 1.2,
                }

                # Record initial buys
                date_based_recommendations["buys"].append(
                    {
                        "stock": ticker,
                        "date": current_date.strftime("%Y-%m-%d"),
                        "action": "buy",
                        "shares": shares,
                        "price": price,
                        "transaction_fee": transaction_fee,
                        "total_cost": total_cost,
                        "profit": 0.0,
                        "profit_percentage": 0.0,
                        "reason": "Initial portfolio allocation",
                    }
                )

    # Simulate price evolution and identify trading opportunities
    cumulative_profit = 0.0

    print(
        f"Starting simulation on {current_date.strftime('%Y-%m-%d')} with ${remaining_cash:.2f} remaining cash..."
    )

    for day in range(1, future_days + 1):
        # Current date in simulation
        sim_date = date_range[day - 1]

        # Simulate price changes
        # Apply more realistic market behavior:
        # 1. Market-wide factors affect all stocks (systematic risk)
        # 2. Individual stock factors (idiosyncratic risk)
        # 3. Sector correlation (stocks in same sector move together)

        # Simulate market-wide movement (affects all stocks)
        market_move = random.normalvariate(
            0.0002, 0.01
        )  # Slight positive bias with volatility

        # Sector correlations (simplified)
        sectors = {
            "tech": ["SHOP.TO", "BB.TO", "OTEX.TO"],
            "energy": [
                "ENB.TO",
                "TRP.TO",
                "CNQ.TO",
                "SU.TO",
                "CVE.TO",
                "BTE.TO",
                "GEI.TO",
            ],
            "materials": ["ABX.TO", "FNV.TO", "K.TO", "AEM.TO", "NXE.TO"],
            "financials": ["RY.TO", "TD.TO", "BNS.TO", "NA.TO", "CM.TO"],
            "utilities": ["FTS.TO", "EMA.TO", "H.TO"],
            "real_estate": ["REI.UN.TO", "HR.UN.TO"],
            "telecom": ["BCE.TO", "T.TO", "RCI-B.TO"],
            "consumer": ["ATD.TO", "L.TO", "MRU.TO", "QSR.TO"],
            "healthcare": ["SIA.TO", "DOC.TO"],
            "industrial": ["CNR.TO", "CP.TO", "WCN.TO"],
        }

        # Generate sector moves
        sector_moves = {
            sector: random.normalvariate(market_move, 0.008) for sector in sectors
        }

        # Process each stock
        for ticker in simulated_prices:
            # Find which sector the stock belongs to
            stock_sector = None
            for sector, stocks in sectors.items():
                if ticker in stocks:
                    stock_sector = sector
                    break

            # Base movement components
            sector_component = sector_moves.get(stock_sector, 0) if stock_sector else 0
            stock_specific_component = random.normalvariate(
                0, 0.02
            )  # Stock-specific volatility

            # Apply trend continuity (momentum effect)
            trend_component = 0
            if ticker in ticker_trends:
                if ticker_trends[ticker]["trend"] == "up":
                    trend_component = random.uniform(
                        0, 0.01
                    )  # More likely to continue up
                elif ticker_trends[ticker]["trend"] == "down":
                    trend_component = random.uniform(
                        -0.01, 0
                    )  # More likely to continue down

            # Combine components for final price change
            pct_change = (
                market_move
                + sector_component
                + stock_specific_component
                + trend_component
            )

            # Apply the change
            old_price = simulated_prices[ticker]
            new_price = old_price * (1 + pct_change)
            simulated_prices[ticker] = new_price

            # Update price history
            if ticker in price_history:
                price_history[ticker].append(new_price)

            # Update trend information
            if ticker in ticker_trends:
                # Calculate daily change
                daily_change = (new_price / old_price) - 1

                # Update price change from starting point
                ticker_trends[ticker]["price_change"] = (
                    new_price / price_history[ticker][0]
                ) - 1

                # Detect trend
                old_trend = ticker_trends[ticker]["trend"]

                if daily_change > 0.02:  # 2% up day
                    if old_trend == "up":
                        ticker_trends[ticker]["days_in_trend"] += 1
                    else:
                        ticker_trends[ticker]["trend"] = "up"
                        ticker_trends[ticker]["days_in_trend"] = 1
                elif daily_change < -0.02:  # 2% down day
                    if old_trend == "down":
                        ticker_trends[ticker]["days_in_trend"] += 1
                    else:
                        ticker_trends[ticker]["trend"] = "down"
                        ticker_trends[ticker]["days_in_trend"] = 1
                else:
                    # Neutral day, continue previous trend but don't increment
                    pass

                # Buy opportunity:
                # 1. Stock has been going down for a while but starts to recover
                # 2. Stock is in strong uptrend
                if (old_trend == "down" and daily_change > 0.04) or (
                    old_trend == "up"
                    and ticker_trends[ticker]["days_in_trend"] >= 3
                    and ticker_trends[ticker]["days_in_trend"]
                    - ticker_trends[ticker]["last_action_day"]
                    >= 3
                ):
                    ticker_trends[ticker]["buy_opportunity"] = True
                else:
                    ticker_trends[ticker]["buy_opportunity"] = False

                # Sell opportunity:
                # 1. Stock has been going up but now turning down
                # 2. Stock hits take profit level
                # 3. Stock hits stop loss level
                # 4. Stock in prolonged downtrend
                if ticker in positions and positions[ticker]["shares"] > 0:
                    if (
                        (old_trend == "up" and daily_change < -0.03)
                        or (new_price >= positions[ticker]["take_profit"])
                        or (new_price <= positions[ticker]["stop_loss"])
                        or (
                            old_trend == "down"
                            and ticker_trends[ticker]["days_in_trend"] >= 5
                        )
                    ):
                        ticker_trends[ticker]["sell_opportunity"] = True
                    else:
                        ticker_trends[ticker]["sell_opportunity"] = False

        # Generate buy/sell/hold recommendations based on trends
        actions_today = []

        # First, process sells to free up cash
        for ticker in list(positions.keys()):
            if (
                ticker in ticker_trends
                and ticker_trends[ticker]["sell_opportunity"]
                and positions[ticker]["shares"] > 0
            ):
                price = simulated_prices[ticker]
                shares = positions[ticker]["shares"]
                avg_price = positions[ticker]["avg_price"]
                cost_basis = positions[ticker]["cost_basis"]

                if shares > 0:
                    # Calculate the actual proceeds after transaction fees
                    transaction_amount = shares * price

                    # Apply transaction fee and other costs
                    transaction_fee = transaction_amount * 0.001  # 0.1% fee
                    tax_rate = 0.15  # 15% tax on gains

                    # Calculate pre-tax profit
                    pre_tax_profit = transaction_amount - cost_basis - transaction_fee

                    # Calculate tax (only on profits, not losses)
                    tax = max(0, pre_tax_profit * tax_rate)

                    # Calculate final profit
                    profit = pre_tax_profit - tax
                    profit_percentage = (profit / cost_basis) * 100

                    # Add to cash balance
                    remaining_cash += transaction_amount - transaction_fee - tax

                    # Record the sell recommendation
                    sell_reason = ""
                    if price >= positions[ticker]["take_profit"]:
                        sell_reason = f"Take profit target reached (+20%)"
                    elif price <= positions[ticker]["stop_loss"]:
                        sell_reason = f"Stop loss triggered (-15%)"
                    elif (
                        ticker_trends[ticker]["trend"] == "up"
                        and ticker_trends[ticker]["price_change"] > 0.15
                    ):
                        sell_reason = "Taking profits after strong rally"
                    elif (
                        ticker_trends[ticker]["trend"] == "down"
                        and ticker_trends[ticker]["days_in_trend"] >= 5
                    ):
                        sell_reason = "Cutting losses in extended downtrend"
                    else:
                        sell_reason = "Technical sell signal"

                    date_based_recommendations["sells"].append(
                        {
                            "stock": ticker,
                            "date": sim_date.strftime("%Y-%m-%d"),
                            "action": "sell",
                            "shares": shares,
                            "price": price,
                            "transaction_fee": transaction_fee,
                            "tax": tax,
                            "gross_proceeds": transaction_amount,
                            "net_proceeds": transaction_amount - transaction_fee - tax,
                            "cost_basis": cost_basis,
                            "profit": profit,
                            "profit_percentage": profit_percentage,
                            "reason": sell_reason,
                        }
                    )

                    # Update cumulative profit
                    cumulative_profit += profit

                    # Mark position as closed
                    positions[ticker]["shares"] = 0

                    # Update last action day
                    ticker_trends[ticker]["last_action_day"] = ticker_trends[ticker][
                        "days_in_trend"
                    ]

                    actions_today.append(
                        f"SELL {ticker}: {shares:.2f} shares @ ${price:.2f} - Profit: ${profit:.2f} ({profit_percentage:.2f}%) - {sell_reason}"
                    )

        # Now, process buys with updated cash
        for ticker in ticker_trends:
            # Buy logic
            if ticker_trends[ticker]["buy_opportunity"] and (
                ticker not in positions or positions[ticker]["shares"] == 0
            ):
                # Calculate how many shares we can buy
                price = simulated_prices[ticker]

                # Realistic position sizing:
                # 1. Never use more than 5% of portfolio on one position
                # 2. Ensure we have enough cash
                max_position_size = min(amount_cad * 0.05, remaining_cash * 0.25)

                # Calculate number of shares (always round down to avoid fractional shares for some brokers)
                shares_to_buy = int(max_position_size / price)

                if shares_to_buy > 0:
                    # Calculate the actual cost including transaction fees
                    transaction_amount = shares_to_buy * price
                    transaction_fee = transaction_amount * 0.001  # 0.1% fee
                    total_cost = transaction_amount + transaction_fee

                    # Make sure we have enough cash
                    if total_cost <= remaining_cash:
                        # Deduct from cash
                        remaining_cash -= total_cost

                        # Record the buy recommendation
                        buy_reason = ""
                        if (
                            ticker_trends[ticker]["trend"] == "down"
                            and ticker_trends[ticker]["price_change"] < -0.1
                        ):
                            buy_reason = "Potential bottom (down trend reversal)"
                        elif (
                            ticker_trends[ticker]["trend"] == "up"
                            and ticker_trends[ticker]["days_in_trend"] >= 3
                        ):
                            buy_reason = "Strong uptrend continuation"
                        else:
                            buy_reason = "Technical buy signal"

                        date_based_recommendations["buys"].append(
                            {
                                "stock": ticker,
                                "date": sim_date.strftime("%Y-%m-%d"),
                                "action": "buy",
                                "shares": shares_to_buy,
                                "price": price,
                                "transaction_fee": transaction_fee,
                                "total_cost": total_cost,
                                "profit": 0.0,
                                "profit_percentage": 0.0,
                                "reason": buy_reason,
                            }
                        )

                        # Record the position
                        positions[ticker] = {
                            "shares": shares_to_buy,
                            "avg_price": price,
                            "cost_basis": total_cost,
                            "transaction_fee": transaction_fee,
                            "buy_date": sim_date,
                            "stop_loss": price * 0.85,
                            "take_profit": price * 1.2,
                        }

                        # Update last action day
                        ticker_trends[ticker]["last_action_day"] = ticker_trends[
                            ticker
                        ]["days_in_trend"]

                        actions_today.append(
                            f"BUY {ticker}: {shares_to_buy:.0f} shares @ ${price:.2f} - Cost: ${total_cost:.2f} - {buy_reason}"
                        )

            # Hold logic - record significant unrealized profits or losses
            elif ticker in positions and positions[ticker]["shares"] > 0:
                price = simulated_prices[ticker]
                shares = positions[ticker]["shares"]
                avg_price = positions[ticker]["avg_price"]
                cost_basis = positions[ticker]["cost_basis"]
                days_held = (sim_date - positions[ticker]["buy_date"]).days

                # Only record holds every 5 days or on significant price movements
                if days_held > 0 and (
                    days_held % 5 == 0 or abs((price / avg_price - 1) * 100) > 10
                ):
                    # Calculate unrealized profit (including estimated fees if sold now)
                    transaction_amount = shares * price
                    transaction_fee = transaction_amount * 0.001  # 0.1% fee
                    pre_tax_profit = transaction_amount - cost_basis - transaction_fee
                    tax = max(0, pre_tax_profit * 0.15)  # 15% tax on gains

                    unrealized_profit = pre_tax_profit - tax
                    unrealized_pct = (unrealized_profit / cost_basis) * 100

                    hold_reason = ""
                    if unrealized_pct > 15:
                        hold_reason = "Holding winning position, approaching target"
                    elif unrealized_pct < -10:
                        hold_reason = (
                            "Holding through drawdown, monitoring for reversal"
                        )
                    else:
                        hold_reason = "Maintaining position, no exit signal"

                    date_based_recommendations["holds"].append(
                        {
                            "stock": ticker,
                            "date": sim_date.strftime("%Y-%m-%d"),
                            "action": "hold",
                            "shares": shares,
                            "price": price,
                            "days_held": days_held,
                            "cost_basis": cost_basis,
                            "current_value": transaction_amount,
                            "unrealized_profit": unrealized_profit,
                            "unrealized_percentage": unrealized_pct,
                            "reason": hold_reason,
                        }
                    )

                    actions_today.append(
                        f"HOLD {ticker}: {shares:.0f} shares, ${unrealized_profit:.2f} unrealized ({unrealized_pct:.2f}%) - {hold_reason}"
                    )

        # Print daily summary if there were actions
        if actions_today:
            print(
                f"\nDay {day} - {sim_date.strftime('%Y-%m-%d')} - Cash: ${remaining_cash:.2f}"
            )
            for action in actions_today:
                print(f"  {action}")

    # Calculate remaining portfolio value
    final_portfolio_value = remaining_cash
    for ticker, position in positions.items():
        if position["shares"] > 0:
            final_portfolio_value += position["shares"] * simulated_prices[ticker]

    portfolio_gain = final_portfolio_value - amount_cad
    portfolio_gain_pct = (portfolio_gain / amount_cad) * 100

    print(f"\nSimulation complete.")
    print(f"Initial investment: ${amount_cad:.2f}")
    print(f"Final portfolio value: ${final_portfolio_value:.2f}")
    print(f"Total portfolio gain: ${portfolio_gain:.2f} ({portfolio_gain_pct:.2f}%)")
    print(f"Realized profits: ${cumulative_profit:.2f}")
    print(f"Remaining cash: ${remaining_cash:.2f}")
    print(f"Total buy recommendations: {len(date_based_recommendations['buys'])}")
    print(f"Total sell recommendations: {len(date_based_recommendations['sells'])}")
    print(f"Total hold notifications: {len(date_based_recommendations['holds'])}")

    return date_based_recommendations, final_portfolio_value, cumulative_profit


def display_date_based_recommendations(self, recommendations):
    """
    Display date-based recommendations in a chronological format

    Args:
        recommendations: Output from simulate_future_recommendations_with_varying_dates
    """
    # Combine all recommendations
    all_recs = []

    for buy in recommendations["buys"]:
        all_recs.append(buy)

    for sell in recommendations["sells"]:
        all_recs.append(sell)

    for hold in recommendations["holds"]:
        all_recs.append(hold)

    # Sort by date
    all_recs.sort(key=lambda x: x["date"])

    # Display chronologically
    print("\n=== CHRONOLOGICAL TRADING RECOMMENDATIONS ===")
    print(
        f"{'Date':<12} {'Stock':<8} {'Action':<6} {'Shares':<10} {'Price':<10} {'Profit':<12} {'Profit %':<10} {'Reason'}"
    )
    print("-" * 120)

    current_date = None

    for rec in all_recs:
        if current_date != rec["date"]:
            current_date = rec["date"]
            print(f"\n--- {current_date} ---")

        # Handle different action types slightly differently
        if rec["action"] == "hold":
            print(
                f"{'':<12} {rec['stock']:<8} {rec['action']:<6} {rec['shares']:<10.2f} ${rec['price']:<9.2f} ${rec['unrealized_profit']:<11.2f} {rec['unrealized_percentage']:<9.2f}% {rec['reason']}"
            )
        else:
            print(
                f"{'':<12} {rec['stock']:<8} {rec['action']:<6} {rec['shares']:<10.2f} ${rec['price']:<9.2f} ${rec.get('profit', 0.0):<11.2f} {rec.get('profit_percentage', 0.0):<9.2f}% {rec['reason']}"
            )

    # Calculate profitability stats
    total_buys = len(recommendations["buys"])
    total_sells = len(recommendations["sells"])
    total_profit = sum(sell.get("profit", 0.0) for sell in recommendations["sells"])
    winning_trades = sum(
        1 for sell in recommendations["sells"] if sell.get("profit", 0.0) > 0
    )
    losing_trades = sum(
        1 for sell in recommendations["sells"] if sell.get("profit", 0.0) <= 0
    )

    win_rate = winning_trades / total_sells * 100 if total_sells > 0 else 0
    avg_win = (
        sum(
            sell.get("profit", 0.0)
            for sell in recommendations["sells"]
            if sell.get("profit", 0.0) > 0
        )
        / winning_trades
        if winning_trades > 0
        else 0
    )
    avg_loss = (
        sum(
            sell.get("profit", 0.0)
            for sell in recommendations["sells"]
            if sell.get("profit", 0.0) <= 0
        )
        / losing_trades
        if losing_trades > 0
        else 0
    )

    print("\n=== TRADING PERFORMANCE SUMMARY ===")
    print(f"Total Buy Recommendations: {total_buys}")
    print(f"Total Sell Recommendations: {total_sells}")
    print(f"Total Realized Profit: ${total_profit:.2f}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Average Win: ${avg_win:.2f}")
    print(f"Average Loss: ${avg_loss:.2f}")
    if avg_loss != 0:
        print(f"Risk-Reward Ratio: {abs(avg_win/avg_loss):.2f}")


def export_date_based_recommendations(
    self, recommendations, filename="date_based_recommendations.csv"
):
    """
    Export date-based recommendations to a CSV file

    Args:
        recommendations: Output from simulate_future_recommendations_with_varying_dates
        filename: Output filename
    """
    # Combine all recommendations
    all_recs = []

    for buy in recommendations["buys"]:
        rec = buy.copy()
        all_recs.append(rec)

    for sell in recommendations["sells"]:
        rec = sell.copy()
        all_recs.append(rec)

    for hold in recommendations["holds"]:
        rec = hold.copy()
        # Rename fields to match the others
        if "unrealized_profit" in rec:
            rec["profit"] = rec["unrealized_profit"]
            del rec["unrealized_profit"]
        if "unrealized_percentage" in rec:
            rec["profit_percentage"] = rec["unrealized_percentage"]
            del rec["unrealized_percentage"]
        all_recs.append(rec)

    # Sort by date
    all_recs.sort(key=lambda x: x["date"])

    # Convert to DataFrame
    df = pd.DataFrame(all_recs)

    # Save to CSV
    df.to_csv(filename, index=False)

    print(f"\nAll recommendations exported to {filename}")

    # Also export a summary by day
    daily_summary = {}
    for rec in all_recs:
        date = rec["date"]
        action = rec["action"]

        if date not in daily_summary:
            daily_summary[date] = {"buys": 0, "sells": 0, "holds": 0, "profit": 0.0}

        daily_summary[date][action + "s"] += 1
        if action == "sell":
            daily_summary[date]["profit"] += rec.get("profit", 0.0)

    # Convert to DataFrame
    df_summary = pd.DataFrame(
        [
            {
                "date": date,
                "buys": data["buys"],
                "sells": data["sells"],
                "holds": data["holds"],
                "daily_profit": data["profit"],
            }
            for date, data in daily_summary.items()
        ]
    )

    # Sort by date
    df_summary = df_summary.sort_values("date")

    # Add cumulative profit
    df_summary["cumulative_profit"] = df_summary["daily_profit"].cumsum()

    # Save to CSV
    summary_filename = "daily_trading_summary.csv"
    df_summary.to_csv(summary_filename, index=False)

    print(f"Daily trading summary exported to {summary_filename}")

    # Return both DataFrames
    return df, df_summary


def plot_trading_summary(self, recommendations):
    """
    Create visualizations of the trading recommendations and performance

    Args:
        recommendations: Output from simulate_future_recommendations_with_varying_dates
    """
    # Combine all recommendations
    all_recs = []

    for buy in recommendations["buys"]:
        rec = buy.copy()
        rec["profit"] = 0.0
        rec["profit_percentage"] = 0.0
        all_recs.append(rec)

    for sell in recommendations["sells"]:
        all_recs.append(sell)

    # Sort by date
    all_recs.sort(key=lambda x: x["date"])

    # Create a DataFrame
    df = pd.DataFrame(all_recs)

    # Convert date to datetime
    df["date"] = pd.to_datetime(df["date"])

    # Create daily summary
    daily_summary = (
        df.groupby(["date", "action"])
        .agg({"stock": "count", "profit": "sum"})
        .reset_index()
    )

    # Pivot to get buys and sells by day
    pivoted = daily_summary.pivot_table(
        index="date", columns="action", values="stock", fill_value=0
    ).reset_index()

    if "sell" not in pivoted.columns:
        pivoted["sell"] = 0
    if "buy" not in pivoted.columns:
        pivoted["buy"] = 0

    # Calculate daily and cumulative profit
    daily_profit = (
        daily_summary[daily_summary["action"] == "sell"]
        .groupby("date")["profit"]
        .sum()
        .reset_index()
    )
    daily_profit["cumulative_profit"] = daily_profit["profit"].cumsum()

    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot buy/sell counts
    ax1.bar(pivoted["date"], pivoted["buy"], color="green", alpha=0.7, label="Buys")
    ax1.bar(pivoted["date"], pivoted["sell"], color="red", alpha=0.7, label="Sells")
    ax1.set_title("Number of Buy/Sell Recommendations by Date")
    ax1.set_ylabel("Count")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Plot profit
    if not daily_profit.empty:
        ax2.bar(daily_profit["date"], daily_profit["profit"], color="blue", alpha=0.7)
        ax2.set_title("Daily Profit from Sells")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Profit ($)")
        ax2.grid(True, linestyle="--", alpha=0.7)

        # Add cumulative profit line
        ax3 = ax2.twinx()
        ax3.plot(
            daily_profit["date"],
            daily_profit["cumulative_profit"],
            "r-",
            label="Cumulative Profit",
        )
        ax3.set_ylabel("Cumulative Profit ($)")
        ax3.legend(loc="upper right")

    plt.tight_layout()
    plt.show()

    # Create performance visualization by stock
    stock_performance = (
        df[df["action"] == "sell"]
        .groupby("stock")
        .agg({"profit": "sum", "profit_percentage": "mean"})
        .reset_index()
    )

    # Sort by total profit
    stock_performance = stock_performance.sort_values("profit", ascending=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot total profit by stock
    ax1.bar(
        stock_performance["stock"], stock_performance["profit"], color="blue", alpha=0.7
    )
    ax1.set_title("Total Profit by Stock")
    ax1.set_xlabel("Stock")
    ax1.set_ylabel("Total Profit ($)")
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.tick_params(axis="x", rotation=90)

    # Plot average profit percentage by stock
    ax2.bar(
        stock_performance["stock"],
        stock_performance["profit_percentage"],
        color="green",
        alpha=0.7,
    )
    ax2.set_title("Average Profit Percentage by Stock")
    ax2.set_xlabel("Stock")
    ax2.set_ylabel("Average Profit (%)")
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.tick_params(axis="x", rotation=90)

    plt.tight_layout()
    plt.show()
