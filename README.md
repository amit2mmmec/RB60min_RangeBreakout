
# Opening Range Breakout (ORB) with VIX Regimes — Backtest Pack

This pack includes a ready-to-run Python script to backtest an Opening Range Breakout strategy on NIFTY 50 with VIX-based regime filtering.

## Files
- `nifty_vix_orb_backtest.py` — the backtester
- (you provide) `NIFTY_50_minute.csv` — columns: date,open,high,low,close,volume
- (you provide) `INDIA_VIX_minute.csv` — columns: date,open,high,low,close,volume

## Strategy (per timeframe)
- Build the **opening range** using the first N minutes from 09:15 IST (N ∈ {5, 15, 60}).
- After the range is set, take the **first breakout**:
  - Long if price hits OR High; Short if it hits OR Low.
  - Entry is approximated at the breakout level (stop order), conservative fill if gap.
- **Stops/Targets** (default): SL = 1.0 × OR range, TP = 1.5 × OR range (conservative intrabar ordering: SL before TP if both touched in a bar).
- Flat any open position by **15:25 IST**.
- **One trade per day**, first breakout wins (configurable).
- Tag each day by VIX at OR end into buckets **<12**, **12–15**, **>15**.

## Outputs
All CSVs go to `./outputs` by default:
- `trades_orb_{5|15|60}m.csv` — one row per trade with details & VIX bucket.
- `daily_orb_{5|15|60}m.csv` — daily P&L (points) and returns (% of base capital).
- `summary_by_vix_and_tf.csv` — metrics per timeframe and per VIX bucket.
- `leaderboard_overall.csv` — top timeframe overall (ALL VIX).

## Metrics
- Trades, Win-rate, Avg/Total P&L (points), Profit Factor, Max Drawdown (points),
  **CAGR%** and a **Sharpe-like** metric (using daily returns vs base capital).

## How to run
Place your two CSV files in the same folder and run:

```bash
python nifty_vix_orb_backtest.py --nifty_csv NIFTY_50_minute.csv --vix_csv INDIA_VIX_minute.csv
```

Common tweaks:

```bash
# Tighter stop, same target
python nifty_vix_orb_backtest.py --sl_mult 0.8 --tp_mult 1.5

# No profit target (ride the day, exit at 15:25)
python nifty_vix_orb_backtest.py --tp_mult 0

# Allow ambiguous same-bar double-touch (not recommended)
python nifty_vix_orb_backtest.py --allow_ambiguous_breakouts
```

## Picking the timeframe
Run once and open `leaderboard_overall.csv` to see the best-performing **timeframe**.
Check `summary_by_vix_and_tf.csv` to see how the same timeframe behaves across **VIX regimes**.
You can narrow usage to the regime(s) where edge is strongest (e.g., only trade when VIX > 15).

## Notes
- This is index **spot** data; if you execute via futures/options, map points to instrument P&L carefully (lot sizes change over time).
- No transaction costs/slippage are included. Add them by subtracting a per-trade fixed cost (e.g., 1–3 points) or %.
- All timestamps assumed in IST and already regular-session filtered to 09:15–15:30 by the script.
