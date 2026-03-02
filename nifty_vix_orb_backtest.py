
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import argparse
import os

# -----------------------------
# Config dataclasses
# -----------------------------

@dataclass
class ORBParams:
    or_minutes: int          # 5, 15, 60 (opening range length in minutes)
    sl_mult: float = 1.0     # Stop loss multiple of opening range
    tp_mult: float = 2.0     # Take profit multiple of opening range; set 0 for no TP
    eod_flat_time: str = "15:25"  # Flat all positions at or before this HH:MM (IST) if not exited
    allow_ambiguous_breakouts: bool = False  # if True, take breakout even if bar hits both sides (rare)
    capital: float = 100000.0   # Notional base capital for % returns (for CAGR/Sharpe comparability)
    max_trades_per_day: int = 2 # 1 = take first breakout of the day and be done

@dataclass
class BacktestResult:
    timeframe: int
    trades: int
    win_rate: float
    avg_pnl_pts: float
    total_pnl_pts: float
    profit_factor: float
    max_dd_pts: float
    dd_start: Optional[str]
    dd_end: Optional[str]
    cagr_pct: float
    sharpe_like: float
    max_win_streak: int
    max_loss_streak: int
    vix_bucket: str

# -----------------------------
# Utility functions
# -----------------------------

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Expect columns: date, open, high, low, close, volume
    if 'date' not in df.columns:
        raise ValueError(f"{path} missing 'date' column")
    df['date'] = pd.to_datetime(df['date'])
    # Sort just in case
    df = df.sort_values('date').reset_index(drop=True)
    return df

def filter_market_hours(df: pd.DataFrame, start="09:15", end="15:30") -> pd.DataFrame:
    # Keep rows within NSE regular session
    df = df.copy()
    df['t'] = df['date'].dt.time
    start_h, start_m = map(int, start.split(':'))
    end_h, end_m = map(int, end.split(':'))
    mask = (df['date'].dt.hour*60 + df['date'].dt.minute >= start_h*60 + start_m) & \
           (df['date'].dt.hour*60 + df['date'].dt.minute <= end_h*60 + end_m)
    return df.loc[mask].drop(columns=['t'])

def vix_bucket(v: float) -> str:
    if v < 12: return "<12"
    if v <= 15: return "12-15"
    return ">15"

def day_key(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=ts.year, month=ts.month, day=ts.day)

def calc_cagr(daily_returns: pd.Series, trading_days_per_year: int = 252) -> float:
    # daily_returns are percentage returns relative to base capital (e.g., pnl/base_capital)
    if len(daily_returns) == 0:
        return 0.0
    cum_return = (1 + daily_returns.fillna(0)).prod()
    years = len(daily_returns) / trading_days_per_year
    if years == 0:
        return 0.0
    return (cum_return ** (1/years) - 1) * 100

def calc_max_dd_from_equity(equity_curve: pd.Series) -> float:
    roll_max = equity_curve.cummax()
    drawdown = equity_curve - roll_max
    return -drawdown.min() if len(drawdown) else 0.0

def safe_first(series: pd.Series) -> Optional[pd.Timestamp]:
    try:
        return series.iloc[0]
    except Exception:
        return None

# -----------------------------
# Core backtest
# -----------------------------

def run_backtest(nifty_df: pd.DataFrame,
                 vix_df: pd.DataFrame,
                 params: ORBParams) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (trades_df, daily_summary_df)
    trades_df: one row per trade with details
    daily_summary_df: per-day pnl and % return for equity curve
    """
    # Filter to market hours
    ndf = filter_market_hours(nifty_df)
    vdf = filter_market_hours(vix_df)

    # Precompute helper columns
    ndf['date_only'] = ndf['date'].dt.date
    vdf['date_only'] = vdf['date'].dt.date

    # Dict day -> opening range [start, end)
    trades = []
    daily_pnl_pts = {}

    grouped = ndf.groupby('date_only', sort=True)
    vix_grouped = vdf.groupby('date_only', sort=True)

    for day, g in grouped:
        g = g.reset_index(drop=True)
        if g.empty: 
            continue

        # Opening time for the day
        day_open_time = pd.Timestamp(year=g['date'].iloc[0].year,
                                     month=g['date'].iloc[0].month,
                                     day=g['date'].iloc[0].day,
                                     hour=9, minute=15)
        or_end_time = day_open_time + pd.Timedelta(minutes=params.or_minutes-1)

        # OR window: inclusive of the last minute (e.g., 9:15 to 9:19 for 5m)
        or_window = g[(g['date'] >= day_open_time) & (g['date'] <= or_end_time)]
        if or_window.empty:
            continue
        or_high = or_window['high'].max()
        or_low = or_window['low'].min()
        opening_range = or_high - or_low
        if opening_range <= 0:
            continue

        # The rest of the day after OR
        after_or = g[g['date'] > or_end_time].copy()
        if after_or.empty:
            continue

        # VIX at OR end (or closest prior minute)
        vix_g = vix_grouped.get_group(day) if day in vix_grouped.groups else pd.DataFrame()
        if not vix_g.empty:
            vix_row = vix_g[vix_g['date'] <= or_end_time].tail(1)
            if vix_row.empty:
                # if none before OR end, take first available in day
                vix_row = vix_g.head(1)
            vix_at_or = float(vix_row['close'].iloc[0])
        else:
            vix_at_or = np.nan
        vbucket = vix_bucket(vix_at_or) if np.isfinite(vix_at_or) else "NA"

        # EOD flat time
        eod_flat_time = pd.Timestamp(year=after_or['date'].iloc[0].year,
                                     month=after_or['date'].iloc[0].month,
                                     day=after_or['date'].iloc[0].day)
        hh, mm = map(int, params.eod_flat_time.split(':'))
        eod_flat_time = eod_flat_time.replace(hour=hh, minute=mm)

        # Identify first breakout(s)
        trades_taken = 0
        position = None  # 'long' or 'short'
        entry_px = None
        sl = None
        tp = None
        entry_time = None

        for i, row in after_or.iterrows():
            ts = row['date']
            o, h, l, c = row['open'], row['high'], row['low'], row['close']

            # check for end-of-day flat if no entry yet
            if ts >= eod_flat_time and position is None:
                break

            if position is None and trades_taken < params.max_trades_per_day:
                hit_up = h >= or_high
                hit_dn = l <= or_low

                if hit_up and hit_dn and not params.allow_ambiguous_breakouts:
                    # Skip ambiguous one-minute bar that touches both levels
                    # (rare with indices, safer to skip)
                    break

                if hit_up and not hit_dn:
                    # entry long at the breakout price approximation (stop order at or_high)
                    # If open gaps above or_high, assume fill at max(or_high, open) conservatively
                    fill = max(or_high, o) if o >= or_high else or_high
                    position = 'long'
                    entry_px = float(fill)
                    rng = opening_range
                    sl = entry_px - params.sl_mult * rng
                    tp = entry_px + params.tp_mult * rng if params.tp_mult > 0 else None
                    entry_time = ts
                    trades_taken += 1
                elif hit_dn and not hit_up:
                    # entry short
                    fill = min(or_low, o) if o <= or_low else or_low
                    position = 'short'
                    entry_px = float(fill)
                    rng = opening_range
                    sl = entry_px + params.sl_mult * rng
                    tp = entry_px - params.tp_mult * rng if params.tp_mult > 0 else None
                    entry_time = ts
                    trades_taken += 1
                elif hit_up and hit_dn and params.allow_ambiguous_breakouts:
                    # If allowed, decide by which boundary is closer to open (proxy for first touch)
                    up_dist = abs(o - or_high)
                    dn_dist = abs(o - or_low)
                    if up_dist <= dn_dist:
                        fill = max(or_high, o) if o >= or_high else or_high
                        position = 'long'
                        entry_px = float(fill)
                        rng = opening_range
                        sl = entry_px - params.sl_mult * rng
                        tp = entry_px + params.tp_mult * rng if params.tp_mult > 0 else None
                        entry_time = ts
                        trades_taken += 1
                    else:
                        fill = min(or_low, o) if o <= or_low else or_low
                        position = 'short'
                        entry_px = float(fill)
                        rng = opening_range
                        sl = entry_px + params.sl_mult * rng
                        tp = entry_px - params.tp_mult * rng if params.tp_mult > 0 else None
                        entry_time = ts
                        trades_taken += 1
                # else: no breakout yet

            elif position is not None:
                # manage position: conservative intrabar order -> SL first, then TP
                exit_reason = None
                exit_px = None

                if position == 'long':
                    if row['low'] <= sl:
                        exit_px = sl
                        exit_reason = 'SL'
                    elif tp is not None and row['high'] >= tp:
                        exit_px = tp
                        exit_reason = 'TP'
                else:  # short
                    if row['high'] >= sl:
                        exit_px = sl
                        exit_reason = 'SL'
                    elif tp is not None and row['low'] <= tp:
                        exit_px = tp
                        exit_reason = 'TP'

                # EOD flat
                if exit_px is None and ts >= eod_flat_time:
                    exit_px = c
                    exit_reason = 'EOD'

                if exit_px is not None:
                    pnl = (exit_px - entry_px) if position == 'long' else (entry_px - exit_px)
                    trades.append({
                        'date': day,
                        'timeframe_min': params.or_minutes,
                        'entry_time': entry_time,
                        'exit_time': ts,
                        'side': position,
                        'entry': entry_px,
                        'exit': exit_px,
                        'pnl_pts': pnl,
                        'exit_reason': exit_reason,
                        'or_high': or_high,
                        'or_low': or_low,
                        'or_range': opening_range,
                        'vix_at_or': vix_at_or,
                        'vix_bucket': vbucket
                    })
                    daily_pnl_pts[day] = daily_pnl_pts.get(day, 0.0) + pnl
                    # Done for the day if only one trade allowed
                    position = None
                    break

        # If position still open after loop, force close at last bar before EOD
        if position is not None:
            last_row = after_or.iloc[-1]
            exit_px = float(last_row['close'])
            pnl = (exit_px - entry_px) if position == 'long' else (entry_px - exit_px)
            trades.append({
                'date': day,
                'timeframe_min': params.or_minutes,
                'entry_time': entry_time,
                'exit_time': last_row['date'],
                'side': position,
                'entry': entry_px,
                'exit': exit_px,
                'pnl_pts': pnl,
                'exit_reason': 'SESSION_END',
                'or_high': or_high,
                'or_low': or_low,
                'or_range': opening_range,
                'vix_at_or': vix_at_or,
                'vix_bucket': vbucket
            })
            daily_pnl_pts[day] = daily_pnl_pts.get(day, 0.0) + pnl

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df, pd.DataFrame()

    # daily summary (points and percentage vs base capital)
    daily = pd.Series(daily_pnl_pts).sort_index()
    base_capital = params.capital
    daily_ret_pct = daily / base_capital
    daily_summary = pd.DataFrame({
        'date': pd.to_datetime(daily.index),
        'pnl_pts': daily.values,
        'ret_pct_of_base': daily_ret_pct.values
    })
    return trades_df, daily_summary

def calc_max_dd_with_period(equity: pd.Series):
    roll_max = equity.cummax()
    drawdown = equity - roll_max
    min_dd = drawdown.min()
    if pd.isna(min_dd) or min_dd == 0:
        return 0.0, None, None
    dd_end = drawdown.idxmin()
    dd_start = equity.loc[:dd_end].idxmax()
    # Convert to string safely
    dd_start_str = str(dd_start.date()) if hasattr(dd_start, "date") else str(dd_start)
    dd_end_str = str(dd_end.date()) if hasattr(dd_end, "date") else str(dd_end)
    
    return -min_dd, dd_start_str, dd_end_str
    #return -min_dd, str(dd_start.date()), str(dd_end.date())

def calc_streaks(pnls: pd.Series):
    max_win, max_loss = 0, 0
    cur_win, cur_loss = 0, 0
    for val in pnls:
        if val > 0:
            cur_win += 1
            cur_loss = 0
        else:
            cur_loss += 1
            cur_win = 0
        max_win = max(max_win, cur_win)
        max_loss = max(max_loss, cur_loss)
    return max_win, max_loss

def summarize(trades_df: pd.DataFrame, daily_summary: pd.DataFrame,
              params: ORBParams, vix_filter: Optional[str]=None) -> BacktestResult:
    df = trades_df.copy()
    if vix_filter is not None:
        df = df[df['vix_bucket'] == vix_filter]
    if df.empty:
        return BacktestResult(params.or_minutes, 0, 0.0, 0.0, 0.0, 0.0, None, None, 0.0, 0.0, 0, 0, vix_filter or "ALL")

    wins = (df['pnl_pts'] > 0).sum()
    trades = len(df)
    win_rate = wins / trades * 100.0 if trades else 0.0
    total_pnl = df['pnl_pts'].sum()
    avg_pnl = df['pnl_pts'].mean()

    # Profit factor
    gross_profit = df.loc[df['pnl_pts'] > 0, 'pnl_pts'].sum()
    gross_loss = -df.loc[df['pnl_pts'] <= 0, 'pnl_pts'].sum()
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.inf

    # Build equity curve in points for max DD
    equity_pts = df.groupby('date')['pnl_pts'].sum().cumsum()
    max_dd_pts, dd_start, dd_end = calc_max_dd_with_period(equity_pts)
    #max_dd_pts = calc_max_dd_from_equity(equity_pts)

    # CAGR & Sharpe-like from daily summary filtered to the same dates
    if daily_summary is not None and not daily_summary.empty:
        ds = daily_summary.copy()
        # Keep only dates in this vix bucket
        keep_days = df['date'].unique()
        ds = ds[ds['date'].dt.date.astype('datetime64[ns]') .isin(pd.to_datetime(keep_days))]
        cagr = calc_cagr(ds['ret_pct_of_base'])
        sharpe_like = ds['ret_pct_of_base'].mean() / (ds['ret_pct_of_base'].std(ddof=1) + 1e-12) * np.sqrt(252) if len(ds) > 1 else 0.0
    else:
        cagr = 0.0
        sharpe_like = 0.0

    max_win_streak, max_loss_streak = calc_streaks(df['pnl_pts'])

    return BacktestResult(params.or_minutes, trades, win_rate, avg_pnl, total_pnl,
                          profit_factor, max_dd_pts, dd_start, dd_end,
                          cagr, sharpe_like, max_win_streak, max_loss_streak,
                          vix_filter or "ALL")
    #return BacktestResult(params.or_minutes, trades, win_rate, avg_pnl, total_pnl,
    #                      profit_factor, max_dd_pts, cagr, sharpe_like, vix_filter or "ALL")

def run_all(nifty_csv: str,
            vix_csv: str,
            out_dir: str = "./outputs",
            sl_mult: float = 1.0,
            tp_mult: float = 2.0,
            max_trades_per_day: int = 2,
            allow_ambiguous_breakouts: bool = False,
            eod_flat_time: str = "15:25",
            capital: float = 100000.0):
    os.makedirs(out_dir, exist_ok=True)

    nifty = load_csv(nifty_csv)
    vix = load_csv(vix_csv)

    timeframes = [5, 15, 60]
    results: List[BacktestResult] = []
    all_trades = []

    for tf in timeframes:
        params = ORBParams(
            or_minutes=tf,
            sl_mult=sl_mult,
            tp_mult=tp_mult,
            eod_flat_time=eod_flat_time,
            allow_ambiguous_breakouts=allow_ambiguous_breakouts,
            capital=capital,
            max_trades_per_day=max_trades_per_day
        )
        trades_df, daily_df = run_backtest(nifty, vix, params)
        if trades_df.empty:
            continue

        # Save trades per TF
        trades_path = os.path.join(out_dir, f"trades_orb_{tf}m.csv")
        trades_df.to_csv(trades_path, index=False)

        # Save daily summary
        daily_path = os.path.join(out_dir, f"daily_orb_{tf}m.csv")
        daily_df.to_csv(daily_path, index=False)

        # Summaries per VIX bucket and ALL
        for vb in ["<12", "12-15", ">15", "ALL"]:
            vbf = None if vb=="ALL" else vb
            res = summarize(trades_df, daily_df, params, vix_filter=vbf)
            results.append(res)

        # Keep for combined leaderboard
        trades_tf = trades_df.copy()
        trades_tf['timeframe'] = tf
        all_trades.append(trades_tf)

    if not results:
        print("No results produced. Check your CSVs and session times.")
        return

    # Results table
    res_df = pd.DataFrame([asdict(r) for r in results])
    res_df = res_df[['timeframe','vix_bucket','trades','win_rate','avg_pnl_pts','total_pnl_pts',
                     'profit_factor','max_dd_pts','dd_start','dd_end','cagr_pct','sharpe_like','max_win_streak','max_loss_streak']]
    res_df.sort_values(['vix_bucket','profit_factor','total_pnl_pts'], ascending=[True,False,False], inplace=True)

    # Save summary
    summary_path = os.path.join(out_dir, "summary_by_vix_and_tf.csv")
    res_df.to_csv(summary_path, index=False)

    # Leaderboard (overall across VIX buckets)
    leader = res_df[res_df['vix_bucket']=="ALL"].sort_values(['profit_factor','total_pnl_pts'], ascending=[False,False]).reset_index(drop=True)
    leader_path = os.path.join(out_dir, "leaderboard_overall.csv")
    leader.to_csv(leader_path, index=False)

    print("Saved:")
    print(" -", summary_path)
    print(" -", leader_path)
    for tf in [5, 15, 60]:
        tf_trades = os.path.join(out_dir, f"trades_orb_{tf}m.csv")
        tf_daily = os.path.join(out_dir, f"daily_orb_{tf}m.csv")
        if os.path.exists(tf_trades):
            print(" -", tf_trades)
        if os.path.exists(tf_daily):
            print(" -", tf_daily)

    # Optional: Print top recommendation
    if len(leader):
        best = leader.iloc[0]
        print("\nTop pick (overall VIX):")
        print(best.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NIFTY ORB backtest with VIX regimes")
    parser.add_argument("--nifty_csv", type=str, default="NIFTY_50_minute.csv")
    parser.add_argument("--vix_csv", type=str, default="INDIA_VIX_minute.csv")
    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--sl_mult", type=float, default=1.0)
    parser.add_argument("--tp_mult", type=float, default=2.0)
    parser.add_argument("--max_trades_per_day", type=int, default=2)
    parser.add_argument("--allow_ambiguous_breakouts", action="store_true")
    parser.add_argument("--eod_flat_time", type=str, default="15:25")
    parser.add_argument("--capital", type=float, default=100000.0)
    args = parser.parse_args()

    run_all(
        nifty_csv=args.nifty_csv,
        vix_csv=args.vix_csv,
        out_dir=args.out_dir,
        sl_mult=args.sl_mult,
        tp_mult=args.tp_mult,
        max_trades_per_day=args.max_trades_per_day,
        allow_ambiguous_breakouts=args.allow_ambiguous_breakouts,
        eod_flat_time=args.eod_flat_time,
        capital=args.capital
    )
