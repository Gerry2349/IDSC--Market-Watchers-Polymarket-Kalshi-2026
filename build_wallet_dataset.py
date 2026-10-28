"""
Build a CSV of wallets and their betting track records.

Wallets come from:
  - the leaderboard (all-time, this month, this week), ranked by profit and by volume
  - the top holders of the highest-volume open markets
  - the top holders of the highest-volume resolved markets
  - whoever made the most recent trades across all of Polymarket

The volume leaderboard and recent traders bring in ordinary and losing users,
not just top earners, so the dataset has something to compare the winners to.

For each wallet we pull its resolved bets (recent closed positions plus lost
bets it never redeemed) and compute win rate, profit, and how that win rate
compares to the prices the wallet paid.

Run with:
    .venv\\Scripts\\python build_wallet_dataset.py
    .venv\\Scripts\\python build_wallet_dataset.py --leaderboard 500 --markets 100 --max-closed 500

Reruns resume: wallets already in the output CSV are skipped. Pass --fresh to start over.
"""

import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from polymarket_api import PolymarketClient

_local = threading.local()

# Fixed column order, so every batch appended to the CSV lines up with the header
# (wallets with no bets return fewer fields).
COLUMNS = [
    "wallet", "name", "sources", "n_bets", "n_redeemed", "n_unredeemed", "n_open_unresolved",
    "wins", "losses", "win_rate", "avg_entry_price", "win_rate_edge", "total_pnl", "total_cost",
    "roi", "median_pnl", "first_end_date", "last_end_date", "history_truncated",
]


def get_client():
    """One client (and HTTP session) per worker thread."""
    if not hasattr(_local, "client"):
        _local.client = PolymarketClient()
    return _local.client


def collect_wallets(client, n_leaderboard, n_markets, holders_per_outcome, n_closed_markets, n_recent_trades):
    """Return {wallet: {"name": ..., "sources": set(...)}} from every wallet source."""
    wallets = {}

    def add(wallet, name, source):
        entry = wallets.setdefault(wallet.lower(), {"name": name, "sources": set()})
        entry["sources"].add(source)
        if not entry["name"] and name:
            entry["name"] = name

    def add_holders(markets, source):
        for m in markets:
            for token in client.get_holders(m["conditionId"], limit=holders_per_outcome):
                for h in token["holders"]:
                    add(h["proxyWallet"], h.get("name") or h.get("pseudonym"), source)

    # Leaderboards: PNL finds top earners, VOL finds heavy traders whether they win or lose.
    for order_by in ("PNL", "VOL"):
        for period in ("ALL", "MONTH", "WEEK"):
            leaders = client.get_leaderboard(time_period=period, order_by=order_by, max_items=n_leaderboard)
            label = "leaderboard" if order_by == "PNL" else "leaderboard_vol"
            for row in leaders:
                add(row["proxyWallet"], row.get("userName"), f"{label}_{period.lower()}")
            print(f"  leaderboard {order_by} {period}: {len(leaders)} wallets")

    markets = client.get_markets(closed=False, order="volumeNum", ascending=False, max_items=n_markets)
    add_holders(markets, "market_holder")
    print(f"  holders of top {len(markets)} open markets collected")

    if n_closed_markets:
        closed_markets = client.get_markets(closed=True, order="volumeNum", ascending=False, max_items=n_closed_markets)
        add_holders(closed_markets, "closed_market_holder")
        print(f"  holders of top {len(closed_markets)} resolved markets collected")

    if n_recent_trades:
        trades = client.get_trades(max_items=n_recent_trades)
        for t in trades:
            add(t["proxyWallet"], t.get("name") or t.get("pseudonym"), "recent_trader")
        print(f"  {len({t['proxyWallet'] for t in trades})} wallets from the last {len(trades)} trades")

    return wallets


def resolved_bets(wallet, max_closed, max_open):
    """
    All of a wallet's resolved bets as one DataFrame with columns
    endDate, avgPrice, totalBought, pnl.

    Closed positions alone are badly biased toward wins: a winning bet gets
    redeemed for cash (and becomes "closed"), but a losing bet pays nothing, so
    many users never redeem it and it stays in /positions as redeemable with
    curPrice 0. We add those back in.
    """
    client = get_client()
    cols = ["endDate", "avgPrice", "totalBought", "pnl"]

    closed = pd.DataFrame(client.get_closed_positions(wallet, max_items=max_closed))
    if not closed.empty:
        closed["pnl"] = closed["realizedPnl"]
        closed = closed[cols]

    open_ = pd.DataFrame(client.get_positions(wallet, max_items=max_open))
    n_unresolved = 0
    if not open_.empty:
        n_unresolved = int((~open_["redeemable"]).sum())
        open_ = open_[open_["redeemable"]].copy()
        # unrealized loss/gain at resolution plus anything realized from earlier partial sells
        open_["pnl"] = open_["cashPnl"] + open_["realizedPnl"]
        open_ = open_[cols]
        # If the closed history was cut off, drop unredeemed bets older than it so both cover the same period.
        if len(closed) >= max_closed:
            open_ = open_[open_["endDate"] >= closed["endDate"].min()]

    # Drop tokens the wallet never bought (e.g. Polymarket's own contracts that mint shares).
    closed = closed[closed["totalBought"] > 0] if not closed.empty else closed
    open_ = open_[open_["totalBought"] > 0] if not open_.empty else open_

    bets = pd.concat([d for d in (closed, open_) if not d.empty], ignore_index=True) if (
        not closed.empty or not open_.empty) else pd.DataFrame(columns=cols)
    return bets, len(closed), len(open_), n_unresolved


def wallet_stats(wallet, max_closed, max_open=2000):
    """Summarize one wallet's resolved bets."""
    df, n_redeemed, n_unredeemed, n_unresolved = resolved_bets(wallet, max_closed, max_open)
    base = {"wallet": wallet, "n_bets": len(df), "n_redeemed": n_redeemed,
            "n_unredeemed": n_unredeemed, "n_open_unresolved": n_unresolved}
    if df.empty:
        return base

    cost = (df["avgPrice"] * df["totalBought"]).sum()
    wins = int((df["pnl"] > 0).sum())
    win_rate = wins / len(df)
    # If the market prices were fair, a bet bought at price p wins with probability p.
    # So the average entry price is the win rate you'd "expect"; beating it is the edge.
    avg_entry = df["avgPrice"].mean()
    return {
        **base,
        "wins": wins,
        "losses": len(df) - wins,
        "win_rate": win_rate,
        "avg_entry_price": avg_entry,
        "win_rate_edge": win_rate - avg_entry,
        "total_pnl": df["pnl"].sum(),
        "total_cost": cost,
        "roi": df["pnl"].sum() / cost if cost else None,
        "median_pnl": df["pnl"].median(),
        "first_end_date": df["endDate"].min(),
        "last_end_date": df["endDate"].max(),
        "history_truncated": n_redeemed >= max_closed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--leaderboard", type=int, default=100, help="top N users from each leaderboard period")
    parser.add_argument("--markets", type=int, default=100, help="number of top open markets to pull holders from")
    parser.add_argument("--holders", type=int, default=100, help="top holders per outcome of each market")
    parser.add_argument("--closed-markets", type=int, default=50, help="number of top resolved markets to pull holders from (0 to skip)")
    parser.add_argument("--recent-trades", type=int, default=2000, help="number of most recent trades to pull traders from (0 to skip)")
    parser.add_argument("--max-closed", type=int, default=200, help="most recent closed positions per wallet")
    parser.add_argument("--workers", type=int, default=4, help="parallel requests (keep low to avoid rate limits)")
    parser.add_argument("--out", default="data/wallets.csv")
    parser.add_argument("--fresh", action="store_true", help="ignore an existing output file and start over")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(args.out) and not args.fresh:
        done = set(pd.read_csv(args.out, usecols=["wallet"])["wallet"])
        print(f"Resuming: {len(done)} wallets already in {args.out}")
    elif os.path.exists(args.out):
        os.remove(args.out)

    print("Collecting wallets...")
    wallets = collect_wallets(get_client(), args.leaderboard, args.markets, args.holders,
                              args.closed_markets, args.recent_trades)
    todo = [w for w in wallets if w not in done]
    print(f"{len(wallets)} unique wallets, {len(todo)} to fetch\n")

    start = time.time()
    batch = []
    failed = 0

    def flush():
        if batch:
            pd.DataFrame(batch).reindex(columns=COLUMNS).to_csv(args.out, mode="a", header=not os.path.exists(args.out), index=False)
            batch.clear()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(wallet_stats, w, args.max_closed): w for w in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            w = futures[fut]
            try:
                row = fut.result()
            except Exception as e:
                failed += 1
                print(f"  failed {w}: {e}")
                continue
            row["name"] = wallets[w]["name"]
            row["sources"] = ";".join(sorted(wallets[w]["sources"]))
            batch.append(row)
            if len(batch) >= 50:
                flush()
            if i % 100 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} wallets  ({time.time() - start:.0f}s)")
    flush()

    df = pd.read_csv(args.out)
    print(f"\nSaved {len(df)} wallets to {args.out} ({failed} failed)")
    solid = df[df["n_bets"] >= 30].sort_values("win_rate_edge", ascending=False)
    print(f"\nTop 10 by win-rate edge (wallets with 30+ resolved bets, {len(solid)} total):")
    print(solid[["wallet", "name", "n_bets", "n_unredeemed", "win_rate", "avg_entry_price", "win_rate_edge", "total_pnl"]]
          .head(10).to_string(index=False))


if __name__ == "__main__":
    main()
