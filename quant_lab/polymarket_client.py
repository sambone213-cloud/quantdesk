"""
Polymarket CLOB + Gamma API Client — Read-Only (no auth required for market data)
CLOB: https://clob.polymarket.com
Gamma: https://gamma-api.polymarket.com

Price format: 0.0 – 1.0 (USDC per share, normalized from cents)
Market ID: condition_id (hex string)
Token ID: token_id (for specific YES/NO sides)
"""

import time
import threading
import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import requests


# ─── Data Models ────────────────────────────────────────────────────────────

@dataclass
class PolymarketMarket:
    condition_id: str
    question: str
    yes_token_id: str
    no_token_id: str
    yes_price: float        # 0.0 – 1.0
    no_price: float
    volume_24h: float
    liquidity: float
    end_date: str
    active: bool

    @property
    def mid(self) -> float:
        return self.yes_price

    @property
    def spread(self) -> float:
        # In Polymarket, yes + no = 1.0 ideally; spread reflects book depth
        return abs(1.0 - self.yes_price - self.no_price)


@dataclass
class PolymarketOrderBook:
    token_id: str
    bids: List[Tuple[float, float]]   # [(price, size_usdc), ...]
    asks: List[Tuple[float, float]]
    timestamp: float = field(default_factory=time.time)

    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else 1.0

    @property
    def mid(self) -> float:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return self.best_bid or self.best_ask or 0.5

    @property
    def depth_imbalance(self) -> float:
        """Order book imbalance: >0.5 = buying pressure."""
        bid_vol = sum(s for _, s in self.bids[:5])
        ask_vol = sum(s for _, s in self.asks[:5])
        total = bid_vol + ask_vol
        return bid_vol / total if total > 0 else 0.5


# ─── Client ──────────────────────────────────────────────────────────────────

class PolymarketClient:
    """
    Read-only Polymarket data client.
    No authentication required for all methods below.
    """

    CLOB_URL  = "https://clob.polymarket.com"
    GAMMA_URL = "https://gamma-api.polymarket.com"

    def __init__(self, timeout: int = 10):
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "QuantPMStack/2.0"})

    def _clob_get(self, path: str, params: dict = None) -> dict:
        resp = self._session.get(self.CLOB_URL + path, params=params, timeout=self._timeout)
        resp.raise_for_status()
        return resp.json()

    def _gamma_get(self, path: str, params: dict = None) -> dict:
        resp = self._session.get(self.GAMMA_URL + path, params=params, timeout=self._timeout)
        resp.raise_for_status()
        return resp.json()

    # ── Market Discovery ──────────────────────────────────────────────────

    def search_markets(
        self,
        query: str = "",
        limit: int = 20,
        active_only: bool = True,
        min_volume: float = 0,
        debug: bool = False,
    ) -> List[PolymarketMarket]:
        """Search markets via Gamma API."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)

        params = {
            "limit": limit,
            "active": str(active_only).lower(),
            "closed": "false",
            "order": "volume24hr",
            "ascending": "false",
        }
        if query:
            params["q"] = query
        try:
            data = self._gamma_get("/markets", params)
            markets_raw = data if isinstance(data, list) else data.get("data", data.get("markets", []))
            if debug:
                print(f"[DEBUG] Raw response type: {type(data)}, items: {len(markets_raw)}")
                if markets_raw:
                    print(f"[DEBUG] First item keys: {list(markets_raw[0].keys())[:20]}")
                    print(f"[DEBUG] clobTokenIds: {markets_raw[0].get('clobTokenIds', 'MISSING')[:80]}")
            markets = []
            parse_errors = 0
            for m in markets_raw:
                # Skip closed or archived markets
                if m.get("closed") or m.get("archived"):
                    continue
                # Skip markets whose end date has already passed
                end_date_str = m.get("endDateIso") or m.get("endDate", "")
                if end_date_str:
                    try:
                        end_date_str_clean = end_date_str[:10]  # take YYYY-MM-DD portion
                        end_dt = datetime.strptime(end_date_str_clean, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                        if end_dt < now:
                            continue
                    except Exception:
                        pass

                parsed = self._parse_gamma_market(m, debug=debug)
                if parsed and parsed.yes_token_id and parsed.volume_24h >= min_volume:
                    markets.append(parsed)
                elif parsed is None:
                    parse_errors += 1
            if debug and parse_errors:
                print(f"[DEBUG] {parse_errors} markets failed to parse")
            return markets
        except Exception as e:
            if debug:
                print(f"[DEBUG] search_markets error: {e}")
            return []

    def get_market(self, condition_id: str) -> Optional[PolymarketMarket]:
        try:
            data = self._gamma_get(f"/markets/{condition_id}")
            return self._parse_gamma_market(data)
        except Exception:
            return None

    def get_order_book(self, token_id: str) -> Optional[PolymarketOrderBook]:
        try:
            data = self._clob_get("/book", {"token_id": token_id})
            bids = [(float(b["price"]), float(b["size"])) for b in data.get("bids", [])]
            asks = [(float(a["price"]), float(a["size"])) for a in data.get("asks", [])]
            bids.sort(key=lambda x: -x[0])
            asks.sort(key=lambda x: x[0])
            return PolymarketOrderBook(token_id=token_id, bids=bids, asks=asks)
        except Exception:
            return None

    def get_midpoint(self, token_id: str) -> Optional[float]:
        try:
            data = self._clob_get("/midpoint", {"token_id": token_id})
            mid = float(data.get("mid", 0))
            return mid if 0 < mid < 1 else None
        except Exception:
            return None

    def get_price_history(self, condition_id: str, fidelity: int = 60) -> List[Tuple[int, float]]:
        """
        Returns (timestamp_ms, price) list from Polymarket's timeseries.
        fidelity: seconds between data points (default 60 = 1 minute)
        """
        try:
            data = self._clob_get(
                "/prices-history",
                {"market": condition_id, "fidelity": fidelity}
            )
            history = data.get("history", [])
            return [(int(h["t"] * 1000), float(h["p"])) for h in history if "t" in h and "p" in h]
        except Exception:
            return []

    def get_multiple_order_books(self, token_ids: List[str]) -> List[Optional[PolymarketOrderBook]]:
        """Batch fetch order books."""
        return [self.get_order_book(tid) for tid in token_ids]

    # ── Parsing ───────────────────────────────────────────────────────────

    def _parse_gamma_market(self, m: dict, debug: bool = False) -> Optional[PolymarketMarket]:
        try:
            # clobTokenIds — may be a stringified JSON array or a real list
            clob_ids_raw = m.get("clobTokenIds", "[]")
            if isinstance(clob_ids_raw, str):
                try:
                    clob_ids = json.loads(clob_ids_raw)
                except Exception:
                    clob_ids = []
            elif isinstance(clob_ids_raw, list):
                clob_ids = clob_ids_raw
            else:
                clob_ids = []

            # Token IDs may be 0x hex strings OR large decimal integer strings — both are valid
            yes_token_id = str(clob_ids[0]) if len(clob_ids) > 0 else ""
            no_token_id  = str(clob_ids[1]) if len(clob_ids) > 1 else ""

            # Fallback: some endpoints return a `tokens` array
            tokens = m.get("tokens", [])
            if not yes_token_id and tokens:
                yes_token = next((t for t in tokens if t.get("outcome", "").lower() == "yes"), tokens[0] if tokens else {})
                no_token  = next((t for t in tokens if t.get("outcome", "").lower() == "no"),  tokens[1] if len(tokens) > 1 else {})
                yes_token_id = str(yes_token.get("token_id", ""))
                no_token_id  = str(no_token.get("token_id", ""))

            # outcomePrices — stringified JSON array or real list
            outcome_prices = m.get("outcomePrices", "[]")
            if isinstance(outcome_prices, str):
                try:
                    prices = json.loads(outcome_prices)
                except Exception:
                    prices = ["0.5", "0.5"]
            elif isinstance(outcome_prices, list):
                prices = outcome_prices
            else:
                prices = ["0.5", "0.5"]

            yes_price = float(prices[0]) if prices else 0.5
            no_price  = float(prices[1]) if len(prices) > 1 else 1.0 - yes_price

            # Use lastTradePrice / bestBid as a better price source when available
            last_trade = m.get("lastTradePrice")
            if last_trade is not None:
                try:
                    yes_price = float(last_trade)
                    no_price  = round(1.0 - yes_price, 6)
                except Exception:
                    pass

            # conditionId is the canonical market ID; fall back to numeric `id`
            condition_id = m.get("conditionId") or m.get("id", "")

            return PolymarketMarket(
                condition_id=str(condition_id),
                question=m.get("question", m.get("title", "")),
                yes_token_id=yes_token_id,
                no_token_id=no_token_id,
                yes_price=yes_price,
                no_price=no_price,
                volume_24h=float(m.get("volume24hr", m.get("volumeNum", 0)) or 0),
                liquidity=float(m.get("liquidityNum", m.get("liquidity", 0)) or 0),
                end_date=str(m.get("endDateIso", m.get("endDate", ""))),
                active=bool(m.get("active", True)),
            )
        except Exception as e:
            if debug:
                print(f"[parse error] {e} — keys: {list(m.keys())[:10]}")
            return None

    def poll_price(self, token_id: str) -> Optional[float]:
        """Returns current YES mid price (0-1) or None."""
        return self.get_midpoint(token_id)


# ─── WebSocket Streaming ─────────────────────────────────────────────────────

class PolymarketWebSocket:
    """
    Streams live price updates from Polymarket's WebSocket.
    Calls price_callback(token_id, mid_price) on each price update.
    No authentication required.
    """

    WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

    def __init__(self, token_ids: List[str], price_callback: Callable):
        self.token_ids = token_ids
        self.price_callback = price_callback
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _run(self):
        try:
            import websocket

            def on_message(ws, message):
                try:
                    events = json.loads(message)
                    if not isinstance(events, list):
                        events = [events]
                    for event in events:
                        event_type = event.get("event_type", "")
                        asset_id = event.get("asset_id", "")
                        if event_type == "price_change" and asset_id:
                            price_str = event.get("price")
                            if price_str:
                                price = float(price_str)
                                if 0 < price < 1:
                                    self.price_callback(asset_id, price)
                        elif event_type == "book" and asset_id:
                            # Extract mid from book snapshot
                            bids = event.get("bids", [])
                            asks = event.get("asks", [])
                            if bids and asks:
                                best_bid = float(bids[0]["price"]) if bids else 0
                                best_ask = float(asks[0]["price"]) if asks else 1
                                mid = (best_bid + best_ask) / 2
                                if 0 < mid < 1:
                                    self.price_callback(asset_id, mid)
                except Exception:
                    pass

            def on_open(ws):
                sub = {
                    "assets_ids": self.token_ids,
                    "type": "market",
                }
                ws.send(json.dumps(sub))

            while self._running:
                try:
                    ws = websocket.WebSocketApp(
                        self.WS_URL,
                        on_message=on_message,
                        on_open=on_open,
                    )
                    ws.run_forever(ping_interval=30, ping_timeout=10)
                except Exception:
                    time.sleep(5)  # reconnect on error

        except ImportError:
            pass  # websocket-client not installed
