#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

# ---------------------------------------------------------------------------
# CONFIG – adjust if needed
# ---------------------------------------------------------------------------

NEU_REPO_PATH_DEFAULT = "/Users/kimkyeongdoun/Desktop/skyblock/NotEnoughUpdates-REPO"

BASE_URL = "https://api.hypixel.net"
HYPIXEL_API_KEY = os.getenv("HYPIXEL_API_KEY", "").strip()
HEADERS = {"API-Key": HYPIXEL_API_KEY} if HYPIXEL_API_KEY else {}

AH_TAX_RATE = 0.01  # 1% fee


class HypixelApiError(Exception):
    pass


@dataclass
class Recipe:
    output_id: str
    output_display: str
    ingredients: Dict[str, int]  # internal id -> count
    is_forge: bool


@dataclass
class FlipRow:
    output_id: str
    display_name: str
    is_forge: bool
    craft_cost: int
    cheapest_bin: int
    avg_bin: float
    listed_count: int
    profit: int
    profit_after_tax: int
    margin_pct: float
    margin_after_tax_pct: float
    confidence: int
    sampled_bins: int


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"{BASE_URL}{path}"
    resp = requests.get(url, headers=HEADERS, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if data.get("success") is False:
        raise HypixelApiError(f"API error at {url}: {data.get('cause')}")
    return data


# ---------------------------------------------------------------------------
# Name helpers / NEU parsing
# ---------------------------------------------------------------------------

def strip_color_codes(s: str) -> str:
    """Remove Minecraft '§x' color/format codes from a string."""
    out = []
    skip = False
    for ch in s:
        if skip:
            skip = False
            continue
        if ch == "§":
            skip = True
            continue
        out.append(ch)
    return "".join(out)


def parse_grid_recipe(recipe_obj: Any) -> Optional[Dict[str, int]]:
    """
    NEU 'recipe' object:
      "A2": "ENCHANTED_EYE_OF_ENDER:16"
    """
    if not isinstance(recipe_obj, dict):
        return None
    ingredients: Dict[str, int] = {}
    for val in recipe_obj.values():
        if not val:
            continue
        parts = str(val).split(":")
        item_id = parts[0].strip()
        if not item_id:
            continue
        try:
            count = int(parts[1]) if len(parts) > 1 else 1
        except ValueError:
            count = 1
        ingredients[item_id] = ingredients.get(item_id, 0) + count
    return ingredients or None


def parse_forge_recipe(rec_obj: Dict[str, Any]) -> Optional[Dict[str, int]]:
    """
    NEU forge recipe format inside 'recipes' array:
      {
        "type": "forge",
        "inputs": ["BEACON_4:1", "REFINED_MITHRIL:40", "PLASMA:5"],
        ...
      }
    """
    inputs = rec_obj.get("inputs")
    if not isinstance(inputs, list):
        return None
    ingredients: Dict[str, int] = {}
    for val in inputs:
        parts = str(val).split(":")
        item_id = parts[0].strip()
        if not item_id:
            continue
        try:
            count = int(parts[1]) if len(parts) > 1 else 1
        except ValueError:
            count = 1
        ingredients[item_id] = ingredients.get(item_id, 0) + count
    return ingredients or None


def load_neu_recipes(
    repo_path: str,
) -> Tuple[Dict[str, str], Dict[str, str], List[Recipe]]:
    """
    Scan NotEnoughUpdates-REPO/items.

    Returns:
      - id_to_display_neu: internal_id -> clean display name
      - neu_name_lower_to_id: clean display name (lower) -> internal_id
      - recipes: list[Recipe]
    """
    items_dir = os.path.join(repo_path, "items")
    if not os.path.isdir(items_dir):
        raise FileNotFoundError(f"NEU items dir not found: {items_dir}")

    id_to_display: Dict[str, str] = {}
    neu_name_lower_to_id: Dict[str, str] = {}
    recipes: List[Recipe] = []

    files_scanned = 0
    recipes_parsed = 0

    for root, _, files in os.walk(items_dir):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            path = os.path.join(root, fname)
            files_scanned += 1
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
            except Exception:
                continue
            if not text:
                continue

            objs: List[Dict[str, Any]] = []
            try:
                data = json.loads(text)
                if isinstance(data, dict):
                    objs.append(data)
                elif isinstance(data, list):
                    objs.extend(x for x in data if isinstance(x, dict))
            except json.JSONDecodeError:
                for line in text.splitlines():
                    line = line.strip().rstrip(",")
                    if not line or line in ("[", "]"):
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            objs.append(obj)
                    except json.JSONDecodeError:
                        continue

            for obj in objs:
                internal = obj.get("internalname")
                if not internal:
                    continue
                # skip level variants etc like "BLAST_PROTECTION;5"
                if ";" in internal:
                    continue

                raw_disp = obj.get("displayname") or internal
                display_clean = strip_color_codes(raw_disp).strip()

                # Skip generic "Enchanted Book" because it collides with thousands of enchants
                if display_clean == "Enchanted Book":
                    continue

                id_to_display[internal] = display_clean
                name_l = display_clean.lower()
                # keep first mapping for a given name
                neu_name_lower_to_id.setdefault(name_l, internal)

                # Normal 3x3 recipe
                if "recipe" in obj:
                    ing = parse_grid_recipe(obj["recipe"])
                    if ing:
                        recipes_parsed += 1
                        recipes.append(
                            Recipe(
                                output_id=internal,
                                output_display=display_clean,
                                ingredients=ing,
                                is_forge=False,
                            )
                        )

                # Forge recipes
                for r in obj.get("recipes", []):
                    if not isinstance(r, dict):
                        continue
                    if str(r.get("type", "")).lower() != "forge":
                        continue
                    ing = parse_forge_recipe(r)
                    if not ing:
                        continue
                    out_id = r.get("overrideOutputId") or internal
                    recipes_parsed += 1
                    recipes.append(
                        Recipe(
                            output_id=out_id,
                            output_display=display_clean,
                            ingredients=ing,
                            is_forge=True,
                        )
                    )

    print(f"NEU scan: {files_scanned} item files, {recipes_parsed} recipes parsed.")
    return id_to_display, neu_name_lower_to_id, recipes


# ---------------------------------------------------------------------------
# Bazaar
# ---------------------------------------------------------------------------

def fetch_bazaar_products() -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    for path in ("/v2/skyblock/bazaar", "/skyblock/bazaar"):
        try:
            data = _get(path)
            products = data.get("products")
            if isinstance(products, dict):
                return products
        except Exception as e:
            last_error = e
    raise HypixelApiError(f"Failed to fetch bazaar products: {last_error}")


def build_bazaar_price_map(products: Dict[str, Any]) -> Dict[str, float]:
    prices: Dict[str, float] = {}
    usable = 0

    for pid, prod in products.items():
        quick = prod.get("quick_status") or {}
        buy_price = float(quick.get("buyPrice") or 0.0)

        buy_summary = prod.get("buy_summary") or []
        if buy_summary:
            try:
                buy_price = float(buy_summary[0].get("pricePerUnit", buy_price) or buy_price)
            except (TypeError, ValueError):
                pass

        if buy_price <= 0:
            continue

        prices[pid] = buy_price
        usable += 1

    print(f"Bazaar products with usable buy-order prices: {usable}")
    return prices


# ---------------------------------------------------------------------------
# Auctions
# ---------------------------------------------------------------------------

def fetch_all_auctions(max_pages: Optional[int] = None) -> List[Dict[str, Any]]:
    auctions: List[Dict[str, Any]] = []
    last_error: Optional[Exception] = None

    for path in ("/v2/skyblock/auctions", "/skyblock/auctions"):
        try:
            first = _get(path, params={"page": 0})
            total_pages = int(first.get("totalPages", 1))
            auctions.extend(first.get("auctions", []))

            if max_pages is not None and max_pages > 0:
                total_pages = min(total_pages, max_pages)

            for page in range(1, total_pages):
                data = _get(path, params={"page": page})
                auctions.extend(data.get("auctions", []))
                time.sleep(0.05)

            print(f"Loaded {len(auctions)} active auctions")
            return auctions
        except Exception as e:
            last_error = e

    raise HypixelApiError(f"Failed to fetch auctions: {last_error}")


def fetch_items_index() -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Hypixel resources index: id -> name, plus name_lower -> id
    """
    print("Fetching Hypixel item index (fallback names)...")
    data = _get("/v2/resources/skyblock/items")
    items = data.get("items") or []

    id_to_name: Dict[str, str] = {}
    name_lower_to_id: Dict[str, str] = {}

    for it in items:
        iid = it.get("id")
        name = it.get("name")
        if not iid or not isinstance(name, str):
            continue
        id_to_name[iid] = name
        name_l = name.lower()
        name_lower_to_id.setdefault(name_l, iid)

    return id_to_name, name_lower_to_id


def build_bin_index(
    auctions: List[Dict[str, Any]],
    hypixel_name_l2id: Dict[str, str],
    neu_name_l2id: Dict[str, str],
) -> Dict[str, List[int]]:
    """
    Build internal_id -> sorted BIN price list using BOTH
    Hypixel item names and NEU names for matching.
    """
    print("Building BIN index...")
    bins: Dict[str, List[int]] = {}

    for auc in auctions:
        if not auc.get("bin"):
            continue
        name = auc.get("item_name")
        if not isinstance(name, str):
            continue
        price = auc.get("starting_bid")
        if not isinstance(price, (int, float)):
            continue
        if price <= 0:
            continue

        # Clean & normalize name
        name_clean = strip_color_codes(name).strip()
        name_l = name_clean.lower()

        internal_id = hypixel_name_l2id.get(name_l)
        if not internal_id:
            internal_id = neu_name_l2id.get(name_l)
        if not internal_id:
            # can't map this auction to a clean internal id
            continue

        bins.setdefault(internal_id, []).append(int(price))

    for iid, plist in bins.items():
        plist.sort()

    print(f"BIN index built for {len(bins)} distinct internal IDs.")
    return bins


# ---------------------------------------------------------------------------
# Craft cost computation
# ---------------------------------------------------------------------------

def build_recipe_maps(recipes: List[Recipe]) -> Dict[str, List[Recipe]]:
    by_output: Dict[str, List[Recipe]] = {}
    for r in recipes:
        by_output.setdefault(r.output_id, []).append(r)
    return by_output


def make_craft_cost_fn(
    recipes_by_output: Dict[str, List[Recipe]],
    bazaar_prices: Dict[str, float],
) -> Tuple[Dict[str, Optional[float]], Any]:
    memo: Dict[str, Optional[float]] = {}
    visiting: set = set()

    def get_cost(item_id: str) -> Optional[float]:
        if item_id in memo:
            return memo[item_id]
        if item_id in visiting:
            memo[item_id] = None
            return None

        visiting.add(item_id)
        best_cost: Optional[float] = None

        # Try crafting
        for recipe in recipes_by_output.get(item_id, []):
            total = 0.0
            ok = True
            for ing_id, count in recipe.ingredients.items():
                ing_cost = get_cost(ing_id)
                if ing_cost is None or ing_cost <= 0:
                    ok = False
                    break
                total += ing_cost * count
            if not ok:
                continue
            if best_cost is None or total < best_cost:
                best_cost = total

        # Fallback to BZ direct
        if best_cost is None:
            price = bazaar_prices.get(item_id)
            if price and price > 0:
                best_cost = price

        memo[item_id] = best_cost
        visiting.remove(item_id)
        return best_cost

    return memo, get_cost


# ---------------------------------------------------------------------------
# Volume / confidence helpers
# ---------------------------------------------------------------------------

def compute_bin_stats(
    bin_index: Dict[str, List[int]],
    item_id: str,
    sample_n: int = 5,
) -> Optional[Tuple[int, float, int, int]]:
    prices = bin_index.get(item_id)
    if not prices:
        return None
    listed_count = len(prices)
    n = min(listed_count, max(1, sample_n))
    subset = prices[:n]
    cheapest = subset[0]
    avg_price = float(sum(subset)) / n
    return cheapest, avg_price, listed_count, n


def compute_confidence(
    margin_pct: float,
    profit_after_tax: float,
    sell_price: float,
    listed_count: int,
) -> int:
    if profit_after_tax <= 0 or sell_price <= 0:
        return 0

    roi = max(0.0, min(margin_pct / 200.0, 1.0))          # saturate ~200%+
    liq = max(0.0, min(listed_count / 20.0, 1.0))          # saturate ~20 listings

    if sell_price <= 5_000_000:
        size = 1.0
    elif sell_price >= 50_000_000:
        size = 0.4
    else:
        size = 1.0 - (sell_price - 5_000_000) / 45_000_000 * 0.6
    size = max(0.4, min(size, 1.0))

    score = 100.0 * (0.45 * roi + 0.35 * liq + 0.20 * size)
    return int(round(score))


# ---------------------------------------------------------------------------
# Flip discovery
# ---------------------------------------------------------------------------

def find_flips(
    recipes: List[Recipe],
    id_to_display_neu: Dict[str, str],
    id_to_name_hypixel: Dict[str, str],
    bazaar_prices: Dict[str, float],
    bin_index: Dict[str, List[int]],
    min_profit: int,
    min_margin_pct: float,
    max_margin_pct: float,
) -> Tuple[List[FlipRow], List[FlipRow]]:
    recipes_by_output = build_recipe_maps(recipes)
    memo_costs, get_cost = make_craft_cost_fn(recipes_by_output, bazaar_prices)

    normal_flips: List[FlipRow] = []
    forge_flips: List[FlipRow] = []

    for r in recipes:
        item_id = r.output_id

        bin_stats = compute_bin_stats(bin_index, item_id, sample_n=5)
        if not bin_stats:
            continue
        cheapest, avg_bin, listed_count, sampled_n = bin_stats

        craft_cost_f = get_cost(item_id)
        if craft_cost_f is None or craft_cost_f <= 0:
            continue
        craft_cost = int(craft_cost_f)

        profit = int(round(cheapest - craft_cost))
        if profit < min_profit:
            continue

        margin_pct = (profit / craft_cost) * 100.0 if craft_cost > 0 else 0.0
        if margin_pct < min_margin_pct or margin_pct > max_margin_pct:
            continue

        sell_after_tax = cheapest * (1.0 - AH_TAX_RATE)
        profit_after_tax = int(round(sell_after_tax - craft_cost))
        margin_after_tax_pct = (
            (profit_after_tax / craft_cost) * 100.0 if craft_cost > 0 else 0.0
        )

        conf = compute_confidence(
            margin_after_tax_pct,
            profit_after_tax,
            cheapest,
            listed_count,
        )

        display_name = (
            id_to_name_hypixel.get(item_id)
            or id_to_display_neu.get(item_id)
            or item_id
        )

        row = FlipRow(
            output_id=item_id,
            display_name=display_name,
            is_forge=r.is_forge,
            craft_cost=craft_cost,
            cheapest_bin=int(cheapest),
            avg_bin=float(avg_bin),
            listed_count=listed_count,
            profit=profit,
            profit_after_tax=profit_after_tax,
            margin_pct=margin_pct,
            margin_after_tax_pct=margin_after_tax_pct,
            confidence=conf,
            sampled_bins=sampled_n,
        )

        if r.is_forge:
            forge_flips.append(row)
        else:
            normal_flips.append(row)

    normal_flips.sort(key=lambda x: x.profit, reverse=True)
    forge_flips.sort(key=lambda x: x.profit, reverse=True)
    return normal_flips, forge_flips


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_flips_csv(path: str, flips: List[FlipRow]) -> None:
    if not flips:
        print(f"[INFO] No flips to write for {path}")
        return

    fieldnames = [
        "output_id",
        "display_name",
        "is_forge",
        "craft_cost",
        "cheapest_bin",
        "avg_bin",
        "listed_count",
        "profit",
        "profit_after_tax",
        "margin_pct",
        "margin_after_tax_pct",
        "confidence",
        "sampled_bins",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for fl in flips:
            writer.writerow(
                {
                    "output_id": fl.output_id,
                    "display_name": fl.display_name,
                    "is_forge": fl.is_forge,
                    "craft_cost": fl.craft_cost,
                    "cheapest_bin": fl.cheapest_bin,
                    "avg_bin": round(fl.avg_bin, 2),
                    "listed_count": fl.listed_count,
                    "profit": fl.profit,
                    "profit_after_tax": fl.profit_after_tax,
                    "margin_pct": round(fl.margin_pct, 2),
                    "margin_after_tax_pct": round(fl.margin_after_tax_pct, 2),
                    "confidence": fl.confidence,
                    "sampled_bins": fl.sampled_bins,
                }
            )
    print(f"Wrote {len(flips)} rows to {path}")


# ---------------------------------------------------------------------------
# Run one scan
# ---------------------------------------------------------------------------

def run_once(args: argparse.Namespace) -> None:
    neu_path = args.neu_path or NEU_REPO_PATH_DEFAULT

    if not HYPIXEL_API_KEY:
        print(
            "NOTE: HYPIXEL_API_KEY is not set. "
            "Bazaar & auctions are public; a key is only needed for other endpoints."
        )

    print("Loading NEU recipes...")
    id_to_display_neu, neu_name_l2id, recipes = load_neu_recipes(neu_path)

    print("Fetching bazaar products...")
    bz_products = fetch_bazaar_products()
    bazaar_prices = build_bazaar_price_map(bz_products)

    print("Fetching auctions...")
    max_pages = None if args.pages <= 0 else args.pages
    auctions = fetch_all_auctions(max_pages=max_pages)

    id_to_name_hypixel, hypixel_name_l2id = fetch_items_index()
    bin_index = build_bin_index(auctions, hypixel_name_l2id, neu_name_l2id)

    normal_flips, forge_flips = find_flips(
        recipes=recipes,
        id_to_display_neu=id_to_display_neu,
        id_to_name_hypixel=id_to_name_hypixel,
        bazaar_prices=bazaar_prices,
        bin_index=bin_index,
        min_profit=args.min_profit,
        min_margin_pct=args.min_margin,
        max_margin_pct=args.max_margin,
    )

    top_normal = normal_flips[: args.top]
    top_forge = forge_flips[: args.top]

    if top_normal:
        print(f"\nTop {len(top_normal)} NON-FORGE flips by raw profit:")
        for i, fl in enumerate(top_normal, 1):
            print(
                f"{i:3d}. {fl.display_name:40s} "
                f"profit={fl.profit:9d} (craft={fl.craft_cost:9d}, "
                f"sell={fl.cheapest_bin:9d}, margin={fl.margin_pct:6.1f}%, "
                f"conf={fl.confidence:3d}, listings={fl.listed_count})"
            )
    else:
        print("\nNo non-forge flips found with current filters.")

    if top_forge:
        print(f"\nTop {len(top_forge)} FORGE flips by raw profit:")
        for i, fl in enumerate(top_forge, 1):
            print(
                f"{i:3d}. {fl.display_name:40s} "
                f"profit={fl.profit:9d} (craft={fl.craft_cost:9d}, "
                f"sell={fl.cheapest_bin:9d}, margin={fl.margin_pct:6.1f}%, "
                f"conf={fl.confidence:3d}, listings={fl.listed_count})"
            )
    else:
        print("\nNo forge flips found with current filters.")

    write_flips_csv(args.out_normal, top_normal)
    write_flips_csv(args.out_forge, top_forge)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hypixel SkyBlock AH flip finder using NEU recipes + Bazaar + clean BINs."
    )
    parser.add_argument(
        "--neu-path",
        type=str,
        default=NEU_REPO_PATH_DEFAULT,
        help="Path to NotEnoughUpdates-REPO (default baked into script).",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=5,
        help="Max auction pages to scan (0 or negative = all pages).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=100,
        help="How many top flips to print/write per category.",
    )
    parser.add_argument(
        "--min-profit",
        type=int,
        default=50_000,
        help="Minimum raw profit (before tax) in coins to consider a flip.",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=10.0,
        help="Minimum margin percentage (before tax) to consider a flip.",
    )
    parser.add_argument(
        "--max-margin",
        type=float,
        default=2000.0,
        help="Maximum margin percentage to keep (filters out crazy noise).",
    )
    parser.add_argument(
        "--out-normal",
        type=str,
        default="clean_craft_flips_normal.csv",
        help="Output CSV for non-forge flips.",
    )
    parser.add_argument(
        "--out-forge",
        type=str,
        default="clean_craft_flips_forge.csv",
        help="Output CSV for forge flips.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="If set, rerun scans in a loop.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Seconds between runs when using --loop.",
    )

    args = parser.parse_args()

    while True:
        print("\n=== New flip scan ===")
        try:
            run_once(args)
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")

        if not args.loop:
            break

        print(f"Sleeping {args.interval} seconds before next scan...")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
