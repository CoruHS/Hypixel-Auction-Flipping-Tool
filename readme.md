# Hypixel SkyBlock AH Flip Finder

This script scans **Hypixel SkyBlock** data (Auctions, Bazaar, and NotEnoughUpdates recipes) to find **profitable crafting & forge flips** based on current prices.

It:

- Reads crafting and forge recipes from a local **NotEnoughUpdates-REPO**.
- Fetches **Bazaar** prices for raw materials.
- Fetches **Auction House** BIN listings.
- Calculates **craft cost vs sell price** for each craftable item.
- Outputs the **best flips** (normal and forge) to the console and to CSV files.

---

## Features

- Uses **NEU recipes** to know how to craft items.
- Uses **Bazaar buy order prices** for ingredient costs.
- Uses **BIN only** auctions (no bids) for clean sell prices.
- Handles **normal crafting recipes** and **forge recipes**.
- Computes:
  - Craft cost
  - Cheapest BIN
  - Average of cheapest N BINs
  - Profit and profit after AH tax
  - Margin %
  - Confidence score (based on margin, liquidity, and item price)
- Supports one-time scans or **continuous looping scans** at a given interval.
- Exports results into **CSV files**.

---

## Requirements

- Python 3.8+
- A local copy of **NotEnoughUpdates-REPO**
- Python packages:
  - `requests`

Install dependencies (for example):

```bash
pip install requests

