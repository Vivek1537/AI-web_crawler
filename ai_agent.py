import sys
import os
import json
import re
import random
import asyncio
import time
from urllib.parse import urljoin, urlparse

import aiohttp
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
import google.generativeai as genai

# ==============================
# Config & Globals
# ==============================
load_dotenv()

API_KEYS = [k for k in [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
] if k]
if not API_KEYS:
    raise ValueError("No Gemini API keys found in .env")

_current_key_index = 0

MAX_DEPTH = 2
MAX_PAGES = 50
MAX_CONCURRENCY = 5

REQUEST_TIMEOUT = 30  # seconds for aiohttp
PLAYWRIGHT_TIMEOUT_MS = 30000
SCROLL_PAUSE = 1.2
MAX_SCROLLS = 8

BATCH_SIZE = 20  # AI batching

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/117.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_3) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
]

# Optional proxies. Use format "http://user:pass@host:port" if needed.
PROXIES = [
    # "http://103.152.100.170:8080",
    # "http://195.201.231.22:8080",
    # "http://64.225.8.115:9991",
]


# ==============================
# Gemini helpers (failover)
# ==============================
def _set_model():
    global _current_key_index
    api_key = API_KEYS[_current_key_index]
    print(f"🔑 Configuring Gemini with key index {_current_key_index}")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")

def safe_generate(prompt: str) -> str:
    global _current_key_index
    for _ in range(len(API_KEYS)):
        try:
            print(f"➡️  Using Gemini key index {_current_key_index}")
            model = _set_model()
            resp = model.generate_content(prompt)
            print(f"✅ Gemini key index {_current_key_index} succeeded")
            return resp.text
        except Exception as e:
            print(f"⚠️  Gemini key index {_current_key_index} failed: {e}")
            _current_key_index = (_current_key_index + 1) % len(API_KEYS)
            print(f"🔄 Switching to Gemini key index {_current_key_index}... (waiting 2s)")
            time.sleep(2)  # avoid hammering too fast
    raise RuntimeError("❌ All Gemini API keys failed.")


# ==============================
# Content extraction & quality check
# ==============================
def extract_content(soup: BeautifulSoup, base_url: str):
    data = []

    for heading in soup.find_all(re.compile(r"^h[1-3]$")):
        title = heading.get_text(strip=True)
        para = heading.find_next("p")
        desc = para.get_text(strip=True) if para else ""
        data.append({"type": "section", "title": title, "description": desc, "url": base_url})

    for link in soup.find_all("a", href=True):
        text = link.get_text(strip=True)
        if not text:
            continue
        full_url = urljoin(base_url, link["href"])
        data.append({"type": "link", "title": text, "url": full_url})

    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            rows.append(cells)
        data.append({"type": "table", "content": rows, "url": base_url})

    for img in soup.find_all("img", src=True):
        alt = img.get("alt", "")
        src = urljoin(base_url, img["src"])
        data.append({"type": "image", "alt": alt, "src": src, "url": base_url})

    return data

def is_sparse(html: str, soup: BeautifulSoup) -> bool:
    """Hybrid-mode sparsity heuristic:
    - html length < 2000
    - AND no h1/p/table/img present
    """
    if not html or len(html) < 2000:
        if not soup.find(["h1", "p", "table", "img"]):
            return True
    return False


# ==============================
# Scraper class (fast/full/hybrid)
# ==============================
class Scraper:
    def __init__(self, mode: str = "hybrid"):
        self.mode = mode.lower()
        self.session: aiohttp.ClientSession | None = None
        self.playwright = None
        self.browser = None

    async def __aenter__(self):
        if self.mode in ("fast", "hybrid"):
            timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
            self.session = aiohttp.ClientSession(timeout=timeout)
        if self.mode in ("full", "hybrid"):
            self.playwright = await async_playwright().start()
            proxy = {"server": random.choice(PROXIES)} if PROXIES else None
            self.browser = await self.playwright.chromium.launch(headless=True, proxy=proxy)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def fetch_fast(self, url: str, retries: int = 2) -> tuple[str, BeautifulSoup] | None:
        if not self.session:
            return None
        headers = {"User-Agent": random.choice(USER_AGENTS)}
        for attempt in range(retries + 1):
            try:
                proxy = random.choice(PROXIES) if PROXIES else None
                async with self.session.get(url, headers=headers, proxy=proxy) as resp:
                    html = await resp.text(errors="ignore")
                    soup = BeautifulSoup(html, "lxml")
                    return html, soup
            except Exception:
                if attempt >= retries:
                    return None
                await asyncio.sleep(2 ** attempt)
        return None

    async def fetch_full(self, url: str, retries: int = 2) -> tuple[str, BeautifulSoup] | None:
        if not self.browser:
            return None
        for attempt in range(retries + 1):
            try:
                context = await self.browser.new_context(user_agent=random.choice(USER_AGENTS))
                page = await context.new_page()
                await page.goto(url, timeout=PLAYWRIGHT_TIMEOUT_MS)
                await page.wait_for_load_state("networkidle")
                for _ in range(MAX_SCROLLS):
                    prev_h = await page.evaluate("document.body.scrollHeight")
                    await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
                    await asyncio.sleep(SCROLL_PAUSE)
                    curr_h = await page.evaluate("document.body.scrollHeight")
                    if curr_h == prev_h:
                        break
                html = await page.content()
                await context.close()
                soup = BeautifulSoup(html, "lxml")
                return html, soup
            except Exception:
                if attempt >= retries:
                    return None
                await asyncio.sleep(2 ** attempt)
        return None

    async def scrape(self, url: str) -> BeautifulSoup:
        if self.mode == "fast":
            res = await self.fetch_fast(url)
            if res:
                return res[1]
            return BeautifulSoup("<html></html>", "lxml")

        if self.mode == "full":
            res = await self.fetch_full(url)
            if res:
                return res[1]
            return BeautifulSoup("<html></html>", "lxml")

        res_fast = await self.fetch_fast(url)
        if res_fast:
            html, soup = res_fast
            if is_sparse(html, soup):
                print("Fast scrape sparse → fallback to Playwright")
                res_full = await self.fetch_full(url)
                if res_full:
                    return res_full[1]
            else:
                return soup

        print("Fast scrape failed → fallback to Playwright")
        res_full = await self.fetch_full(url)
        if res_full:
            return res_full[1]
        return BeautifulSoup("<html></html>", "lxml")


# ==============================
# Parallel crawler (BFS with queue)
# ==============================
async def crawl_site_async(start_url: str, mode: str = "hybrid",
                           max_depth: int = MAX_DEPTH, max_pages: int = MAX_PAGES):
    visited = set()
    results = []
    results_lock = asyncio.Lock()
    queue = asyncio.Queue()
    await queue.put((start_url, 0))
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async with Scraper(mode) as scraper:
        async def worker():
            while True:
                try:
                    url, depth = await queue.get()
                except asyncio.CancelledError:
                    return

                if url in visited or len(visited) >= max_pages or depth > max_depth:
                    queue.task_done()
                    continue

                visited.add(url)
                base = "{0.scheme}://{0.netloc}".format(urlparse(url))
                print(f"Crawling: {url} (depth {depth})")

                async with sem:
                    soup = await scraper.scrape(url)

                page_data = extract_content(soup, base)
                async with results_lock:
                    results.extend(page_data)

                if depth < max_depth and len(visited) < max_pages:
                    internal_links = set()
                    for a in soup.find_all("a", href=True):
                        href = urljoin(base, a["href"])
                        if href.startswith(base) and href not in visited:
                            internal_links.add(href)
                    for link in internal_links:
                        if len(visited) + queue.qsize() >= max_pages:
                            break
                        await queue.put((link, depth + 1))

                queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(MAX_CONCURRENCY)]
        await queue.join()
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    return results


# ==============================
# AI Processing (batched)
# ==============================
def ai_process_batched(data, user_request, batch_size=BATCH_SIZE):
    results = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        prompt = (
            "You are given structured snippets extracted from webpages.\n"
            "Return ONLY a valid JSON array (no code fences, no commentary) answering the task.\n\n"
            f"Snippets:\n{json.dumps(batch, indent=2)}\n\n"
            f"Task: {user_request}\n"
        )
        try:
            text = safe_generate(prompt).strip()
            m = re.search(r"\[\s*[\s\S]*\]\s*$", text)
            if m:
                parsed = json.loads(m.group(0))
                if isinstance(parsed, list):
                    results.extend(parsed)
        except Exception as e:
            print(f"AI batch error: {e}")
            continue
    return results


# ==============================
# CLI
# ==============================
def print_usage():
    print("Usage:")
    print("  Scrape:  python ai_agent.py scrape <url> [--mode fast|full|hybrid] [--max_pages N] [--max_depth D]")
    print("  Process: python ai_agent.py process \"<your request>\" [--output file.json]")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    main_mode = sys.argv[1].lower()

    if main_mode == "scrape":
        if len(sys.argv) < 3:
            print_usage()
            sys.exit(1)

        start_url = sys.argv[2]
        scrape_mode = "hybrid"
        mp = MAX_PAGES
        md = MAX_DEPTH

        if "--mode" in sys.argv:
            try:
                scrape_mode = sys.argv[sys.argv.index("--mode") + 1].lower()
            except Exception:
                pass
        if "--max_pages" in sys.argv:
            try:
                mp = int(sys.argv[sys.argv.index("--max_pages") + 1])
            except Exception:
                pass
        if "--max_depth" in sys.argv:
            try:
                md = int(sys.argv[sys.argv.index("--max_depth") + 1])
            except Exception:
                pass

        data = asyncio.run(crawl_site_async(start_url, mode=scrape_mode, max_depth=md, max_pages=mp))
        with open("scraped_data.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print("Scraping complete. Saved to scraped_data.json")

    elif main_mode == "process":
        if len(sys.argv) < 3:
            print_usage()
            sys.exit(1)

        if "--output" in sys.argv:
            idx = sys.argv.index("--output")
            user_request = " ".join(sys.argv[2:idx]).strip()
            output_file = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "ai_filtered_data.json"
        else:
            user_request = " ".join(sys.argv[2:]).strip()
            output_file = "ai_filtered_data.json"

        with open("scraped_data.json", "r", encoding="utf-8") as f:
            scraped_data = json.load(f)

        processed = ai_process_batched(scraped_data, user_request, batch_size=BATCH_SIZE)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(processed, f, indent=2)
        print(f"AI processing complete. Saved to {output_file}")

    else:
        print_usage()
        sys.exit(1)
