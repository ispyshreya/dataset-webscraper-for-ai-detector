import argparse
import csv
import hashlib
import json
import os
import random
import re
import time
import urllib.parse
import urllib.robotparser
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Tuple, cast

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from PIL import Image, ExifTags
import shutil

# Heuristic keywords for initial label suggestions.
AI_HINTS = [
    "ai", "generated", "stable diffusion", "midjourney", "dalle", "gpt-image", "sdxl", "synthetic",
    "cgi", "render", "3d render", "digital art", "ai art", "neural network", "deepfakes"
]
REAL_HINTS = [
    "photo", "photograph", "real", "camera", "shot", "portrait", "landscape", "street", "travel",
    "wedding", "snapshot", "snapshot"
]

KNOWN_AI_DOMAINS = [
    "midjourney.com",
    "stablediffusionweb.com",
    "huggingface.co",
    "artbreeder.com",
    "nightcafe.studio",
    "dalle.com",
    "lexica.art",
    "dreamstudio.ai",
]

KNOWN_REAL_DOMAINS = [
    "unsplash.com",
    "pexels.com",
    "pixabay.com",
    "flickr.com",
    "500px.com",
    "stocksnap.io",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Scrape images from a list of pages, save them locally, and collect dataset metadata."
    )
    parser.add_argument(
        "--sources",
        default="sources.txt",
        help="Text file containing one page URL per line to scrape for images.",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset",
        help="Directory to save images and metadata files.",
    )
    parser.add_argument(
        "--csv",
        default="dataset.csv",
        help="Output CSV file path for dataset metadata.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Maximum number of expanded source pages to scrape. Use 0 for no limit.",
    )
    parser.add_argument(
        "--pages-per-source",
        type=int,
        default=1,
        help="Generate this many paginated variants for each source URL when possible.",
    )
    parser.add_argument(
        "--max-images-per-page",
        type=int,
        default=0,
        help="Maximum number of images to inspect from each page. Use 0 for no limit.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between requests in seconds.",
    )
    parser.add_argument(
        "--real-sources",
        default=None,
        help="Text file containing one page URL per line for reliable real image sources.",
    )
    parser.add_argument(
        "--ai-sources",
        default=None,
        help="Text file containing one page URL per line for reliable AI image sources.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=0,
        help="Optional cumulative target number of images per label (real and ai). Use 0 for no cumulative cap.",
    )
    parser.add_argument(
        "--target-new-images",
        type=int,
        default=10000,
        help="Target number of new, non-duplicate images to add during this run.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of kept images assigned to the train split.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.0,
        help="Fraction of kept images assigned to the validation split.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=224,
        help="Resize images to this square dimension before saving.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Timeout in seconds for HTTP requests.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries for failed HTTP requests.",
    )
    parser.add_argument(
        "--backoff-factor",
        type=float,
        default=0.5,
        help="Backoff factor for retry delays.",
    )
    parser.add_argument(
        "--perceptual-threshold",
        type=int,
        default=4,
        help="Maximum Hamming distance for perceptual duplicate detection.",
    )
    parser.add_argument(
        "--ignore-robots",
        action="store_true",
        help="Ignore robots.txt checks and scrape page content regardless of robots rules.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete existing dataset files and metadata before starting a fresh run.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used for nondeterministic fallback behavior.",
    )
    return parser.parse_args()


def read_source_urls(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Source file not found: {path}")

    urls = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                urls.append(stripped)
    return urls


def expand_paginated_url(url: str, page_number: int) -> str:
    if page_number <= 1:
        return url

    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    query["page"] = [str(page_number)]
    expanded_query = urllib.parse.urlencode(query, doseq=True)
    return urllib.parse.urlunparse(parsed._replace(query=expanded_query))


def is_allowed_by_robots(url: str, user_agent: str = "*") -> bool:
    parsed = urllib.parse.urlparse(url)
    robots_url = urllib.parse.urljoin(f"{parsed.scheme}://{parsed.netloc}", "/robots.txt")
    rp = urllib.robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception:
        return True


def normalize_image_url(page_url: str, src: str) -> str:
    src = src.strip()
    if not src:
        return ""
    return urllib.parse.urljoin(page_url, src)


def create_session(max_retries: int, backoff_factor: float) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(HEADERS)
    return session


def fetch_page(url: str, session: requests.Session, timeout: int = 15) -> str:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def parse_images_from_html(page_url: str, html: str):
    soup = BeautifulSoup(html, "html.parser")
    images = []
    page_title = (soup.title.string or "").strip() if soup.title else ""
    surrounding_text = " ".join(
        [tag.get_text(separator=" ", strip=True) for tag in soup.find_all(["p", "span", "div", "section"])][:10]
    )

    for img in soup.find_all("img"):
        src = (
            img.get("src")
            or img.get("data-src")
            or img.get("data-lazy-src")
            or img.get("data-srcset")
            or img.get("srcset")
        )
        src = str(src or "").strip()
        if not src:
            continue

        if "," in src and "srcset" in img.attrs:
            src = src.split(",")[0].strip().split(" ")[0]

        image_url = normalize_image_url(page_url, src)
        if not image_url:
            continue
        normalized_lower = image_url.lower()
        if normalized_lower.startswith("data:") or normalized_lower.endswith(".svg"):
            continue

        alt_text = str(img.get("alt") or "").strip()
        title_text = str(img.get("title") or "").strip()
        images.append(
            {
                "page_url": page_url,
                "page_title": page_title,
                "page_context": surrounding_text,
                "image_url": image_url,
                "img_alt": alt_text,
                "img_title": title_text,
            }
        )
    return images


def download_image(image_url: str, session: requests.Session, timeout: int = 20):
    response = session.get(image_url, timeout=timeout)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "").lower()
    if "svg" in content_type or image_url.lower().endswith(".svg"):
        raise ValueError(f"Unsupported SVG image skipped: {image_url}")
    if not content_type.startswith("image"):
        raise ValueError(f"Not an image: {image_url} ({content_type})")

    return response.content


def resize_and_save_image(image_bytes: bytes, output_path: Path, size: int):
    try:
        img = Image.open(BytesIO(image_bytes))
        img = img.convert("RGB")
        img = img.resize((size, size), resample=Image.Resampling.LANCZOS)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, format="JPEG", quality=95)
    except Exception as exc:
        raise ValueError(f"Failed to resize or save image: {exc}")


def compute_perceptual_hash(image_bytes: bytes, hash_size: int = 8) -> str:
    img = Image.open(BytesIO(image_bytes)).convert("L")
    img = img.resize((hash_size, hash_size), resample=Image.Resampling.LANCZOS)
    get_pixels = getattr(img, "get_flattened_data", None)
    raw_pixels: Any = get_pixels() if callable(get_pixels) else img.getdata()
    raw_pixels = list(raw_pixels)
    pixels = []
    for pixel in raw_pixels:
        if isinstance(pixel, tuple):
            pixels.append(int(pixel[0]))
        elif pixel is None:
            pixels.append(0)
        else:
            pixels.append(int(pixel))
    avg = sum(pixels) / len(pixels)
    bits = ["1" if pixel > avg else "0" for pixel in pixels]
    return "".join(bits)


def hamming_distance(hash_a: str, hash_b: str) -> int:
    return sum(ch1 != ch2 for ch1, ch2 in zip(hash_a, hash_b))


def load_existing_dataset_state(output_dir: Path, csv_path: Path):
    counts = {"ai": {"train": 0, "val": 0, "test": 0}, "real": {"train": 0, "val": 0, "test": 0}}
    existing_hashes = set()
    existing_phashes = []

    for label in counts:
        for split in counts[label]:
            candidate_dir = output_dir / label / split
            if candidate_dir.exists():
                counts[label][split] += sum(1 for _ in candidate_dir.iterdir() if _.is_file())

    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row.get("image_hash"):
                    existing_hashes.add(row["image_hash"])
                if row.get("perceptual_hash"):
                    existing_phashes.append(row["perceptual_hash"])

    if not existing_hashes or not existing_phashes:
        for label in counts:
            for split in counts[label]:
                candidate_dir = output_dir / label / split
                if candidate_dir.exists():
                    for image_file in candidate_dir.iterdir():
                        if image_file.is_file():
                            try:
                                content = image_file.read_bytes()
                                existing_hashes.add(hashlib.sha256(content).hexdigest())
                                existing_phashes.append(compute_perceptual_hash(content))
                            except Exception:
                                continue

    return counts, existing_hashes, existing_phashes


def determine_split(image_hash: str, train_ratio: float, val_ratio: float) -> str:
    bucket = int(image_hash[:8], 16) % 100
    train_end = int(train_ratio * 100)
    val_end = train_end + int(val_ratio * 100)

    if bucket < train_end:
        return "train"
    if bucket < val_end:
        return "val"
    return "test"


def select_split(image_hash: str, label: str, counts: dict, target_count: int, train_ratio: float, val_ratio: float):
    if target_count <= 0:
        return determine_split(image_hash, train_ratio, val_ratio)

    train_target = int(target_count * train_ratio)
    val_target = int(target_count * val_ratio)
    test_target = target_count - train_target - val_target
    targets = {"train": train_target, "val": val_target, "test": test_target}
    label_counts = counts[label]

    if all(label_counts[split] >= targets[split] for split in targets):
        return None

    preferred = determine_split(image_hash, train_ratio, val_ratio)
    if label_counts[preferred] < targets[preferred]:
        return preferred

    for split in ["train", "val", "test"]:
        if label_counts[split] < targets[split]:
            return split

    return None


def extract_exif(image_bytes: bytes) -> dict:
    try:
        img = Image.open(BytesIO(image_bytes))
        raw_exif: Any = {}
        exif_data = getattr(img, "getexif", None)
        if callable(exif_data):
            raw_exif = exif_data() or {}
        if not raw_exif:
            return {}

        exif = {}
        if hasattr(raw_exif, "items"):
            for key, value in cast(dict, raw_exif).items():
                name = ExifTags.TAGS.get(key, key)
                exif[name] = value
        return exif
    except Exception:
        return {}


def find_label_and_reason(record: dict) -> tuple[str, str]:
    text = " ".join(
        [record.get("page_title", ""), record.get("page_context", ""), record.get("img_alt", ""), record.get("img_title", "")]
    ).lower()
    domain = urllib.parse.urlparse(record["page_url"]).netloc.lower()
    exif = record.get("exif", {})
    reasons = []

    if any(domain.endswith(known) for known in KNOWN_AI_DOMAINS):
        reasons.append("source domain suggests AI content")
    if any(domain.endswith(known) for known in KNOWN_REAL_DOMAINS):
        reasons.append("source domain suggests real photo")

    for hint in AI_HINTS:
        if hint in text:
            reasons.append(f"context contains AI hint '{hint}'")
            break

    for hint in REAL_HINTS:
        if hint in text:
            reasons.append(f"context contains real/photo hint '{hint}'")
            break

    software = str(exif.get("Software", "")).lower()
    if software and any(term in software for term in ["stable", "diffusion", "midjourney", "dalle", "photoshop", "gimp"]):
        reasons.append(f"metadata software is '{software}'")

    if exif.get("Make") or exif.get("Model"):
        reasons.append("camera EXIF metadata present")

    if any("ai hint" in r for r in reasons) or any(term in software for term in ["stable", "diffusion", "midjourney", "dalle"]):
        return "ai", "; ".join(reasons)

    if any("real/photo hint" in r for r in reasons) or exif.get("Make") or exif.get("Model"):
        return "real", "; ".join(reasons)

    if reasons:
        return "unknown", "; ".join(reasons)

    return "unknown", "no strong heuristics found"


def ensure_directory(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def next_image_index(directory: Path) -> int:
    max_index = 0
    if directory.exists():
        for image_file in directory.iterdir():
            if image_file.is_file() and image_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                if image_file.stem.isdigit():
                    max_index = max(max_index, int(image_file.stem))
    return max_index + 1


def build_source_list(args):
    source_urls = []
    source_label_map = {}
    seen_urls = set()

    def add_source_file(path: str, label: str):
        for url in read_source_urls(path):
            for page_number in range(1, args.pages_per_source + 1):
                expanded_url = expand_paginated_url(url, page_number)
                if expanded_url in seen_urls:
                    continue
                seen_urls.add(expanded_url)
                source_urls.append(expanded_url)
                source_label_map[expanded_url] = label

    if args.real_sources:
        add_source_file(args.real_sources, "real")
    if args.ai_sources:
        add_source_file(args.ai_sources, "ai")

    if not args.real_sources and not args.ai_sources:
        for url in read_source_urls(args.sources):
            for page_number in range(1, args.pages_per_source + 1):
                expanded_url = expand_paginated_url(url, page_number)
                if expanded_url not in seen_urls:
                    seen_urls.add(expanded_url)
                    source_urls.append(expanded_url)
                    source_label_map[expanded_url] = "unknown"
    elif args.sources and os.path.exists(args.sources):
        for url in read_source_urls(args.sources):
            for page_number in range(1, args.pages_per_source + 1):
                expanded_url = expand_paginated_url(url, page_number)
                if expanded_url not in seen_urls:
                    seen_urls.add(expanded_url)
                    source_urls.append(expanded_url)
                    source_label_map[expanded_url] = "unknown"

    if args.max_pages > 0:
        return source_urls[: args.max_pages], source_label_map
    return source_urls, source_label_map


def dataset_filled(counts: dict, target_count: int, train_ratio: float, val_ratio: float) -> bool:
    if target_count <= 0:
        return False

    train_target = int(target_count * train_ratio)
    val_target = int(target_count * val_ratio)
    test_target = target_count - train_target - val_target
    for label in ["ai", "real"]:
        if counts[label]["train"] < train_target:
            return False
        if counts[label]["val"] < val_target:
            return False
        if counts[label]["test"] < test_target:
            return False
    return True


def write_csv_header(path: str):
    csv_path = Path(path)
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return

    with csv_path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "image_id",
                "image_hash",
                "perceptual_hash",
                "image_url",
                "page_url",
                "page_title",
                "img_alt",
                "img_title",
                "source_domain",
                "source_label",
                "split",
                "local_path",
                "predicted_label",
                "label_reason",
                "exif_json",
            ],
        )
        writer.writeheader()


def append_csv_row(path: str, row: dict):
    with open(path, "a", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row.keys())
        writer.writerow(row)


def main():
    args = parse_arguments()
    if args.train_ratio < 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train-ratio and val-ratio must sum to less than 1.0 and be non-negative.")
    if args.pages_per_source < 1:
        raise ValueError("pages-per-source must be at least 1.")
    if args.max_pages < 0:
        raise ValueError("max-pages must be 0 or greater.")
    if args.max_images_per_page < 0:
        raise ValueError("max-images-per-page must be 0 or greater.")
    if args.target_new_images < 1:
        raise ValueError("target-new-images must be at least 1.")
    if args.target_count < 0:
        raise ValueError("target-count must be 0 or greater.")

    random.seed(args.random_seed)
    source_urls, source_label_map = build_source_list(args)
    output_dir = Path(args.output_dir)
    metadata_dir = output_dir / "metadata"

    if args.fresh:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        if Path(args.csv).exists():
            Path(args.csv).unlink()

    ensure_directory(output_dir)
    ensure_directory(metadata_dir)

    for label in ["ai", "real"]:
        for split in ["train", "val", "test"]:
            ensure_directory(output_dir / label / split)

    write_csv_header(args.csv)
    counts, existing_hashes, existing_phashes = load_existing_dataset_state(output_dir, Path(args.csv))

    image_id = 0
    if Path(args.csv).exists():
        with open(args.csv, "r", encoding="utf-8", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    image_id = max(image_id, int(row.get("image_id", 0)))
                except ValueError:
                    continue

    session = create_session(args.max_retries, args.backoff_factor)
    added_this_run = 0
    downloaded_this_run = 0
    skipped_duplicates = 0

    for page_index, page_url in enumerate(source_urls, start=1):
        if added_this_run >= args.target_new_images:
            print(f"Per-run target reached: {added_this_run}/{args.target_new_images} new images.")
            break

        print(f"[{page_index}/{len(source_urls)}] Scraping page: {page_url}")
        if not args.ignore_robots and not is_allowed_by_robots(page_url, user_agent=HEADERS["User-Agent"]):
            print(f"  Skipping due to robots.txt rules: {page_url}")
            print("  Use --ignore-robots to override this behavior if you understand the policy implications.")
            continue

        if dataset_filled(counts, args.target_count, args.train_ratio, args.val_ratio):
            print("Target counts for all labels and splits reached. Stopping early.")
            break

        try:
            html = fetch_page(page_url, session, timeout=args.timeout)
        except Exception as exc:
            print(f"  Failed to fetch page: {exc}")
            continue

        images = parse_images_from_html(page_url, html)
        if not images:
            print("  No images found on this page.")
            continue

        images_to_process = images if args.max_images_per_page == 0 else images[: args.max_images_per_page]
        for record_index, image_record in enumerate(images_to_process, start=1):
            if added_this_run >= args.target_new_images:
                print(f"Per-run target reached: {added_this_run}/{args.target_new_images} new images.")
                break

            if dataset_filled(counts, args.target_count, args.train_ratio, args.val_ratio):
                print("Target counts for all labels and splits reached. Stopping early.")
                break

            print(f"  [{record_index}/{len(images_to_process)}] {image_record['image_url']}")
            try:
                image_bytes = download_image(image_record["image_url"], session, timeout=args.timeout)
                downloaded_this_run += 1
            except Exception as exc:
                print(f"    Download failed: {exc}")
                continue

            image_hash = hashlib.sha256(image_bytes).hexdigest()
            perceptual_hash = compute_perceptual_hash(image_bytes)

            if image_hash in existing_hashes:
                print("    Exact duplicate image skipped.")
                skipped_duplicates += 1
                continue

            duplicate_found = False
            for existing in existing_phashes:
                if hamming_distance(perceptual_hash, existing) <= args.perceptual_threshold:
                    print("    Perceptual duplicate image skipped.")
                    duplicate_found = True
                    break
            if duplicate_found:
                skipped_duplicates += 1
                continue

            exif = extract_exif(image_bytes)
            image_record["exif"] = exif
            source_label = source_label_map.get(page_url, "unknown")
            if source_label in ["ai", "real"]:
                label = source_label
                reason = f"source file label '{label}'"
            else:
                label, reason = find_label_and_reason(image_record)
                if label not in ["ai", "real"]:
                    print(f"    Skipping unknown/uncertain label: {label}")
                    continue

            split = select_split(image_hash, label, counts, args.target_count, args.train_ratio, args.val_ratio)
            if split is None:
                print(f"    {label} target reached for all splits. Skipping.")
                continue

            image_id += 1
            local_index = next_image_index(output_dir / label / split)
            relative_path = output_dir / label / split / f"{local_index}.jpg"
            try:
                resize_and_save_image(image_bytes, relative_path, args.size)
            except Exception as exc:
                print(f"    Failed to resize image: {exc}")
                continue

            counts[label][split] += 1
            added_this_run += 1
            existing_hashes.add(image_hash)
            existing_phashes.append(perceptual_hash)

            row = {
                "image_id": image_id,
                "image_hash": image_hash,
                "perceptual_hash": perceptual_hash,
                "image_url": image_record["image_url"],
                "page_url": image_record["page_url"],
                "page_title": image_record["page_title"],
                "img_alt": image_record["img_alt"],
                "img_title": image_record["img_title"],
                "source_domain": urllib.parse.urlparse(image_record["page_url"]).netloc,
                "source_label": source_label,
                "split": split,
                "local_path": str(relative_path),
                "predicted_label": label,
                "label_reason": reason,
                "exif_json": json.dumps(exif if source_label != "unknown" else image_record["exif"], ensure_ascii=False),
            }
            append_csv_row(args.csv, row)

            metadata_path = metadata_dir / f"{image_id}.json"
            with open(metadata_path, "w", encoding="utf-8") as metadata_file:
                json.dump(
                    {
                        **image_record,
                        "predicted_label": label,
                        "label_reason": reason,
                        "split": split,
                        "image_hash": image_hash,
                        "perceptual_hash": perceptual_hash,
                        "source_label": source_label,
                    },
                    metadata_file,
                    ensure_ascii=False,
                    indent=2,
                )

            time.sleep(args.delay)

        else:
            continue
        break

    print(f"Dataset collection complete. CSV saved to: {args.csv}")
    print(f"Images saved to: {output_dir}")
    print(f"Metadata saved to: {metadata_dir}")
    print(f"New images added this run: {added_this_run}/{args.target_new_images}")
    print(f"Images downloaded this run: {downloaded_this_run}")
    print(f"Duplicate images skipped this run: {skipped_duplicates}")
    print(f"Current counts: {json.dumps(counts, sort_keys=True)}")
    if added_this_run < args.target_new_images:
        print(
            "Warning: target-new-images was not reached. Add more source URLs, increase --pages-per-source, "
            "raise --max-images-per-page, or lower duplicate strictness with --perceptual-threshold."
        )


if __name__ == "__main__":
    main()
