# Dataset Webscraper for AI Detector

This repository contains a Python scraper that collects image datasets for AI vs real image classification.

## Features

- Scrapes image URLs from HTML pages listed in source files
- Downloads images and resizes them to `224x224`
- Saves images into separate folders for `real` and `ai`
- Splits dataset into `train`, `val`, and `test`
- Uses exact and perceptual duplicate detection
- Stores metadata in `dataset.csv` and JSON metadata files

## Files

- `Script.py` — main scraper script
- `sources.txt` — generic source list for fallback scraping
- `.gitignore` — excludes virtual environment, dataset output, and temporary files
- `README.md` — usage instructions

## Usage

### Install dependencies

```powershell
pip install requests beautifulsoup4 pillow urllib3
```

### Run the scraper

Use explicit source files for real and AI sources whenever possible:

```powershell
& 'C:\Program Files\Git\cmd\git.exe' git config --global user.name "Your Name"
& 'C:\Program Files\Git\cmd\git.exe' git config --global user.email "you@example.com"
python Script.py --real-sources real_sources.txt --ai-sources ai_sources.txt --output-dir dataset --csv dataset.csv --target-count 10000 --train-ratio 0.8 --val-ratio 0.1 --size 224
```

If you prefer a generic source list, use:

```powershell
python Script.py --sources sources.txt --output-dir dataset --csv dataset.csv --target-count 10000 --train-ratio 0.8 --val-ratio 0.1 --size 224
```

## Output structure

The script saves images as:

- `dataset/real/train`
- `dataset/real/val`
- `dataset/real/test`
- `dataset/ai/train`
- `dataset/ai/val`
- `dataset/ai/test`

Images are named sequentially as `1.jpg`, `2.jpg`, etc.

## Notes

- The dataset labels are derived from source files and heuristics.
- Use curated real and AI source files for better label quality.
- The script uses hashes only for deduplication, not as model training data.
- Respect site terms of service and copyright when scraping images.
