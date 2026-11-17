# py

Standalone Python utility modules.

## Modules

| Module | Purpose |
| :--- | :--- |
| `data_io.py` | Symmetric serialization and multi-backend I/O across **local disk**, **direct URLs**, and the **GitHub Contents / LFS API**. |

## Installation

**Option 1: Install via `pip`**
```bash
# Core dependencies only (requests)
pip install git+[https://github.com/](https://github.com/)scripts-vault/py.git

# With pandas, numpy, Pillow, matplotlib, and scipy support
pip install "py-utils[full] @ git+[https://github.com/](https://github.com/)scripts-vault/py.git"
```

**Option 2: Grab the module directly (no `pip` required)**
```bash
curl -O [https://raw.githubusercontent.com/](https://raw.githubusercontent.com/)scripts-vault/py/main/data_io.py
```

## Quickstart

```python
from data_io import (
    save_content_to_disk,
    load_content_from_disk,
    load_content_from_url,
    upload_content_to_github_api,
    load_content_from_github_api,
)

# Local disk
save_content_to_disk("out/data.csv", df)
df = load_content_from_disk("out/data.csv")

# Direct URL
remote_df = load_content_from_url("[https://example.com/data.csv](https://example.com/data.csv)")

# GitHub repository (uses GITHUB_OWNER, GITHUB_REPO, GITHUB_TOKEN env vars or explicit kwargs)
upload_content_to_github_api("data/results.csv", df)
loaded_df = load_content_from_github_api("data/results.csv")
```

## Running Tests

```bash
# Fast offline unit & mocked network tests
python data_io.py

# Live GitHub integration tests (requires GITHUB_OWNER, GITHUB_REPO, GITHUB_TOKEN)
python data_io.py --live-github
```
