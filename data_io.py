"""
Symmetric serialization and multi-backend I/O for Python data, media, and artifacts.

Supports three distinct storage backends with explicit separation of concerns:
  - Local disk:   load_content_from_disk() / save_content_to_disk()
  - Direct URLs:  load_content_from_url()
  - GitHub API:   load_content_from_github_api() / upload_content_to_github_api()
"""

from __future__ import annotations

import base64
import io
import json
import os
import pickle
import sys
import traceback
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None

__all__ = [
    # Constants
    "DEFAULT_USER_AGENT",
    # Core serialization / deserialization
    "content_to_bytes",
    "parse_by_extension",
    # Local disk I/O
    "load_content_from_disk",
    "save_content_to_disk",
    # Direct HTTP(S) URL I/O
    "load_content_from_url",
    # GitHub Contents API & Git LFS I/O
    "upload_content_to_github_api",
    "load_content_from_github_api",
    "get_repo_file_raw",
    "is_lfs_pointer",
    # Convenience helpers
    "fetch_raw_content",
    "load_csv_universal",
    # Test runners
    "run_offline_checks",
    "run_github_integration_tests",
]


# ---------------------------------------------------------------------------
# Constants & Extension Rules
# ---------------------------------------------------------------------------

DEFAULT_USER_AGENT = "Python-DataLoader/1.0"

PICKLE_PROTOCOL = 4
PICKLE_EXTENSIONS = (".pkl", ".pickle")
RESERVED_EXTENSIONS = (".csv", ".tsv", ".npy") + PICKLE_EXTENSIONS

IMAGE_EXTS = (
    ".jpg", ".jpeg", ".jpe", ".jfif",
    ".png", ".gif",
    ".bmp", ".dib",
    ".webp",
    ".tif", ".tiff",
    ".pbm", ".pgm", ".ppm", ".pnm",
    ".ico",
    ".heic", ".heif",
)

PIL_FORMAT_MAP = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".jpe": "JPEG",
    ".jfif": "JPEG",
    ".tif": "TIFF",
    ".tiff": "TIFF",
}


def _ext(target_path: str | Path | None) -> str:
    return Path(str(target_path)).suffix.lower() if target_path else ""


def _resolve_github_context(
    target_path: str,
    owner: str | None = None,
    repo: str | None = None,
    branch: str | None = None,
    token: str | None = None,
    require_token: bool = False,
) -> tuple[str, str, str, str | None, str]:
    """Resolve GitHub repo metadata from explicit arguments or environment variables."""
    resolved_owner = owner or os.environ["GITHUB_OWNER"]
    resolved_repo = repo or os.environ["GITHUB_REPO"]
    resolved_branch = branch or os.environ.get("GITHUB_BRANCH", "main")
    resolved_token = token or os.environ.get("GITHUB_TOKEN")

    if require_token and not resolved_token:
        raise ValueError("A GitHub token is required for this operation.")

    clean_path = target_path.lstrip("/")
    api_url = f"https://api.github.com/repos/{resolved_owner}/{resolved_repo}/contents/{clean_path}"
    return resolved_owner, resolved_repo, resolved_branch, resolved_token, api_url


# ---------------------------------------------------------------------------
# Serialization & Deserialization (Symmetric Conversion Layer)
# ---------------------------------------------------------------------------

def content_to_bytes(content: Any, target_path: str | Path | None = None) -> bytes:
    """
    Convert supported Python objects to raw bytes for storage or upload.

    Rules:
    - Path requires matching source/target extension.
    - Raw bytes/bytearray are rejected for reserved extensions (.csv, .tsv, .npy, .pkl, .pickle).
    - .csv / .tsv accept pandas DataFrame only.
    - .npy accepts numpy ndarray only.
    - .pkl / .pickle accept any picklable Python object.
    - PIL.Image.Image objects are saved using the extension's image format.
    - Matplotlib/figure-like objects with .savefig() are rendered to the target extension.
    - dict / list / str are serialized for non-reserved extensions.
    """
    ext = _ext(target_path)

    if isinstance(content, Path):
        src_ext = content.suffix.lower()
        if src_ext != ext:
            raise ValueError(
                f"Path extension must match target extension: "
                f"source {str(content)!r} ({src_ext or 'none'}) vs "
                f"target {target_path!r} ({ext or 'none'})"
            )
        return content.read_bytes()

    if isinstance(content, (bytes, bytearray)):
        if ext in RESERVED_EXTENSIONS:
            raise ValueError(
                f"Raw bytes cannot be saved to {ext}. "
                f"Use a matching Path for .csv/.tsv/.npy/.pkl/.pickle, or use a non-reserved extension such as .bin."
            )
        return bytes(content)

    if ext in PICKLE_EXTENSIONS:
        try:
            return pickle.dumps(content, protocol=PICKLE_PROTOCOL)
        except (pickle.PicklingError, TypeError, AttributeError, RecursionError) as e:
            raise ValueError(
                f"Object of type {type(content).__name__} cannot be pickled for {target_path!r}: {e}"
            ) from e

    if pd is not None and isinstance(content, pd.DataFrame):
        if ext not in (".csv", ".tsv"):
            raise ValueError(
                f"DataFrame content must be saved to .csv or .tsv, got {target_path!r}"
            )
        sep = "\t" if ext == ".tsv" else ","
        return content.to_csv(sep=sep, index=False).encode("utf-8")

    if np is not None and isinstance(content, np.ndarray):
        if ext != ".npy":
            raise ValueError(f"numpy array content must be saved to .npy, got {target_path!r}")
        buf = io.BytesIO()
        np.save(buf, content, allow_pickle=False)
        return buf.getvalue()

    if ext in (".csv", ".tsv"):
        raise ValueError(f"{ext} is reserved for pandas DataFrames, got {type(content).__name__}")
    if ext == ".npy":
        raise ValueError(f".npy is reserved for numpy arrays, got {type(content).__name__}")

    if isinstance(content, (dict, list)):
        return json.dumps(content, indent=2).encode("utf-8")

    if isinstance(content, str):
        return content.encode("utf-8")

    if hasattr(content, "savefig"):
        if target_path is None:
            raise ValueError("target_path is required when content is a figure")
        fmt = ext.lstrip(".")
        if not fmt:
            raise ValueError("Could not infer figure format from target_path")
        buf = io.BytesIO()
        content.savefig(buf, format=fmt, bbox_inches="tight")
        return buf.getvalue()

    if PILImage is not None and isinstance(content, PILImage.Image):
        if target_path is None or not ext:
            raise ValueError("target_path with a valid image extension is required for PIL.Image")
        if ext not in IMAGE_EXTS:
            raise ValueError(f"Unsupported image extension {ext!r} for PIL.Image")

        fmt = PIL_FORMAT_MAP.get(ext, ext.lstrip(".").upper())
        img_to_save = content

        # JPEG does not support alpha channels (RGBA / LA / P)
        if fmt == "JPEG" and img_to_save.mode in ("RGBA", "LA", "P"):
            img_to_save = img_to_save.convert("RGB")

        buf = io.BytesIO()
        img_to_save.save(buf, format=fmt)
        return buf.getvalue()

    raise TypeError(
        f"Unsupported content type: {type(content).__name__}. "
        "Supported: str, dict, list, Path (matching extension), "
        "pandas DataFrame (.csv/.tsv), numpy ndarray (.npy), "
        "picklable objects (.pkl/.pickle), PIL.Image, and figure-like objects with savefig()."
    )


def parse_by_extension(raw: bytes, target_path: str | Path) -> Any:
    """
    Convert raw bytes back into a native Python object based on the target_path extension.
    Symmetric counterpart to content_to_bytes().
    """
    ext = _ext(target_path)

    if ext in (".txt", ".md", ".log", ".xml", ".html", ".yaml", ".yml"):
        return raw.decode("utf-8", errors="replace")

    if ext == ".json":
        return json.loads(raw.decode("utf-8"))

    if ext in (".csv", ".tsv"):
        if pd is None:
            raise ImportError("pandas is required to parse .csv/.tsv files")
        return pd.read_csv(io.BytesIO(raw), sep="\t" if ext == ".tsv" else ",")

    if ext == ".parquet":
        if pd is None:
            raise ImportError("pandas is required to parse .parquet files")
        return pd.read_parquet(io.BytesIO(raw))

    if ext in (".xlsx", ".xls"):
        if pd is None:
            raise ImportError("pandas is required to parse Excel files")
        return pd.read_excel(io.BytesIO(raw))

    if ext == ".npy":
        if np is None:
            raise ImportError("numpy is required to parse .npy files")
        return np.load(io.BytesIO(raw), allow_pickle=False)

    if ext == ".npz":
        if np is None:
            raise ImportError("numpy is required to parse .npz files")
        z = np.load(io.BytesIO(raw), allow_pickle=False)
        return {k: z[k] for k in z.files}

    if ext == ".mat":
        import scipy.io
        return scipy.io.loadmat(io.BytesIO(raw))

    if ext in IMAGE_EXTS:
        if PILImage is None:
            raise ImportError("Pillow (PIL) is required to parse image files")
        img = PILImage.open(io.BytesIO(raw))
        img.load()
        return img

    if ext in PICKLE_EXTENSIONS:
        return pickle.loads(raw)  # trusted sources only

    return raw


# ---------------------------------------------------------------------------
# Local Disk I/O
# ---------------------------------------------------------------------------

def load_content_from_disk(path: str | Path) -> Any:
    """Read raw bytes from a local filesystem path and parse into a native object by extension."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Local file not found: {file_path}")
    return parse_by_extension(file_path.read_bytes(), file_path)


def save_content_to_disk(path: str | Path, content: Any) -> Path:
    """Serialize a Python object via content_to_bytes() and write it to local disk."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(content_to_bytes(content, target_path=file_path))
    return file_path


# ---------------------------------------------------------------------------
# Direct HTTP(S) URL & Convenience Helpers
# ---------------------------------------------------------------------------

def load_content_from_url(
    url: str,
    target_path: str | Path | None = None,
    headers: dict[str, str] | None = None,
    user_agent: str | None = DEFAULT_USER_AGENT,
    timeout: int = 120,
) -> Any:
    """
    Download raw bytes from a direct HTTP(S) URL and parse by extension.
    Uses the URL path extension unless an explicit target_path is provided.
    """
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"Expected an http:// or https:// URL, got: {url!r}")

    parsed = urlparse(url)
    if parsed.netloc == "github.com" and "/blob/" in parsed.path:
        raise ValueError(
            "github.com '/blob/' URLs point to an HTML page, not raw file bytes. "
            "Pass a raw.githubusercontent.com URL or use load_content_from_github_api()."
        )

    req_headers = dict(headers or {})
    if user_agent and not any(k.lower() == "user-agent" for k in req_headers):
        req_headers["User-Agent"] = user_agent

    res = requests.get(url, headers=req_headers, timeout=timeout)
    res.raise_for_status()

    effective_path = target_path if target_path is not None else parsed.path
    return parse_by_extension(res.content, effective_path)


def fetch_raw_content(
    path_or_url: str,
    github_token: str | None = None,
    user_agent: str | None = DEFAULT_USER_AGENT,
) -> str:
    """Read a text file from local disk or a remote URL (rewriting GitHub blob URLs)."""
    if os.path.exists(path_or_url):
        with open(path_or_url, encoding="utf-8") as f:
            return f.read()

    if not path_or_url.startswith(("http://", "https://")):
        raise ValueError(f"Invalid path or URL: {path_or_url}")

    parsed = urlparse(path_or_url)
    url = path_or_url
    if parsed.netloc == "github.com":
        url = url.replace("github.com", "raw.githubusercontent.com", 1).replace("/blob/", "/", 1)

    headers: dict[str, str] = {}
    if user_agent:
        headers["User-Agent"] = user_agent

    if parsed.netloc in ("github.com", "raw.githubusercontent.com"):
        token = github_token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"

    res = requests.get(url, headers=headers, timeout=60)
    res.raise_for_status()
    return res.text


def load_csv_universal(
    path_or_url: str | Path,
    github_token: str | None = None,
    user_agent: str | None = DEFAULT_USER_AGENT,
):
    """Fetch a CSV or TSV from disk or URL and parse into a pandas DataFrame."""
    if pd is None:
        raise ImportError("pandas is required for load_csv_universal()")

    target_str = str(path_or_url)
    if target_str.startswith(("http://", "https://")):
        parsed = urlparse(target_str)
        url = target_str
        if parsed.netloc == "github.com":
            url = url.replace("github.com", "raw.githubusercontent.com", 1).replace("/blob/", "/", 1)

        ext = _ext(urlparse(url).path)
        if ext not in (".csv", ".tsv"):
            raise ValueError(f"Expected a .csv or .tsv URL, got {target_str!r}")

        headers: dict[str, str] = {}
        if urlparse(url).netloc == "raw.githubusercontent.com":
            token = github_token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
            if token:
                headers["Authorization"] = f"Bearer {token}"

        return load_content_from_url(url, headers=headers, user_agent=user_agent)

    ext = _ext(target_str)
    if ext not in (".csv", ".tsv"):
        raise ValueError(f"Expected a .csv or .tsv path, got {target_str!r}")
    return load_content_from_disk(target_str)


# ---------------------------------------------------------------------------
# Uploading to GitHub (Contents API)
# ---------------------------------------------------------------------------

def upload_content_to_github_api(
    target_path: str,
    content: Any,
    owner: str | None = None,
    repo: str | None = None,
    branch: str | None = None,
    token: str | None = None,
    commit_message: str | None = None,
) -> dict[str, Any]:
    """Create or update a file in a GitHub repository via the Contents API."""
    clean_path = target_path.lstrip("/")
    _, _, branch, token, url = _resolve_github_context(
        clean_path, owner=owner, repo=repo, branch=branch, token=token, require_token=True
    )

    raw = content_to_bytes(content, target_path=clean_path)
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": DEFAULT_USER_AGENT,
    }

    get_resp = requests.get(url, headers=headers, params={"ref": branch}, timeout=30)
    if get_resp.status_code == 200:
        sha = get_resp.json()["sha"]
    elif get_resp.status_code == 404:
        sha = None
    else:
        get_resp.raise_for_status()

    payload: dict[str, Any] = {
        "message": commit_message or f"Upload {clean_path}",
        "content": base64.b64encode(raw).decode("utf-8"),
        "branch": branch,
    }
    if sha is not None:
        payload["sha"] = sha

    put_resp = requests.put(url, headers=headers, json=payload, timeout=30)
    put_resp.raise_for_status()
    result = put_resp.json()

    return {
        "ok": True,
        "path": result["content"]["path"],
        "commit_sha": result["commit"]["sha"],
        "html_url": result["content"]["html_url"],
    }


# ---------------------------------------------------------------------------
# Downloading from GitHub (Contents API + Git LFS)
# ---------------------------------------------------------------------------

def get_repo_file_raw(
    target_path: str,
    owner: str | None = None,
    repo: str | None = None,
    branch: str | None = None,
    token: str | None = None,
    chunk_size: int = 1024 * 1024,
) -> bytes:
    """Download raw bytes for a file from a GitHub repository with multi-step fallbacks."""
    _, _, branch, token, url = _resolve_github_context(
        target_path, owner=owner, repo=repo, branch=branch, token=token, require_token=False
    )

    # 1) Raw media response path
    headers_raw = {
        "Accept": "application/vnd.github.raw",
        "User-Agent": DEFAULT_USER_AGENT,
    }
    if token:
        headers_raw["Authorization"] = f"Bearer {token}"

    with requests.get(
        url, headers=headers_raw, params={"ref": branch}, stream=True, timeout=120
    ) as r:
        if r.ok:
            return b"".join(c for c in r.iter_content(chunk_size=chunk_size) if c)

    # 2) JSON content API fallback (base64)
    headers_json = {
        "Accept": "application/vnd.github+json",
        "User-Agent": DEFAULT_USER_AGENT,
    }
    if token:
        headers_json["Authorization"] = f"Bearer {token}"

    resp = requests.get(url, headers=headers_json, params={"ref": branch}, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    if data.get("encoding") == "base64" and data.get("content"):
        return base64.b64decode(data["content"])

    # 3) download_url fallback
    if data.get("download_url"):
        dl_headers = {"User-Agent": DEFAULT_USER_AGENT}
        if token:
            dl_headers["Authorization"] = f"Bearer {token}"
        with requests.get(
            data["download_url"],
            headers=dl_headers,
            stream=True,
            timeout=120,
        ) as raw:
            raw.raise_for_status()
            return b"".join(c for c in raw.iter_content(chunk_size=chunk_size) if c)

    return b""


def is_lfs_pointer(raw_bytes: bytes) -> bool:
    head = raw_bytes[:512].decode("utf-8", errors="ignore")
    return (
        "version https://git-lfs.github.com/spec/v1" in head
        and "oid sha256:" in head
        and "size " in head
    )


def _parse_lfs_pointer(raw: bytes) -> tuple[str, int]:
    fields = {}
    for line in raw.decode("utf-8", errors="ignore").splitlines():
        if " " in line:
            k, v = line.split(" ", 1)
            fields[k] = v.strip()
    oid = fields["oid"].split(":", 1)[1]
    size = int(fields["size"])
    return oid, size


def _download_lfs_object(
    pointer_raw: bytes,
    owner: str,
    repo: str,
    token: str,
    chunk_size: int = 1024 * 1024,
) -> bytes:
    oid, size = _parse_lfs_pointer(pointer_raw)

    batch_url = f"https://github.com/{owner}/{repo}.git/info/lfs/objects/batch"
    r = requests.post(
        batch_url,
        auth=("x-access-token", token),
        headers={
            "Accept": "application/vnd.git-lfs+json",
            "Content-Type": "application/vnd.git-lfs+json",
            "User-Agent": DEFAULT_USER_AGENT,
        },
        json={
            "operation": "download",
            "transfers": ["basic"],
            "objects": [{"oid": oid, "size": size}],
        },
        timeout=60,
    )
    r.raise_for_status()
    obj = r.json()["objects"][0]

    if "error" in obj:
        raise RuntimeError(f"LFS error: {obj['error']}")

    action = obj["actions"]["download"]
    dl_headers = {"User-Agent": DEFAULT_USER_AGENT, **action.get("header", {})}
    with requests.get(
        action["href"],
        headers=dl_headers,
        stream=True,
        timeout=300,
    ) as d:
        d.raise_for_status()
        return b"".join(c for c in d.iter_content(chunk_size=chunk_size) if c)


def load_content_from_github_api(
    target_path: str,
    follow_lfs: bool = False,
    **kwargs: Any,
) -> Any:
    """Download a file from GitHub and automatically parse it based on its extension."""
    raw = get_repo_file_raw(target_path, **kwargs)

    if is_lfs_pointer(raw):
        if not follow_lfs:
            return raw.decode("utf-8")

        owner, repo, _, token, _ = _resolve_github_context(
            target_path,
            owner=kwargs.get("owner"),
            repo=kwargs.get("repo"),
            branch=kwargs.get("branch"),
            token=kwargs.get("token"),
            require_token=True,
        )
        raw = _download_lfs_object(raw, owner, repo, token)

    return parse_by_extension(raw, target_path)


# ---------------------------------------------------------------------------
# Offline Unit Tests (Serialization, Disk I/O, Mocked URL/GitHub/LFS)
# ---------------------------------------------------------------------------

def run_offline_checks() -> None:
    import tempfile
    from unittest.mock import MagicMock, patch

    if pd is None:
        raise RuntimeError("pandas is required for offline checks")
    if np is None:
        raise RuntimeError("numpy is required for offline checks")
    if PILImage is None:
        raise RuntimeError("Pillow (PIL) is required for offline checks")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        arr = np.arange(6).reshape(2, 3)

        txt_file = tmp_path / "src.txt"
        txt_file.write_text("hello\n", encoding="utf-8")

        csv_file = tmp_path / "src.csv"
        csv_file.write_text("a,b\n1,x\n", encoding="utf-8")

        tsv_file = tmp_path / "src.tsv"
        tsv_file.write_text("a\tb\n1\tx\n", encoding="utf-8")

        npy_file = tmp_path / "src.npy"
        np.save(npy_file, arr, allow_pickle=False)

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9], marker="o")

        # Sample PIL images (RGB and RGBA)
        pil_rgb = PILImage.new("RGB", (16, 16), color=(255, 0, 0))
        pil_rgba = PILImage.new("RGBA", (16, 16), color=(0, 255, 0, 128))

        def expect(exc: type[Exception], fn, *args, label: str = "", **kwargs) -> None:
            try:
                fn(*args, **kwargs)
            except exc:
                return
            raise AssertionError(f"Expected {exc.__name__} for {label}")

        # 1. Raw bytes rejected for reserved targets (including .tsv)
        for ext in (".csv", ".tsv", ".npy", ".pkl", ".pickle"):
            expect(ValueError, content_to_bytes, b"hello", target_path=f"x{ext}", label=f"bytes -> {ext}")
            expect(ValueError, content_to_bytes, bytearray(b"hello"), target_path=f"x{ext}", label=f"bytearray -> {ext}")

        # 2. Path extension must match
        expect(ValueError, content_to_bytes, txt_file, target_path="x.csv", label="Path .txt -> .csv")
        expect(ValueError, content_to_bytes, csv_file, target_path="x.tsv", label="Path .csv -> .tsv")
        expect(ValueError, content_to_bytes, tsv_file, target_path="x.csv", label="Path .tsv -> .csv")
        expect(ValueError, content_to_bytes, csv_file, target_path="x.npy", label="Path .csv -> .npy")
        expect(ValueError, content_to_bytes, npy_file, target_path="x.csv", label="Path .npy -> .csv")
        expect(ValueError, content_to_bytes, txt_file, target_path="x", label="Path .txt -> no extension")

        # 3. Typed objects must go to their proper formats
        expect(ValueError, content_to_bytes, df, target_path="x.npy", label="DataFrame -> .npy")
        expect(ValueError, content_to_bytes, df, target_path="x.txt", label="DataFrame -> .txt")
        expect(ValueError, content_to_bytes, arr, target_path="x.csv", label="array -> .csv")
        expect(ValueError, content_to_bytes, arr, target_path="x.tsv", label="array -> .tsv")
        expect(ValueError, content_to_bytes, arr, target_path="x.txt", label="array -> .txt")
        expect(ValueError, content_to_bytes, {"a": 1}, target_path="x.csv", label="dict -> .csv")
        expect(ValueError, content_to_bytes, {"a": 1}, target_path="x.tsv", label="dict -> .tsv")
        expect(ValueError, content_to_bytes, [1, 2], target_path="x.npy", label="list -> .npy")
        expect(ValueError, content_to_bytes, "hello", target_path="x.npy", label="str -> .npy")
        expect(ValueError, content_to_bytes, lambda x: x, target_path="x.pkl", label="lambda -> .pkl")

        # 4. CSV & TSV round-trip serialization + parser check
        csv_bytes = content_to_bytes(df, "data.csv")
        assert b"a,b" in csv_bytes
        assert parse_by_extension(csv_bytes, "data.csv").equals(df)

        tsv_bytes = content_to_bytes(df, "data.tsv")
        assert b"a\tb" in tsv_bytes
        assert parse_by_extension(tsv_bytes, "data.tsv").equals(df)

        # 5. Pickle should work for picklable objects
        assert pickle.loads(content_to_bytes({"a": 1}, "x.pkl")) == {"a": 1}
        assert pickle.loads(content_to_bytes([1, 2], "x.pickle")) == [1, 2]
        assert pickle.loads(content_to_bytes("hello", "x.pkl")) == "hello"
        assert pickle.loads(content_to_bytes(None, "x.pkl")) is None

        # 6. Non-reserved raw bytes are fine
        assert content_to_bytes(b"raw", "x.bin") == b"raw"
        assert content_to_bytes(b"raw", "x") == b"raw"

        # 7. Matplotlib figures: validate by file signature/content
        svg_b = content_to_bytes(fig, "figure.svg")
        assert b"<svg" in svg_b, f"SVG output did not contain <svg; head={svg_b[:80]!r}"

        png_b = content_to_bytes(fig, "figure.png")
        assert png_b.startswith(b"\x89PNG\r\n\x1a\n"), f"PNG signature wrong: {png_b[:12]!r}"

        jpg_b = content_to_bytes(fig, "figure.jpg")
        assert jpg_b.startswith(b"\xff\xd8\xff"), f"JPG signature wrong: {jpg_b[:12]!r}"

        jpeg_b = content_to_bytes(fig, "figure.jpeg")
        assert jpeg_b.startswith(b"\xff\xd8\xff"), f"JPEG signature wrong: {jpeg_b[:12]!r}"

        pdf_b = content_to_bytes(fig, "figure.pdf")
        assert pdf_b.startswith(b"%PDF"), f"PDF signature wrong: {pdf_b[:12]!r}"

        expect(ValueError, content_to_bytes, fig, target_path="figure", label="figure without extension")

        # 8. PIL.Image serialization & round-trip parsing
        pil_png_b = content_to_bytes(pil_rgba, "img.png")
        assert pil_png_b.startswith(b"\x89PNG\r\n\x1a\n")
        parsed_png = parse_by_extension(pil_png_b, "img.png")
        assert parsed_png.size == (16, 16) and parsed_png.mode == "RGBA"

        # RGBA image saved as .jpg should auto-convert to RGB instead of crashing
        pil_jpg_b = content_to_bytes(pil_rgba, "img.jpg")
        assert pil_jpg_b.startswith(b"\xff\xd8\xff")
        parsed_jpg = parse_by_extension(pil_jpg_b, "img.jpg")
        assert parsed_jpg.size == (16, 16) and parsed_jpg.mode == "RGB"

        pil_webp_b = content_to_bytes(pil_rgb, "img.webp")
        parsed_webp = parse_by_extension(pil_webp_b, "img.webp")
        assert parsed_webp.size == (16, 16)

        expect(ValueError, content_to_bytes, pil_rgb, target_path="img", label="PIL.Image without extension")
        expect(ValueError, content_to_bytes, pil_rgb, target_path="img.csv", label="PIL.Image -> .csv")
        expect(ValueError, content_to_bytes, pil_rgb, target_path="img.unknown", label="PIL.Image -> unsupported extension")

        # 9. Local disk I/O round-trips (save_content_to_disk & load_content_from_disk)
        disk_df_path = save_content_to_disk(tmp_path / "nested" / "table.csv", df)
        assert load_content_from_disk(disk_df_path).equals(df)

        disk_tsv_path = save_content_to_disk(tmp_path / "nested" / "table.tsv", df)
        assert load_content_from_disk(disk_tsv_path).equals(df)

        disk_arr_path = save_content_to_disk(tmp_path / "matrix.npy", arr)
        assert np.array_equal(load_content_from_disk(disk_arr_path), arr)

        disk_img_path = save_content_to_disk(tmp_path / "avatar.png", pil_rgba)
        loaded_img = load_content_from_disk(disk_img_path)
        assert loaded_img.size == (16, 16) and loaded_img.mode == "RGBA"

        disk_json_path = save_content_to_disk(tmp_path / "config.json", {"ok": True, "items": [1, 2]})
        assert load_content_from_disk(disk_json_path) == {"ok": True, "items": [1, 2]}

        assert load_csv_universal(disk_df_path).equals(df)
        assert load_csv_universal(disk_tsv_path).equals(df)
        expect(ValueError, load_csv_universal, disk_img_path, label="non-csv/tsv in load_csv_universal")
        expect(FileNotFoundError, load_content_from_disk, tmp_path / "does_not_exist.csv", label="missing local file")

        # 10. GitHub path normalization & URL validation guardrails (offline)
        _, _, _, _, resolved_url = _resolve_github_context(
            "/data/file.csv", owner="octocat", repo="hello-world", branch="main"
        )
        assert resolved_url == "https://api.github.com/repos/octocat/hello-world/contents/data/file.csv"

        for bad_url in (
            "not-a-url",
            "ftp://example.com/file.csv",
            "https://github.com/org/repo/blob/main/data.csv",
        ):
            expect(ValueError, load_content_from_url, bad_url, label=f"invalid url {bad_url!r}")

        # 11. Additional parse_by_extension formats (.npz, text, unknown bytes)
        npz_buf = io.BytesIO()
        np.savez(npz_buf, first=arr, second=np.array([10, 20]))
        parsed_npz = parse_by_extension(npz_buf.getvalue(), "bundle.npz")
        assert set(parsed_npz.keys()) == {"first", "second"}
        assert np.array_equal(parsed_npz["first"], arr)

        assert parse_by_extension(b"hello \xff world", "notes.md") == "hello \ufffd world"
        assert parse_by_extension(b"\x00\x01\x02", "blob.unknown") == b"\x00\x01\x02"

        # 12. Git LFS pointer detection
        sample_pointer = (
            b"version https://git-lfs.github.com/spec/v1\n"
            b"oid sha256:4d7a214614ab2935c943f9e0ff69d22eadbb8f32b1258daaa5e2ca24d17e2393\n"
            b"size 12345\n"
        )
        assert is_lfs_pointer(sample_pointer) is True
        assert is_lfs_pointer(b"just regular file content") is False

        # 13. Mocked HTTP tests for load_content_from_url & GitHub API flows
        def make_resp(status=200, content=b"", json_data=None):
            r = MagicMock()
            r.status_code = status
            r.ok = 200 <= status < 300
            r.content = content
            r.text = content.decode("utf-8", errors="replace")
            r.json.return_value = json_data if json_data is not None else {}
            r.iter_content.return_value = [content] if content else []
            r.raise_for_status.side_effect = (
                None if r.ok else requests.HTTPError(f"HTTP {status}")
            )
            r.__enter__.return_value = r
            r.__exit__.return_value = False
            return r

        with patch("requests.get", return_value=make_resp(200, content=csv_bytes)) as mock_get:
            loaded_url_df = load_content_from_url(
                "https://example.com/export?id=42", target_path="override.csv"
            )
            assert loaded_url_df.equals(df)
            assert mock_get.call_args.kwargs["headers"]["User-Agent"] == DEFAULT_USER_AGENT

            load_content_from_url("https://example.com/data.csv", user_agent="CustomAgent/2.0")
            assert mock_get.call_args.kwargs["headers"]["User-Agent"] == "CustomAgent/2.0"

            load_content_from_url(
                "https://example.com/data.csv",
                headers={"user-agent": "HeaderWins/3.0"},
                user_agent="Ignored/1.0",
            )
            assert mock_get.call_args.kwargs["headers"]["user-agent"] == "HeaderWins/3.0"

        with patch(
            "requests.get",
            side_effect=[
                make_resp(404),  # File doesn't exist yet -> create
                make_resp(200, json_data={"sha": "old_sha_123"}),  # File exists -> update
            ],
        ), patch(
            "requests.put",
            return_value=make_resp(
                200,
                json_data={
                    "content": {
                        "path": "data/df.csv",
                        "html_url": "https://github.com/o/r/blob/main/data/df.csv",
                    },
                    "commit": {"sha": "new_commit_456"},
                },
            ),
        ) as mock_put:
            res_create = upload_content_to_github_api(
                "data/df.csv", df, owner="o", repo="r", token="tok"
            )
            assert res_create["ok"] is True
            assert "sha" not in mock_put.call_args.kwargs["json"]

            upload_content_to_github_api("data/df.csv", df, owner="o", repo="r", token="tok")
            assert mock_put.call_args.kwargs["json"]["sha"] == "old_sha_123"

        with patch("requests.get", return_value=make_resp(200, content=sample_pointer)):
            ptr_text = load_content_from_github_api("big.npy", owner="o", repo="r")
            assert "git-lfs.github.com" in ptr_text

        with patch(
            "requests.get",
            side_effect=[
                make_resp(200, content=sample_pointer),  # Contents API returns LFS pointer
                make_resp(200, content=content_to_bytes(arr, "big.npy")),  # LFS download href
            ],
        ), patch(
            "requests.post",
            return_value=make_resp(
                200,
                json_data={
                    "objects": [{"actions": {"download": {"href": "https://lfs.example.com/obj"}}}]
                },
            ),
        ):
            lfs_arr = load_content_from_github_api(
                "big.npy", follow_lfs=True, owner="o", repo="r", token="tok"
            )
            assert np.array_equal(lfs_arr, arr)

        plt.close(fig)
        print("All offline checks passed.")


# ---------------------------------------------------------------------------
# Live GitHub Integration Tests (Requires GITHUB_OWNER, GITHUB_REPO, GITHUB_TOKEN)
# ---------------------------------------------------------------------------

def create_test_files_in_repo(
    owner: str,
    repo: str,
    branch: str,
    token: str,
    prefix: str = "samples",
) -> dict[str, str]:
    """Create or update deterministic sample files in a live GitHub repo under `prefix/`."""
    if pd is None or np is None or PILImage is None:
        raise RuntimeError("pandas, numpy, and Pillow are required for GitHub integration tests")

    prefix = prefix.strip("/")
    img = PILImage.new("RGB", (32, 24), color=(255, 0, 0))

    npz_buf = io.BytesIO()
    np.savez(npz_buf, x=np.arange(5), y=np.array([[1, 2], [3, 4]]))

    pdf_bytes = (
        b"%PDF-1.1\n"
        b"1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n"
        b"2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1 >>endobj\n"
        b"3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >>endobj\n"
        b"xref\n0 4\n0000000000 65535 f \n0000000010 00000 n \n0000000060 00000 n \n0000000117 00000 n \n"
        b"trailer<< /Size 4 /Root 1 0 R >>\nstartxref\n178\n%%EOF\n"
    )

    samples: dict[str, tuple[str, Any]] = {
        "text": (f"{prefix}/hello.txt", "hello from github loader test\n"),
        "json": (f"{prefix}/config.json", {"name": "demo", "version": 1, "ok": True}),
        "csv": (f"{prefix}/table.csv", pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.5, 6.25]})),
        "image_jpg": (f"{prefix}/photo.jpg", img),
        "image_tif": (f"{prefix}/scan.tif", img),
        "npy": (f"{prefix}/arr.npy", np.arange(12, dtype=np.float32).reshape(3, 4)),
        "npz": (f"{prefix}/arrs.npz", npz_buf.getvalue()),
        "pdf": (f"{prefix}/report.pdf", pdf_bytes),
        "pickle": (f"{prefix}/model.pkl", {"kind": "demo", "weights": [1, 2, 3]}),
    }

    try:
        import scipy.io
        mat_buf = io.BytesIO()
        scipy.io.savemat(mat_buf, {"sim": np.array([[1.1, 2.2], [3.3, 4.4]])})
        samples["mat"] = (f"{prefix}/sim.mat", mat_buf.getvalue())
    except ImportError:
        print("Skipping samples/sim.mat (scipy not installed).")

    for path, content in samples.values():
        upload_content_to_github_api(
            target_path=path,
            content=content,
            owner=owner,
            repo=repo,
            branch=branch,
            token=token,
            commit_message=f"test: add/update {path}",
        )

    print(f"Created/updated {len(samples)} files in {owner}/{repo}@{branch}")
    return {label: path for label, (path, _) in samples.items()}


def run_github_integration_tests(prefix: str = "samples") -> None:
    """Upload sample files to GitHub and verify round-trip loading with load_content_from_github_api."""
    owner = os.environ["GITHUB_OWNER"]
    repo = os.environ["GITHUB_REPO"]
    branch = os.environ.get("GITHUB_BRANCH", "main")
    token = os.environ["GITHUB_TOKEN"]

    print("Step 1: creating/updating test files in GitHub...")
    test_paths = create_test_files_in_repo(owner, repo, branch, token, prefix=prefix)

    def check(name: str, fn) -> Any:
        print(f"\n=== {name} ===")
        try:
            out = fn()
            print("PASS")
            return out
        except Exception as e:
            print("FAIL:", e)
            traceback.print_exc()
            return None

    def describe(obj: Any) -> str:
        if isinstance(obj, str):
            return f"str len={len(obj)}"
        if isinstance(obj, bytes):
            return f"bytes len={len(obj)}"
        if isinstance(obj, dict):
            return f"dict keys={list(obj.keys())[:5]}"
        if pd is not None and isinstance(obj, pd.DataFrame):
            return f"DataFrame shape={obj.shape}"
        if np is not None and isinstance(obj, np.ndarray):
            return f"ndarray shape={obj.shape}, dtype={obj.dtype}"
        if PILImage is not None and isinstance(obj, PILImage.Image):
            return f"PIL.Image size={obj.size}, mode={obj.mode}, format={obj.format}"
        return f"{type(obj)}"

    print("\nStep 2: loading/parsing with load_content_from_github_api...")
    results = {
        label: check(
            label,
            lambda p=path: load_content_from_github_api(
                p, owner=owner, repo=repo, branch=branch, token=token
            ),
        )
        for label, path in test_paths.items()
    }

    print("\n=== SUMMARY ===")
    for k, v in results.items():
        print(f"{k}: {'FAIL' if v is None else describe(v)}")

    if any(v is None for v in results.values()):
        raise RuntimeError("One or more live GitHub integration tests failed.")


if __name__ == "__main__":
    run_offline_checks()
    if "--live-github" in sys.argv:
        run_github_integration_tests()
