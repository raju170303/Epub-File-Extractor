from __future__ import annotations
import os
import re
import sys
import json
import uuid
import shutil
import zipfile
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import html as _html

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QProgressBar, QTextEdit, QLineEdit, QListWidget
)
from PySide6.QtCore import Qt

CHAPTER_REGEX = re.compile(r"^(?:ch|chp)[-_]?(\d{1,3})\.(xhtml|html)$", re.IGNORECASE)

FM_KEYWORDS: Dict[str, List[str]] = {
    "Cover Page": ["cover", "cover-page"],
    "Half Title Page": ["half", "half-title", "halftitle"],
    "Title Page": ["title", "tit", "title-page"],
    "Copyright": ["copyright", "copyright-page", "copy"],
    "About the Editor": ["about-editor", "about_the_editor", "editor"],
    "About the Contributors": ["about-contributors", "contributors", "about_the_contributors"],
}
FM_ORDER = [
    "Cover Page", "Half Title Page", "Title Page",
    "Copyright", "About the Editor", "About the Contributors"
]
INDEX_KEYWORDS = ["index"]

DEBUG = True

@dataclass
class RunLog:
    messages: List[str]
    warnings: List[str]

    def log(self, msg: str):
        print(msg)
        self.messages.append(msg)

    def warn(self, msg: str):
        w = f"WARNING: {msg}"
        print(w, file=sys.stderr)
        self.warnings.append(w)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def write_text(p: Path, text: str):
    ensure_dir(p.parent)
    p.write_text(text, encoding="utf-8")

def get_subdir_case_insensitive(parent: Path, preferred: str) -> Path:
    pref = parent / preferred
    if pref.exists():
        return pref
    if not parent.exists():
        return pref
    want = preferred.lower()
    for child in parent.iterdir():
        if child.is_dir() and child.name.lower() == want:
            return child
    return pref

LI_PATTERN = re.compile(r'<li\b[^>]*>(.*?)</li>', re.DOTALL | re.IGNORECASE)
HREF_PATTERN = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)
A_TAG_PATTERN = re.compile(r'<a\b([^>]*)>(.*?)</a>', re.IGNORECASE | re.DOTALL)

def extract_original_nav_and_raw(xhtml_dir: Path, log: RunLog) -> Tuple[str, str]:
    if not xhtml_dir.exists():
        log.warn("xhtml directory not found in EPUB.")
        return "", ""
    cands = sorted(list(xhtml_dir.glob("nav.*")))
    if not cands:
        cands = [p for p in xhtml_dir.iterdir() if p.is_file() and p.name.lower().startswith("nav.")]
    if not cands:
        log.warn("No nav file found in input xhtml directory.")
        return "", ""
    try:
        raw = cands[0].read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        log.warn(f"Failed to read nav file: {e}")
        return "", ""
    m = re.search(r'<nav\b[^>]*\bepub:type\s*=\s*["\']toc["\'][^>]*>(.*?)</nav>', raw, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        m = re.search(r'<nav\b[^>]*(?:\brole\s*=\s*["\']doc-toc["\']|\baria-labelledby\s*=\s*["\']contents["\'])[^(>]*>(.*?)</nav>',
                      raw, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        m = re.search(r'(<nav\b[^>]*>[\s\S]*?</nav>)', raw, flags=re.IGNORECASE | re.DOTALL)
    nav_html = m.group(0) if m else ""
    return nav_html, raw

def parse_nav_items_from_nav_html(nav_raw: str) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    if not nav_raw:
        return items
    ol_m = re.search(r'<ol\b[^>]*>(.*?)</ol>', nav_raw, flags=re.IGNORECASE | re.DOTALL)
    ol_content = ol_m.group(1) if ol_m else nav_raw
    for li in LI_PATTERN.findall(ol_content):
        href_m = HREF_PATTERN.search(li)
        href = href_m.group(1) if href_m else ""
        text = re.sub(r'<[^>]+>', '', li).strip()
        text = re.sub(r'\s+', ' ', text)
        items.append((text, href))
    return items

def sanitize_nav_xhtml(raw: str) -> str:
    s = (raw or "").lstrip()
    if not s.startswith("<?xml"):
        s = '<?xml version="1.0" encoding="utf-8"?>\n' + s
    s = re.sub(r'(?i)<!doctype\s+html[^>]*>', '<!DOCTYPE html>', s, count=1)
    m = re.search(r'<html\b([^>]*)>', s, flags=re.IGNORECASE)
    if m:
        attrs = m.group(1)
        def has_attr(rx): return re.search(rx, attrs, flags=re.IGNORECASE) is not None
        adds = []
        if not has_attr(r'\bxmlns\b'):
            adds.append('xmlns="http://www.w3.org/1999/xhtml"')
        if not has_attr(r'\bxmlns:epub\b'):
            adds.append('xmlns:epub="http://www.idpf.org/2007/ops"')
        if not has_attr(r'\bxml:lang\b'):
            adds.append('xml:lang="en"')
        if not has_attr(r'\blang\b'):
            adds.append('lang="en"')
        if adds:
            new_attrs = attrs.rstrip() + ' ' + ' '.join(adds)
            s = s[:m.start()] + '<html' + new_attrs + '>' + s[m.end():]
    else:
        body_m = re.search(r'(<body\b[\s\S]*</body>)', s, flags=re.IGNORECASE)
        body = body_m.group(1) if body_m else '<body></body>'
        s = '<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE html>\n' \
            '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="en" lang="en">\n' \
            '  <head><meta charset="utf-8"/><title>nav</title></head>\n' + body + '\n</html>\n'
    if not s.endswith("\n"):
        s += "\n"
    return s

def _clean_html_text(s: str) -> str:
    s2 = re.sub(r'<[^>]+>', '', s)
    s2 = re.sub(r'\s+', ' ', s2).strip()
    return _html.unescape(s2)

def extract_title_from_xhtml(xhtml_path: Path) -> str:
    try:
        txt = xhtml_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return xhtml_path.name
    m = re.search(r'<h1[^>]*>(.*?)</h1>', txt, flags=re.IGNORECASE | re.DOTALL)
    if m:
        title = _clean_html_text(m.group(1))
        if title:
            return title
    m = re.search(r'<([a-z0-9]+)[^>]*\bepub:type=["\']title["\'][^>]*>(.*?)</\1>', txt, flags=re.IGNORECASE | re.DOTALL)
    if m:
        title = _clean_html_text(m.group(2))
        if title:
            return title
    m = re.search(r'<([a-z0-9]+)[^>]*\bclass=["\'][^"\']*(chapter[-_ ]?title|chapterTitle|title|book-title|part-title)[^"\']*["\'][^>]*>(.*?)</\1>',
                  txt, flags=re.IGNORECASE | re.DOTALL)
    if m:
        title = _clean_html_text(m.group(2))
        if title:
            return title
    m = re.search(r'<h[23][^>]*>(.*?)</h[23]>', txt, flags=re.IGNORECASE | re.DOTALL)
    if m:
        title = _clean_html_text(m.group(1))
        if title:
            return title
    m = re.search(r'<title[^>]*>(.*?)</title>', txt, flags=re.IGNORECASE | re.DOTALL)
    if m:
        title = _clean_html_text(m.group(1))
        if title:
            return title
    return xhtml_path.name

def classify_nav_items(items: List[Tuple[str, str]], log: RunLog):
    fm_items: List[Tuple[str,str]] = []
    chapter_map: Dict[int, Tuple[str,str]] = {}
    index_item: Optional[Tuple[str,str]] = None
    first_chapter_idx = None
    for idx, (text, href) in enumerate(items):
        href_basename = Path(href).name if href else ""
        if href_basename:
            if CHAPTER_REGEX.match(href_basename):
                first_chapter_idx = idx
                break
            if re.match(r'(?i)^(ch|chp|chapter)', href_basename) and re.search(r'(\d{1,3})', href_basename):
                first_chapter_idx = idx
                break
        if text and re.search(r'\bchapter\b', text, flags=re.IGNORECASE) and re.search(r'\b(\d{1,3})\b', text):
            first_chapter_idx = idx
            break
    for idx, (text, href) in enumerate(items):
        lower_text = (text or "").lower()
        href_basename = Path(href).name if href else ""
        href_lower = href_basename.lower()
        is_before_chapters = (first_chapter_idx is None) or (idx < first_chapter_idx)
        matched_fm = False
        if is_before_chapters:
            for label, keys in FM_KEYWORDS.items():
                for k in keys:
                    if k in lower_text or k in href_lower:
                        fm_items.append((label, href))
                        matched_fm = True
                        break
                if matched_fm:
                    break
        if matched_fm:
            continue
        if any(k in lower_text or k in href_lower for k in INDEX_KEYWORDS):
            index_item = (text, href)
            continue
        num = None
        if href_basename:
            m_href = re.search(r'(\d{1,3})', href_basename)
            if m_href:
                num = int(m_href.group(1))
        if num is None:
            m_text = re.search(r'\b(\d{1,3})\b', text or "")
            if m_text:
                num = int(m_text.group(1))
        if num is not None:
            if num in chapter_map:
                log.warn(f"Duplicate nav entry for chapter {num}: keeping previous entry, ignored '{text}'")
            else:
                chapter_map[num] = (text, href)
            continue
        m_file = CHAPTER_REGEX.match(href_basename)
        if m_file:
            num = int(m_file.group(1))
            if num in chapter_map:
                log.warn(f"Duplicate nav entry for chapter {num} (from filename): keeping previous entry, ignored '{text}'")
            else:
                chapter_map[num] = (text, href)
            continue
        log.warn(f"Unclassified nav item: text='{text}', href='{href}'")
    return fm_items, chapter_map, index_item

def find_file_case_insensitive(base_dir: Path, name: str) -> Optional[Path]:
    if not base_dir.exists():
        return None
    candidate = base_dir / name
    if candidate.exists():
        return candidate
    low = name.lower()
    for p in base_dir.iterdir():
        if p.is_file() and p.name.lower() == low:
            return p
    return None

def normalize_href_to_actual(href: str, xhtml_dir: Path, log: RunLog) -> str:
    if not href:
        return ""
    href_only = href.split('#', 1)[0].split('?', 1)[0]
    base = Path(href_only).name
    if not base:
        return href
    found = find_file_case_insensitive(xhtml_dir, base)
    if found:
        return found.name
    log.warn(f"Could not resolve href '{href}' to a file in xhtml_dir")
    return href

def find_best_anchor_text_for_href_in_nav(raw_nav_full: str, target_filename: str) -> str:
    if not raw_nav_full or not target_filename:
        return ""
    candidates: List[str] = []
    for m in A_TAG_PATTERN.finditer(raw_nav_full):
        attrs = m.group(1)
        inner = m.group(2)
        href_m = HREF_PATTERN.search(attrs)
        if not href_m:
            opening_tag = m.group(0)[:m.group(0).find('>')]
            href_m2 = HREF_PATTERN.search(opening_tag)
            if href_m2:
                href_val = href_m2.group(1)
            else:
                continue
        else:
            href_val = href_m.group(1)
        href_only = href_val.split('#',1)[0].split('?',1)[0]
        if Path(href_only).name.lower() == target_filename.lower():
            txt = re.sub(r'<[^>]+>', '', inner).strip()
            txt = re.sub(r'\s+', ' ', txt)
            if txt:
                candidates.append(txt)
    if not candidates:
        return ""
    candidates = sorted(candidates, key=lambda s: (any(c.isalpha() for c in s), len(s)), reverse=True)
    return candidates[0]

def pack_epub_dir(src_dir: Path, out_path: Path):
    with zipfile.ZipFile(out_path, 'w') as z:
        mimetype_path = src_dir / "mimetype"
        if mimetype_path.exists():
            z.writestr("mimetype", mimetype_path.read_bytes(), compress_type=zipfile.ZIP_STORED)
        for root, _, files in os.walk(src_dir):
            files_sorted = sorted(files)
            for f in files_sorted:
                full = Path(root) / f
                rel = full.relative_to(src_dir)
                if str(rel) == "mimetype":
                    continue
                z.write(full, arcname=str(rel), compress_type=zipfile.ZIP_DEFLATED)

def create_staged_epub(epub_name: str, files_map: Dict[Path, Path], meta_inf: Path, fonts_dir: Path, styles_dir: Path, contents_opf: Path, toc_ncx: Path, log: RunLog) -> Path:
    work = Path(tempfile.mkdtemp(prefix=f"staged-{uuid.uuid4().hex}-"))
    try:
        ensure_dir(work / "META-INF")
        ensure_dir(work / "OEBPS" / "fonts")
        ensure_dir(work / "OEBPS" / "styles")
        ensure_dir(work / "OEBPS" / "xhtml")
        ensure_dir(work / "OEBPS" / "images")
        if meta_inf.exists():
            shutil.copytree(meta_inf, work / "META-INF", dirs_exist_ok=True)
        if fonts_dir.exists():
            shutil.copytree(fonts_dir, work / "OEBPS" / "fonts", dirs_exist_ok=True)
        if styles_dir.exists():
            shutil.copytree(styles_dir, work / "OEBPS" / "styles", dirs_exist_ok=True)
        for src, rel in files_map.items():
            dest = work / rel
            ensure_dir(dest.parent)
            if src.exists():
                shutil.copy2(src, dest)
            else:
                log.warn(f"Missing file when creating {epub_name}: {src}")
        if contents_opf.exists():
            shutil.copy2(contents_opf, work / "OEBPS" / "contents.opf")
        if toc_ncx.exists():
            shutil.copy2(toc_ncx, work / "OEBPS" / "toc.ncx")
        write_text(work / "mimetype", "application/epub+zip")
        staged = Path(tempfile.gettempdir()) / f"{epub_name}-{uuid.uuid4().hex}.epub"
        pack_epub_dir(work, staged)
        return staged
    finally:
        shutil.rmtree(work, ignore_errors=True)

def stage_chapter_epub(num: int, text: str, href: str, chapter_file: Path, xhtml_dir: Path, images_dir: Path, meta_inf: Path, fonts_dir: Path, styles_dir: Path, contents_opf: Path, toc_ncx: Path, log: RunLog) -> Tuple[str, Path]:
    epub_name = f"CH{num:02d}.epub"
    files_map: Dict[Path, Path] = {}
    files_map[chapter_file] = Path("OEBPS") / "xhtml" / chapter_file.name
    raw_nav = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="en" xml:lang="en">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8"/>
<title>Navigational Table of Contents</title>
<link rel="stylesheet" type="text/css" href="../styles/stylesheet.css"/>
</head>
<body epub:type="frontmatter">
<nav epub:type="toc" id="toc" role="doc-toc" aria-labelledby="contents">
<h1 id="contents">Contents</h1>
<ol>
  <li><a href="{href}">{text}</a></li>
</ol>
</nav>
</body>
</html>
'''
    nav_tmp = Path(tempfile.gettempdir()) / f"nav-{uuid.uuid4().hex}.xhtml"
    nav_tmp.write_text(sanitize_nav_xhtml(raw_nav), encoding="utf-8")
    files_map[nav_tmp] = Path("OEBPS") / "xhtml" / "nav.xhtml"
    if images_dir.exists():
        prefix1 = f"fig{num}_"
        prefix2 = f"fig{num:02d}_"
        for img in images_dir.iterdir():
            if not img.is_file(): continue
            ln = img.name.lower()
            if ln.startswith(prefix1) or ln.startswith(prefix2):
                files_map[img] = Path("OEBPS") / "images" / img.name
    staged = create_staged_epub(epub_name, files_map, meta_inf, fonts_dir, styles_dir, contents_opf, toc_ncx, log)
    return epub_name, staged

def stage_fm_epub(fm_items: List[Tuple[str,str]], fm_files: List[Path], images_dir: Path, meta_inf: Path, fonts_dir: Path, styles_dir: Path, contents_opf: Path, toc_ncx: Path, log: RunLog) -> Tuple[str, Path]:
    epub_name = "FM.epub"
    files_map: Dict[Path, Path] = {}
    for f in fm_files:
        files_map[f] = Path("OEBPS") / "xhtml" / f.name
    canonical_filenames = [
        ("cover", "Cover Page"),
        ("half", "Half Title Page"),
        ("tit", "Title Page"),
        ("copy", "Copyright"),
        ("toc", "Contents"),
        ("fm1", "About the Editor"),
        ("fm2", "About the Contributors"),
    ]
    fm_lookup = {p.name.lower(): p for p in fm_files}
    chosen_li: List[str] = []
    for key, label in canonical_filenames:
        found_path = None
        for name_lower, p in fm_lookup.items():
            if name_lower == f"{key}.xhtml" or name_lower == f"{key}.html":
                found_path = p
                break
        if not found_path:
            for name_lower, p in fm_lookup.items():
                if key in name_lower:
                    found_path = p
                    break
        if found_path:
            chosen_li.append(f'  <li><a href="{found_path.name}">{label}</a></li>')
    if not chosen_li:
        for p in fm_files:
            lab = p.stem
            chosen_li.append(f'  <li><a href="{p.name}">{lab}</a></li>')
    lis = "\n".join(chosen_li)
    raw_nav = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="en" xml:lang="en">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8"/>
<title>Navigational Table of Contents</title>
<link rel="stylesheet" type="text/css" href="../styles/stylesheet.css"/>
</head>
<body epub:type="frontmatter">
<nav epub:type="toc" id="toc" role="doc-toc" aria-labelledby="contents">
<h1 id="contents">Contents</h1>
<ol>
{lis}
</ol>
</nav>
</body>
</html>
'''
    nav_tmp = Path(tempfile.gettempdir()) / f"fm-nav-{uuid.uuid4().hex}.xhtml"
    nav_tmp.write_text(sanitize_nav_xhtml(raw_nav), encoding="utf-8")
    files_map[nav_tmp] = Path("OEBPS") / "xhtml" / "nav.xhtml"
    if images_dir.exists():
        for img in images_dir.iterdir():
            if not img.is_file(): continue
            if re.match(r'^fig\d+_', img.name, re.IGNORECASE): continue
            files_map[img] = Path("OEBPS") / "images" / img.name
    staged = create_staged_epub(epub_name, files_map, meta_inf, fonts_dir, styles_dir, contents_opf, toc_ncx, log)
    return epub_name, staged

def stage_index_epub(index_item: Tuple[str,str], index_file: Path, meta_inf: Path, fonts_dir: Path, styles_dir: Path, contents_opf: Path, toc_ncx: Path, log: RunLog) -> Tuple[str, Path]:
    epub_name = "Index.epub"
    files_map: Dict[Path, Path] = {}
    files_map[index_file] = Path("OEBPS") / "xhtml" / index_file.name
    text, href = index_item
    raw_nav = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="en" xml:lang="en">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8"/>
<title>Navigational Table of Contents</title>
<link rel="stylesheet" type="text/css" href="../styles/stylesheet.css"/>
</head>
<body epub:type="frontmatter">
<nav epub:type="toc" id="toc" role="doc-toc" aria-labelledby="contents">
<h1 id="contents">Contents</h1>
<ol>
  <li><a href="{href}">{text}</a></li>
</ol>
</nav>
</body>
</html>
'''
    nav_tmp = Path(tempfile.gettempdir()) / f"idx-nav-{uuid.uuid4().hex}.xhtml"
    nav_tmp.write_text(sanitize_nav_xhtml(raw_nav), encoding="utf-8")
    files_map[nav_tmp] = Path("OEBPS") / "xhtml" / "nav.xhtml"
    staged = create_staged_epub(epub_name, files_map, meta_inf, fonts_dir, styles_dir, contents_opf, toc_ncx, log)
    return epub_name, staged

def process_single_epub_to_final_zip(input_epub: Path, output_dir: Path, log: RunLog, progress_cb=None) -> dict:
    ensure_dir(output_dir)
    log.log(f"Unzipping input EPUB: {input_epub.name}")
    tmp = Path(tempfile.mkdtemp(prefix="input-unzip-"))
    try:
        with zipfile.ZipFile(input_epub, 'r') as zin:
            zin.extractall(tmp)
    except Exception as e:
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError(f"Failed to unzip input EPUB: {e}")
    try:
        meta_inf = tmp / "META-INF"
        oebps = tmp / "OEBPS"
        xhtml_dir = get_subdir_case_insensitive(oebps, "xhtml")
        images_dir = get_subdir_case_insensitive(oebps, "images")
        fonts_dir = get_subdir_case_insensitive(oebps, "fonts")
        styles_dir = get_subdir_case_insensitive(oebps, "styles")
        contents_opf = oebps / "contents.opf"
        toc_ncx = oebps / "toc.ncx"
        nav_html, nav_raw_full = extract_original_nav_and_raw(xhtml_dir, log)
        items = parse_nav_items_from_nav_html(nav_html)
        fm_items, chapter_map, index_item = classify_nav_items(items, log)
        if DEBUG:
            log.log(f"DEBUG: parsed nav items count={len(items)} fm_items={fm_items} chapter_map_keys={list(chapter_map.keys())} index_item={index_item}")
        for ch_num, (t, h) in list(chapter_map.items()):
            chapter_map[ch_num] = (t, normalize_href_to_actual(h or "", xhtml_dir, log))
        fm_items = [(label, normalize_href_to_actual(h or "", xhtml_dir, log)) for (label, h) in fm_items]
        if index_item:
            t, h = index_item
            index_item = (t, normalize_href_to_actual(h or "", xhtml_dir, log))
        chapter_files: List[Tuple[int, Path]] = []
        if xhtml_dir.exists():
            for p in xhtml_dir.glob("*.*html"):
                m = CHAPTER_REGEX.match(p.name)
                if m:
                    chapter_files.append((int(m.group(1)), p))
        for ch_num, (t, href) in chapter_map.items():
            candidate = None
            if href:
                candidate = find_file_case_insensitive(xhtml_dir, href)
            if candidate is None and xhtml_dir.exists():
                for p in xhtml_dir.glob("*.*html"):
                    if re.search(r'\b' + re.escape(str(ch_num)) + r'\b', p.name):
                        candidate = p
                        break
            if candidate and not any(n == ch_num for n, _ in chapter_files):
                chapter_files.append((ch_num, candidate))
            elif not candidate:
                log.warn(f"Chapter {ch_num} referenced in nav but file not found (href='{href}')")
        chapter_files.sort(key=lambda x: x[0])
        log.log(f"Detected chapters: {[n for n,_ in chapter_files]}")
        total_steps = max(1, len(chapter_files) + 2)
        step = 0
        def bump():
            nonlocal step
            step += 1
            if progress_cb:
                progress_cb(int(step * 100 / total_steps))
        final_zip = output_dir / "final-output.zip"
        produced: List[str] = []
        with zipfile.ZipFile(final_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            for num, ch_file in chapter_files:
                orig = chapter_map.get(num)
                orig_text = orig[0] if orig else ""
                orig_href = orig[1] if orig else ""
                href_candidate = normalize_href_to_actual(orig_href or ch_file.name, xhtml_dir, log)
                extracted = extract_title_from_xhtml(ch_file)
                def _is_short_numeric_or_roman(s: str) -> bool:
                    s2 = (s or "").strip()
                    if not s2:
                        return True
                    if re.fullmatch(r'\d{1,4}', s2):
                        return True
                    if re.fullmatch(r'(?i)(?:m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3}))', s2):
                        return len(s2) <= 4
                    if len(s2) <= 2:
                        return True
                    return False
                chosen_label = ""
                if orig_text and orig_text.strip():
                    bare_chap_rx = r'(?ix)^\s*chapter[\s\-–—]*0*' + str(num) + r'[\s\.:,-]*\s$'
                    if not re.match(bare_chap_rx, orig_text) and not _is_short_numeric_or_roman(orig_text):
                        chosen_label = orig_text
                if not chosen_label:
                    candidate_fname = ""
                    if orig_href:
                        candidate_fname = Path(orig_href).name
                    if not candidate_fname:
                        candidate_fname = ch_file.name
                    best_anchor = find_best_anchor_text_for_href_in_nav(nav_raw_full, candidate_fname)
                    if best_anchor and not _is_short_numeric_or_roman(best_anchor):
                        chosen_label = best_anchor
                if not chosen_label:
                    if extracted and extracted.strip():
                        if re.search(r'\bchapter[\s\-–—]*0*' + str(num) + r'\b', extracted, flags=re.IGNORECASE):
                            chosen_label = extracted
                        else:
                            chosen_label = f"Chapter {num} {extracted}"
                    else:
                        chosen_label = f"Chapter {num}"
                text = chosen_label
                href = normalize_href_to_actual(orig_href or ch_file.name, xhtml_dir, log)
                diag = f"[CHAPTER {num}] nav='{orig_text}' href='{orig_href}' anchor-scan='{find_best_anchor_text_for_href_in_nav(nav_raw_full, ch_file.name)}' extracted='{extracted}' chosen='{text}'"
                log.log(diag)
                if DEBUG:
                    print(diag)
                epub_name, staged_path = stage_chapter_epub(num, text, href, ch_file, xhtml_dir, images_dir, meta_inf, fonts_dir, styles_dir, contents_opf, toc_ncx, log)
                try:
                    zf.write(staged_path, arcname=epub_name)
                except Exception as e:
                    log.warn(f"Failed to write {epub_name} into final zip: {e}")
                    raise
                try:
                    staged_path.unlink()
                except Exception:
                    pass
                produced.append(epub_name)
                log.log(f"Added chapter {num} as {epub_name}")
                bump()
            fm_files: List[Path] = []
            if xhtml_dir.exists():
                for f in xhtml_dir.glob("*.*html"):
                    fn = f.name.lower()
                    for keys in FM_KEYWORDS.values():
                        for k in keys:
                            if k in fn:
                                fm_files.append(f)
                                break
                for item_label, href in fm_items:
                    if not href:
                        continue
                    href_name = Path(href.split('#', 1)[0].split('?', 1)[0]).name
                    found = find_file_case_insensitive(xhtml_dir, href_name)
                    if found:
                        fm_files.append(found)
                    else:
                        for f in xhtml_dir.glob("*.*html"):
                            try:
                                txt = f.read_text(encoding="utf-8", errors="ignore")
                            except Exception:
                                txt = ""
                            if item_label and item_label.lower() in txt.lower():
                                fm_files.append(f)
                                break
                toc_candidate = find_file_case_insensitive(xhtml_dir, "toc.xhtml")
                if toc_candidate and toc_candidate not in fm_files:
                    fm_files.append(toc_candidate)
            fm_files = list(sorted({p.resolve(): p for p in fm_files}.values(), key=lambda p: p.name))
            if DEBUG:
                log.log(f"DEBUG: fm_files found = {[p.name for p in fm_files]} fm_items={fm_items}")
            fm_name, fm_staged = stage_fm_epub(fm_items, fm_files, images_dir, meta_inf, fonts_dir, styles_dir, contents_opf, toc_ncx, log)
            try:
                zf.write(fm_staged, arcname=fm_name)
            except Exception as e:
                log.warn(f"Failed to write {fm_name} into final zip: {e}")
                raise
            try:
                fm_staged.unlink()
            except Exception:
                pass
            produced.append(fm_name)
            log.log("Added FM.epub")
            bump()
            if index_item:
                t, href = index_item
                target = (xhtml_dir / href) if (xhtml_dir / href).exists() else None
                if not target:
                    for f in xhtml_dir.glob("index.*html"):
                        target = f
                        break
                if target:
                    idx_name, idx_staged = stage_index_epub(index_item, target, meta_inf, fonts_dir, styles_dir, contents_opf, toc_ncx, log)
                    try:
                        zf.write(idx_staged, arcname=idx_name)
                    except Exception as e:
                        log.warn(f"Failed to write {idx_name} into final zip: {e}")
                        raise
                    try:
                        idx_staged.unlink()
                    except Exception:
                        pass
                    produced.append(idx_name)
                    log.log("Added Index.epub")
                else:
                    log.warn("Index referenced in nav but file not found; skipping Index.epub")
            else:
                log.warn("No Index entry found in nav.xhtml")
        summary = {"produced": produced, "warnings": log.warnings, "bundle": str(final_zip)}
        log.log(f"Created final bundle: {final_zip.name}")
        return summary
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EPUB → final-output.zip")
        self.setMinimumSize(920, 560)
        self.log = RunLog(messages=[], warnings=[])
        self._init_ui()
    def _init_ui(self):
        root = QHBoxLayout(self)
        left = QVBoxLayout(); left.setAlignment(Qt.AlignTop)
        left.addWidget(QLabel("Input EPUB (single file):"))
        self.input_line = QLineEdit(); left.addWidget(self.input_line)
        btn_in = QPushButton("Browse EPUB..."); btn_in.clicked.connect(self.browse_epub); left.addWidget(btn_in)
        left.addSpacing(8)
        left.addWidget(QLabel("Output folder (final-output.zip will be placed here):"))
        self.output_line = QLineEdit("final-output"); left.addWidget(self.output_line)
        btn_out = QPushButton("Browse Output Folder..."); btn_out.clicked.connect(self.browse_output); left.addWidget(btn_out)
        left.addSpacing(12)
        self.start_btn = QPushButton("▶ Start (Create final-output.zip)")
        self.start_btn.clicked.connect(self.on_start); left.addWidget(self.start_btn)
        left.addStretch()
        root.addLayout(left, 1)
        right = QVBoxLayout()
        right.addWidget(QLabel("Progress:"))
        self.progress = QProgressBar(); self.progress.setValue(0); right.addWidget(self.progress)
        right.addWidget(QLabel("Console:"))
        self.console = QTextEdit(); self.console.setReadOnly(True); right.addWidget(self.console)
        right.addWidget(QLabel("Generated files:"))
        self.results = QListWidget(); right.addWidget(self.results)
        root.addLayout(right, 2)
        self.setLayout(root)
    def browse_epub(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select EPUB", "", "EPUB Files (*.epub)")
        if p:
            self.input_line.setText(p)
            self.quick_scan(p)
    def browse_output(self):
        p = QFileDialog.getExistingDirectory(self, "Select output folder")
        if p:
            self.output_line.setText(p)
    def quick_scan(self, epub_path):
        try:
            with tempfile.TemporaryDirectory(prefix="scan-") as td:
                tmp = Path(td)
                with zipfile.ZipFile(epub_path, 'r') as z:
                    z.extractall(tmp)
                xhtml_dir = get_subdir_case_insensitive(tmp / "OEBPS", "xhtml")
                if xhtml_dir.exists():
                    chs = [p.name for p in xhtml_dir.glob("*.*html") if CHAPTER_REGEX.match(p.name)]
                    others = [p.name for p in xhtml_dir.glob("*.*html") if not CHAPTER_REGEX.match(p.name)]
                    self.console.append(f"Quick scan - Chapters: {', '.join(sorted(chs)) or 'none'}")
                    self.console.append(f"Quick scan - Other XHTML: {', '.join(sorted(others)[:10])}{' …' if len(others) > 10 else ''}")
                else:
                    self.console.append("Quick scan - OEBPS/xhtml not found")
        except Exception as e:
            self.console.append(f"Quick scan failed: {e}")
    def append_console(self, s: str):
        self.console.append(s)
        QApplication.processEvents()
    def on_start(self):
        inp = self.input_line.text().strip()
        out = self.output_line.text().strip() or "final-output"
        if not inp:
            self.append_console("Select an input EPUB first.")
            return
        epub_path = Path(inp)
        if not epub_path.exists():
            self.append_console("Input EPUB does not exist.")
            return
        output_dir = Path(out)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.append_console(f"Could not create output folder: {e}")
            return
        self.console.clear(); self.results.clear(); self.progress.setValue(0)
        self.log = RunLog(messages=[], warnings=[])
        def progress_cb(pct:int):
            self.progress.setValue(pct); QApplication.processEvents()
        try:
            summary = process_single_epub_to_final_zip(epub_path, output_dir, self.log, progress_cb=progress_cb)
        except Exception as e:
            self.append_console(f"ERROR: {e}")
            return
        self.append_console("=== SUMMARY ===")
        self.append_console(json.dumps(summary, indent=2))
        if (output_dir / "final-output.zip").exists():
            self.results.addItem("final-output.zip")
        self.append_console(f"Done. final-output.zip placed at: {output_dir.resolve()}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = App()
    win.show()
    sys.exit(app.exec())
