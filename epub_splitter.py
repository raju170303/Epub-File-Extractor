#!/usr/bin/env python3
"""
Final App: EPUB → Per-section ZIP Splitter (GUI + Backend)

What this app does:
- Lets the user pick an input .epub.
- Unzips it to a temp workspace and reads its structure.
- Detects chapters by filename pattern: ^ch(\d{2})\.(xhtml|html)$ in OEBPS/xhtml/.
- For each detected chapter N, creates a ZIP named chNN.zip containing:
  - mimetype, META-INF/* (copied),
  - OEBPS/fonts/* and OEBPS/styles/* (copied whole),
  - OEBPS/xhtml/chNN.(x)html only,
  - OEBPS/images/figNN_* only,
  - contents.opf and toc.ncx (copied from input; no trimming).
- Creates FM.zip that contains front-matter xhtml files (resolved by names) and a generated OEBPS/xhtml/nav.html:
  - <h1>Contents</h1> at the top
  - A list item "Contents" that links to toc.(x)html if found
  - Then: Cover Page, Half Title Page, Title Page, Copyright,
          About the Editor, About the Contributors (linked where resolvable)
- Creates index.zip that contains the existing index.(x)html (no generated nav in index; we copy the real index page you have).
- Produces a master final-output.zip that contains chNN.zip, FM.zip, index.zip.
- Writes run-log.txt and summary.json into the chosen output folder (next to final-output.zip).

Notes:
- No HTML parsing to find <img>; images are selected purely by name prefix figNN_.
- Warnings only (no auto-fixes) if expected files are missing.
- You can tweak CHAPTER_REGEX and FM_RESOLUTION_TABLE below if your naming differs.

Requires:
  PySide6
Tested on Python 3.10+.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# -----------------------------
# PySide6 GUI imports
# -----------------------------
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QProgressBar, QListWidget, QTextEdit, QStackedWidget, QCheckBox, QLineEdit
)
from PySide6.QtCore import Qt, QTimer

# -----------------------------
# Config / rules
# -----------------------------
CHAPTER_REGEX = re.compile(r"^(?:ch|chp)[-_]?(\d{2,3})\.(xhtml|html)$", re.IGNORECASE)
IMG_PREFIX_FMT = "fig{num:02d}_"  # chapter images: figNN_

# FM label → candidate basenames (without extension)
FM_RESOLUTION_TABLE: Dict[str, List[str]] = {
    "Contents": ["toc", "contents"],
    "Cover Page": ["cover"],
    "Half Title Page": ["half", "half-title"],
    "Title Page": ["tit", "title"],
    "Copyright": ["copyright", "copyright-page"],
    "About the Editor": ["about-editor", "about_the_editor", "editor"],
    "About the Contributors": ["about-contributors", "about_the_contributors", "contributors"],
}

FM_CANONICAL_ORDER = [
    "Contents",
    "Cover Page",
    "Half Title Page",
    "Title Page",
    "Copyright",
    "About the Editor",
    "About the Contributors",
]

BACK_MATTER_INDEX_CANDIDATES = ["index"]  # for index.zip

# -----------------------------
# Logging helpers
# -----------------------------
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
    def dump_files(self, out_root: Path):
        (out_root / "run-log.txt").write_text("\n".join(self.messages + self.warnings), encoding="utf-8")

# -----------------------------
# Backend: EPUB processing
# -----------------------------

def unzip_epub(epub_path: Path, workdir: Path, log: RunLog) -> Path:
    dst = workdir / f"unzipped-{uuid.uuid4().hex}"
    dst.mkdir(parents=True, exist_ok=True)
    log.log(f"Unzipping EPUB: {epub_path}")
    with zipfile.ZipFile(epub_path, 'r') as z:
        z.extractall(dst)
    return dst


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def detect_chapters(xhtml_dir: Path, log: RunLog) -> List[Tuple[int, Path]]:
    chapters: List[Tuple[int, Path]] = []
    for p in xhtml_dir.glob("*.*html"):
        m = CHAPTER_REGEX.match(p.name)
        if m:
            num = int(m.group(1))
            chapters.append((num, p))
    chapters.sort(key=lambda t: t[0])
    log.log(f"Detected chapters: {[f'ch{n:02d}' for n,_ in chapters]} (in {xhtml_dir})")
    return chapters


def find_first_existing(base_dir: Path, names_wo_ext: List[str]) -> Optional[Path]:
    for stem in names_wo_ext:
        for ext in (".xhtml", ".html"):
            cand = base_dir / f"{stem}{ext}"
            if cand.exists():
                return cand
        # fallback: contains match
        for ext in (".xhtml", ".html"):
            hits = list(base_dir.glob(f"*{stem}*{ext}"))
            if hits:
                return hits[0]
    return None


def resolve_fm_files(xhtml_dir: Path, log: RunLog) -> List[Path]:
    out: List[Path] = []
    for label in FM_CANONICAL_ORDER:
        cands = FM_RESOLUTION_TABLE.get(label, [])
        found = find_first_existing(xhtml_dir, cands)
        if found:
            out.append(found)
            log.log(f"FM: '{label}' → {found.name}")
        else:
            log.warn(f"FM missing: '{label}' (searched {cands} in {xhtml_dir})")
    return out


def resolve_index_file(xhtml_dir: Path, log: RunLog) -> Optional[Path]:
    found = find_first_existing(xhtml_dir, BACK_MATTER_INDEX_CANDIDATES)
    if found:
        log.log(f"Index resolved → {found.name}")
    else:
        log.warn("Index xhtml not found (searched: index(.xhtml|.html))")
    return found


def write_text(path: Path, text: str):
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def build_zip_structure(temp_dir: Path):
    ensure_dir(temp_dir / "META-INF")
    ensure_dir(temp_dir / "OEBPS" / "fonts")
    ensure_dir(temp_dir / "OEBPS" / "styles")
    ensure_dir(temp_dir / "OEBPS" / "xhtml")
    ensure_dir(temp_dir / "OEBPS" / "images")
    write_text(temp_dir / "mimetype", "application/epub+zip")


def zip_directory(src_dir: Path, out_zip: Path):
    with zipfile.ZipFile(out_zip, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(src_dir):
            root_p = Path(root)
            for f in files:
                fpath = root_p / f
                arc = fpath.relative_to(src_dir)
                z.write(fpath, arcname=str(arc))


def generate_fm_nav_html(xhtml_dir: Path, log: RunLog) -> str:
    """Create FM nav.html with: H1 'Contents' + list entries (linked when resolvable)."""
    def link_if_exists(label: str, stems: List[str]) -> str:
        tgt = find_first_existing(xhtml_dir, stems)
        if tgt:
            return f'<li><a href="{tgt.name}">{label}</a></li>'
        else:
            log.warn(f"FM nav: link target missing for '{label}' (searched {stems})")
            return f"<li>{label}</li>"

    parts: List[str] = []
    parts.append("<h1>Contents</h1>")  # headline at the top
    parts.append("<ul>")
    parts.append(link_if_exists("Contents", FM_RESOLUTION_TABLE["Contents"]))
    parts.append(link_if_exists("Cover Page", FM_RESOLUTION_TABLE["Cover Page"]))
    parts.append(link_if_exists("Half Title Page", FM_RESOLUTION_TABLE["Half Title Page"]))
    parts.append(link_if_exists("Title Page", FM_RESOLUTION_TABLE["Title Page"]))
    parts.append(link_if_exists("Copyright", FM_RESOLUTION_TABLE["Copyright"]))
    parts.append(link_if_exists("About the Editor", FM_RESOLUTION_TABLE["About the Editor"]))
    parts.append(link_if_exists("About the Contributors", FM_RESOLUTION_TABLE["About the Contributors"]))
    parts.append("</ul>")

    html_doc = f"""<!doctype html>
<html xmlns="http://www.w3.org/1999/xhtml" lang="en">
<head>
  <meta charset="utf-8" />
  <title>Front Matter</title>
</head>
<body>
  {''.join(parts)}
</body>
</html>
"""
    return html_doc


@dataclass
class BuildSummary:
    chapters_built: List[str]
    fm_built: bool
    index_built: bool
    warnings: List[str]


def build_all(epub_path: Path, output_root: Path, log: RunLog, progress_cb=None) -> BuildSummary:
    ensure_dir(output_root)

    with tempfile.TemporaryDirectory(prefix="epubsplit-") as tmpd:
        tmpdir = Path(tmpd)
        unzipped = unzip_epub(epub_path, tmpdir, log)

        # Expected paths
        meta_inf = unzipped / "META-INF"
        oebps = unzipped / "OEBPS"
        xhtml_dir = oebps / "xhtml"
        styles_dir = oebps / "styles"
        fonts_dir = oebps / "fonts"
        images_dir = oebps / "images"
        contents_opf = oebps / "contents.opf"
        toc_ncx = oebps / "toc.ncx"

        if not xhtml_dir.exists():
            log.warn(f"Missing OEBPS/xhtml in {unzipped}")

        chapters = detect_chapters(xhtml_dir, log)
        total_steps = max(1, len(chapters) + 2)  # + FM + index
        step = 0

        def bump():
            nonlocal step
            step += 1
            if progress_cb:
                progress_cb(int(step * 100 / total_steps))

        chapter_zip_names: List[str] = []

        # Build per-chapter ZIPs
        for num, ch_file in chapters:
            ch_tag = f"ch{num:02d}"
            with tempfile.TemporaryDirectory(prefix=f"{ch_tag}-") as ch_tmp:
                ch_dir = Path(ch_tmp)
                build_zip_structure(ch_dir)

                # Copy constants
                if meta_inf.exists():
                    shutil.copytree(meta_inf, ch_dir / "META-INF", dirs_exist_ok=True)
                if fonts_dir.exists():
                    shutil.copytree(fonts_dir, ch_dir / "OEBPS" / "fonts", dirs_exist_ok=True)
                if styles_dir.exists():
                    shutil.copytree(styles_dir, ch_dir / "OEBPS" / "styles", dirs_exist_ok=True)

                # Chapter XHTML
                ensure_dir(ch_dir / "OEBPS" / "xhtml")
                shutil.copy2(ch_file, ch_dir / "OEBPS" / "xhtml" / ch_file.name)

                # Chapter images (figNN_*)
                ensure_dir(ch_dir / "OEBPS" / "images")
                prefix = IMG_PREFIX_FMT.format(num=num)
                matched = []
                if images_dir.exists():
                    matched = list(images_dir.glob(f"{prefix}*"))
                    if not matched:
                        log.warn(f"No images matched for {ch_tag}: prefix '{prefix}'")
                    for img in matched:
                        shutil.copy2(img, ch_dir / "OEBPS" / "images" / img.name)
                else:
                    log.warn("Images directory missing in input")

                # OPF & NCX copied (MVP copies, no trim)
                if contents_opf.exists():
                    shutil.copy2(contents_opf, ch_dir / "OEBPS" / "contents.opf")
                else:
                    log.warn("contents.opf missing in input")
                if toc_ncx.exists():
                    shutil.copy2(toc_ncx, ch_dir / "OEBPS" / "toc.ncx")
                else:
                    log.warn("toc.ncx missing in input")

                # Zip → chNN.zip in output_root
                out_zip = output_root / f"{ch_tag}.zip"
                zip_directory(ch_dir, out_zip)
                log.log(f"Wrote {out_zip.name}")
                chapter_zip_names.append(out_zip.name)

            bump()

        # Build FM.zip
        with tempfile.TemporaryDirectory(prefix="FM-") as fm_tmp:
            fm_dir = Path(fm_tmp)
            build_zip_structure(fm_dir)

            if meta_inf.exists():
                shutil.copytree(meta_inf, fm_dir / "META-INF", dirs_exist_ok=True)
            if fonts_dir.exists():
                shutil.copytree(fonts_dir, fm_dir / "OEBPS" / "fonts", dirs_exist_ok=True)
            if styles_dir.exists():
                shutil.copytree(styles_dir, fm_dir / "OEBPS" / "styles", dirs_exist_ok=True)

            # FM XHTMLs
            fm_files = resolve_fm_files(xhtml_dir, log)
            for f in fm_files:
                shutil.copy2(f, fm_dir / "OEBPS" / "xhtml" / f.name)

            # FM nav.html per spec
            fm_nav_html = generate_fm_nav_html(xhtml_dir, log)
            write_text(fm_dir / "OEBPS" / "xhtml" / "nav.html", fm_nav_html)

            # FM images: include non-chapter images (exclude fig\d+_)
            if images_dir.exists():
                for img in images_dir.iterdir():
                    if not img.is_file():
                        continue
                    if re.match(r"^fig\d+_", img.name, re.IGNORECASE):
                        continue
                    ensure_dir(fm_dir / "OEBPS" / "images")
                    shutil.copy2(img, fm_dir / "OEBPS" / "images" / img.name)

            # OPF & NCX copied
            if contents_opf.exists():
                shutil.copy2(contents_opf, fm_dir / "OEBPS" / "contents.opf")
            if toc_ncx.exists():
                shutil.copy2(toc_ncx, fm_dir / "OEBPS" / "toc.ncx")

            out_zip = output_root / "FM.zip"
            zip_directory(fm_dir, out_zip)
            log.log("Wrote FM.zip")

        bump()

        # Build index.zip — copy existing index.(x)html only; no generated nav
        index_src = resolve_index_file(xhtml_dir, log)
        index_built = False
        if index_src:
            with tempfile.TemporaryDirectory(prefix="INDEX-") as idx_tmp:
                idx_dir = Path(idx_tmp)
                build_zip_structure(idx_dir)

                if meta_inf.exists():
                    shutil.copytree(meta_inf, idx_dir / "META-INF", dirs_exist_ok=True)
                if fonts_dir.exists():
                    shutil.copytree(fonts_dir, idx_dir / "OEBPS" / "fonts", dirs_exist_ok=True)
                if styles_dir.exists():
                    shutil.copytree(styles_dir, idx_dir / "OEBPS" / "styles", dirs_exist_ok=True)

                shutil.copy2(index_src, idx_dir / "OEBPS" / "xhtml" / index_src.name)

                if contents_opf.exists():
                    shutil.copy2(contents_opf, idx_dir / "OEBPS" / "contents.opf")
                if toc_ncx.exists():
                    shutil.copy2(toc_ncx, idx_dir / "OEBPS" / "toc.ncx")

                out_zip = output_root / "index.zip"
                zip_directory(idx_dir, out_zip)
                log.log("Wrote index.zip")
                index_built = True
        else:
            log.warn("index.zip skipped: index xhtml not found")

        bump()

        # Create master final-output.zip that contains all produced .zip files
        all_zips = [str(p.name) for p in sorted(output_root.glob("*.zip"))]
        final_zip = output_root / "final-output.zip"
        with zipfile.ZipFile(final_zip, 'w', compression=zipfile.ZIP_DEFLATED) as fzip:
            for name in all_zips:
                fzip.write(output_root / name, arcname=name)
        log.log(f"Master package written → {final_zip.name}")

        # write logs + summary
        log.dump_files(output_root)
        summary = BuildSummary(
            chapters_built=[n for n in chapter_zip_names],
            fm_built=True,
            index_built=index_built,
            warnings=log.warnings,
        )
        (output_root / "summary.json").write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")

        return summary

# -----------------------------
# GUI
# -----------------------------
class EpubSplitterApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EPUB → ZIP Splitter (Final App)")
        self.setMinimumSize(1000, 640)
        self.log = RunLog(messages=[], warnings=[])
        self.init_ui()

    def init_ui(self):
        root = QHBoxLayout(self)

        # Sidebar
        sidebar = QVBoxLayout(); sidebar.setAlignment(Qt.AlignTop)
        title = QLabel("📘 EPUB → ZIP Splitter")
        title.setStyleSheet("font-size: 22px; font-weight: bold; margin-bottom: 12px;")
        sidebar.addWidget(title)
        for s in ["1️⃣ Select EPUB", "2️⃣ Output & Options", "3️⃣ Run", "4️⃣ Results"]:
            lbl = QLabel("• " + s)
            lbl.setStyleSheet("font-size: 15px; margin: 4px 0;")
            sidebar.addWidget(lbl)
        sidebar.addStretch()
        root.addLayout(sidebar, 1)

        # Pages
        self.pages = QStackedWidget()
        self.pages.addWidget(self.page_select())
        self.pages.addWidget(self.page_options())
        self.pages.addWidget(self.page_run())
        self.pages.addWidget(self.page_results())
        root.addWidget(self.pages, 4)

    # ---- Pages ----
    def page_select(self):
        w = QWidget(); lay = QVBoxLayout(w)
        lay.addWidget(QLabel("Select an EPUB file:"))
        self.input_path = QLineEdit(); self.input_path.setPlaceholderText("/path/to/book.epub")
        btn = QPushButton("Browse EPUB…"); btn.clicked.connect(self.on_browse_epub)
        lay.addWidget(self.input_path); lay.addWidget(btn)
        self.info_detect = QLabel("")
        self.info_detect.setStyleSheet("color:#666;")
        lay.addWidget(self.info_detect)
        next_btn = QPushButton("Next →"); next_btn.clicked.connect(lambda: self.pages.setCurrentIndex(1))
        lay.addWidget(next_btn, alignment=Qt.AlignRight)
        return w

    def page_options(self):
        w = QWidget(); lay = QVBoxLayout(w)
        lay.addWidget(QLabel("Output Folder:"))
        self.output_path = QLineEdit("final-output")
        out_btn = QPushButton("Select Folder…"); out_btn.clicked.connect(self.on_browse_output)
        lay.addWidget(self.output_path); lay.addWidget(out_btn)

        lay.addWidget(QLabel("What to generate:"))
        self.chk_chapters = QCheckBox("Per-chapter ZIPs"); self.chk_chapters.setChecked(True)
        self.chk_fm = QCheckBox("FM.zip (Front Matter)"); self.chk_fm.setChecked(True)
        self.chk_index = QCheckBox("index.zip (Index)"); self.chk_index.setChecked(True)
        for chk in (self.chk_chapters, self.chk_fm, self.chk_index):
            lay.addWidget(chk)

        lay.addWidget(QLabel("Detection Rule (regex for chapter files):"))
        self.rule = QLineEdit(r"^(?:ch|chp)[-_]?(\d{2,3})\.(xhtml|html)$")
        lay.addWidget(self.rule)

        next_btn = QPushButton("Next →"); next_btn.clicked.connect(lambda: self.pages.setCurrentIndex(2))
        lay.addWidget(next_btn, alignment=Qt.AlignRight)
        return w

    def page_run(self):
        w = QWidget(); lay = QVBoxLayout(w)
        lay.addWidget(QLabel("Run"))
        self.progress = QProgressBar(); self.progress.setValue(0)
        lay.addWidget(self.progress)
        self.console = QTextEdit(); self.console.setReadOnly(True)
        lay.addWidget(self.console)
        start_btn = QPushButton("▶ Start Processing")
        start_btn.clicked.connect(self.on_start)
        lay.addWidget(start_btn, alignment=Qt.AlignCenter)
        next_btn = QPushButton("Next →"); next_btn.clicked.connect(lambda: self.pages.setCurrentIndex(3))
        lay.addWidget(next_btn, alignment=Qt.AlignRight)
        return w

    def page_results(self):
        w = QWidget(); lay = QVBoxLayout(w)
        lay.addWidget(QLabel("Results"))
        self.results_list = QListWidget(); lay.addWidget(self.results_list)
        open_btn = QPushButton("📂 Open Output Folder")
        open_btn.clicked.connect(self.on_open_output)
        lay.addWidget(open_btn, alignment=Qt.AlignCenter)
        restart_btn = QPushButton("← Back to Start")
        restart_btn.clicked.connect(lambda: self.pages.setCurrentIndex(0))
        lay.addWidget(restart_btn, alignment=Qt.AlignRight)
        return w

    # ---- Handlers ----
    def on_browse_epub(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select EPUB", "", "EPUB Files (*.epub)")
        if not path:
            return
        self.input_path.setText(path)
        # Light scan just to show quick info
        try:
            with tempfile.TemporaryDirectory(prefix="scan-") as t:
                tmp = Path(t)
                with zipfile.ZipFile(path, 'r') as z:
                    z.extractall(tmp)
                xhtml_dir = tmp / "OEBPS" / "xhtml"
                if xhtml_dir.exists():
                    chapters = [p.name for p in xhtml_dir.glob("*.*html") if CHAPTER_REGEX.match(p.name)]
                    others = [p.name for p in xhtml_dir.glob("*.*html") if not CHAPTER_REGEX.match(p.name)]
                    self.info_detect.setText(
                        f"Chapters detected: {', '.join(sorted(chapters)) or 'none'}\nOther XHTML: {', '.join(sorted(others)[:10])}{' …' if len(others)>10 else ''}"
                    )
                else:
                    self.info_detect.setText("No OEBPS/xhtml/ found in EPUB")
        except Exception as e:
            self.info_detect.setText(f"Failed to scan EPUB: {e}")

    def on_browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self.output_path.setText(path)

    def on_start(self):
        epub = Path(self.input_path.text().strip())
        out = Path(self.output_path.text().strip() or "final-output")
        if not epub.exists():
            self.append_console("Please select a valid EPUB file.")
            return
        out.mkdir(parents=True, exist_ok=True)

        # Reset UI
        self.console.clear()
        self.progress.setValue(0)
        self.results_list.clear()
        self.log = RunLog(messages=[], warnings=[])

        def progress_cb(pct: int):
            self.progress.setValue(pct)
            QApplication.processEvents()

        try:
            summary = build_all(epub, out, self.log, progress_cb=progress_cb)
        except Exception as e:
            self.append_console(f"ERROR: {e}")
            return

        # Show results
        self.append_console("=== SUMMARY ===")
        self.append_console(json.dumps(asdict(summary), indent=2))
        for z in sorted(out.glob("*.zip")):
            self.results_list.addItem(z.name)
        self.append_console(f"Output written to: {out.resolve()}")

    def on_open_output(self):
        path = self.output_path.text().strip()
        if not path:
            return
        if sys.platform.startswith('win'):
            os.startfile(path)
        elif sys.platform == 'darwin':
            os.system(f'open "{path}"')
        else:
            os.system(f'xdg-open "{path}"')

    def append_console(self, text: str):
        self.console.append(text)
        QApplication.processEvents()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = EpubSplitterApp()
    win.show()
    sys.exit(app.exec())
