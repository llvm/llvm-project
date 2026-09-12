"""Shared Furo theme assets for LLVM Sphinx projects."""

import zlib
from pathlib import Path

from sphinx.util.fileutil import copy_asset

_SHARED_STATIC_PREFIX = "llvm-sphinx"
_SHARED_STATIC_DIR = Path(__file__).parents[1] / "_static"


def _furo_pygments_stylesheet():
    from furo import get_pygments_stylesheet

    return get_pygments_stylesheet()


def _rebuild_on_pygments_change(app, env, added, changed, removed):
    if app.builder.format != "html":
        return ()

    stylesheet = _furo_pygments_stylesheet().encode()
    checksum = zlib.crc32(stylesheet.translate(None, b"\r"))
    previous_checksum = getattr(env, "_llvm_furo_pygments_checksum", None)
    env._llvm_furo_pygments_checksum = checksum
    if checksum != previous_checksum:
        return env.found_docs
    return ()


def _install_furo_pygments_writer(app):
    if app.builder.format != "html":
        return

    # Sphinx normally writes pygments.css immediately before rendering HTML,
    # which lets it add a checksum of that file to the stylesheet URL. Furo
    # replaces the file at build-finished to add dark-mode styles, after Sphinx
    # has already calculated the checksum. Generate Furo's final contents at
    # Sphinx's normal asset-writing point so the checksum matches the file that
    # Furo writes again at build-finished.
    def create_pygments_style_file():
        pygments_css = Path(app.builder.outdir) / "_static" / "pygments.css"
        pygments_css.parent.mkdir(parents=True, exist_ok=True)
        pygments_css.write_text(_furo_pygments_stylesheet(), encoding="utf-8")

    app.builder.create_pygments_style_file = create_pygments_style_file


def _add_shared_static_files(app):
    if app.builder.format != "html":
        return

    app.add_js_file(f"{_SHARED_STATIC_PREFIX}/copybutton.js")
    app.add_css_file(f"{_SHARED_STATIC_PREFIX}/copybutton.css")


def _copy_shared_static_files(app, exception):
    if exception is not None or app.builder.format != "html":
        return

    copy_asset(
        str(_SHARED_STATIC_DIR),
        str(Path(app.builder.outdir) / "_static" / _SHARED_STATIC_PREFIX),
    )


def setup(app):
    # Furo initializes the light and dark Pygments styles in its own
    # builder-inited handler at the default priority (500).
    app.connect("builder-inited", _install_furo_pygments_writer, priority=600)
    app.connect("env-get-outdated", _rebuild_on_pygments_change)
    app.connect("builder-inited", _add_shared_static_files)
    app.connect("build-finished", _copy_shared_static_files)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
