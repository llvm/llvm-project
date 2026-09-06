"""Shared Furo theme assets for LLVM Sphinx projects."""

from pathlib import Path

from sphinx.util.fileutil import copy_asset

_SHARED_STATIC_PREFIX = "llvm-sphinx"
_SHARED_STATIC_DIR = Path(__file__).parents[1] / "_static"


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
    app.connect("builder-inited", _add_shared_static_files)
    app.connect("build-finished", _copy_shared_static_files)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
