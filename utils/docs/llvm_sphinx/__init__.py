# -*- coding: utf-8 -*-

"""Shared configuration and extensions for llvm-project sphinx documentation

Every sphinx `conf.py` in llvm-project is intended to use `common_conf` as a
baseline configuration. The sphinx module-as-conf model means the easiest
way to achieve this is by updating `globals()` directly, as in:

    from llvm_sphinx import *  # see llvm-project/utils/docs/README.md

    globals().update(common_conf(tags))

Note: common settings like `extensions` should not be reassigned after the
call to `common_conf`, they should be modified/appended to, as in:

    extensions += ["foo"]

"""

import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from enum import Enum, auto
from sphinx.util.tags import Tags
from llvm_sphinx.help import venv_help

_SHARED_STATIC_DIR = Path(__file__).parent / "_static"


class Markdown(Enum):
    ALWAYS = auto()
    EXCEPT_MAN = auto()
    NEVER = auto()


def common_conf(tags: Tags, markdown=Markdown.ALWAYS) -> Dict[str, Any]:
    # If your documentation needs a minimal Sphinx version, state it here.
    # needs_sphinx = '1.0'
    # The encoding of source files.
    # source_encoding = 'utf-8-sig'
    extensions = ["llvm_sphinx.ext.mlir_pygments"]
    source_suffix = {".rst": "restructuredtext"}
    if markdown != Markdown.NEVER:
        # When building man pages, we do not use the markdown pages,
        # So, we can continue without the myst_parser dependencies.
        # Doing so reduces dependencies of some packaged llvm distributions.
        try:
            import myst_parser
        except ImportError as err:
            if markdown == Markdown.ALWAYS or not tags.has("builder-man"):
                print(venv_help(err), file=sys.stderr)
                raise
        else:
            extensions.append("myst_parser")
            source_suffix[".md"] = "markdown"
    myst_enable_extensions = ["substitution", "colon_fence"]
    myst_heading_anchors = 6
    myst_heading_slug_func = "llvm_sphinx.make_slug"
    templates_path = ["_templates"]
    master_doc = "index"
    # The sphinx implementation of numfig seems to be incredibly slow for
    # larger projects, and has an outsized impact on no-op builds or builds
    # with just a single file changed (i.e. the cases which most impact the
    # documentation writer's experience). There were only every 2 uses in 1
    # document, so we just forbid using it until the performance can be fixed.
    numfig = False

    return locals()


def _append_unique(target, entries):
    for entry in entries:
        if entry not in target:
            target.append(entry)


def configure_furo(
    conf: Dict[str, Any],
    *,
    source_directory: str,
    html_title: str,
    html_logo: Optional[str] = None,
    local_static_path: Iterable[str] = (),
    extra_css_files: Iterable[str] = (),
    extra_js_files: Iterable[str] = (),
) -> None:
    """Configure the shared Furo theme setup for LLVM Sphinx projects."""
    extensions = conf.setdefault("extensions", [])
    _append_unique(extensions, ["llvm_sphinx.ext.furo"])

    conf["html_theme"] = "furo"
    conf["html_theme_options"] = {
        "source_repository": "https://github.com/llvm/llvm-project",
        "source_branch": "main",
        "source_directory": source_directory,
    }
    conf["html_title"] = html_title
    if html_logo is not None:
        conf["html_logo"] = html_logo
    conf["html_static_path"] = list(local_static_path)
    conf["html_css_files"] = list(extra_css_files)
    conf["html_js_files"] = list(extra_js_files)


def shared_static_asset(filename: str) -> str:
    """Return an absolute path to a shared LLVM Sphinx static asset."""
    return str(_SHARED_STATIC_DIR / filename)


# Some of our markdown documentation numbers section titles
# This helpers is used by myst to remove that numbering from the anchor links.
def make_slug(s: str) -> str:
    from docutils.nodes import make_id
    from re import sub

    s = sub(r"^\s*(\w\.)+\w\s", "", s)
    s = sub(r"^\s*\w\.\s", "", s)
    return make_id(s)
