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
from typing import Any, Dict, TYPE_CHECKING
from enum import Enum, auto
from sphinx.util.tags import Tags
from llvm_sphinx.help import venv_help


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
    myst_enable_extensions = ["substitution"]
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


# Some of our markdown documentation numbers section titles
# This helpers is used by myst to remove that numbering from the anchor links.
def make_slug(s: str) -> str:
    from docutils.nodes import make_id
    from re import sub

    s = sub(r"^\s*(\w\.)+\w\s", "", s)
    s = sub(r"^\s*\w\.\s", "", s)
    return make_id(s)
