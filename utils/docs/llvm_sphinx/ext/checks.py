# -*- coding: utf-8 -*-

"""Sphinx extension for llvm-project checks/lints.

Enable by adding "llvm_sphinx.checks" to the sphinx `extensions` list.
"""

import sys
from typing import Dict, List
from llvm_sphinx.help import venv_help

try:
    from sphinx.application import Sphinx
    from sphinx.environment import BuildEnvironment
    from sphinx.util import logging
    from sphinx.errors import ExtensionError
except ImportError as err:
    print(venv_help(err), file=sys.stderr)
    raise

__version__ = "1.0"

logger = logging.getLogger("llvm_sphinx.ext.checks")


def setup(app: Sphinx) -> Dict[str, object]:
    app.connect("env-updated", check_env)
    return {
        "version": __version__,
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }


def check_env(app: Sphinx, env: BuildEnvironment) -> None:
    toc_parents: Dict[str, List[str]] = {}
    for parent, children in env.toctree_includes.items():
        for child in children:
            toc_parents.setdefault(child, []).append(parent)

    for doc, parents in sorted(toc_parents.items()):
        if len(parents) > 1:
            # sphinx considers this an `info` only, and silently picks one
            # parent arbitrarily. We upgrade this to an `error` to ensure
            # we don't accumulate a growing list of offending toctrees.
            logger.error(
                "document is referenced in multiple toctrees: %s",
                parents,
                location=doc,
            )
