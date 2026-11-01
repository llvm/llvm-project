# -*- coding: utf-8 -*-

"""Sphinx extension for llvm-project LLVM IR syntax highlighting."""

import importlib.util
from pathlib import Path
import sys
from typing import Dict
from llvm_sphinx.help import venv_help

try:
    from sphinx.application import Sphinx
except ImportError as err:
    print(venv_help(err), file=sys.stderr)
    raise

__version__ = "1.0"


def _load_llvm_lexer():
    lexer_path = (
        Path(__file__).resolve().parents[4]
        / "llvm"
        / "utils"
        / "pygments"
        / "llvm_lexer.py"
    )
    spec = importlib.util.spec_from_file_location("llvm_sphinx_llvm_lexer", lexer_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LlvmLexer


def setup(app: Sphinx) -> Dict[str, object]:
    app.add_lexer("llvm", _load_llvm_lexer())
    return {
        "version": __version__,
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
