# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Pygments lexer for LLVM IR.

Pygments ships an LLVM IR lexer with a fixed keyword list that lags behind the
IR. When it meets a keyword it does not know, it emits error tokens, and Sphinx
drops highlighting for the whole code block.

This lexer extends the stock one with the keywords the IR parser accepts. They
are read from the source tree, so new instructions, types and attributes are
highlighted without touching this file.
"""

import re
from pathlib import Path
from typing import List, Set

from pygments.lexer import inherit, words
from pygments.lexers.asm import LlvmLexer as _PygmentsLlvmLexer
from pygments.token import Keyword, Name, Number, Punctuation

__all__ = ["LlvmLexer"]

_LLVM_DIR = Path(__file__).resolve().parents[2]
_LLLEXER = _LLVM_DIR / "lib" / "AsmParser" / "LLLexer.cpp"
_ATTRIBUTES = _LLVM_DIR / "include" / "llvm" / "IR" / "Attributes.td"


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _ir_types() -> Set[str]:
    return set(re.findall(r'\bTYPEKEYWORD\(\s*"(\w+)"', _read(_LLLEXER)))


def _ir_keywords() -> Set[str]:
    # KEYWORD(foo) and INSTKEYWORD(foo, Foo) in LLLexer.cpp.
    kws = set(re.findall(r"\b(?:INST)?KEYWORD\(\s*([A-Za-z_]\w*)", _read(_LLLEXER)))
    # Attributes are pulled into LLLexer.cpp from the generated Attributes.inc.
    # Read their names from Attributes.td instead. String attributes
    # (StrBoolAttr) are quoted in the IR, so they are not keywords.
    kws |= set(
        re.findall(
            r'\b(?:Enum|Int|Type|ConstantRange|ConstantRangeList)Attr<"(\w+)"',
            _read(_ATTRIBUTES),
        )
    )
    # Macro parameter names, matched by the #define lines.
    kws -= {"STR", "DISPLAY_NAME"}
    return kws


def _keyword_rules():
    rules: List[tuple] = []
    for kws, token in ((_ir_types(), Keyword.Type), (_ir_keywords(), Keyword)):
        if kws:
            rules.append((words(sorted(kws), suffix=r"\b"), token))
    return rules


class LlvmLexer(_PygmentsLlvmLexer):
    tokens = {
        "root": [
            # Hex FP literals, e.g. `0x1.8p1`. They must come before the stock
            # rules, which would split them at the `.`.
            (r"[-+]?0x[0-9A-Fa-f]+\.[0-9A-Fa-f]*[pP][-+]?\d+", Number.Float),
            inherit,
            # Comdat names, e.g. `$foo = comdat any`. They come after the
            # stock label rule, so `$foo:` is still a label.
            (r'\$(?:[-a-zA-Z$._][\w\-$.]*|"[^"]*?")', Name.Variable),
            # Debug records, e.g. `#dbg_value(...)`.
            (r"#dbg_\w+", Keyword),
            # Summary entries, e.g. `^0 = module: ...`.
            (r"\^\d+", Name.Variable.Anonymous),
            # Integers with a sign prefix, e.g. `u0x8000`, and `f0x` floats.
            (r"[usf]0x[0-9A-Fa-f]+", Number.Hex),
            # Numbers with a leading `+`, e.g. `+1.0`.
            (r"\+\d+(?:[.]\d+)?(?:[eE][-+]?\d+(?:[.]\d+)?)?", Number),
            # DWARF and debug info constants, e.g. `DW_TAG_member`.
            (r"(?:DW_|DIFlag|DISPFlag|CSK_)\w+", Name.Constant),
            # Flag separator, e.g. `DIFlagPublic | DIFlagVector`.
            (r"\|", Punctuation),
            # Any other word, e.g. a placeholder in an example. Without this,
            # one unknown word disables highlighting for the whole block.
            (r"[A-Za-z_][\w.]*", Name),
        ],
        "keyword": _keyword_rules()
        + [
            # Byte types, e.g. `b8`.
            (r"b[1-9]\d*\b", Keyword.Type),
            # Numbered calling conventions, e.g. `cc10`.
            (r"cc\d+\b", Keyword),
            inherit,
        ],
    }
