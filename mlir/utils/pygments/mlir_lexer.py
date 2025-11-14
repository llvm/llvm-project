# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pygments.lexer import RegexLexer, bygroups, include, using
from pygments.token import *
import re


class MlirLexer(RegexLexer):
    """Pygments lexer for MLIR.

    This lexer focuses on accurate tokenization of common MLIR constructs:
    - SSA values (%%... / %...)
    - attribute and type aliases (#name =, !name =)
    - types (builtin and dialect types, parametric types)
    - attribute dictionaries and nested containers to a reasonable depth
    - numbers (ints, floats with exponents, hex)
    - strings with common escapes
    - line comments (// ...)
    - block labels (^foo) and operations
    """

    name = "MLIR"
    aliases = ["mlir"]
    filenames = ["*.mlir"]

    flags = re.MULTILINE

    class VariableList(RegexLexer):
        """Lexer for lists of SSA variables separated by commas."""

        tokens = {
            "root": [
                (r"\s+", Text),
                (r",", Punctuation),
                (r"%[_A-Za-z0-9\.\$\-:#]+", Name.Variable),
            ]
        }

    tokens = {
        "root": [
            # Comments
            (r"//.*?$", Comment.Single),
            # operation name with assignment: %... = op.name
            (
                r"^(\s*)(%[\%_A-Za-z0-9\:#\,\s]+)(=)(\s*)([A-Za-z0-9_\.\$\-]+)\b",
                bygroups(Text, using(VariableList), Operator, Text, Name.Builtin),
            ),
            # operation name without result
            (r"^(\s*)([A-Za-z0-9_\.\$\-]+)\b(?=[^<:])", bygroups(Text, Name.Builtin)),
            # Attribute alias definition:  #name =
            (
                r"^(\s*)(#[_A-Za-z0-9\$\-\.]+)(\b)(\s*=)",
                bygroups(Text, Name.Constant, Text, Operator),
            ),
            # Type alias definition: !name =
            (
                r"^(\s*)(![_A-Za-z0-9\$\-\.]+)(\b)(\s*=)",
                bygroups(Text, Keyword.Type, Text, Operator),
            ),
            # SSA values (uses)
            (r"%[_A-Za-z0-9\.\$\-:#]+", Name.Variable),
            # attribute refs, constants and named attributes
            (r"#[_A-Za-z0-9\$\-\.]+\b", Name.Constant),
            # symbol refs / function-like names
            (r"@[_A-Za-z][_A-Za-z0-9\$\-\.]*\b", Name.Function),
            # blocks
            (r"\^[A-Za-z0-9_\$\.\-]+", Name.Label),
            # types by exclamation or builtin names
            (r"![_A-Za-z0-9\$\-\.]+\b", Keyword.Type),
            # NOTE: please sync changes to corresponding builtin type rule in "angled-type"
            (r"\b(bf16|f16|f32|f64|f80|f128|index|none|(u|s)?i[0-9]+)\b", Keyword.Type),
            # container-like dialect types (tensor<...>, memref<...>, vector<...>)
            (
                r"\b(complex|memref|tensor|tuple|vector)\s*(<)",
                bygroups(Keyword.Type, Punctuation),
                "angled-type",
            ),
            # affine constructs
            (r"\b(affine_map|affine_set)\b", Keyword.Reserved),
            # common builtin operators / functions inside affine_map
            (r"\b(ceildiv|floordiv|mod|symbol)\b", Name.Other),
            # identifiers / bare words
            (r"\b[_A-Za-z][_A-Za-z0-9\.-]*\b", Name.Other),
            # numbers: hex, float (with exponent), integer
            (r"\b0x[0-9A-Fa-f]+\b", Number.Hex),
            (r"\b([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?\b", Number.Float),
            (r"\b[0-9]+\b", Number.Integer),
            # strings
            (r'"', String.Double, "string"),
            # punctuation and arrow-like tokens
            (r"->|>=|<=|\>=|\<=|\->|\=>", Operator),
            (r"[()\[\]{}<>,.:=]", Punctuation),
            # operators
            (r"[-+*/%]", Operator),
        ],
        # string state with common escapes
        "string": [
            (r'\\[ntr"\\]', String.Escape),
            (r'[^"\\]+', String.Double),
            (r'"', String.Double, "#pop"),
        ],
        # angled-type content
        "angled-type": [
            # match nested '<' and '>'
            (r"<", Punctuation, "#push"),
            (r">", Punctuation, "#pop"),
            # dimensions like 3x or 3x3x... and standalone numbers:
            # - match numbers that are followed by an 'x' (dimension separator)
            (r"([0-9]+)(?=(?:x))", Number.Integer),
            # - match bare numbers (sizes)
            (r"[0-9]+", Number.Integer),
            # dynamic dimension '?'
            (r"\?", Name.Integer),
            # the 'x' dimension separator (treat as punctuation)
            (r"x", Punctuation),
            # element / builtin types inside angle brackets (no word-boundary)
            # NOTE: please sync changes to corresponding builtin type rule in "root"
            (
                r"(?:bf16|f16|f32|f64|f80|f128|index|none|(?:[us]?i[0-9]+))",
                Keyword.Type,
            ),
            # also allow nested container-like types to be recognized
            (
                r"\b(complex|memref|tensor|tuple|vector)\s*(<)",
                bygroups(Keyword.Type, Punctuation),
                "angled-type",
            ),
            # fall back to root rules for anything else
            include("root"),
        ],
    }
