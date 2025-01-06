# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pygments.lexer import RegexLexer
from pygments.token import *


class MlirLexer(RegexLexer):
    name = "MLIR"
    aliases = ["mlir"]
    filenames = ["*.mlir"]

    tokens = {
        "root": [
            (r"%[a-zA-Z0-9_]+", Name.Variable),
            (r"@[a-zA-Z_][a-zA-Z0-9_]+", Name.Function),
            (r"\^[a-zA-Z0-9_]+", Name.Label),
            (r"#[a-zA-Z0-9_]+", Name.Constant),
            (r"![a-zA-Z0-9_]+", Keyword.Type),
            (r"[a-zA-Z_][a-zA-Z0-9_]*\.", Name.Entity),
            (r"memref[^.]", Keyword.Type),
            (r"index", Keyword.Type),
            (r"i[0-9]+", Keyword.Type),
            (r"f[0-9]+", Keyword.Type),
            (r"[0-9]+", Number.Integer),
            (r"[0-9]*\.[0-9]*", Number.Float),
            (r'"[^"]*"', String.Double),
            (r"affine_map", Keyword.Reserved),
            # TODO: this should be within affine maps only
            (r"\+-\*\/", Operator),
            (r"floordiv", Operator.Word),
            (r"ceildiv", Operator.Word),
            (r"mod", Operator.Word),
            (r"()\[\]<>,{}", Punctuation),
            (r"\/\/.*\n", Comment.Single),
        ]
    }
