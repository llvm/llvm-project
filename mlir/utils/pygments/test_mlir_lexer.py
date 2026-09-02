# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for mlir_lexer.py. Run with `python -m unittest` from this directory.

Requires Pygments (`pip install Pygments`), as the lexer itself does.
"""

import unittest

from mlir_lexer import MlirLexer
from pygments.token import Error, Keyword, Name

# Every builtin float type keyword, per mlir/include/mlir/IR/BuiltinTypes.td.
FLOAT_TYPES = "bf16 f16 f32 f64 f80 f128 tf32 f8E3M4 f8E4M3 f8E4M3B11FNUZ \
f8E4M3FN f8E4M3FNUZ f8E5M2 f8E5M2FNUZ f8E8M0FNU f6E2M3FN f6E3M2FN f4E2M1FN".split()


def lex(source):
    """Lex source, dropping whitespace-only tokens."""
    return [(t, v) for t, v in MlirLexer().get_tokens(source) if v.strip()]


class TestMlirLexer(unittest.TestCase):
    def test_indented_operation_names(self):
        """Whitespace rules must not consume the indent the op-name rules anchor on."""
        got = lex("func.func @f() {\n  %0 = arith.constant 1 : i32\n  return\n}\n")
        self.assertEqual(
            [v for t, v in got if t is Name.Builtin],
            ["func.func", "arith.constant", "return"],
        )

    def test_builtin_float_types(self):
        """Each float type is one token, bare and as a tensor element type."""
        for name in FLOAT_TYPES:
            for source in ("%%0 = a.b %%c : %s\n", "%%0 = a.b %%c : tensor<4x8x%s>\n"):
                with self.subTest(type=name, source=source):
                    got = lex(source % name)
                    self.assertIn((Keyword.Type, name), got)
                    self.assertEqual([v for t, v in got if t is Error], [])


if __name__ == "__main__":
    unittest.main()
