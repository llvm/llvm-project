#!/usr/bin/env python3
# ===----------------------------------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===#
"""Generate the powers-of-five table used by APFloat.cpp.

Run from the llvm source directory:

  utils/generate-apfloat-pow5-table.py > lib/Support/APFloatPowerOfFiveTable.inc
"""

MAX_EXPONENT = 16383
WORD_BITS = 64


def words(value):
    """Return VALUE as little-endian 64-bit words."""
    result = []
    while value:
        result.append(value & ((1 << WORD_BITS) - 1))
        value >>= WORD_BITS
    return result


def main():
    table = []
    exponent = 8
    # Python integers have arbitrary precision, so this and the repeated
    # squaring below compute each power exactly.
    value = 5**exponent
    while exponent <= MAX_EXPONENT:
        value_words = words(value)
        table.append((exponent, value_words))
        exponent *= 2
        value *= value

    # This is included within an initializer.  Keep the generated layout in
    # the form produced by clang-format: after the first entry, two 64-bit
    # words fit on each indented line.
    for index, (exponent, value_words) in enumerate(table):
        indent = "" if index == 0 else "    "
        print(f"{indent}// 5^{exponent}")
        for word_index in range(0, len(value_words), 2):
            line = ", ".join(
                f"UINT64_C(0x{word:016X})"
                for word in value_words[word_index : word_index + 2]
            )
            print(f"{indent}{line},")


if __name__ == "__main__":
    main()
