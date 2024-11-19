//===- NativeFormatting.h - Low level formatting helpers ---------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_NATIVEFORMATTING_H
#define LLVM_SUPPORT_NATIVEFORMATTING_H

#include "llvm/Support/Compiler.h"
#include <cstdint>
#include <optional>

namespace llvm {
class raw_ostream;
enum class FloatStyle { Exponent, ExponentUpper, Fixed, Percent };
enum class IntegerStyle {
  Integer,
  Number,
};
enum class HexPrintStyle { Upper, Lower, PrefixUpper, PrefixLower };

LLVM_ABI size_t getDefaultPrecision(FloatStyle Style);

LLVM_ABI bool isPrefixedHexStyle(HexPrintStyle S);

LLVM_ABI void write_integer(raw_ostream &S, unsigned int N, size_t MinDigits,
                            IntegerStyle Style);
LLVM_ABI void write_integer(raw_ostream &S, int N, size_t MinDigits,
                            IntegerStyle Style);
LLVM_ABI void write_integer(raw_ostream &S, unsigned long N, size_t MinDigits,
                            IntegerStyle Style);
LLVM_ABI void write_integer(raw_ostream &S, long N, size_t MinDigits,
                            IntegerStyle Style);
LLVM_ABI void write_integer(raw_ostream &S, unsigned long long N,
                            size_t MinDigits, IntegerStyle Style);
LLVM_ABI void write_integer(raw_ostream &S, long long N, size_t MinDigits,
                            IntegerStyle Style);

LLVM_ABI void write_hex(raw_ostream &S, uint64_t N, HexPrintStyle Style,
                        std::optional<size_t> Width = std::nullopt);
LLVM_ABI void write_double(raw_ostream &S, double D, FloatStyle Style,
                           std::optional<size_t> Precision = std::nullopt);
}

#endif

