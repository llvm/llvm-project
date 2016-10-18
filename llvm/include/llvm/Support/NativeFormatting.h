//===- NativeFormatting.h - Low level formatting helpers ---------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_NATIVE_FORMATTING_H
#define LLVM_SUPPORT_NATIVE_FORMATTING_H

#include "llvm/ADT/Optional.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>

namespace llvm {
enum class FloatStyle { Exponent, ExponentUpper, Fixed, Percent };
enum class IntegerStyle {
  Exponent,
  ExponentUpper,
  Integer,
  Fixed,
  Number,
  Percent,
  HexUpperPrefix,
  HexUpperNoPrefix,
  HexLowerPrefix,
  HexLowerNoPrefix
};
enum class HexPrintStyle { Upper, Lower, PrefixUpper, PrefixLower };

IntegerStyle hexStyleToIntHexStyle(HexPrintStyle S);

size_t getDefaultPrecision(FloatStyle Style);
size_t getDefaultPrecision(IntegerStyle Style);
size_t getDefaultPrecision(HexPrintStyle Style);

void write_integer(raw_ostream &S, unsigned int N, IntegerStyle Style,
                   Optional<size_t> Precision = None,
                   Optional<int> Width = None);
void write_integer(raw_ostream &S, int N, IntegerStyle Style,
                   Optional<size_t> Precision = None,
                   Optional<int> Width = None);
void write_integer(raw_ostream &S, unsigned long N, IntegerStyle Style,
                   Optional<size_t> Precision = None,
                   Optional<int> Width = None);
void write_integer(raw_ostream &S, long N, IntegerStyle Style,
                   Optional<size_t> Precision = None,
                   Optional<int> Width = None);
void write_integer(raw_ostream &S, unsigned long long N, IntegerStyle Style,
                   Optional<size_t> Precision = None,
                   Optional<int> Width = None);
void write_integer(raw_ostream &S, long long N, IntegerStyle Style,
                   Optional<size_t> Precision = None,
                   Optional<int> Width = None);

void write_hex(raw_ostream &S, uint64_t N, HexPrintStyle Style,
               Optional<size_t> Precision = None, Optional<int> Width = None);
void write_double(raw_ostream &S, double D, FloatStyle Style,
                  Optional<size_t> Precision = None,
                  Optional<int> Width = None);
}

#endif