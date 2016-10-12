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

#include "llvm/Support/raw_ostream.h"

#include <cstdint>

namespace llvm {
enum class FloatStyle { Exponent, Decimal };

void write_ulong(raw_ostream &S, unsigned long N, std::size_t MinWidth);
void write_long(raw_ostream &S, long N, std::size_t MinWidth);
void write_ulonglong(raw_ostream &S, unsigned long long N,
                     std::size_t MinWidth);
void write_longlong(raw_ostream &S, long long N, std::size_t MinWidth);
void write_hex(raw_ostream &S, unsigned long long N, std::size_t MinWidth,
               bool Upper, bool Prefix);
void write_double(raw_ostream &S, double D, std::size_t MinWidth,
                  std::size_t MinDecimals, FloatStyle Style);
}

#endif