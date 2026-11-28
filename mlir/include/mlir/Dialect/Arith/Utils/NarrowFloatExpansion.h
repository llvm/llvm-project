//===- NarrowFloatExpansion.h - Narrow float truncf/extf --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ARITH_UTILS_NARROWFLOATEXPANSION_H
#define MLIR_DIALECT_ARITH_UTILS_NARROWFLOATEXPANSION_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include <cstdint>

namespace llvm {
struct fltSemantics;
} // namespace llvm

namespace mlir {
namespace arith {
/// The encoding of a float format, as floatFormatSpecFromSemantics derives it.
struct FloatFormatSpec {
  unsigned sizeInBits = 0;
  unsigned mantBits = 0;
  unsigned expBits = 0;
  /// UINT_MAX for a format without a sign.
  unsigned signBitPos = 0;
  /// 1 - the minimum exponent.
  int bias = 0;
  unsigned maxStoredExp = 0;
  /// APFloat::getNaN's encoding; max-finite for a format without a NaN.
  uint64_t quietNaNBits = 0;
  uint64_t maxFiniteBitsAbs = 0;
  bool hasInf = false;
  bool isSigned = true;
  bool hasAllOnesNaN = false;
  bool isFNUZ = false;

  bool hasNaN() const { return hasInf || hasAllOnesNaN || isFNUZ; }
};

/// Returns the spec of `sem`, one of APFloat's semantics, which must have a
/// zero, subnormals, an implicit integer bit and at most 64 bits (asserted).
FloatFormatSpec floatFormatSpecFromSemantics(const llvm::fltSemantics &sem);

/// Returns floatFormatSpecFromSemantics of `floatTy`'s semantics.
FloatFormatSpec floatFormatSpecFromFloatType(FloatType floatTy);
} // namespace arith
} // namespace mlir

#endif // MLIR_DIALECT_ARITH_UTILS_NARROWFLOATEXPANSION_H
