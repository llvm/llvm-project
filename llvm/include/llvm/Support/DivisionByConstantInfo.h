//===- llvm/Support/DivisionByConstantInfo.h ---------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file implements support for optimizing divisions by a constant
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_DIVISIONBYCONSTANTINFO_H
#define LLVM_SUPPORT_DIVISIONBYCONSTANTINFO_H

#include "llvm/ADT/APInt.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

/// Standard integer bitwidths that division strength-reduction may widen to.
/// The numeric value is the actual bit count, so arithmetic on it is valid.
enum class IntegerBitWidth : unsigned {
  None = 0,
  I8 = 8,
  I16 = 16,
  I32 = 32,
  I64 = 64,
  I128 = 128,
};

/// Widening strategies for unsigned division by a constant.
enum class UnsignedDivisionByConstantWidening {
  None,
  /// Use a widened high-half multiply and truncate the result.
  MulHigh,
  /// Use a widened full multiply followed by an explicit right shift.
  FullMultiply,
};

/// Magic data for optimising signed division by a constant.
struct SignedDivisionByConstantInfo {
  LLVM_ABI static SignedDivisionByConstantInfo get(const APInt &D);
  APInt Magic;          ///< magic number
  unsigned ShiftAmount; ///< shift amount
};

/// Magic data for optimising unsigned division by a constant.
struct UnsignedDivisionByConstantInfo {
  LLVM_ABI static UnsignedDivisionByConstantInfo
  get(const APInt &D, unsigned LeadingZeros = 0,
      bool AllowEvenDivisorOptimization = true,
      IntegerBitWidth MaxBitWidth = IntegerBitWidth::None);
  APInt Magic;          ///< magic number
  bool IsAdd;           ///< add indicator
  unsigned PostShift;   ///< post-shift amount
  unsigned PreShift;    ///< pre-shift amount
  UnsignedDivisionByConstantWidening Widening;
};

} // namespace llvm

#endif
