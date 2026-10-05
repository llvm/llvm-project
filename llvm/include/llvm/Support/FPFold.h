//===- FPFold.h - Floating-point constant-folding helpers -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file provides fallible floating-point folding operations with explicit
/// rounding and denormal modes.
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_FPFOLD_H
#define LLVM_SUPPORT_FPFOLD_H

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FloatingPointMode.h"
#include <optional>

namespace llvm {

/// The floating-point operations supported by the folding helpers.
enum class FPOp {
  Add,
  Sub,
  Mul,
  Div,
  FRem,
  MinNum,
  MaxNum,
  Minimum,
  Maximum,
  MinimumNum,
  MaximumNum,
  FMA,
  Ceil,
  Floor,
  Trunc,
  Round,
  RoundEven
};

/// A unique floating-point result and the IEEE exceptions raised computing it.
struct FPFoldResult {
  APFloat Value;
  APFloat::opStatus Status;
};

/// Return the value of a successful fold, or std::nullopt.
LLVM_ABI std::optional<APFloat> getFPValue(std::optional<FPFoldResult> Res);

/// Return whether folding to NaN should be avoided by default.
LLVM_ABI bool shouldAvoidFoldingToNaN();

/// Try to evaluate a floating-point operation with the specified inputs.
/// All inputs must have the same floating-point semantics. Unary rounding
/// operations take one input, FMA takes three, and other operations take two.
/// Ceil, Floor, Trunc, Round, and RoundEven select their own rounding mode;
/// Other rounding-dependent operations use RN.
/// FRem and min/max do not depend on the rounding mode.
///
/// Returns std::nullopt when the modes do not specify one result. In
/// particular, this occurs when a dynamic denormal mode could change an input
/// or output value.
LLVM_ABI std::optional<FPFoldResult>
tryFoldFP(FPOp Opcode, ArrayRef<APFloat> Args, DenormalMode Denorms,
          bool AvoidFoldingToNaN = shouldAvoidFoldingToNaN());

/// Try to evaluate a floating-point operation with an explicit rounding mode.
/// Invalid and Dynamic modes are evaluated using RN, but folding is declined
/// if evaluation is inexact. Fixed-rounding operations still select their own
/// rounding mode.
LLVM_ABI std::optional<FPFoldResult>
tryFoldFPWithRM(FPOp Opcode, ArrayRef<APFloat> Args, RoundingMode RM,
                DenormalMode Denorms,
                bool AvoidFoldingToNaN = shouldAvoidFoldingToNaN());

/// Try to compare two floating-point values.
///
/// A comparison has no rounding or output-denormal mode, but is not foldable
/// when dynamic input-denormal handling could alter an operand.
LLVM_ABI std::optional<APFloat::cmpResult>
tryFoldFCmp(const APFloat &LHS, const APFloat &RHS, DenormalMode Denorms);

} // namespace llvm

#endif // LLVM_SUPPORT_FPFOLD_H
