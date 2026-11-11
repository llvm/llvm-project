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
#include "llvm/Support/Compiler.h"
#include <cstddef>
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
  RoundEven,
  FCmp // Used by tryFoldFCmp when querying target preferences.
};

/// A non-owning adapter for target-specific floating-point folding preferences.
/// Construct from a TTI or TargetLowering pointer, which must outlive the
/// adapter. A missing target conservatively uses dynamic denormal handling.
class TargetFPInfo {
  using GetTargetConstFoldDenormModeFn = DenormalMode (*)(const void *, FPOp,
                                                          ArrayRef<APFloat>,
                                                          DenormalMode);

  const void *Target = nullptr;
  GetTargetConstFoldDenormModeFn GetTargetConstFoldDenormMode = nullptr;

public:
  TargetFPInfo() = default;
  TargetFPInfo(std::nullptr_t) {}

  template <typename T> TargetFPInfo(const T *Target) : Target(Target) {
    if (!Target)
      return;
    GetTargetConstFoldDenormMode = [](const void *Target, FPOp Opcode,
                                      ArrayRef<APFloat> Args, DenormalMode DM) {
      return static_cast<const T *>(Target)->getTargetConstFoldDenormMode(
          Opcode, Args, DM);
    };
  }

  DenormalMode getTargetConstFoldDenormMode(FPOp Opcode, ArrayRef<APFloat> Args,
                                            DenormalMode DM) const {
    return Target ? GetTargetConstFoldDenormMode(Target, Opcode, Args, DM)
                  : DenormalMode::getDynamic();
  }
};

/// A unique floating-point result and the IEEE exceptions raised computing it.
struct FPFoldResult {
  APFloat Value;
  APFloat::opStatus Status;
};

/// Return the value of a successful fold, or std::nullopt.
LLVM_ABI std::optional<APFloat> getFPValue(std::optional<FPFoldResult> Res);

/// Try to evaluate a floating-point operation with the specified inputs.
/// Ceil, Floor, Trunc, Round, and RoundEven select their own rounding mode;
/// Other rounding-dependent operations use RN.
/// Min/max and FRem do not depend on the rounding mode.
///
/// Returns std::nullopt when the modes do not specify one result. In
/// particular, this occurs when a dynamic denormal mode could change an input
/// or output value.
/// In target-default mode, TFPI selects the folding environment. If no target
/// is available, use dynamic denormal handling.
/// OverrideDefaultFlushingMode bypasses -const-folding-denormal-mode and uses
/// Denorms unchanged, for operations with explicit denormal semantics.
LLVM_ABI std::optional<FPFoldResult>
tryFoldFP(FPOp Opcode, ArrayRef<APFloat> Args, DenormalMode Denorms,
          TargetFPInfo TFPI = {}, bool OverrideDefaultFlushingMode = false);

/// Try to evaluate a floating-point operation with an explicit rounding mode.
/// Dynamic is evaluated using RN, but folding is declined if evaluation is
/// inexact. Fixed-rounding operations still select their own rounding mode.
/// OverrideDefaultFlushingMode has the same meaning as in tryFoldFP.
LLVM_ABI std::optional<FPFoldResult>
tryFoldFPWithRM(FPOp Opcode, ArrayRef<APFloat> Args, RoundingMode RM,
                DenormalMode Denorms, TargetFPInfo TFPI = {},
                bool OverrideDefaultFlushingMode = false);

/// Try to compare two floating-point values.
///
/// A comparison has no rounding or output-denormal mode, but is not foldable
/// when dynamic input-denormal handling could alter an operand.
/// OverrideDefaultFlushingMode has the same meaning as in tryFoldFP.
LLVM_ABI std::optional<APFloat::cmpResult>
tryFoldFCmp(const APFloat &LHS, const APFloat &RHS, DenormalMode Denorms,
            TargetFPInfo TFPI = {}, bool OverrideDefaultFlushingMode = false);

} // namespace llvm

#endif // LLVM_SUPPORT_FPFOLD_H
