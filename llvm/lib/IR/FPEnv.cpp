//===-- FPEnv.cpp ---- FP Environment -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// @file
/// This file contains the implementations of entities that describe floating
/// point environment.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/FPEnv.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include <optional>

namespace llvm {

std::optional<RoundingMode> convertStrToRoundingMode(StringRef RoundingArg,
                                                     bool InBundle) {
  if (InBundle)
    return StringSwitch<std::optional<RoundingMode>>(RoundingArg)
        .Case("dyn", RoundingMode::Dynamic)
        .Case("rte", RoundingMode::NearestTiesToEven)
        .Case("rmm", RoundingMode::NearestTiesToAway)
        .Case("rtn", RoundingMode::TowardNegative)
        .Case("rtp", RoundingMode::TowardPositive)
        .Case("rtz", RoundingMode::TowardZero)
        .Default(std::nullopt);

  // For dynamic rounding mode, we use round to nearest but we will set the
  // 'exact' SDNodeFlag so that the value will not be rounded.
  return StringSwitch<std::optional<RoundingMode>>(RoundingArg)
      .Case("round.dynamic", RoundingMode::Dynamic)
      .Case("round.tonearest", RoundingMode::NearestTiesToEven)
      .Case("round.tonearestaway", RoundingMode::NearestTiesToAway)
      .Case("round.downward", RoundingMode::TowardNegative)
      .Case("round.upward", RoundingMode::TowardPositive)
      .Case("round.towardzero", RoundingMode::TowardZero)
      .Default(std::nullopt);
}

std::optional<StringRef> convertRoundingModeToStr(RoundingMode UseRounding,
                                                  bool InBundle) {
  std::optional<StringRef> RoundingStr;
  switch (UseRounding) {
  case RoundingMode::Dynamic:
    RoundingStr = InBundle ? "dyn" : "round.dynamic";
    break;
  case RoundingMode::NearestTiesToEven:
    RoundingStr = InBundle ? "rte" : "round.tonearest";
    break;
  case RoundingMode::NearestTiesToAway:
    RoundingStr = InBundle ? "rmm" : "round.tonearestaway";
    break;
  case RoundingMode::TowardNegative:
    RoundingStr = InBundle ? "rtn" : "round.downward";
    break;
  case RoundingMode::TowardPositive:
    RoundingStr = InBundle ? "rtp" : "round.upward";
    break;
  case RoundingMode::TowardZero:
    RoundingStr = InBundle ? "rtz" : "round.towardzero";
    break;
  default:
    break;
  }
  return RoundingStr;
}

std::optional<fp::ExceptionBehavior>
convertStrToExceptionBehavior(StringRef ExceptionArg, bool InBundle) {
  if (InBundle)
    return StringSwitch<std::optional<fp::ExceptionBehavior>>(ExceptionArg)
        .Case("ignore", fp::ebIgnore)
        .Case("maytrap", fp::ebMayTrap)
        .Case("strict", fp::ebStrict)
        .Default(std::nullopt);

  return StringSwitch<std::optional<fp::ExceptionBehavior>>(ExceptionArg)
      .Case("fpexcept.ignore", fp::ebIgnore)
      .Case("fpexcept.maytrap", fp::ebMayTrap)
      .Case("fpexcept.strict", fp::ebStrict)
      .Default(std::nullopt);
}

std::optional<StringRef>
convertExceptionBehaviorToStr(fp::ExceptionBehavior UseExcept, bool InBundle) {
  std::optional<StringRef> ExceptStr;
  switch (UseExcept) {
  case fp::ebStrict:
    ExceptStr = InBundle ? "strict" : "fpexcept.strict";
    break;
  case fp::ebIgnore:
    ExceptStr = InBundle ? "ignore" : "fpexcept.ignore";
    break;
  case fp::ebMayTrap:
    ExceptStr = InBundle ? "maytrap" : "fpexcept.maytrap";
    break;
  }
  return ExceptStr;
}

Intrinsic::ID getConstrainedIntrinsicID(const Instruction &Instr) {
  Intrinsic::ID IID = Intrinsic::not_intrinsic;
  switch (Instr.getOpcode()) {
  case Instruction::FCmp:
    // Unlike other instructions FCmp can be mapped to one of two intrinsic
    // functions. We choose the non-signaling variant.
    IID = Intrinsic::experimental_constrained_fcmp;
    break;

    // Instructions
#define INSTRUCTION(NAME, NARG, ROUND_MODE, INTRINSIC)                         \
  case Instruction::NAME:                                                      \
    IID = Intrinsic::INTRINSIC;                                                \
    break;
#define FUNCTION(NAME, NARG, ROUND_MODE, INTRINSIC)
#define CMP_INSTRUCTION(NAME, NARG, ROUND_MODE, INTRINSIC, DAGN)
#include "llvm/IR/ConstrainedOps.def"

  // Intrinsic calls.
  case Instruction::Call:
    if (auto *IntrinCall = dyn_cast<IntrinsicInst>(&Instr)) {
      switch (IntrinCall->getIntrinsicID()) {
#define FUNCTION(NAME, NARG, ROUND_MODE, INTRINSIC)                            \
  case Intrinsic::NAME:                                                        \
    IID = Intrinsic::INTRINSIC;                                                \
    break;
#define INSTRUCTION(NAME, NARG, ROUND_MODE, INTRINSIC)
#define CMP_INSTRUCTION(NAME, NARG, ROUND_MODE, INTRINSIC, DAGN)
#include "llvm/IR/ConstrainedOps.def"
      default:
        break;
      }
    }
    break;
  default:
    break;
  }

  return IID;
}

} // namespace llvm
