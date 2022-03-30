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

namespace llvm {

Optional<RoundingMode> convertStrToRoundingMode(StringRef RoundingArg) {
  // For dynamic rounding mode, we use round to nearest but we will set the
  // 'exact' SDNodeFlag so that the value will not be rounded.
  return StringSwitch<Optional<RoundingMode>>(RoundingArg)
      .Case("round.dynamic", RoundingMode::Dynamic)
      .Case("round.tonearest", RoundingMode::NearestTiesToEven)
      .Case("round.tonearestaway", RoundingMode::NearestTiesToAway)
      .Case("round.downward", RoundingMode::TowardNegative)
      .Case("round.upward", RoundingMode::TowardPositive)
      .Case("round.towardzero", RoundingMode::TowardZero)
      .Default(None);
}

Optional<StringRef> convertRoundingModeToStr(RoundingMode UseRounding) {
  Optional<StringRef> RoundingStr = None;
  switch (UseRounding) {
  case RoundingMode::Dynamic:
    RoundingStr = "round.dynamic";
    break;
  case RoundingMode::NearestTiesToEven:
    RoundingStr = "round.tonearest";
    break;
  case RoundingMode::NearestTiesToAway:
    RoundingStr = "round.tonearestaway";
    break;
  case RoundingMode::TowardNegative:
    RoundingStr = "round.downward";
    break;
  case RoundingMode::TowardPositive:
    RoundingStr = "round.upward";
    break;
  case RoundingMode::TowardZero:
    RoundingStr = "round.towardzero";
    break;
  default:
    break;
  }
  return RoundingStr;
}

Optional<fp::ExceptionBehavior>
convertStrToExceptionBehavior(StringRef ExceptionArg) {
  return StringSwitch<Optional<fp::ExceptionBehavior>>(ExceptionArg)
      .Case("fpexcept.ignore", fp::ebIgnore)
      .Case("fpexcept.maytrap", fp::ebMayTrap)
      .Case("fpexcept.strict", fp::ebStrict)
      .Default(None);
}

Optional<StringRef>
convertExceptionBehaviorToStr(fp::ExceptionBehavior UseExcept) {
  Optional<StringRef> ExceptStr = None;
  switch (UseExcept) {
  case fp::ebStrict:
    ExceptStr = "fpexcept.strict";
    break;
  case fp::ebIgnore:
    ExceptStr = "fpexcept.ignore";
    break;
  case fp::ebMayTrap:
    ExceptStr = "fpexcept.maytrap";
    break;
  }
  return ExceptStr;
}
} // namespace llvm
