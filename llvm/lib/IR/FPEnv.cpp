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

#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/FPEnv.h"

namespace llvm {

Optional<fp::RoundingMode> StrToRoundingMode(StringRef RoundingArg) {
  // For dynamic rounding mode, we use round to nearest but we will set the
  // 'exact' SDNodeFlag so that the value will not be rounded.
  return StringSwitch<Optional<fp::RoundingMode>>(RoundingArg)
      .Case("round.dynamic", fp::rmDynamic)
      .Case("round.tonearest", fp::rmToNearest)
      .Case("round.downward", fp::rmDownward)
      .Case("round.upward", fp::rmUpward)
      .Case("round.towardzero", fp::rmTowardZero)
      .Default(None);
}

Optional<StringRef> RoundingModeToStr(fp::RoundingMode UseRounding) {
  Optional<StringRef> RoundingStr = None;
  switch (UseRounding) {
  case fp::rmDynamic:
    RoundingStr = "round.dynamic";
    break;
  case fp::rmToNearest:
    RoundingStr = "round.tonearest";
    break;
  case fp::rmDownward:
    RoundingStr = "round.downward";
    break;
  case fp::rmUpward:
    RoundingStr = "round.upward";
    break;
  case fp::rmTowardZero:
    RoundingStr = "round.towardzero";
    break;
  }
  return RoundingStr;
}

Optional<fp::ExceptionBehavior> StrToExceptionBehavior(StringRef ExceptionArg) {
  return StringSwitch<Optional<fp::ExceptionBehavior>>(ExceptionArg)
      .Case("fpexcept.ignore", fp::ebIgnore)
      .Case("fpexcept.maytrap", fp::ebMayTrap)
      .Case("fpexcept.strict", fp::ebStrict)
      .Default(None);
}

Optional<StringRef> ExceptionBehaviorToStr(fp::ExceptionBehavior UseExcept) {
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

Optional<APFloatBase::roundingMode>
getAPFloatRoundingMode(fp::RoundingMode RM) {
  switch (RM) {
  case fp::rmDynamic:
    return None;
  case fp::rmToNearest:
    return APFloat::rmNearestTiesToEven;
  case fp::rmDownward:
    return APFloat::rmTowardNegative;
  case fp::rmUpward:
    return APFloat::rmTowardPositive;
  case fp::rmTowardZero:
    return APFloat::rmTowardZero;
  }
  llvm_unreachable("Unexpected rounding mode");
}
}
