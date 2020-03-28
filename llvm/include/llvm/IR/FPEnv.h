//===- FPEnv.h ---- FP Environment ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// @file
/// This file contains the declarations of entities that describe floating
/// point environment and related functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_FLOATINGPOINT_H
#define LLVM_IR_FLOATINGPOINT_H

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include <stdint.h>

namespace llvm {

namespace fp {

/// Rounding mode used for floating point operations.
///
/// Each of these values correspond to some metadata argument value of a
/// constrained floating point intrinsic. See the LLVM Language Reference Manual
/// for details.
enum RoundingMode : uint8_t {
  rmDynamic,   ///< This corresponds to "fpround.dynamic".
  rmToNearest, ///< This corresponds to "fpround.tonearest".
  rmDownward,  ///< This corresponds to "fpround.downward".
  rmUpward,    ///< This corresponds to "fpround.upward".
  rmTowardZero ///< This corresponds to "fpround.tozero".
};

/// Exception behavior used for floating point operations.
///
/// Each of these values correspond to some metadata argument value of a
/// constrained floating point intrinsic. See the LLVM Language Reference Manual
/// for details.
enum ExceptionBehavior : uint8_t {
  ebIgnore,  ///< This corresponds to "fpexcept.ignore".
  ebMayTrap, ///< This corresponds to "fpexcept.maytrap".
  ebStrict   ///< This corresponds to "fpexcept.strict".
};

}

/// Returns a valid RoundingMode enumerator when given a string
/// that is valid as input in constrained intrinsic rounding mode
/// metadata.
Optional<fp::RoundingMode> StrToRoundingMode(StringRef);

/// For any RoundingMode enumerator, returns a string valid as input in
/// constrained intrinsic rounding mode metadata.
Optional<StringRef> RoundingModeToStr(fp::RoundingMode);

/// Returns a valid ExceptionBehavior enumerator when given a string
/// valid as input in constrained intrinsic exception behavior metadata.
Optional<fp::ExceptionBehavior> StrToExceptionBehavior(StringRef);

/// For any ExceptionBehavior enumerator, returns a string valid as
/// input in constrained intrinsic exception behavior metadata.
Optional<StringRef> ExceptionBehaviorToStr(fp::ExceptionBehavior);

/// Converts rounding mode represented by fp::RoundingMode to the rounding mode
/// index used by APFloat. For fp::rmDynamic it returns None.
Optional<APFloatBase::roundingMode> getAPFloatRoundingMode(fp::RoundingMode);
}
#endif
