//===- llvm/Support/FloatingPointMode.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for dealing with flags related to floating point mode controls.
//
//===----------------------------------------------------------------------===/

#ifndef LLVM_FLOATINGPOINTMODE_H
#define LLVM_FLOATINGPOINTMODE_H

#include "llvm/ADT/StringSwitch.h"

namespace llvm {

/// Represent handled modes for denormal (aka subnormal) modes in the floating
/// point environment.
enum class DenormalMode {
  Invalid = -1,

  /// IEEE-754 denormal numbers preserved.
  IEEE,

  /// The sign of a flushed-to-zero number is preserved in the sign of 0
  PreserveSign,

  /// Denormals are flushed to positive zero.
  PositiveZero
};

/// Parse the expected names from the denormal-fp-math attribute.
inline DenormalMode parseDenormalFPAttribute(StringRef Str) {
  // Assume ieee on unspecified attribute.
  return StringSwitch<DenormalMode>(Str)
    .Cases("", "ieee", DenormalMode::IEEE)
    .Case("preserve-sign", DenormalMode::PreserveSign)
    .Case("positive-zero", DenormalMode::PositiveZero)
    .Default(DenormalMode::Invalid);
}

/// Return the name used for the denormal handling mode used by the the
/// expected names from the denormal-fp-math attribute.
inline StringRef denormalModeName(DenormalMode Mode) {
  switch (Mode) {
  case DenormalMode::IEEE:
    return "ieee";
  case DenormalMode::PreserveSign:
    return "preserve-sign";
  case DenormalMode::PositiveZero:
    return "positive-zero";
  default:
    return "";
  }
}

}

#endif // LLVM_FLOATINGPOINTMODE_H
