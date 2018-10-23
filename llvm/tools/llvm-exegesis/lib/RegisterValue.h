//===-- RegisterValue.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
///
/// Defines a Target independent value for a Register. This is useful to explore
/// the influence of the instruction input values on its execution time.
///
//===----------------------------------------------------------------------===//

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>

namespace llvm {
namespace exegesis {

// A simple object storing the value for a particular register.
struct RegisterValue {
  unsigned Register;
  llvm::APInt Value;
};

enum class PredefinedValues {
  POS_ZERO,       // Positive zero
  NEG_ZERO,       // Negative zero
  ONE,            // 1.0
  TWO,            // 2.0
  INF,            // Infinity
  QNAN,           // Quiet NaN
  ULP,            // One Unit in the last place
  SMALLEST = ULP, // The minimum subnormal number
  SMALLEST_NORM,  // The minimum normal number
  LARGEST,        // The maximum normal number
  ONE_PLUS_ULP,   // The value just after 1.0
};

llvm::APInt bitcastFloatValue(const llvm::fltSemantics &FltSemantics,
                              PredefinedValues Value);

} // namespace exegesis
} // namespace llvm
