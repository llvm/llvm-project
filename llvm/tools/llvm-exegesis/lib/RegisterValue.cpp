//===-- RegisterValue.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterValue.h"
#include "llvm/ADT/APFloat.h"

namespace llvm {
namespace exegesis {

static llvm::APFloat getFloatValue(const llvm::fltSemantics &FltSemantics,
                                   PredefinedValues Value) {
  switch (Value) {
  case PredefinedValues::POS_ZERO:
    return llvm::APFloat::getZero(FltSemantics);
  case PredefinedValues::NEG_ZERO:
    return llvm::APFloat::getZero(FltSemantics, true);
  case PredefinedValues::ONE:
    return llvm::APFloat(FltSemantics, "1");
  case PredefinedValues::TWO:
    return llvm::APFloat(FltSemantics, "2");
  case PredefinedValues::INF:
    return llvm::APFloat::getInf(FltSemantics);
  case PredefinedValues::QNAN:
    return llvm::APFloat::getQNaN(FltSemantics);
  case PredefinedValues::SMALLEST_NORM:
    return llvm::APFloat::getSmallestNormalized(FltSemantics);
  case PredefinedValues::LARGEST:
    return llvm::APFloat::getLargest(FltSemantics);
  case PredefinedValues::ULP:
    return llvm::APFloat::getSmallest(FltSemantics);
  case PredefinedValues::ONE_PLUS_ULP:
    auto Output = getFloatValue(FltSemantics, PredefinedValues::ONE);
    Output.next(false);
    return Output;
  }
  llvm_unreachable("Unhandled exegesis::PredefinedValues");
}

llvm::APInt bitcastFloatValue(const llvm::fltSemantics &FltSemantics,
                              PredefinedValues Value) {
  return getFloatValue(FltSemantics, Value).bitcastToAPInt();
}

} // namespace exegesis
} // namespace llvm
