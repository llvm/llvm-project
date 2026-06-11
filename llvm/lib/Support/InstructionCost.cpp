//===- InstructionCost.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file includes the function definitions for the InstructionCost class
/// that is used when calculating the cost of an instruction, or a group of
/// instructions.
//===----------------------------------------------------------------------===//

#include "llvm/Support/InstructionCost.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

void InstructionCost::print(raw_ostream &OS) const {
  using UnsignedCostType = std::make_unsigned_t<CostType>;
  if (isValid()) {
    UnsignedCostType AbsValue =
        (Value < 0) ? -((UnsignedCostType)Value) : ((UnsignedCostType)Value);
    UnsignedCostType WholeNumber = AbsValue / CostGranularity;
    UnsignedCostType Remainder = AbsValue % CostGranularity;
    if (Value < 0)
      OS << "-";
    UnsignedCostType RemainderHundreds = (Remainder * 100) / CostGranularity;
    while (RemainderHundreds % 10 == 0 && RemainderHundreds)
      RemainderHundreds /= 10;
    OS << WholeNumber;
    if (RemainderHundreds)
      OS << "." << RemainderHundreds;
  } else {
    OS << "Invalid";
  }
}
