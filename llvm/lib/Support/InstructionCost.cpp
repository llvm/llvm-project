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
  if (isValid()) {
    CostType WholeNumber = Value / CostGranularity;
    CostType Remainder = Value % CostGranularity;
    OS << WholeNumber;
    assert(CostGranularity == 4 && "Hardcoded for CostGranularity=4");
    switch (Remainder) {
    case 1:
      OS << ".25";
      break;
    case 2:
      OS << ".5";
      break;
    case 3:
      OS << ".75";
      break;
    }
  } else {
    OS << "Invalid";
  }
}
