//===-- LoongArchCallLowering.cpp - Call lowering ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file implements the lowering of LLVM calls to machine code calls for
/// GlobalISel.
//
//===----------------------------------------------------------------------===//

#include "LoongArchCallLowering.h"
#include "LoongArchISelLowering.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"

using namespace llvm;

LoongArchCallLowering::LoongArchCallLowering(const LoongArchTargetLowering &TLI)
    : CallLowering(&TLI) {}

bool LoongArchCallLowering::lowerReturn(MachineIRBuilder &MIRBuilder,
                                        const Value *Val,
                                        ArrayRef<Register> VRegs,
                                        FunctionLoweringInfo &FLI) const {
  if (Val != nullptr)
    return false;

  MIRBuilder.buildInstr(LoongArch::PseudoRET);
  return true;
}

bool LoongArchCallLowering::lowerFormalArguments(
    MachineIRBuilder &MIRBuilder, const Function &F,
    ArrayRef<ArrayRef<Register>> VRegs, FunctionLoweringInfo &FLI) const {
  return F.arg_empty();
}

bool LoongArchCallLowering::lowerCall(MachineIRBuilder &MIRBuilder,
                                      CallLoweringInfo &Info) const {
  return false;
}
