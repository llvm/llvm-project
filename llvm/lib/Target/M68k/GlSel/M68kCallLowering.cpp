//===-- M68kCallLowering.cpp - Call lowering -------------------*- C++ -*-===//
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

#include "M68kCallLowering.h"
#include "M68kISelLowering.h"
#include "M68kInstrInfo.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
using namespace llvm;

M68kCallLowering::M68kCallLowering(const M68kTargetLowering &TLI)
    : CallLowering(&TLI) {}

bool M68kCallLowering::lowerReturn(MachineIRBuilder &MIRBuilder,
                                   const Value *Val, ArrayRef<Register> VRegs,
                                   FunctionLoweringInfo &FLI,
                                   Register SwiftErrorVReg) const {

  if (Val)
    return false;
  MIRBuilder.buildInstr(M68k::RTS);
  return true;
}

bool M68kCallLowering::lowerFormalArguments(MachineIRBuilder &MIRBuilder,
                                            const Function &F,
                                            ArrayRef<ArrayRef<Register>> VRegs,
                                            FunctionLoweringInfo &FLI) const {

  if (F.arg_empty())
    return true;

  return false;
}

bool M68kCallLowering::lowerCall(MachineIRBuilder &MIRBuilder,
                                 CallLoweringInfo &Info) const {
  return false;
}

bool M68kCallLowering::enableBigEndian() const { return true; }
