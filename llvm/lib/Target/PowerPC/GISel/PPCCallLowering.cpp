//===-- PPCCallLowering.h - Call lowering for GlobalISel -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the lowering of LLVM calls to machine code calls for
/// GlobalISel.
///
//===----------------------------------------------------------------------===//

#include "PPCCallLowering.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ppc-call-lowering"

using namespace llvm;

PPCCallLowering::PPCCallLowering(const PPCTargetLowering &TLI)
    : CallLowering(&TLI) {}

bool PPCCallLowering::lowerReturn(MachineIRBuilder &MIRBuilder,
                                  const Value *Val, ArrayRef<Register> VRegs,
                                  Register SwiftErrorVReg) const {
  assert(((Val && !VRegs.empty()) || (!Val && VRegs.empty())) &&
         "Return value without a vreg");
  if (VRegs.size() > 0)
    return false;

  MIRBuilder.buildInstr(PPC::BLR8);
  return true;
}

bool PPCCallLowering::lowerFormalArguments(
    MachineIRBuilder &MIRBuilder, const Function &F,
    ArrayRef<ArrayRef<Register>> VRegs) const {

  // If VRegs is empty, then there are no formal arguments to lower and thus can
  // always return true. If there are formal arguments, we currently do not
  // handle them and thus return false.
  return VRegs.empty();
}

bool PPCCallLowering::lowerCall(MachineIRBuilder &MIRBuilder,
                                CallLoweringInfo &Info) const {
  return false;
}
