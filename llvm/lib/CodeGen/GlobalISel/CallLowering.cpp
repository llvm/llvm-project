//===-- lib/CodeGen/GlobalISel/CallLowering.cpp - Call lowering -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements some simple delegations needed for call lowering.
///
//===----------------------------------------------------------------------===//


#include "llvm/CodeGen/GlobalISel/CallLowering.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

bool CallLowering::lowerCall(
    MachineIRBuilder &MIRBuilder, const CallInst &CI, unsigned ResReg,
    ArrayRef<unsigned> ArgRegs, std::function<unsigned()> GetCalleeReg) const {
  // First step is to marshall all the function's parameters into the correct
  // physregs and memory locations. Gather the sequence of argument types that
  // we'll pass to the assigner function.
  SmallVector<MVT, 8> ArgTys;
  for (auto &Arg : CI.arg_operands())
    ArgTys.push_back(MVT::getVT(Arg->getType()));

  MachineOperand Callee = MachineOperand::CreateImm(0);
  if (Function *F = CI.getCalledFunction())
    Callee = MachineOperand::CreateGA(F, 0);
  else
    Callee = MachineOperand::CreateReg(GetCalleeReg(), false);

  return lowerCall(MIRBuilder, Callee, MVT::getVT(CI.getType()),
                   ResReg ? ResReg : ArrayRef<unsigned>(), ArgTys, ArgRegs);
}
