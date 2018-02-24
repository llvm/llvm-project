//===- MipsCallLowering.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file implements the lowering of LLVM calls to machine code calls for
/// GlobalISel.
//
//===----------------------------------------------------------------------===//

#include "MipsCallLowering.h"
#include "MipsISelLowering.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"

using namespace llvm;

MipsCallLowering::MipsCallLowering(const MipsTargetLowering &TLI)
    : CallLowering(&TLI) {}

bool MipsCallLowering::lowerReturn(MachineIRBuilder &MIRBuilder,
                                   const Value *Val, unsigned VReg) const {

  MachineInstrBuilder Ret = MIRBuilder.buildInstrNoInsert(Mips::RetRA);

  if (Val != nullptr) {
    return false;
  }
  MIRBuilder.insertInstr(Ret);
  return true;
}

bool MipsCallLowering::lowerFormalArguments(MachineIRBuilder &MIRBuilder,
                                            const Function &F,
                                            ArrayRef<unsigned> VRegs) const {

  // Quick exit if there aren't any args.
  if (F.arg_empty())
    return true;

  // Function had args, but we didn't lower them.
  return false;
}
