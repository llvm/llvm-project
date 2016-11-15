//===-- llvm/lib/Target/X86/X86CallLowering.cpp - Call lowering -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the lowering of LLVM calls to machine code calls for
/// GlobalISel.
///
//===----------------------------------------------------------------------===//

#include "X86CallLowering.h"
#include "X86ISelLowering.h"
#include "X86InstrInfo.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"

using namespace llvm;

#ifndef LLVM_BUILD_GLOBAL_ISEL
#error "This shouldn't be built without GISel"
#endif

X86CallLowering::X86CallLowering(const X86TargetLowering &TLI)
    : CallLowering(&TLI) {}

bool X86CallLowering::lowerReturn(MachineIRBuilder &MIRBuilder,
                                  const Value *Val, unsigned VReg) const {
  // TODO: handle functions returning non-void values.
  if (Val)
    return false;

  MIRBuilder.buildInstr(X86::RET).addImm(0);

  return true;
}

bool X86CallLowering::lowerFormalArguments(MachineIRBuilder &MIRBuilder,
                                           const Function &F,
                                           ArrayRef<unsigned> VRegs) const {
  // TODO: handle functions with one or more arguments.
  return F.arg_empty();
}
