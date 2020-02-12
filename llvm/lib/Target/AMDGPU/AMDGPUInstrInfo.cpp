//===-- AMDGPUInstrInfo.cpp - Base class for AMD GPU InstrInfo ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Implementation of the TargetInstrInfo class that is common to all
/// AMD GPUs.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUInstrInfo.h"
#include "AMDGPUTargetMachine.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

// Pin the vtable to this file.
//void AMDGPUInstrInfo::anchor() {}

AMDGPUInstrInfo::AMDGPUInstrInfo(const GCNSubtarget &ST) { }


// TODO: Should largely merge with AMDGPUTTIImpl::isSourceOfDivergence.
bool AMDGPUInstrInfo::isUniformMMO(const MachineMemOperand *MMO) {
  const Value *Ptr = MMO->getValue();
  // UndefValue means this is a load of a kernel input.  These are uniform.
  // Sometimes LDS instructions have constant pointers.
  // If Ptr is null, then that means this mem operand contains a
  // PseudoSourceValue like GOT.
  if (!Ptr || isa<UndefValue>(Ptr) ||
      isa<Constant>(Ptr) || isa<GlobalValue>(Ptr))
    return true;

  if (MMO->getAddrSpace() == AMDGPUAS::CONSTANT_ADDRESS_32BIT)
    return true;

  if (const Argument *Arg = dyn_cast<Argument>(Ptr))
    return AMDGPU::isArgPassedInSGPR(Arg);

  const Instruction *I = dyn_cast<Instruction>(Ptr);
  return I && I->getMetadata("amdgpu.uniform");
}
