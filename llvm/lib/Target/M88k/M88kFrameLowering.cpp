//===-- M88kFrameLowering.cpp - Frame lowering for M88k -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "M88kFrameLowering.h"
//#include "M88kCallingConv.h"
//#include "M88kInstrBuilder.h"
//#include "M88kInstrInfo.h"
//#include "M88kMachineFunctionInfo.h"
#include "M88kRegisterInfo.h"
#include "M88kSubtarget.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/IR/Function.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

M88kFrameLowering::M88kFrameLowering()
    : TargetFrameLowering(TargetFrameLowering::StackGrowsDown, Align(8), 0,
                          Align(8), false /* StackRealignable */),
      RegSpillOffsets(0) {}

void M88kFrameLowering::emitPrologue(MachineFunction &MF,
                                     MachineBasicBlock &MBB) const {}

void M88kFrameLowering::emitEpilogue(MachineFunction &MF,
                                     MachineBasicBlock &MBB) const {}

bool M88kFrameLowering::hasFP(const MachineFunction &MF) const { return false; }
