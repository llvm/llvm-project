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
#include "llvm/CodeGen/MachineFrameInfo.h"
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

bool M88kFrameLowering::spillCalleeSavedRegisters(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
    ArrayRef<CalleeSavedInfo> CSI, const TargetRegisterInfo *TRI) const {

  // TODO Check correct handling of R1.
  // R1 is marked as used in the RET pseudo instruction.
  // R1 is also part of the callee-saved register list.
  // In case R1 needs to be saved, it becomes part of CSI,
  // and is automatically marked as live-in.

  // Mark the return address register as live in.
  MBB.addLiveIn(M88k::R1);
  return false;
}

void M88kFrameLowering::determineCalleeSaves(MachineFunction &MF,
                                             BitVector &SavedRegs,
                                             RegScavenger *RS) const {
  TargetFrameLowering::determineCalleeSaves(MF, SavedRegs, RS);

  MachineFrameInfo &MFFrame = MF.getFrameInfo();

  // If the function calls other functions, record that the return
  // address register will be clobbered.
  if (MFFrame.hasCalls())
    SavedRegs.set(M88k::R1);
}
