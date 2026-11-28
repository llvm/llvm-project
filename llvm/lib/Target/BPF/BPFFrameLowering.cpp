//===-- BPFFrameLowering.cpp - BPF Frame Information ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the BPF implementation of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#include "BPFFrameLowering.h"
#include "BPFSubtarget.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/IR/DiagnosticInfo.h"

using namespace llvm;

BPFFrameLowering::BPFFrameLowering(const BPFSubtarget &STI)
    : TargetFrameLowering(STI.useStackSpilling() ? StackGrowsUp
                                                 : StackGrowsDown,
                          STI.useStackSpilling() ? Align(64) : Align(8), 0,
                          STI.useStackSpilling() ? Align(64) : Align(1)) {}

bool BPFFrameLowering::hasFPImpl(const MachineFunction &MF) const {
  return true;
}

void BPFFrameLowering::emitPrologue(MachineFunction &MF,
                                    MachineBasicBlock &MBB) const {
  const BPFSubtarget &STI = MF.getSubtarget<BPFSubtarget>();
  if (STI.useStackSpilling() &&
      MF.getFrameInfo().getStackSize() >
          static_cast<uint64_t>(STI.getStackSizeLimit()))
    MF.getFunction().getContext().diagnose(DiagnosticInfoUnsupported(
        MF.getFunction(), "BPF stack frame exceeds the stack size limit"));
}

void BPFFrameLowering::emitEpilogue(MachineFunction &MF,
                                    MachineBasicBlock &MBB) const {}

void BPFFrameLowering::determineCalleeSaves(MachineFunction &MF,
                                            BitVector &SavedRegs,
                                            RegScavenger *RS) const {
  TargetFrameLowering::determineCalleeSaves(MF, SavedRegs, RS);
  SavedRegs.reset(BPF::R6);
  SavedRegs.reset(BPF::R7);
  SavedRegs.reset(BPF::R8);
  SavedRegs.reset(BPF::R9);
}

StackOffset BPFFrameLowering::getFrameIndexReference(const MachineFunction &MF,
                                                     int FI,
                                                     Register &FrameReg) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const TargetRegisterInfo *RI = MF.getSubtarget().getRegisterInfo();

  FrameReg = RI->getFrameRegister(MF);

  int64_t Offset = MFI.getObjectOffset(FI);
  const BPFSubtarget &STI = MF.getSubtarget<BPFSubtarget>();
  if (STI.useStackSpilling()) {
    if (MFI.isFixedObjectIndex(FI) && Offset < 0) {
      // Outgoing arguments belong to the next frame. Their negative fixed
      // offsets keep them out of this function's upward-growing frame.
      Offset = -Offset - MFI.getObjectSize(FI);
    } else {
      // Incoming arguments, locals and register spills grow upwards from
      // the bottom of the fixed-size frame supplied by the runtime.
      Offset -= STI.getStackSizeLimit();
    }
  }
  return StackOffset::getFixed(Offset);
}
