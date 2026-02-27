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

using namespace llvm;

#define DEBUG_TYPE "bpf-frame-lowering"

bool BPFFrameLowering::hasFPImpl(const MachineFunction &MF) const {
  return true;
}

void BPFFrameLowering::emitPrologue(MachineFunction &MF,
                                    MachineBasicBlock &MBB) const {}

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

void
BPFFrameLowering::processFunctionBeforeFrameFinalized(MachineFunction &MF,
                                                      RegScavenger *RS) const {
  MachineFrameInfo &MFI = MF.getFrameInfo();
  int Offset = 0;
  for (unsigned FI = 0, E = MFI.getObjectIndexEnd(); FI != E; ++FI) {
    // Objects with TargetStackID::Default are handled by
    // PEIImpl::calculateFrameObjectOffsets().
    if (MFI.getStackID(FI) == TargetStackID::Default ||
        MFI.isDeadObjectIndex(FI))
      continue;
    Offset += MFI.getObjectSize(FI);
    Align Alignment = MFI.getObjectAlign(FI);
    Offset = alignTo(Offset, Alignment);
    LLVM_DEBUG(dbgs() << "alloc FI(" << FI << ") at SP[" << -Offset << "]\n");
    MFI.setObjectOffset(FI, -Offset);
  }
}
