//===-- M88kFrameLowering.h - Frame lowering for M88k -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_M88K_M88KFRAMELOWERING_H
#define LLVM_LIB_TARGET_M88K_M88KFRAMELOWERING_H

#include "llvm/ADT/IndexedMap.h"
#include "llvm/CodeGen/TargetFrameLowering.h"

namespace llvm {
class M88kTargetMachine;
class M88kSubtarget;

class M88kFrameLowering : public TargetFrameLowering {
  IndexedMap<unsigned> RegSpillOffsets;

public:
  M88kFrameLowering();

  // Override TargetFrameLowering.
  void emitPrologue(MachineFunction &MF, MachineBasicBlock &MBB) const override;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const override;
  bool hasFP(const MachineFunction &MF) const override;
  bool spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator MI,
                                 ArrayRef<CalleeSavedInfo> CSI,
                                 const TargetRegisterInfo *TRI) const override;
  void determineCalleeSaves(MachineFunction &MF, BitVector &SavedRegs,
                            RegScavenger *RS) const;
};
} // end namespace llvm

#endif
