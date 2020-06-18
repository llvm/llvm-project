//===-- SystemZFrameLowering.h - Frame lowering for SystemZ -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SYSTEMZ_SYSTEMZFRAMELOWERING_H
#define LLVM_LIB_TARGET_SYSTEMZ_SYSTEMZFRAMELOWERING_H

#include "llvm/ADT/IndexedMap.h"
#include "llvm/CodeGen/TargetFrameLowering.h"

namespace llvm {
class SystemZTargetMachine;
class SystemZSubtarget;

class SystemZFrameLowering : public TargetFrameLowering {
  IndexedMap<unsigned> RegSpillOffsets;

public:
  SystemZFrameLowering();

  // Override TargetFrameLowering.
  bool isFPCloseToIncomingSP() const override { return false; }
  bool
  assignCalleeSavedSpillSlots(MachineFunction &MF,
                              const TargetRegisterInfo *TRI,
                              std::vector<CalleeSavedInfo> &CSI) const override;
  void determineCalleeSaves(MachineFunction &MF, BitVector &SavedRegs,
                            RegScavenger *RS) const override;
  bool spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator MBBI,
                                 ArrayRef<CalleeSavedInfo> CSI,
                                 const TargetRegisterInfo *TRI) const override;
  bool
  restoreCalleeSavedRegisters(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MBBII,
                              MutableArrayRef<CalleeSavedInfo> CSI,
                              const TargetRegisterInfo *TRI) const override;
  void processFunctionBeforeFrameFinalized(MachineFunction &MF,
                                           RegScavenger *RS) const override;
  void emitPrologue(MachineFunction &MF, MachineBasicBlock &MBB) const override;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const override;
  void inlineStackProbe(MachineFunction &MF,
                        MachineBasicBlock &PrologMBB) const override;
  bool hasFP(const MachineFunction &MF) const override;
  bool hasReservedCallFrame(const MachineFunction &MF) const override;
  int getFrameIndexReference(const MachineFunction &MF, int FI,
                             Register &FrameReg) const override;
  MachineBasicBlock::iterator
  eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator MI) const override;

  // Return the byte offset from the incoming stack pointer of Reg's
  // ABI-defined save slot.  Return 0 if no slot is defined for Reg.  Adjust
  // the offset in case MF has packed-stack.
  unsigned getRegSpillOffset(MachineFunction &MF, Register Reg) const;

  // Get or create the frame index of where the old frame pointer is stored.
  int getOrCreateFramePointerSaveIndex(MachineFunction &MF) const;

  bool usePackedStack(MachineFunction &MF) const;
};
} // end namespace llvm

#endif
