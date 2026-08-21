//===- SuperHFrameLowering.h - Define frame lowering for SuperH -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the SuperHTargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SUPERH_SUPERHFRAMELOWERING_H
#define LLVM_LIB_TARGET_SUPERH_SUPERHFRAMELOWERING_H

#include "llvm/CodeGen/TargetFrameLowering.h"

namespace llvm {

class SuperHSubtarget;

class SuperHFrameLowering : public TargetFrameLowering {
protected:
  const SuperHSubtarget &STI;

public:
  explicit SuperHFrameLowering(const SuperHSubtarget &STI)
    : TargetFrameLowering(TargetFrameLowering::StackGrowsDown,
                          /*StackAlignment*/Align(4),
                          /*LocalAreaOffset*/0,
                          /*TransAl*/Align(4)),
      STI(STI) {}
  bool canSimplifyCallFramePseudos(const MachineFunction &MF) const override;
  bool hasReservedCallFrame(const MachineFunction &MF) const override;

  void emitPrologue(MachineFunction &MF, MachineBasicBlock &MBB) const override;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const override;
  bool spillCalleeSavedRegisters(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                                 ArrayRef<CalleeSavedInfo> CSI, const TargetRegisterInfo *TRI) const override;
  bool restoreCalleeSavedRegisters(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                                   MutableArrayRef<CalleeSavedInfo> CSI, const TargetRegisterInfo *TRI) const override;
  
  MachineBasicBlock::iterator
  eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator I) const override;

  void determineCalleeSaves(MachineFunction &MF, BitVector &SavedRegs,
                            RegScavenger *RS) const override;
protected:
  bool hasFPImpl(const MachineFunction &MF) const override;
};

} // end namespace llvm


#endif // end LLVM_LIB_TARGET_SUPERH_SUPERHFRAMELOWERING_H