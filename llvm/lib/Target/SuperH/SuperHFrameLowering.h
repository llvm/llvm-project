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
private:
  using Block = MachineBasicBlock;
  using BlockIt = MachineBasicBlock::iterator;

protected:
  const SuperHSubtarget &STI;

public:
  explicit SuperHFrameLowering(const SuperHSubtarget &STI)
      : TargetFrameLowering(TargetFrameLowering::StackGrowsDown,
                            /*StackAlignment*/ Align(4),
                            /*LocalAreaOffset*/ 0,
                            /*TransAl*/ Align(1),
                            /*StackReal*/ false),
        STI(STI) {}
  bool canSimplifyCallFramePseudos(const MachineFunction &MF) const override;
  bool hasReservedCallFrame(const MachineFunction &MF) const override;

  StackOffset getFrameIndexReference(const MachineFunction &MF, int FI,
                                     Register &FrameReg) const override;

  void emitPrologue(MachineFunction &MF, Block &MBB) const override;
  void emitEpilogue(MachineFunction &MF, Block &MBB) const override;
  bool spillCalleeSavedRegisters(Block &MBB, BlockIt MI,
                                 ArrayRef<CalleeSavedInfo> CSI,
                                 const TargetRegisterInfo *TRI) const override;
  bool
  restoreCalleeSavedRegisters(Block &MBB, BlockIt MI,
                              MutableArrayRef<CalleeSavedInfo> CSI,
                              const TargetRegisterInfo *TRI) const override;

  void determineCalleeSaves(MachineFunction &MF, BitVector &SavedRegs,
                            RegScavenger *RS) const override;

  void emitFrameAdjust(Register Base, Block &MBB, BlockIt MI, 
                       int32_t AdjValue, MachineInstr::MIFlag MIFlag) const;

  BlockIt
  eliminateCallFramePseudoInstr(MachineFunction &MF, Block &MBB, 
                                BlockIt MI) const override;

protected:
  bool hasFPImpl(const MachineFunction &MF) const override;
};

} // end namespace llvm

#endif // end LLVM_LIB_TARGET_SUPERH_SUPERHFRAMELOWERING_H