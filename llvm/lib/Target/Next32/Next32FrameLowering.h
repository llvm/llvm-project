//===-- Next32FrameLowering.h - Define frame lowering for Next32 -----*- C++
//-*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class implements Next32-specific bits of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_Next32_Next32FRAMELOWERING_H
#define LLVM_LIB_TARGET_Next32_Next32FRAMELOWERING_H

#include "llvm/CodeGen/TargetFrameLowering.h"

namespace llvm {
class Next32Subtarget;

class Next32FrameLowering : public TargetFrameLowering {
public:
  explicit Next32FrameLowering(const Next32Subtarget &sti);

  void emitPrologue(MachineFunction &MF, MachineBasicBlock &MBB) const override;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const override;

  bool hasFP(const MachineFunction &MF) const override;
  void determineCalleeSaves(MachineFunction &MF, BitVector &SavedRegs,
                            RegScavenger *RS) const override;

  MachineBasicBlock::iterator addArgumentFeeders(MachineFunction &MF,
                                                 MachineBasicBlock &MBB) const;
  void debugMarkParameterFeeders(MachineFunction &MF, MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator MBBI) const;
  MachineBasicBlock::iterator
  eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator MI) const override {
    return MBB.erase(MI);
  }

  StackOffset getFrameIndexReference(const MachineFunction &MF, int FI,
                                     Register &FrameReg) const override;
};
} // namespace llvm
#endif
