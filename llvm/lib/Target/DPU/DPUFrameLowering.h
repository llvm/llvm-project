//===-- DPUFrameLowering.h - Define frame lowering for DPU -----*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class implements DPU-specific bits of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_DPU_DPUFRAMELOWERING_H
#define LLVM_LIB_TARGET_DPU_DPUFRAMELOWERING_H

#include "llvm/CodeGen/TargetFrameLowering.h"

namespace llvm {

class DPUSubtarget;

class DPUFrameLowering : public TargetFrameLowering {
public:
  explicit DPUFrameLowering(const DPUSubtarget &SubtargetInfo)
      : TargetFrameLowering(TargetFrameLowering::StackGrowsUp, 8, 0),
        STI(SubtargetInfo) {}

  void emitPrologue(MachineFunction &MF, MachineBasicBlock &MBB) const override;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const override;

  bool hasFP(const MachineFunction &MF) const override;

  MachineBasicBlock::iterator
  eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator MI) const override;

  int getFrameIndexReference(const MachineFunction &MF, int FI,
                             unsigned &FrameReg) const override;

private:
  const DPUSubtarget &STI;
};
} // namespace llvm
#endif
