//===-- RISCVFrameLowering.h - Define frame lowering for RISCV -*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class implements RISCV-specific bits of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCVFRAMELOWERING_H
#define LLVM_LIB_TARGET_RISCV_RISCVFRAMELOWERING_H

#include "llvm/CodeGen/TargetFrameLowering.h"

namespace llvm {
class RISCVSubtarget;

class RISCVFrameLowering : public TargetFrameLowering {
public:
  explicit RISCVFrameLowering(const RISCVSubtarget &STI)
      : TargetFrameLowering(StackGrowsDown,
                            /*StackAlignment=*/16,
                            /*LocalAreaOffset=*/0) {}

  void emitPrologue(MachineFunction &MF, MachineBasicBlock &MBB) const override;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const override;

  bool hasFP(const MachineFunction &MF) const override;

  MachineBasicBlock::iterator
  eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator MI) const override {
    return MBB.erase(MI);
  }
};
}
#endif
