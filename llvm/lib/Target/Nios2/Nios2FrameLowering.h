//===-- Nios2FrameLowering.h - Define frame lowering for Nios2 --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIB_TARGET_NIOS2_NIOS2FRAMELOWERING_H
#define LLVM_LIB_TARGET_NIOS2_NIOS2FRAMELOWERING_H

#include "Nios2.h"
#include "llvm/CodeGen/TargetFrameLowering.h"

namespace llvm {
class Nios2Subtarget;

class Nios2FrameLowering : public TargetFrameLowering {
protected:
  const Nios2Subtarget &STI;

public:
  explicit Nios2FrameLowering(const Nios2Subtarget &sti)
      : TargetFrameLowering(TargetFrameLowering::StackGrowsDown, 4, 0, 4),
        STI(sti) {}

  bool hasFP(const MachineFunction &MF) const override;
  /// emitProlog/emitEpilog - These methods insert prolog and epilog code into
  /// the function.
  void emitPrologue(MachineFunction &MF, MachineBasicBlock &MBB) const override;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const override;
};
} // namespace llvm

#endif
