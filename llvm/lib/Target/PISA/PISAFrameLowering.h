//===-- PISAFrameLowering.h - Define frame lowering for PISA --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PISA_PISAFRAMELOWERING_H
#define LLVM_LIB_TARGET_PISA_PISAFRAMELOWERING_H

#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/Support/Alignment.h"

namespace llvm {
class PISASubtarget;

class PISAFrameLowering : public TargetFrameLowering {
public:
  explicit PISAFrameLowering(const PISASubtarget &Sti)
      : TargetFrameLowering(TargetFrameLowering::StackGrowsDown, Align(8), 0) {}

  void emitPrologue(MachineFunction &MF,
                    MachineBasicBlock &MBB) const override {}
  void emitEpilogue(MachineFunction &MF,
                    MachineBasicBlock &MBB) const override {}

  bool hasFPImpl(const MachineFunction &MF) const override { return false; }
};
} // namespace llvm
#endif // LLVM_LIB_TARGET_PISA_PISAFRAMELOWERING_H
