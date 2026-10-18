//===-- AGXFrameLowering.h - Define frame lowering for AGX ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AGX_AGXFRAMELOWERING_H
#define LLVM_LIB_TARGET_AGX_AGXFRAMELOWERING_H

#include "llvm/CodeGen/TargetFrameLowering.h"

namespace llvm {

class AGXSubtarget;

/// Each thread has a private stack addressed by byte offsets. Objects are
/// laid out upwards from offset zero; the stack is allocated with
/// stack_adjust at kernel entry.
class AGXFrameLowering : public TargetFrameLowering {
public:
  explicit AGXFrameLowering(const AGXSubtarget &ST)
      : TargetFrameLowering(TargetFrameLowering::StackGrowsUp, Align(4), 0,
                            Align(4)) {}

  void emitPrologue(MachineFunction &MF, MachineBasicBlock &MBB) const override;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const override {}

  /// Size of the stack in the 32-bit units of stack_adjust, or zero.
  static unsigned getStackAdjustUnits(const MachineFunction &MF);

protected:
  bool hasFPImpl(const MachineFunction &MF) const override { return false; }
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AGX_AGXFRAMELOWERING_H
