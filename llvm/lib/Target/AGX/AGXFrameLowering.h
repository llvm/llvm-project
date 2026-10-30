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
  // Experimental call ABI: each static call depth has a separate region of
  // the kernel's per-thread scratch allocation: the kernel's own first, then
  // one per depth. Normally all of CallFrameRegionBytes; the only kernel of a
  // module may have a larger one, the depths' then sharing what remains of the
  // 32 KB that frame offsets can address (getCallFrameLayout).
  static constexpr unsigned CallFrameRegionBytes = 4096;

  static unsigned getCallDepth(const MachineFunction &MF);

  /// The offset of a called function's region (zero for depth 0).
  static uint64_t getCalleeFrameOffset(const MachineFunction &MF);

  explicit AGXFrameLowering(const AGXSubtarget &ST)
      : TargetFrameLowering(TargetFrameLowering::StackGrowsUp, Align(4), 0,
                            Align(4)) {}

  void emitPrologue(MachineFunction &MF, MachineBasicBlock &MBB) const override;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const override;

  /// Size of the stack in the 32-bit units of stack_adjust, or zero.
  static unsigned getStackAdjustUnits(const MachineFunction &MF);

protected:
  bool hasFPImpl(const MachineFunction &MF) const override { return false; }
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AGX_AGXFRAMELOWERING_H
