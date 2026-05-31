//===-- AIRFrameLowering.h - Frame lowering for AIR ------*- C++ ---*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class implements AIR-specific bits of TargetFrameLowering class.
// This is just a stub because the AIR backend does not lower through the
// MC layer; it serializes to AIR bitcode in a .metallib container instead.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIRFRAMELOWERING_H
#define LLVM_LIB_TARGET_AIR_AIRFRAMELOWERING_H

#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/Support/Alignment.h"

namespace llvm {
class AIRSubtarget;

class AIRFrameLowering : public TargetFrameLowering {
public:
  explicit AIRFrameLowering(const AIRSubtarget &STI)
      : TargetFrameLowering(TargetFrameLowering::StackGrowsDown, Align(8), 0) {}

  void emitPrologue(MachineFunction &, MachineBasicBlock &) const override {}
  void emitEpilogue(MachineFunction &, MachineBasicBlock &) const override {}

protected:
  bool hasFPImpl(const MachineFunction &) const override { return false; }
};
} // namespace llvm
#endif // LLVM_LIB_TARGET_AIR_AIRFRAMELOWERING_H
