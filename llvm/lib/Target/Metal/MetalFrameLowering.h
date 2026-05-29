//===-- MetalFrameLowering.h - Frame lowering for Metal ------*- C++ ---*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class implements Metal-specific bits of TargetFrameLowering class.
// This is just a stub because the Metal/AIR backend does not lower through the
// MC layer; it serializes to AIR bitcode in a .metallib container instead.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_METAL_METALFRAMELOWERING_H
#define LLVM_LIB_TARGET_METAL_METALFRAMELOWERING_H

#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/Support/Alignment.h"

namespace llvm {
class MetalSubtarget;

class MetalFrameLowering : public TargetFrameLowering {
public:
  explicit MetalFrameLowering(const MetalSubtarget &STI)
      : TargetFrameLowering(TargetFrameLowering::StackGrowsDown, Align(8), 0) {}

  void emitPrologue(MachineFunction &, MachineBasicBlock &) const override {}
  void emitEpilogue(MachineFunction &, MachineBasicBlock &) const override {}

protected:
  bool hasFPImpl(const MachineFunction &) const override { return false; }
};
} // namespace llvm
#endif // LLVM_LIB_TARGET_METAL_METALFRAMELOWERING_H
