//=- LoongArchFrameLowering.h - TargetFrameLowering for LoongArch -*- C++ -*--//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class implements LoongArch-specific bits of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_LOONGARCH_LOONGARCHFRAMELOWERING_H
#define LLVM_LIB_TARGET_LOONGARCH_LOONGARCHFRAMELOWERING_H

#include "llvm/CodeGen/TargetFrameLowering.h"

namespace llvm {
class LoongArchSubtarget;

class LoongArchFrameLowering : public TargetFrameLowering {
  const LoongArchSubtarget &STI;

public:
  explicit LoongArchFrameLowering(const LoongArchSubtarget &STI)
      : TargetFrameLowering(StackGrowsDown,
                            /*StackAlignment=*/Align(16),
                            /*LocalAreaOffset=*/0),
        STI(STI) {}

  void emitPrologue(MachineFunction &MF, MachineBasicBlock &MBB) const override;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const override;

  bool hasFP(const MachineFunction &MF) const override;
  bool hasBP(const MachineFunction &MF) const;
};
} // namespace llvm
#endif // LLVM_LIB_TARGET_LOONGARCH_LOONGARCHFRAMELOWERING_H
