//===- SIShrinkInstructions.h -----------------------------------*- C++- *-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_SISHRINKINSTRUCTIONS_H
#define LLVM_LIB_TARGET_AMDGPU_SISHRINKINSTRUCTIONS_H

#include "llvm/CodeGen/MachinePassManager.h"

namespace llvm {

class SIShrinkInstructionsPass
    : public PassInfoMixin<SIShrinkInstructionsPass> {
public:
  SIShrinkInstructionsPass() = default;
  PreservedAnalyses run(MachineFunction &MF, MachineFunctionAnalysisManager &);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_SISHRINKINSTRUCTIONS_H
