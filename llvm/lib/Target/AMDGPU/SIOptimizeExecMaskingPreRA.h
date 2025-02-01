//===- SIOptimizeExecMaskingPreRA.h.h ---------------------------------------*-
//C++- *-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_SIOPTIMIZEEXECMASKINGPRERA_H
#define LLVM_LIB_TARGET_AMDGPU_SIOPTIMIZEEXECMASKINGPRERA_H

#include "llvm/CodeGen/MachinePassManager.h"

namespace llvm {
class SIOptimizeExecMaskingPreRAPass
    : public PassInfoMixin<SIOptimizeExecMaskingPreRAPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};
} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_SIOPTIMIZEEXECMASKINGPRERA_H
