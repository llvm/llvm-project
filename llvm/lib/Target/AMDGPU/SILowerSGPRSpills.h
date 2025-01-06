//===- SILowerSGPRSpills.h --------------------------------------*- C++- *-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_SILOWERSGPRSPILLS_H
#define LLVM_LIB_TARGET_AMDGPU_SILOWERSGPRSPILLS_H

#include "llvm/CodeGen/MachinePassManager.h"

namespace llvm {
class SILowerSGPRSpillsPass : public PassInfoMixin<SILowerSGPRSpillsPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);

  MachineFunctionProperties getClearedProperties() const {
    // SILowerSGPRSpills introduces new Virtual VGPRs for spilling SGPRs.
    return MachineFunctionProperties()
        .set(MachineFunctionProperties::Property::IsSSA)
        .set(MachineFunctionProperties::Property::NoVRegs);
  }
};
} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_SILOWERSGPRSPILLS_H
