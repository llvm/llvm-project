//===-- SIPinVGPR.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pre-RA pass that biases the register allocator to keep values marked by
// llvm.amdgcn.internal.vgpr.pin VGPR-resident. See SIPinVGPR.cpp.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_SIPINVGPR_H
#define LLVM_LIB_TARGET_AMDGPU_SIPINVGPR_H

#include "llvm/CodeGen/MachinePassManager.h"

namespace llvm {

class SIPinVGPRPass : public PassInfoMixin<SIPinVGPRPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_SIPINVGPR_H
