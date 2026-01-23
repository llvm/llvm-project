//===- AMDGPUPreloadKernargProlog.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_PRELOAD_KERNARG_PROLOG_H
#define LLVM_LIB_TARGET_AMDGPU_PRELOAD_KERNARG_PROLOG_H

#include "llvm/CodeGen/MachinePassManager.h"

namespace llvm {

class AMDGPUPreloadKernArgPrologPass
    : public PassInfoMixin<AMDGPUPreloadKernArgPrologPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_PRELOAD_KERNARG_PROLOG_H
