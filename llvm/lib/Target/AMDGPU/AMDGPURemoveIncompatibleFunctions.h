//===- AMDGPURemoveIncompatibleFunctions.h ----------------------*- C++- *-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_REMOVEINCOMPATIBLEFUNCTIONS_H
#define LLVM_LIB_TARGET_AMDGPU_REMOVEINCOMPATIBLEFUNCTIONS_H

#include "llvm/IR/PassManager.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
class AMDGPURemoveIncompatibleFunctionsPass
    : public PassInfoMixin<AMDGPURemoveIncompatibleFunctionsPass> {
  const TargetMachine *TM;

public:
  AMDGPURemoveIncompatibleFunctionsPass(const TargetMachine &TM) : TM(&TM) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};
} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_REMOVEINCOMPATIBLEFUNCTIONS_H
