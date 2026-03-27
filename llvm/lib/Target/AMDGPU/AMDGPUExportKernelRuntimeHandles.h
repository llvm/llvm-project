//===- AMDGPUExportKernelRuntimeHandles.h -----------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_EXPORTKERNELRUNTIMEHANDLES_H
#define LLVM_LIB_TARGET_AMDGPU_EXPORTKERNELRUNTIMEHANDLES_H

#include "llvm/IR/PassManager.h"

namespace llvm {
class AMDGPUExportKernelRuntimeHandlesPass
    : public PassInfoMixin<AMDGPUExportKernelRuntimeHandlesPass> {
public:
  AMDGPUExportKernelRuntimeHandlesPass() = default;
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};
} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_EXPORTKERNELRUNTIMEHANDLES_H
