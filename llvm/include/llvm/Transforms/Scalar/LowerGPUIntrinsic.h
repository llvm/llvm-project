//===--- LowerGPUIntrinsic.h - Lower GPU intrinsics -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers GPU intrinsics.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_SCALAR_LOWERGPUINTRINSIC_H
#define LLVM_TRANSFORMS_SCALAR_LOWERGPUINTRINSIC_H

#include "llvm/IR/PassManager.h"

namespace llvm {

struct LowerGPUIntrinsicPass : public PassInfoMixin<LowerGPUIntrinsicPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

  static bool isRequired() { return true; } // otherwise O0 doesn't run it
};
} // namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_LOWERGPUINTRINSIC_H
