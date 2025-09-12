//===- AMDGPUVectorIdiom.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AMDGPU-specific vector idiom canonicalizations to unblock SROA and
// subsequent scalarization/vectorization.
//
// This pass rewrites memcpy with select-fed operands into either:
//  - a value-level select (two loads + select + store), when safe to
//    speculatively load both arms, or
//  - a conservative CFG split around the condition to isolate each arm.
//
// Run this pass early, before SROA.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPVECTORIDIOM_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPVECTORIDIOM_H

#include "AMDGPU.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class AMDGPUVectorIdiomCombinePass
    : public PassInfoMixin<AMDGPUVectorIdiomCombinePass> {
  unsigned MaxBytes;

public:
  /// \p MaxBytes is max memcpy size (in bytes) to transform in
  /// AMDGPUVectorIdiom
  AMDGPUVectorIdiomCombinePass(unsigned MaxBytes);

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPVECTORIDIOM_H
