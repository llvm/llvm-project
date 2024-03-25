//===- AArch64LoopIdiomTransform.h --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AARCH64_AARCH64LOOPIDIOMTRANSFORM_H
#define LLVM_LIB_TARGET_AARCH64_AARCH64LOOPIDIOMTRANSFORM_H

#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"

namespace llvm {

struct AArch64LoopIdiomTransformPass
    : PassInfoMixin<AArch64LoopIdiomTransformPass> {
  PreservedAnalyses run(Loop &L, LoopAnalysisManager &AM,
                        LoopStandardAnalysisResults &AR, LPMUpdater &U);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AARCH64_AARCH64LOOPIDIOMTRANSFORM_H
