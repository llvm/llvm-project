//==- CanonicalizeFreezeInLoop.h - Canonicalize freezes in a loop-*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file canonicalizes freeze instructions in a loop.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_CANONICALIZE_FREEZES_IN_LOOPS_H
#define LLVM_TRANSFORMS_UTILS_CANONICALIZE_FREEZES_IN_LOOPS_H

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"

namespace llvm {

/// A pass that canonicalizes freeze instructions in a loop.
class CanonicalizeFreezeInLoopsPass
    : public PassInfoMixin<CanonicalizeFreezeInLoopsPass> {
public:
  PreservedAnalyses run(Loop &L, LoopAnalysisManager &AM,
                        LoopStandardAnalysisResults &AR, LPMUpdater &U);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_CANONICALIZE_FREEZES_IN_LOOPS_H
