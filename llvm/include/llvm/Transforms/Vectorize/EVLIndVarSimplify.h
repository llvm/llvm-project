//===------ EVLIndVarSimplify.h - Optimize vectorized loops w/ EVL IV------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass optimizes a vectorized loop with canonical IV to using EVL-based
// IV if it was tail-folded by predicated EVL.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_EVLINDVARSIMPLIFY_H
#define LLVM_TRANSFORMS_VECTORIZE_EVLINDVARSIMPLIFY_H

#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/PassManager.h"

namespace llvm {
class Loop;
class LPMUpdater;

/// Turn vectorized loops with canonical induction variables into loops that
/// only use a single EVL-based induction variable.
struct EVLIndVarSimplifyPass : public PassInfoMixin<EVLIndVarSimplifyPass> {
  PreservedAnalyses run(Loop &L, LoopAnalysisManager &LAM,
                        LoopStandardAnalysisResults &AR, LPMUpdater &U);
};
} // namespace llvm
#endif
