//===- LoopReduceMotion.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass is designed to sink `ReduceCall` operations out of loops to reduce
// the number of instructions within the loop body.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_VECTORIZE_LOOPREDUCEMOTION_H
#define LLVM_TRANSFORMS_VECTORIZE_LOOPREDUCEMOTION_H

#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"

namespace llvm {
class LoopReduceMotionPass : public PassInfoMixin<LoopReduceMotionPass> {
public:
  bool compareCost(LoopStandardAnalysisResults &AR, Loop &L, VectorType *VecTy);
  bool matchAndTransform(LoopStandardAnalysisResults &AR, Loop &L,
                         DominatorTree *DT, LoopInfo *LI);
  PreservedAnalyses run(Loop &L, LoopAnalysisManager &AM,
                        LoopStandardAnalysisResults &AR, LPMUpdater &U);
};
} // namespace llvm
#endif // LLVM_TRANSFORMS_VECTORIZE_LOOPREDUCEMOTION_H
