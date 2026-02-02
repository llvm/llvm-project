//===- LoopReduceMotion.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass is designed to hoist `ReduceCall` operations out of loops to reduce
// the number of instructions within the loop body.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_VECTORIZE_LOOPREDUCEMOTION_H
#define LLVM_TRANSFORMS_VECTORIZE_LOOPREDUCEMOTION_H
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/PassManager.h"
namespace llvm {
class LoopReduceMotionPass : public PassInfoMixin<LoopReduceMotionPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
  bool matchAndTransform(Loop &L, DominatorTree &DT, LoopInfo &LI);
};
} // namespace llvm
#endif
