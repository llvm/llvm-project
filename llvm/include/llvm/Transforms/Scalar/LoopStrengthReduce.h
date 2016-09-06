//===- LoopStrengthReduce.h - Loop Strength Reduce Pass -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This transformation analyzes and transforms the induction variables (and
// computations derived from them) into forms suitable for efficient execution
// on the target.
//
// This pass performs a strength reduction on array references inside loops that
// have as one or more of their components the loop induction variable, it
// rewrites expressions to take advantage of scaled-index addressing modes
// available on the target, and it performs a variety of other optimizations
// related to loop induction variables.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_LOOPSTRENGTHREDUCE_H
#define LLVM_TRANSFORMS_SCALAR_LOOPSTRENGTHREDUCE_H

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPassManager.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

/// Performs Loop Strength Reduce Pass.
class LoopStrengthReducePass : public PassInfoMixin<LoopStrengthReducePass> {
public:
  PreservedAnalyses run(Loop &L, LoopAnalysisManager &AM);
};
} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_LOOPSTRENGTHREDUCE_H
