//===- LoopNoOpElimination.h - Loop No-Op Elimination pass ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass eliminates no-op operations in loop bodies
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_LOOPNOOPELIMINATION_H
#define LLVM_TRANSFORMS_SCALAR_LOOPNOOPELIMINATION_H

#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class DominatorTree;
class Function;
class Instruction;
class Loop;
class LoopAccessInfoManager;
class LoopInfo;
class ScalarEvolution;
class TargetLibraryInfo;
class TargetTransformInfo;
class OptimizationRemarkEmitter;
class DataLayout;
class SCEVExpander;

/// Performs Loop No-Op Elimination Pass.
class LoopNoOpEliminationPass : public PassInfoMixin<LoopNoOpEliminationPass> {
public:
  ScalarEvolution *SE;
  LoopInfo *LI;
  TargetTransformInfo *TTI;
  DominatorTree *DT;
  TargetLibraryInfo *TLI;
  OptimizationRemarkEmitter *ORE;


  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
private:
  bool runImpl(Function &F);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_LOOPNOOPELIMINATION_H
