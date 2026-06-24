//===------- LoopPeelPass.h - Loop Peeling Pass -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/LoopPeelPass.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"

#define DEBUG_TYPE "loop-peel-pass"

using namespace llvm;
using namespace llvm::PatternMatch;

namespace {

static bool canPeelLoop(const Loop &L, const DominatorTree &DT,
                        ScalarEvolution &SE) {
  // Skip function with optsize.
  if (L.getHeader()->getParent()->hasOptSize())
    return false;

  // Split only innermost loop.
  if (!L.isInnermost())
    return false;

  // Check loop is in simplified form.
  if (!L.isLoopSimplifyForm())
    return false;

  // Check loop is in LCSSA form.
  if (!L.isLCSSAForm(DT))
    return false;

  // Skip loop that cannot be cloned.
  if (!L.isSafeToClone())
    return false;

  BasicBlock *ExitingBB = L.getExitingBlock();
  // Assumed only one exiting block.
  if (!ExitingBB)
    return false;

  // Exiting block terminator should be conditional
  BranchInst *ExitingBI = dyn_cast<BranchInst>(ExitingBB->getTerminator());
  if (!ExitingBI || ExitingBI->isUnconditional())
    return false;

  BranchInst *ExitingBI = dyn_cast<BranchInst>(ExitingBB->getTerminator());
  if (!ExitingBI)
    return false;

  // TODO: Analyze Predicate of ExitingBI of loop to among them -> ugt, sgt,
  // ult, slt If sge, uge, sle, sge, see if can be coverted into above ones by
  // modifiying SCEVExpr. If this holds than loop peel is suppoerted.

  LLVM_DEBUG(dbgs() << "Can Peel Loop : " << L << '\n');
  return true;
}
} // namespace

PreservedAnalyses LoopPeelPass::run(Loop &L, LoopAnalysisManager &AM,
                                    LoopStandardAnalysisResults &AR,
                                    LPMUpdater &LU) {
  dbgs() << "\nStart here\n";
  Function &F = *L.getHeader()->getParent();
  dbgs() << "Peel loop in function : " << F.getName() << " ->\n " << L << "\n";

  auto &LI = AR.LI;
  auto &DT = AR.DT;
  auto &SE = AR.SE;

  // const TargetTransformInfo &TTI = AR.TTI;

  // if (canPeelLoop(L, DT, SE))
  // Process Loop now!!!

  PreservedAnalyses PA;
  PA.areAllPreserved();
  return PA;
}