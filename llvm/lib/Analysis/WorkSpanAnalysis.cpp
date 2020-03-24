//===- WorkSpanAnalysis.cpp - Analysis to estimate work and span ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements an analysis pass to estimate the work and span of the
// program.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/WorkSpanAnalysis.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/Support/BranchProbability.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "work-span"

// Get a constant trip count for the given loop.
unsigned llvm::getConstTripCount(const Loop *L, ScalarEvolution &SE) {
  int64_t ConstTripCount = 0;
  // If there are multiple exiting blocks but one of them is the latch, use
  // the latch for the trip count estimation. Otherwise insist on a single
  // exiting block for the trip count estimation.
  BasicBlock *ExitingBlock = L->getLoopLatch();
  if (!ExitingBlock || !L->isLoopExiting(ExitingBlock))
    ExitingBlock = L->getExitingBlock();
  if (ExitingBlock)
    ConstTripCount = SE.getSmallConstantTripCount(L, ExitingBlock);
  return ConstTripCount;
}

/// Recursive helper routine to estimate the amount of work in a loop.
static void estimateLoopCostHelper(const Loop *L, CodeMetrics &Metrics,
                                   WSCost &LoopCost, LoopInfo *LI,
                                   ScalarEvolution *SE) {
  if (LoopCost.UnknownCost)
    return;

  // TODO: Handle control flow within the loop intelligently, using
  // BlockFrequencyInfo.
  for (Loop *SubL : *L) {
    WSCost SubLoopCost;
    estimateLoopCostHelper(SubL, Metrics, SubLoopCost, LI, SE);
    // Quit early if the size of this subloop is already too big.
    if (std::numeric_limits<int64_t>::max() == SubLoopCost.Work)
      LoopCost.Work = std::numeric_limits<int64_t>::max();

    // Find a constant trip count if available
    int64_t ConstTripCount = SE ? getConstTripCount(SubL, *SE) : 0;
    // TODO: Use a more precise analysis to account for non-constant trip
    // counts.
    if (!ConstTripCount) {
      LoopCost.UnknownCost = true;
      // If we cannot compute a constant trip count, assume this subloop
      // executes at least once.
      ConstTripCount = 1;
    }

    // Check if the total size of this subloop is huge.
    if (std::numeric_limits<int64_t>::max() / ConstTripCount > SubLoopCost.Work)
      LoopCost.Work = std::numeric_limits<int64_t>::max();

    // Check if this subloop suffices to make loop L huge.
    if (std::numeric_limits<int64_t>::max() - LoopCost.Work <
        (SubLoopCost.Work * ConstTripCount))
      LoopCost.Work = std::numeric_limits<int64_t>::max();

    // Add in the size of this subloop.
    LoopCost.Work += (SubLoopCost.Work * ConstTripCount);
  }

  // After looking at all subloops, if we've concluded we have a huge loop size,
  // return early.
  if (std::numeric_limits<int64_t>::max() == LoopCost.Work)
    return;

  for (BasicBlock *BB : L->blocks())
    if (LI->getLoopFor(BB) == L) {
      // Check if this BB suffices to make loop L huge.
      if (std::numeric_limits<int64_t>::max() - LoopCost.Work <
          Metrics.NumBBInsts[BB]) {
        LoopCost.Work = std::numeric_limits<int64_t>::max();
        return;
      }
      LoopCost.Work += Metrics.NumBBInsts[BB];
    }
}

void llvm::estimateLoopCost(WSCost &LoopCost, const Loop *L, LoopInfo *LI,
                            ScalarEvolution *SE, const TargetTransformInfo &TTI,
                            TargetLibraryInfo *TLI,
                            const SmallPtrSetImpl<const Value *> &EphValues) {
  // TODO: Use more precise analysis to estimate the work in each call.
  // TODO: Use vectorizability to enhance cost analysis.

  // Gather code metrics for all basic blocks in the loop.
  for (BasicBlock *BB : L->blocks())
    LoopCost.Metrics.analyzeBasicBlock(BB, TTI, EphValues, TLI);

  estimateLoopCostHelper(L, LoopCost.Metrics, LoopCost, LI, SE);
}
