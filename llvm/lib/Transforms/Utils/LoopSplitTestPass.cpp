//===- LoopSplitTestPass.cpp - Test driver for LoopSplitUtils -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass drives LoopSplitUtils from `opt` for testing. For every eligible
// loop it builds partitions from the -loop-split-points offsets and splits the
// loop.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/LoopSplitTestPass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/LoopSplitUtils.h"

using namespace llvm;

#define DEBUG_TYPE "loop-split-test"

static cl::list<unsigned>
    SplitPoints("loop-split-points",
                cl::desc("Iteration offsets (relative to the induction start) "
                         "at which to split each loop"),
                cl::CommaSeparated);

static cl::list<unsigned> UnguardedPartitions(
    "loop-split-unguarded",
    cl::desc("Partition indices whose entry guard is omitted (the caller "
             "guarantees they run at least one iteration)"),
    cl::CommaSeparated);

static cl::opt<bool> PrintPartitionMap(
    "loop-split-print-partition-map",
    cl::desc("After splitting, print each original loop instruction's "
             "counterpart in every partition (LoopSplitUtils::getPartitionValue)"),
    cl::init(false));

/// Build the partition list for \p L from the command-line split offsets and
/// run the transform. Returns true if the loop was split.
static bool splitLoop(Loop *L, ScalarEvolution &SE, DominatorTree &DT,
                      LoopInfo &LI) {
  LoopSplitUtils LSU(L, &LI, &SE, &DT);
  if (!LSU.isLegal()) {
    LLVM_DEBUG(dbgs() << "loop-split-test: loop is not legal for splitting\n");
    return false;
  }

  const auto *IndAR =
      dyn_cast<SCEVAddRecExpr>(SE.getSCEV(LSU.getInductionVariable()));
  if (!IndAR)
    return false;

  const SCEV *Start = IndAR->getStart();
  const SCEV *BTC = SE.getBackedgeTakenCount(L);
  const SCEV *End = IndAR->evaluateAtIteration(BTC, SE);
  Type *Ty = Start->getType();
  if (End->getType() != Ty)
    End = SE.getTruncateExpr(End, Ty);

  // Build boundaries in iteration order, stepping away from Start by each
  // offset (down for a descending loop). Each offset opens a new partition at
  // iteration `Start +/- offset`; the previous partition ends one step before.
  bool Descending = false;
  if (const auto *StepC = dyn_cast<SCEVConstant>(IndAR->getStepRecurrence(SE)))
    Descending = StepC->getValue()->isMinusOne();

  const SCEV *PrevStart = Start;
  const SCEV *One = SE.getOne(Ty);
  for (unsigned Offset : SplitPoints) {
    const SCEV *Off = SE.getConstant(Ty, Offset);
    const SCEV *Point =
        Descending ? SE.getMinusSCEV(Start, Off) : SE.getAddExpr(Start, Off);
    const SCEV *PrevEnd =
        Descending ? SE.getAddExpr(Point, One) : SE.getMinusSCEV(Point, One);
    LSU.addPartition(PrevStart, PrevEnd);
    PrevStart = Point;
  }
  // The final partition runs to the iteration-space end.
  LSU.addPartition(PrevStart, End);

  // Suppress guards for the partitions the caller listed (out-of-range indices
  // are ignored).
  for (unsigned Idx : UnguardedPartitions)
    if (Idx < LSU.getNumPartitions())
      LSU.avoidPartitionGuard(Idx);

  if (LSU.getNumPartitions() < 2)
    return false;

  // Snapshot the original loop's named instructions before the transform so we
  // can query their per-partition counterparts afterwards (handles track any
  // that the transform deletes).
  SmallVector<WeakTrackingVH, 16> OrigValues;
  if (PrintPartitionMap)
    for (BasicBlock *BB : L->blocks())
      for (Instruction &I : *BB)
        if (I.hasName())
          OrigValues.push_back(&I);

  if (!LSU.split())
    return false;

  if (PrintPartitionMap) {
    const unsigned N = LSU.getNumPartitions();
    for (unsigned P = 0; P < N; ++P) {
      outs() << "LS-MAP partition " << P << ":\n";
      for (WeakTrackingVH &VH : OrigValues) {
        if (!VH)
          continue;
        Value *M = LSU.getPartitionValue(VH, P);
        outs() << "LS-MAP   " << VH->getName() << " -> "
               << (M ? M->getName() : "<none>") << "\n";
      }
    }
  }
  return true;
}

PreservedAnalyses LoopSplitTestPass::run(Function &F,
                                         FunctionAnalysisManager &AM) {
  if (SplitPoints.empty())
    return PreservedAnalyses::all();

  auto &LI = AM.getResult<LoopAnalysis>(F);
  auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);

  // Collect the original top-level loops up front; the transform creates new
  // sub-loops that we must not revisit.
  SmallVector<Loop *, 4> Worklist(LI.begin(), LI.end());

  bool Changed = false;
  for (Loop *L : Worklist) {
    SE.forgetLoop(L);
    Changed |= splitLoop(L, SE, DT, LI);
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
