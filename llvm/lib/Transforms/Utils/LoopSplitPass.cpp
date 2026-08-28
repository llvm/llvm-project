//===- LoopSplitPass.cpp - Test driver for LoopSplit ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass drives LoopSplit from `opt` for testing. For every eligible loop it
// builds partitions from the -loop-split-points offsets and splits the loop.
// Which loops are eligible is chosen by -loop-split-depth; the default is the
// innermost ones.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/LoopSplitPass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ScalarEvolutionPatternMatch.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/LoopSplit.h"

using namespace llvm;
using namespace llvm::SCEVPatternMatch;

#define DEBUG_TYPE "loop-split"

static cl::list<unsigned>
    SplitPoints("loop-split-points",
                cl::desc("Iteration offsets (relative to the induction start) "
                         "at which to split each loop"),
                cl::CommaSeparated);

static cl::opt<unsigned> SplitDepth(
    "loop-split-depth",
    cl::desc(
        "Split the loops at this nesting depth (1 is outermost) instead of "
        "the innermost ones"),
    cl::init(0));

// Build the partition list for \p L from the command-line split offsets and run
// the transform. Returns true if the loop was split.
static bool splitLoop(Loop *L, ScalarEvolution &SE, DominatorTree &DT,
                      LoopInfo &LI) {
  std::optional<LoopSplit> LS = LoopSplit::get(L, &LI, &SE, &DT);
  if (!LS) {
    LLVM_DEBUG(dbgs() << DEBUG_TYPE ": loop is not legal for splitting\n");
    return false;
  }

  // Legality analysis has already established this shape.
  const SCEV *IndVarSCEV = SE.getSCEV(LS->getInductionVariable());
  const SCEV *Start;
  const APInt *StepC;
  [[maybe_unused]] bool Matched =
      match(IndVarSCEV,
            m_scev_AffineAddRec(m_SCEV(Start), m_scev_APInt(StepC))) &&
      (StepC->isOne() || StepC->isAllOnes());
  assert(Matched && "expected unit-step affine induction");

  const SCEV *BTC = SE.getBackedgeTakenCount(L);
  const SCEV *End = LS->getInductionEnd();
  Type *Ty = Start->getType();
  unsigned BitWidth = Ty->getIntegerBitWidth();
  // The backedge-taken count is a separate expression and need not share the
  // induction's width, so coerce it before doing arithmetic in that type.
  const SCEV *Count = SE.getTruncateOrZeroExtend(BTC, Ty);

  // Build boundaries in iteration order, stepping away from Start by each
  // offset (down for a descending loop). Each offset opens a new partition at
  // iteration `Start +/- offset`; the previous partition ends one step before.
  bool Descending = StepC->isAllOnes();

  // Boundaries must be increasing and distinct to tile the space, so sort and
  // unique the offsets. Drop any that do not fit the induction type; truncating
  // would reorder them and the partitions would overlap.
  SmallVector<unsigned, 4> Offsets;
  for (unsigned Offset : SplitPoints)
    if (BitWidth >= 32 || Offset < (1u << BitWidth))
      Offsets.push_back(Offset);
  llvm::sort(Offsets);
  Offsets.erase(llvm::unique(Offsets), Offsets.end());

  const SCEV *PrevStart = Start;
  const SCEV *One = SE.getOne(Ty);
  for (unsigned Offset : Offsets) {
    // Clamp into [1, BTC] so each boundary stays in the space; Start +/- BTC is
    // the last iteration. The umax reaches one past it when BTC is zero, which
    // Legality analysis proved representable.
    const SCEV *Off = SE.getConstant(Ty, Offset);
    Off = SE.getUMaxExpr(One, SE.getUMinExpr(Off, Count));
    const SCEV *Point =
        Descending ? SE.getMinusSCEV(Start, Off) : SE.getAddExpr(Start, Off);
    const SCEV *PrevEnd =
        Descending ? SE.getAddExpr(Point, One) : SE.getMinusSCEV(Point, One);
    LS->addPartition(PrevStart, PrevEnd);
    PrevStart = Point;
  }
  // The final partition runs to the iteration-space end.
  LS->addPartition(PrevStart, End);

  if (LS->getNumPartitions() < 2)
    return false;

  return LS->split();
}

// Split the selected loops in \p F at the command-line offsets.
PreservedAnalyses LoopSplitPass::run(Function &F, FunctionAnalysisManager &AM) {
  if (SplitPoints.empty())
    return PreservedAnalyses::all();

  auto &LI = AM.getResult<LoopAnalysis>(F);
  auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);

  // Collect the loops up front: the transform creates new sibling loops that we
  // must not revisit. Depth 0 means no depth was given, so take the innermost
  // loops; LoopSplit itself works at any depth.
  SmallVector<Loop *, 4> Worklist;
  for (Loop *L : LI.getLoopsInPreorder())
    if (SplitDepth ? L->getLoopDepth() == SplitDepth : L->isInnermost())
      Worklist.push_back(L);

  bool Changed = false;
  for (Loop *L : Worklist)
    Changed |= splitLoop(L, SE, DT, LI);

  if (!Changed)
    return PreservedAnalyses::all();

  // LoopSplit patches the dominator tree and loop info as it goes, so keeping
  // them lets `verify<domtree>` and `verify<loops>` check those updates instead
  // of a freshly recomputed copy.
  PreservedAnalyses PA;
  PA.preserve<DominatorTreeAnalysis>();
  PA.preserve<LoopAnalysis>();
  return PA;
}
