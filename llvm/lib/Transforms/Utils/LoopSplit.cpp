//===- LoopSplit.cpp - Split a loop's iteration space ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Splits a counted loop's iteration space into a chain of per-partition
// sub-loops. See LoopSplit.h for the high-level usage guidelines.
//
// Structure produced for partitions [S0,E0], [S1,E1], ... where E is the loop's
// last iteration and each clamped end sel_i is min(E_i, E), or max descending:
//
//   guard0:                            ; every S_i and sel_i is computed here
//     if (S0 <= sel0) goto preheader0 else goto guard1
//   loop0: ...                         ; latch iterates while i < sel0
//   exit0 -> guard1
//   guard1:
//     if (S1 <= sel1) goto preheader1 else goto guard2
//   loop1: ...                         ; latch iterates while i < sel1
//   exit1 -> guard2
//     ...
//   final.exit:
//
// Each guard holds the "S_i <= sel_i" check and skips an empty partition by
// falling through to the next guard. All S_i/sel_i are materialized once in
// guard0, and the end clamp keeps the "runs at least once" iteration in the
// right partition.
//
// The latch keeps iterating while the value the next iteration would use is
// still in the partition. That is written as the strict "i < sel_i" on the
// induction PHI rather than "i + 1 <= sel_i" on the step value; the two agree
// because legality analysis has established that the space does not wrap, and
// the strict form never forms i + 1, so it remains a real test even when sel_i
// is the last value of the type, where the inclusive one would be a tautology
// and the partition would never exit.
//
// A descending (step -1) loop uses the same structure mirrored: partitions run
// high-to-low and the clamp and predicates flip (>=/>).
//
// Usage guidelines:
//  - Caller bounds must not wrap the induction type. The clamp absorbs a bound
//    past the runtime trip count, and legality analysis reserves the one step
//    past the induction start that an empty partition needs, but a bound
//    reaching any further wraps in the bound arithmetic and cannot be repaired
//    here.
//  - Bounds must be loop-invariant: they are expanded in guard0, so a bound
//    depending on a value defined inside the loop cannot be placed.
//  - The partitions must tile the original iteration space exactly -- same
//    iterations, same order -- so the split preserves program behaviour.
//
// The transform is structural: inside a partition it only seeds the induction
// PHI with that partition's start and replaces the latch test. It never
// rebuilds a value that flows between partitions, so no SSA reconstruction is
// needed.
//
// Not yet supported, and rejected during legality analysis: loop-carried
// values, values that escape the loop (exit values), non-unit and non-integer
// inductions, top-tested loops, and multiple exits. Also rejected is an
// induction start at the extreme of the iteration direction, which leaves
// nowhere to put a boundary. An induction *end* at that extreme is fine,
// because the latch stays strict.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/LoopSplit.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ScalarEvolutionPatternMatch.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/ProfDataUtils.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <optional>

using namespace llvm;
using namespace llvm::SCEVPatternMatch;

#define DEBUG_TYPE "loop-split"

//===----------------------------------------------------------------------===//
// LoopSplit - construction, partition list, induction analysis
//===----------------------------------------------------------------------===//

/// Per-split() scratch shared by the phase helpers; lives for one split() call.
/// Everything derived from the induction lives on LoopSplit itself, filled in
/// by legality analysis; this holds only what the transform creates.
struct LoopSplit::SplitState {
  // Partition 0 reuses the original loop's preheader, exit, and entry guard;
  // those blocks live in Partitions[0] rather than being duplicated here.
  BasicBlock *FinalExit = nullptr; // where the partition chain converges.
  Loop *OuterLoop = nullptr;       // parent of the new blocks, if any.
  PHINode *Induction = nullptr;    // the loop's induction variable.
  MDNode *OrigLoopID = nullptr;    // !llvm.loop on the loop before splitting.
};

// Record a new partition with the given inclusive iteration range.
void LoopSplit::addPartition(const SCEV *Start, const SCEV *End) {
  assert(InductionEnd && "addPartition() requires prior legality analysis");
  // The bounds are combined with the induction end and expanded in its type. A
  // mismatch would otherwise surface either as a bare "Operand types don't
  // match!" from inside ScalarEvolution, or worse, as a silent cast.
  assert(Start->getType() == InductionEnd->getType() &&
         End->getType() == InductionEnd->getType() &&
         "partition bounds must have the induction type");
  if (Partitions.empty()) {
    assert((Start == InductionStart ||
            SE->isKnownPredicate(ICmpInst::ICMP_EQ, Start, InductionStart)) &&
           "first partition Start must match the induction start");
  }
  Partitions.emplace_back(Start, End);
}

// Return the induction's add-recurrence, or null unless the induction is an
// integer with a unit step that the latch compares.
static const SCEVAddRecExpr *analyzeInduction(Loop *L, ScalarEvolution *SE) {
  ICmpInst *LatchCmp = L->getLatchCmpInst();

  // SCEV's induction variable, restricted to a unit-step affine recurrence.
  PHINode *Induction = L->getInductionVariable(*SE);
  if (!Induction)
    return nullptr;
  // Partition bounds are integer arithmetic on the induction type, so a loop
  // whose only induction is a pointer is out of scope.
  if (!Induction->getType()->isIntegerTy())
    return nullptr;
  const SCEV *IndSCEV = SE->getSCEV(Induction);
  // Match an affine add-recurrence and capture its constant step; accept a unit
  // step in either direction: +1 (ascending) or -1 (descending).
  const APInt *Step;
  if (!match(IndSCEV, m_scev_AffineAddRec(m_SCEV(), m_scev_APInt(Step))))
    return nullptr;
  if (!Step->isOne() && !Step->isAllOnes())
    return nullptr;
  const auto *AR = cast<SCEVAddRecExpr>(IndSCEV);

  // The induction's "next" value (i + 1), produced in the latch.
  auto *StepInst = dyn_cast<Instruction>(
      Induction->getIncomingValueForBlock(L->getLoopLatch()));
  if (!StepInst)
    return nullptr;

  // One compare operand must be the induction, either the PHI or its step. The
  // rebuilt latch always compares the PHI, so which operand it was is not used.
  if (any_of(LatchCmp->operands(),
             [&](Value *Op) { return Op == Induction || Op == StepInst; }))
    return AR;
  return nullptr;
}

// Decide whether the iteration ordering is signed or unsigned; returns the
// signedness, or nullopt if it cannot be proven.
static std::optional<bool> computeSignedness(ScalarEvolution &SE, Loop *L,
                                             const SCEVAddRecExpr *IndAR) {
  ICmpInst::Predicate P = L->getLatchCmpInst()->getPredicate();
  // A relational predicate gives the ordering directly; for eq/ne fall back to
  // the recurrence's no-wrap flags.
  if (ICmpInst::isRelational(P))
    return ICmpInst::isSigned(P);
  if (IndAR->hasNoSignedWrap() && IndAR->hasNoUnsignedWrap()) {
    const ConstantRange UR = SE.getUnsignedRange(IndAR);
    const ConstantRange SR = SE.getSignedRange(IndAR);
    if (UR.isFullSet() && SR.isFullSet()) {
      LLVM_DEBUG(dbgs() << DEBUG_TYPE
                 ": ambiguous iteration ordering with both nsw and nuw\n");
      return std::nullopt;
    }
    if (!SR.isFullSet())
      return true;
    if (!UR.isFullSet())
      return false;
    LLVM_DEBUG(dbgs() << DEBUG_TYPE
               ": cannot prove iteration ordering signedness\n");
    return std::nullopt;
  }
  if (IndAR->hasNoSignedWrap())
    return true;
  if (IndAR->hasNoUnsignedWrap())
    return false;
  LLVM_DEBUG(dbgs() << DEBUG_TYPE
             ": cannot prove iteration ordering signedness\n");
  return std::nullopt;
}

// Prove \p Pred between \p LHS and \p RHS on entry to \p L, with loop guards
// folded in so a bound fixed by a dominating condition is seen.
static bool isEntryGuardedByCond(ScalarEvolution &SE, Loop *L,
                                 ICmpInst::Predicate Pred, const SCEV *LHS,
                                 const SCEV *RHS) {
  return SE.isLoopEntryGuardedByCond(L, Pred, SE.applyLoopGuards(LHS, L),
                                     SE.applyLoopGuards(RHS, L));
}

// Latch "keep iterating" predicate, comparing the induction PHI against the
// partition end: `i < sel` ascending, `i > sel` descending.
static ICmpInst::Predicate continuePredicate(bool Signed, bool Descending) {
  ICmpInst::Predicate P = Descending ? ICmpInst::ICMP_UGT : ICmpInst::ICMP_ULT;
  return Signed ? ICmpInst::getSignedPredicate(P) : P;
}

// Guard "enter this partition" predicate: the latch test made non-strict, so
// `start <= sel` ascending and `start >= sel` descending. Also used during
// legality analysis to prove the iteration space is monotonic.
static ICmpInst::Predicate guardPredicate(bool Signed, bool Descending) {
  return ICmpInst::getNonStrictPredicate(continuePredicate(Signed, Descending));
}

struct LoopSplitAnalysis {
  const SCEV *InductionStart;
  const SCEV *InductionEnd;
  bool InductionIsSigned;
  bool Descending;
};

// Check every structural precondition and record the induction analysis.
static std::optional<LoopSplitAnalysis>
analyzeLegality(Loop *L, LoopInfo *LI, ScalarEvolution *SE, DominatorTree *DT) {
  // Require a bottom-tested single-exit loop in LCSSA form. Simplify form gives
  // the preheader, single latch and dedicated exits; the rest pin the exit to
  // the latch, so the latch compare can be rewritten per partition.
  if (!L->isLoopSimplifyForm() || !L->isLCSSAForm(*DT) ||
      L->getExitingBlock() != L->getLoopLatch() || !L->getExitBlock()) {
    LLVM_DEBUG(dbgs() << DEBUG_TYPE ": loop not in expected form\n");
    return std::nullopt;
  }

  // The latch compare must exist and reside in the latch: it is rewritten in
  // place, once per partition.
  ICmpInst *LatchCmp = L->getLatchCmpInst();
  if (!LatchCmp || LatchCmp->getParent() != L->getLoopLatch()) {
    LLVM_DEBUG(dbgs() << DEBUG_TYPE ": latch compare not in the loop latch\n");
    return std::nullopt;
  }

  // Exit values are unsupported. Look for an LCSSA PHI and for a use outside
  // the loop: a token-like type cannot appear in a PHI, so LCSSA can leave a
  // value escaping with no PHI to find.
  if (!L->getExitBlock()->phis().empty()) {
    LLVM_DEBUG(dbgs() << DEBUG_TYPE ": loop has exit values\n");
    return std::nullopt;
  }
  if (!findDefsUsedOutsideOfLoop(L).empty()) {
    LLVM_DEBUG(dbgs() << DEBUG_TYPE ": loop has exit values\n");
    return std::nullopt;
  }

  // Guard-gated clones require isSafeToCloneConditionally().
  if (!L->isSafeToCloneConditionally(*DT)) {
    LLVM_DEBUG(dbgs() << DEBUG_TYPE ": loop not safe to clone conditionally\n");
    return std::nullopt;
  }

  // A computable backedge-taken count fixes the iteration space we rebuild.
  const SCEV *BTC = SE->getBackedgeTakenCount(L);
  if (isa<SCEVCouldNotCompute>(BTC)) {
    LLVM_DEBUG(dbgs() << DEBUG_TYPE ": loop trip count uncomputable\n");
    return std::nullopt;
  }

  const SCEVAddRecExpr *IndAR = analyzeInduction(L, SE);
  if (!IndAR) {
    LLVM_DEBUG(dbgs() << DEBUG_TYPE
               ": no unique unit-step integer induction\n");
    return std::nullopt;
  }

  PHINode *Induction = L->getInductionVariable(*SE);

  // Loop-carried values are unsupported: a later partition would have to resume
  // the previous one's value, which needs SSA reconstruction. The induction is
  // the exception, seeded per partition from its own start bound.
  for (PHINode &HeaderPHI : L->getHeader()->phis())
    if (&HeaderPHI != Induction) {
      LLVM_DEBUG(dbgs() << DEBUG_TYPE ": loop has carried values\n");
      return std::nullopt;
    }

  std::optional<bool> Signed = computeSignedness(*SE, L, IndAR);
  if (!Signed)
    return std::nullopt;
  const bool InductionIsSigned = *Signed;
  const bool Descending =
      cast<SCEVConstant>(IndAR->getStepRecurrence(*SE))->getAPInt().isAllOnes();

  // Start and end must share the induction type; reject any width mismatch.
  // evaluateAtIteration coerces to the start's type for an affine recurrence,
  // so this is defensive rather than reachable.
  const SCEV *InductionEnd = IndAR->evaluateAtIteration(BTC, *SE);
  if (InductionEnd->getType() != IndAR->getStart()->getType()) {
    LLVM_DEBUG(dbgs() << DEBUG_TYPE ": induction end/start type mismatch\n");
    return std::nullopt;
  }

  // Partition bounds and entry guards assume the space runs monotonically from
  // start to end, so refuse one that wraps past the type extreme. A no-wrap
  // flag on the recurrence asserts that directly.
  bool NoWrap =
      InductionIsSigned ? IndAR->hasNoSignedWrap() : IndAR->hasNoUnsignedWrap();
  if (!NoWrap && !isEntryGuardedByCond(
                     *SE, L, guardPredicate(InductionIsSigned, Descending),
                     IndAR->getStart(), InductionEnd)) {
    LLVM_DEBUG(dbgs() << DEBUG_TYPE
               ": iteration space may wrap past the type extreme\n");
    return std::nullopt;
  }

  // A boundary can sit one step beyond the start, so that step has to be
  // representable: from the type extreme it wraps and still compares in range.
  // Such a loop runs one iteration anyway, which cannot be divided.
  const SCEV *Start = IndAR->getStart();
  if (!(Descending ? cannotBeMinInLoop(Start, L, *SE, InductionIsSigned)
                   : cannotBeMaxInLoop(Start, L, *SE, InductionIsSigned))) {
    LLVM_DEBUG(dbgs() << DEBUG_TYPE
               ": induction start at a type extreme, no room for a boundary\n");
    return std::nullopt;
  }

  return LoopSplitAnalysis{Start, InductionEnd, InductionIsSigned, Descending};
}

std::optional<LoopSplit>
LoopSplit::get(Loop *L, LoopInfo *LI, ScalarEvolution *SE, DominatorTree *DT) {
  std::optional<LoopSplitAnalysis> Analysis = analyzeLegality(L, LI, SE, DT);
  if (!Analysis)
    return std::nullopt;
  return LoopSplit(L, LI, SE, DT, Analysis->InductionStart,
                   Analysis->InductionEnd, Analysis->InductionIsSigned,
                   Analysis->Descending);
}

//===----------------------------------------------------------------------===//
// Transform
//===----------------------------------------------------------------------===//

static void buildEntryGuard(BasicBlock *&Preheader, BasicBlock *&EntryGuard,
                            DominatorTree *DT, LoopInfo *LI);

const SCEV *LoopSplit::getClampedEndSCEV(const SCEV *EndExpr) const {
  if (Descending)
    return InductionIsSigned ? SE->getSMaxExpr(EndExpr, InductionEnd)
                             : SE->getUMaxExpr(EndExpr, InductionEnd);
  return InductionIsSigned ? SE->getSMinExpr(EndExpr, InductionEnd)
                           : SE->getUMinExpr(EndExpr, InductionEnd);
}

bool LoopSplit::arePartitionBoundsSafeToExpand(SCEVExpander &Expander,
                                               Instruction *InsertPt) const {
  for (const PartitionInfo &P : Partitions) {
    if (!Expander.isSafeToExpandAt(P.StartExpr, InsertPt))
      return false;
    if (!Expander.isSafeToExpandAt(getClampedEndSCEV(P.EndExpr), InsertPt))
      return false;
  }
  return true;
}

// Drive the whole transform: set up scratch state and run each phase in order.
bool LoopSplit::split() {
  PHINode *Induction = L->getInductionVariable(*SE);
  assert(Induction && "split() requires prior legality analysis");
  if (getNumPartitions() < 2)
    return false;

  // Check expansion safety at the preheader terminator before any CFG change.
  Instruction *ExpandAt = L->getLoopPreheader()->getTerminator();
  SCEVExpander Expander(*SE, DEBUG_TYPE);
  if (!arePartitionBoundsSafeToExpand(Expander, ExpandAt)) {
    LLVM_DEBUG(dbgs() << DEBUG_TYPE
                      << ": partition bounds not safe to expand\n");
    return false;
  }

  SplitState S;
  // Partition 0 reuses the original loop; record its preheader/exit/guard up
  // front.
  PartitionInfo &P0 = Partitions[0];
  P0.Preheader = L->getLoopPreheader();
  P0.Exit = L->getExitBlock();
  P0.SubLoop = L;
  P0.IndPHI = Induction;
  S.OuterLoop = LI->getLoopFor(P0.Exit);
  S.Induction = Induction;
  S.OrigLoopID = L->getLoopID();

  splitFinalExit(S);
  buildEntryGuard(P0.Preheader, P0.GuardBlock, DT, LI);

  // Keep the expander and its cleaner alive for the whole transform: the bounds
  // it materializes are consumed by later phases. markResultUsed() below keeps
  // them; without it the cleaner reclaims them.
  SCEVExpanderCleaner ExpanderCleaner(Expander);
  expandPartitionBounds(S, Expander);
  clonePartitions(S);
  chainPartitions(S);
  ExpanderCleaner.markResultUsed();

  // The iteration space and the surrounding block structure both changed.
  SE->forgetLoop(L);
  SE->forgetBlockAndLoopDispositions();
  return true;
}

// Split the final exit off the loop exit block, so the original exit can serve
// as partition 0's dedicated exit and branch on into the guard chain.
void LoopSplit::splitFinalExit(SplitState &S) {
  BasicBlock *OrigExit = Partitions[0].Exit;

  // Splitting at begin() moves everything into FinalExit; the exit block has no
  // PHIs because legality analysis rejects escaping values. SplitBlock also
  // re-parents the dominator-tree children of the exit onto FinalExit.
  S.FinalExit = SplitBlock(OrigExit, OrigExit->begin(), DT, LI,
                           /*MSSAU=*/nullptr, "ls.final.exit");
}

// Insert the entry guard ahead of partition 0's preheader, updating the
// dominator tree and loop info. On return \p Preheader is the clean preheader
// and \p EntryGuard is the new guard block dominating the chain.
static void buildEntryGuard(BasicBlock *&Preheader, BasicBlock *&EntryGuard,
                            DominatorTree *DT, LoopInfo *LI) {
  // Split the preheader: the upper half becomes the guard dominating the chain,
  // the lower half a clean preheader.
  BasicBlock *NewPreheader =
      SplitBlock(Preheader, Preheader->getTerminator(), DT, LI);
  EntryGuard = Preheader;
  Preheader = NewPreheader;
  // Move the original preheader's name onto the new preheader, then name the
  // guard.
  Preheader->takeName(EntryGuard);
  EntryGuard->setName("ls.guard0");
}

// Materialize each partition's start and clamped end in the entry guard.
void LoopSplit::expandPartitionBounds(SplitState &S, SCEVExpander &Expander) {
  Type *IndTy = S.Induction->getType();
  Instruction *EntryGuardTerm = Partitions[0].GuardBlock->getTerminator();

  // Expand all partition bounds in the entry guard, which dominates the whole
  // chain (a skipped partition bypasses the original preheader).
  const unsigned N = getNumPartitions();
  for (unsigned I = 0; I < N; ++I) {
    PartitionInfo &P = Partitions[I];

    P.StartVal = Expander.expandCodeFor(P.StartExpr, IndTy, EntryGuardTerm);

    // Clamp the end to the induction end (min ascending, max descending) so a
    // short trip count keeps the last iteration in the right partition.
    const SCEV *ClampedEndSCEV = getClampedEndSCEV(P.EndExpr);
    P.SelEnd = Expander.expandCodeFor(ClampedEndSCEV, IndTy, EntryGuardTerm);
  }
}

// Clone each later partition's sub-loop and create its guard and exit blocks
// (partition 0 reuses the original loop).
void LoopSplit::clonePartitions(SplitState &S) {
  Function &F = *L->getHeader()->getParent();
  LLVMContext &Ctx = SE->getContext();

  const unsigned N = getNumPartitions();
  // Partition 0 reuses the original loop; clone the rest off its preheader.
  BasicBlock *OrigPreheader = Partitions[0].Preheader;

  for (unsigned I = 1; I < N; ++I) {
    PartitionInfo &P = Partitions[I];
    ValueToValueMapTy VMap;
    SmallVector<BasicBlock *, 8> ClonedBlocks;
    Loop *PL = cloneLoopWithPreheader(S.FinalExit, OrigPreheader, L, VMap,
                                      ".ls" + Twine(I), LI, DT, ClonedBlocks);
    remapInstructionsInBlocks(ClonedBlocks, VMap);
    BasicBlock *PHi = PL->getLoopPreheader();

    BasicBlock *Exiti =
        BasicBlock::Create(Ctx, "ls.exit" + Twine(I), &F, S.FinalExit);
    BasicBlock *Guardi =
        BasicBlock::Create(Ctx, "ls.guard" + Twine(I), &F, PHi);
    if (S.OuterLoop) {
      S.OuterLoop->addBasicBlockToLoop(Exiti, *LI);
      S.OuterLoop->addBasicBlockToLoop(Guardi, *LI);
    }
    // Placeholder terminators; both are re-pointed at the merge in pass 2.
    UncondBrInst::Create(S.FinalExit, Exiti);
    UncondBrInst::Create(S.FinalExit, Guardi);

    // Seed the clone's induction PHI with this partition's start value.
    auto *ClonedInduction = cast<PHINode>(VMap[S.Induction]);
    ClonedInduction->setIncomingValueForBlock(PHi, P.StartVal);

    P.GuardBlock = Guardi;
    P.Preheader = PHi;
    P.Exit = Exiti;
    P.SubLoop = PL;
    P.IndPHI = ClonedInduction;
  }
}

// Replace a partition's latch test so it iterates only within [start, SelEnd],
// re-point the exit edge, and carry the original branch weights over. See the
// file comment for why the test is the strict one on the PHI.
static void rewriteLatch(Loop *PL, PHINode *IndPHI, Value *SelEnd,
                         BasicBlock *Exit, bool Signed, bool Descending) {
  auto *Term = cast<CondBrInst>(PL->getLoopLatch()->getTerminator());
  auto *Cmp = cast<ICmpInst>(Term->getCondition());
  IRBuilder<> B(Cmp);
  // The bound was expanded in the induction type, which is the PHI's type.
  assert(SelEnd->getType() == IndPHI->getType() &&
         "latch operand type mismatch");
  ICmpInst::Predicate Pred = continuePredicate(Signed, Descending);
  Value *NewCmp = B.CreateICmp(Pred, IndPHI, SelEnd, "itr.chk");
  B.SetInsertPoint(Term);
  auto *NewBr = B.CreateCondBr(NewCmp, PL->getHeader(), Exit);
  // Carry the original latch's weights over, mapping by which successor stayed
  // in the loop.
  uint64_t TrueW, FalseW;
  if (extractBranchWeights(*Term, TrueW, FalseW)) {
    bool Succ0InLoop = PL->contains(Term->getSuccessor(0));
    setFittedBranchWeights(
        *NewBr, {Succ0InLoop ? TrueW : FalseW, Succ0InLoop ? FalseW : TrueW},
        /*IsExpected=*/false);
  }
  Term->eraseFromParent();
  if (Cmp->use_empty())
    Cmp->eraseFromParent();
}

// Attach a distinct !llvm.loop that inherits \p OrigLoopID's attributes.
static void assignPartitionLoopID(Loop *PL, MDNode *OrigLoopID) {
  if (!OrigLoopID)
    return;
  MDNode *NewLoopID = makePostTransformationMetadata(
      PL->getHeader()->getContext(), OrigLoopID, {}, {});
  PL->setLoopID(NewLoopID);
}

// Emit each partition's guard branch, clamp its latch, wire the partitions into
// a chain, and update the dominator tree.
void LoopSplit::chainPartitions(SplitState &S) {
  const ICmpInst::Predicate GuardPred =
      guardPredicate(InductionIsSigned, Descending);

  // Emit each guard, clamp each latch, and chain partitions; a skipped
  // partition falls through to the next guard.
  const unsigned N = getNumPartitions();
  IRBuilder<> B(SE->getContext());

  for (unsigned I = 0; I < N; ++I) {
    PartitionInfo &P = Partitions[I];
    // Where control goes when this partition is skipped or after it finishes:
    // the next partition's guard, or the final merge for the last partition.
    BasicBlock *MergeAfter =
        I + 1 == N ? S.FinalExit : Partitions[I + 1].GuardBlock;

    Instruction *GuardTerm = P.GuardBlock->getTerminator();
    B.SetInsertPoint(GuardTerm);
    // An empty partition needs no special case: its check is false and control
    // falls through. Emitting the branch either way keeps both CFG edges, so
    // the dominator tree stays consistent with one computed from scratch.
    Value *Enter = B.CreateICmp(GuardPred, P.StartVal, P.SelEnd, "itr.chk");
    auto *GuardBr = B.CreateCondBr(Enter, P.Preheader, MergeAfter);
    // New control flow with no source profile; record the weights as unknown
    // so profile-tracking passes are not misled.
    setExplicitlyUnknownBranchWeightsIfProfiled(*GuardBr, DEBUG_TYPE);
    GuardTerm->eraseFromParent();

    rewriteLatch(P.SubLoop, P.IndPHI, P.SelEnd, P.Exit, InductionIsSigned,
                 Descending);
    assignPartitionLoopID(P.SubLoop, S.OrigLoopID);
    P.Exit->getTerminator()->setSuccessor(0, MergeAfter);
  }

  // Patch the dominator tree directly: every partition is guarded, so a merge
  // target is dominated by the guard that can branch straight to it.
  for (unsigned I = 1; I < N; ++I) {
    PartitionInfo &Cur = Partitions[I];
    DT->addNewBlock(Cur.GuardBlock, Partitions[I - 1].GuardBlock);
    DT->changeImmediateDominator(Cur.Preheader, Cur.GuardBlock);
    DT->addNewBlock(Cur.Exit, Cur.SubLoop->getLoopLatch());
  }
  // The final exit is the last partition's merge target.
  DT->changeImmediateDominator(S.FinalExit, Partitions.back().GuardBlock);
}
