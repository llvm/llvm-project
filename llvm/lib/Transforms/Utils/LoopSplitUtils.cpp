//===- LoopSplitUtils.cpp - Split a loop's iteration space ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Splits a counted loop's iteration space into a chain of per-partition
// sub-loops. See LoopSplitUtils.h for the high-level usage guidelines.
//
// Structure produced for partitions [S0,E0], [S1,E1], ... where E is the loop's
// last iteration and each clamped end sel_i = min(E_i, E):
//
//   guard0:                            ; every S_i and sel_i is computed here
//     if (S0 <= sel0) goto preheader0 else goto guard1   ; default guard check
//   loop0: ...                         ; latch stops at sel0
//   exit0 -> guard1
//   guard1:
//     if (S1 <= sel1) goto preheader1 else goto guard2   ; default guard check
//   loop1: ...                         ; latch stops at sel1
//   exit1 -> guard2
//     ...
//   final.exit:                        ; merges every partition's live-outs
//
// Each guard holds the "S_i <= sel_i" check and skips an empty partition by
// falling through to the next guard. The check is replaced by an unconditional
// branch when a partition is proven empty (to the next guard) or the caller
// exempts it via avoidPartitionGuard() (to its preheader). All S_i/sel_i are
// materialized once in guard0; the end clamp keeps the "runs at least once"
// iteration in the right partition; live-outs are rebuilt one SSAUpdater each.
//
// A descending (step -1) loop uses the same structure mirrored: partitions run
// high-to-low and the empty test, clamp, and predicates flip (>=/>).
//
// Usage guidelines:
//  - Caller bounds must not wrap the induction type. The clamp absorbs a bound
//    past the runtime trip count, but a Start +/- offset that overshoots the
//    type extreme wraps in the bound arithmetic and cannot be repaired here.
//  - Bounds must be loop-invariant: they are expanded in guard0 (the preheader),
//    so a bound depending on a value defined inside the loop cannot be placed.
//  - The partitions must tile the original iteration space exactly -- same
//    iterations, same order -- so the split preserves program behaviour.
//  - A caller that drops a guard via avoidPartitionGuard() must itself ensure
//    that partition runs at least once, or the result is a spurious iteration.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/LoopSplitUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace llvm;

#define DEBUG_TYPE "loop-split-utils"

//===----------------------------------------------------------------------===//
// LoopSplitUtils - construction, partition list, induction analysis
//===----------------------------------------------------------------------===//

/// Per-split() scratch shared by the phase helpers; lives for one split() call.
struct LoopSplitUtils::SplitState {
  BasicBlock *OrigPreheader = nullptr; // also partition 0's preheader.
  BasicBlock *ExitBlock = nullptr;     // also partition 0's exit.
  BasicBlock *FinalExit = nullptr;     // where live-outs merge.
  BasicBlock *EntryGuard = nullptr;    // guard ahead of partition 0.
  Loop *OuterLoop = nullptr;           // parent of the new blocks, if any.

  /// A value that must be reconstructed after cloning because it is
  /// loop-carried (feeds a later partition), live-out (used after the loop), or
  /// both.
  struct EscapingValue {
    /// The value as it exists in partition 0 (the original).
    Value *Def = nullptr;
    /// The carried header PHI in partition 0, or null if \c Def needs no
    /// per-partition start value seeded.
    PHINode *CarriedHeaderPHI = nullptr;
    /// True if \c Def is used outside the loop and must be merged at the final
    /// exit.
    bool EscapesOutside = false;
    /// \c Def and \c CarriedHeaderPHI cloned into each partition (index 0 is the
    /// original; \c PerPartitionPHI[0] is unused).
    SmallVector<Value *, 4> PerPartitionDef;
    SmallVector<PHINode *, 4> PerPartitionPHI;
  };

  /// Values that must survive across partitions (carried and/or live-out).
  SmallVector<EscapingValue, 8> Escaping;

  EscapingValue &addEscaping(Value *Def) {
    Escaping.emplace_back();
    Escaping.back().Def = Def;
    return Escaping.back();
  }
};

// Record a new partition with the given inclusive iteration range.
void LoopSplitUtils::addPartition(const SCEV *Start, const SCEV *End) {
  PartitionInfo &P = Partitions.emplace_back();
  P.StartExpr = Start;
  P.EndExpr = End;
}

// Mark a partition so split() emits no entry guard for it.
void LoopSplitUtils::avoidPartitionGuard(unsigned PartitionIndex) {
  assert(PartitionIndex < Partitions.size() &&
         "avoidPartitionGuard() called for an unknown partition");
  Partitions[PartitionIndex].Guarded = false;
}

// Return a partition's original-to-clone map, or null if it has none.
const ValueToValueMapTy *
LoopSplitUtils::getPartitionValueMap(unsigned PartitionIndex) const {
  if (PartitionIndex >= Partitions.size())
    return nullptr;
  return Partitions[PartitionIndex].VMap.get();
}

// Look up the counterpart of an original value in a given partition.
Value *LoopSplitUtils::getPartitionValue(const Value *V,
                                         unsigned PartitionIndex) const {
  assert(PartitionIndex < getNumPartitions() && "partition index out of range");
  // Partition 0 reuses the original loop: every value maps to itself.
  if (PartitionIndex == 0)
    return const_cast<Value *>(V);
  const ValueToValueMapTy *VMap = getPartitionValueMap(PartitionIndex);
  if (!VMap)
    return nullptr;
  ValueToValueMapTy::const_iterator It = VMap->find(V);
  return It != VMap->end() ? It->second : nullptr;
}

// Find the induction variable and the latch operand it is compared against;
// returns the induction's add-recurrence, or null if the loop is unsuitable.
const SCEVAddRecExpr *LoopSplitUtils::analyzeInduction() {
  // The loop must exit on an integer compare living in the latch.
  LatchCmp = L->getLatchCmpInst();
  if (!LatchCmp || LatchCmp->getParent() != L->getLoopLatch())
    return nullptr;

  // SCEV's induction variable, restricted to a unit-step affine recurrence.
  Induction = L->getInductionVariable(*SE);
  if (!Induction)
    return nullptr;
  const auto *AR = dyn_cast<SCEVAddRecExpr>(SE->getSCEV(Induction));
  if (!AR || !AR->isAffine())
    return nullptr;
  // Accept a unit step in either direction: +1 (ascending) or -1 (descending).
  const auto *Step = dyn_cast<SCEVConstant>(AR->getStepRecurrence(*SE));
  if (!Step || !(Step->getValue()->isOne() || Step->getValue()->isMinusOne()))
    return nullptr;
  InductionIsDescending = Step->getValue()->isMinusOne();

  // The induction's "next" value (i + 1), produced in the latch.
  auto *StepInst = dyn_cast<Instruction>(
      Induction->getIncomingValueForBlock(L->getLoopLatch()));
  if (!StepInst)
    return nullptr;

  // Select the compare operand that is the induction (PHI or its step).
  if (LatchCmp->getOperand(0) == Induction ||
      LatchCmp->getOperand(0) == StepInst)
    LatchIndOperand = LatchCmp->getOperand(0);
  else if (LatchCmp->getOperand(1) == Induction ||
           LatchCmp->getOperand(1) == StepInst)
    LatchIndOperand = LatchCmp->getOperand(1);
  else
    return nullptr;
  LatchUsesInductionPHI = (LatchIndOperand == Induction);
  return AR;
}

// Decide whether the iteration ordering is signed or unsigned.
bool LoopSplitUtils::computeSignedness(const SCEVAddRecExpr *IndAR) {
  ICmpInst::Predicate P = LatchCmp->getPredicate();
  // Relational predicate gives the ordering; for eq/ne use the no-wrap flags.
  if (ICmpInst::isSigned(P))
    InductionIsSigned = true;
  else if (ICmpInst::isUnsigned(P))
    InductionIsSigned = false;
  else if (IndAR->hasNoSignedWrap())
    InductionIsSigned = true;
  else if (IndAR->hasNoUnsignedWrap())
    InductionIsSigned = false;
  else {
    LLVM_DEBUG(dbgs() << "LS: cannot prove iteration ordering signedness\n");
    return false;
  }
  return true;
}

// Check every structural precondition and record the induction analysis.
bool LoopSplitUtils::isLegal() {
  if (!L->getLoopPreheader() || !L->getLoopLatch()) {
    LLVM_DEBUG(dbgs() << "LS: missing preheader/latch\n");
    return false;
  }
  if (!L->getExitingBlock() || !L->getExitBlock() ||
      L->getExitingBlock() != L->getLoopLatch()) {
    LLVM_DEBUG(dbgs() << "LS: not a bottom-tested single-exit loop\n");
    return false;
  }
  if (!L->isLCSSAForm(*DT)) {
    LLVM_DEBUG(dbgs() << "LS: loop is not in LCSSA form\n");
    return false;
  }

  // A computable backedge-taken count fixes the iteration space we rebuild.
  const SCEV *BTC = SE->getBackedgeTakenCount(L);
  if (isa<SCEVCouldNotCompute>(BTC)) {
    LLVM_DEBUG(dbgs() << "LS: no computable trip count\n");
    return false;
  }

  const SCEVAddRecExpr *IndAR = analyzeInduction();
  if (!IndAR) {
    LLVM_DEBUG(dbgs() << "LS: no unique unit-step integer induction\n");
    return false;
  }

  if (!computeSignedness(IndAR))
    return false;

  InductionEnd = IndAR->evaluateAtIteration(BTC, *SE);
  // Start and end must share the induction type; reject any width mismatch.
  if (InductionEnd->getType() != IndAR->getStart()->getType()) {
    LLVM_DEBUG(dbgs() << "LS: induction end/start type mismatch\n");
    return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Transform
//===----------------------------------------------------------------------===//

// Clone of \p V from \p VMap, or \p V itself if it was not cloned.
static Value *remapValue(ValueToValueMapTy &VMap, Value *V) {
  auto It = VMap.find(V);
  if (It == VMap.end())
    return V;
  return It->second;
}

// Latch "keep iterating" predicate (ascending </<=, descending >/>=); inclusive
// when the latch compares the step value, strict when it compares the PHI.
static ICmpInst::Predicate continuePredicate(bool Signed, bool Descending,
                                             bool Inclusive) {
  if (Descending)
    return Inclusive ? (Signed ? ICmpInst::ICMP_SGE : ICmpInst::ICMP_UGE)
                     : (Signed ? ICmpInst::ICMP_SGT : ICmpInst::ICMP_UGT);
  return Inclusive ? (Signed ? ICmpInst::ICMP_SLE : ICmpInst::ICMP_ULE)
                   : (Signed ? ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT);
}

// Guard "enter this partition" predicate: Start <= sel ascending, Start >= sel
// descending.
static ICmpInst::Predicate guardPredicate(bool Signed, bool Descending) {
  if (Descending)
    return Signed ? ICmpInst::ICMP_SGE : ICmpInst::ICMP_UGE;
  return Signed ? ICmpInst::ICMP_SLE : ICmpInst::ICMP_ULE;
}

// Drive the whole transform: set up scratch state and run each phase in order.
bool LoopSplitUtils::split() {
  assert(Induction && "split() requires a successful isLegal()");
  if (getNumPartitions() < 2)
    return false;

  if (!L->hasDedicatedExits() &&
      !formDedicatedExitBlocks(L, DT, LI, /*MSSAU=*/nullptr,
                               /*PreserveLCSSA=*/true))
    return false;

  SplitState S;
  S.OrigPreheader = L->getLoopPreheader();
  S.ExitBlock = L->getExitBlock();
  S.OuterLoop = LI->getLoopFor(S.ExitBlock);

  collectEscapingValues(S);
  buildEntryGuard(S);
  expandPartitionBounds(S);
  clonePartitions(S);
  chainPartitions(S);
  reconstructSSA(S);
  return true;
}

// Find loop-carried and live-out values and split the final-exit block off the
// loop exit, seeding partition 0's slots for each escaping value.
void LoopSplitUtils::collectEscapingValues(SplitState &S) {
  BasicBlock *Latch = L->getLoopLatch();

  // Separate FinalExit from the loop exit. Split at begin() so the LCSSA PHIs
  // move into FinalExit (SplitBlock would advance past them).
  S.FinalExit =
      S.ExitBlock->splitBasicBlock(S.ExitBlock->begin(), "ls.final.exit");
  if (S.OuterLoop)
    S.OuterLoop->addBasicBlockToLoop(S.FinalExit, *LI);
  // splitBasicBlock does not update the dominator tree; the new exit's sole
  // predecessor is the original exit block.
  DT->addNewBlock(S.FinalExit, S.ExitBlock);

  // (1) Carried values: each non-induction header PHI whose backedge value
  // differs from its initial value must resume in later partitions.
  DenseMap<Value *, unsigned> CarriedDefToEscapingIdx;
  for (PHINode &HeaderPHI : L->getHeader()->phis()) {
    if (&HeaderPHI == Induction)
      continue;
    Value *CarriedValue = HeaderPHI.getIncomingValueForBlock(Latch);
    Value *InitialValue = HeaderPHI.getIncomingValueForBlock(S.OrigPreheader);
    if (CarriedValue == InitialValue)
      continue; // invariant and equal to the initial value: nothing to carry.
    auto &EV = S.addEscaping(CarriedValue);
    EV.CarriedHeaderPHI = &HeaderPHI;
    // Track in-loop carried defs so a matching live-out in (2) merges onto
    // them.
    if (auto *CarriedInst = dyn_cast<Instruction>(CarriedValue);
        CarriedInst && L->contains(CarriedInst))
      CarriedDefToEscapingIdx[CarriedValue] = S.Escaping.size() - 1;
  }

  // (2) Live-outs: dissolve each LCSSA PHI into its def and mark it escaping,
  // merging onto a pass-(1) entry if also carried. Uses are repaired later.
  for (PHINode &LCSSAPhi : make_early_inc_range(S.FinalExit->phis())) {
    assert(LCSSAPhi.getNumIncomingValues() == 1 &&
           "exit block not in LCSSA form");
    Value *LiveOutDef = LCSSAPhi.getIncomingValue(0);
    auto Existing = CarriedDefToEscapingIdx.find(LiveOutDef);
    auto &EV = Existing != CarriedDefToEscapingIdx.end()
                   ? S.Escaping[Existing->second]
                   : S.addEscaping(LiveOutDef);
    EV.EscapesOutside = true;
    LCSSAPhi.replaceAllUsesWith(LiveOutDef);
    LCSSAPhi.eraseFromParent();
  }

  // Seed partition 0 with the originals; later partitions are filled when
  // cloned.
  const unsigned N = getNumPartitions();
  for (auto &EV : S.Escaping) {
    EV.PerPartitionDef.assign(N, nullptr);
    EV.PerPartitionPHI.assign(N, nullptr);
    EV.PerPartitionDef[0] = EV.Def;
    EV.PerPartitionPHI[0] = EV.CarriedHeaderPHI;
  }
}

// Insert the entry guard ahead of partition 0's preheader and update the
// dominator tree.
void LoopSplitUtils::buildEntryGuard(SplitState &S) {
  // Split the preheader: the upper half becomes the guard dominating the chain,
  // the lower half a clean preheader.
  std::string PreheaderName = S.OrigPreheader->getName().str();
  BasicBlock *NewPreheader =
      SplitBlock(S.OrigPreheader, S.OrigPreheader->getTerminator(), DT, LI);
  S.EntryGuard = S.OrigPreheader;
  S.OrigPreheader = NewPreheader;
  S.EntryGuard->setName("ls.guard0");
  S.OrigPreheader->setName(PreheaderName);
}

// Materialize each partition's start and clamped end in the entry guard and
// flag the partitions that are provably empty at compile time.
void LoopSplitUtils::expandPartitionBounds(SplitState &S) {
  Type *IndTy = Induction->getType();
  Instruction *EntryGuardTerm = S.EntryGuard->getTerminator();
  SCEVExpander Expander(*SE, "ls");

  // Expand all partition bounds in the entry guard, which dominates the whole
  // chain (a skipped partition bypasses the original preheader).
  const unsigned N = getNumPartitions();
  for (unsigned I = 0; I < N; ++I) {
    PartitionInfo &P = Partitions[I];

    // Provably empty when Start overshoots End by exactly one step. Compile-time
    // only: a runtime overshoot wraps at the type extreme and would falsely enter.
    const SCEV *PartWidth = SE->getMinusSCEV(P.StartExpr, P.EndExpr);
    if (auto *PartWidthConst = dyn_cast<SCEVConstant>(PartWidth)) {
      const APInt &W = PartWidthConst->getAPInt();
      P.Empty = InductionIsDescending ? W.isAllOnes() : W.isOne();
    }

    P.StartVal = Expander.expandCodeFor(P.StartExpr, IndTy, EntryGuardTerm);

    // Clamp the end to the induction end (min ascending, max descending) so a
    // short trip count keeps the last iteration in the right partition.
    const SCEV *ClampedEndSCEV;
    if (InductionIsDescending)
      ClampedEndSCEV = InductionIsSigned ? SE->getSMaxExpr(P.EndExpr, InductionEnd)
                                         : SE->getUMaxExpr(P.EndExpr, InductionEnd);
    else
      ClampedEndSCEV = InductionIsSigned ? SE->getSMinExpr(P.EndExpr, InductionEnd)
                                         : SE->getUMinExpr(P.EndExpr, InductionEnd);
    P.SelEnd = Expander.expandCodeFor(ClampedEndSCEV, IndTy, EntryGuardTerm);
  }
}

// Pass 1: clone each later partition's sub-loop and create its guard and exit
// blocks (partition 0 reuses the original loop).
void LoopSplitUtils::clonePartitions(SplitState &S) {
  Function &F = *L->getHeader()->getParent();
  LLVMContext &Ctx = F.getContext();

  const unsigned N = getNumPartitions();
  // Partition 0 reuses the original blocks; its map stays null (identity).
  PartitionInfo &P0 = Partitions[0];
  P0.GuardBlock = S.EntryGuard;
  P0.Preheader = S.OrigPreheader;
  P0.Exit = S.ExitBlock;
  P0.SubLoop = L;
  P0.LatchIndOp = LatchIndOperand;

  for (unsigned I = 1; I < N; ++I) {
    PartitionInfo &P = Partitions[I];
    // Persist this partition's original-to-clone map so callers can later
    // query the counterpart of an original loop value (getPartitionValue()).
    P.VMap = std::make_unique<ValueToValueMapTy>();
    ValueToValueMapTy &VMap = *P.VMap;
    SmallVector<BasicBlock *, 8> ClonedBlocks;
    Loop *PL = cloneLoopWithPreheader(S.FinalExit, S.OrigPreheader, L, VMap,
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
    IRBuilder<>(Exiti).CreateBr(S.FinalExit);
    IRBuilder<>(Guardi).CreateBr(S.FinalExit);

    // Seed the clone's induction PHI with this partition's start value.
    auto *ClonedInduction = cast<PHINode>(VMap[Induction]);
    ClonedInduction->setIncomingValueForBlock(PHi, P.StartVal);

    P.GuardBlock = Guardi;
    P.Preheader = PHi;
    P.Exit = Exiti;
    P.SubLoop = PL;
    P.LatchIndOp = remapValue(VMap, LatchIndOperand);

    for (auto &EV : S.Escaping) {
      EV.PerPartitionDef[I] = remapValue(VMap, EV.Def);
      if (EV.CarriedHeaderPHI)
        EV.PerPartitionPHI[I] = cast<PHINode>(VMap[EV.CarriedHeaderPHI]);
    }
  }
}

// Replace a partition's latch test so it iterates only within [start, SelEnd].
void LoopSplitUtils::rewriteLatch(Loop *PL, Value *IndOp, Value *SelEnd,
                                  BasicBlock *Exit) {
  auto *Term = cast<CondBrInst>(PL->getLoopLatch()->getTerminator());
  auto *Cmp = cast<ICmpInst>(Term->getCondition());
  IRBuilder<> B(Cmp);
  Value *Bound = SelEnd;
  if (Bound->getType() != IndOp->getType())
    Bound = B.CreateIntCast(Bound, IndOp->getType(), InductionIsSigned);
  // Strict when the PHI itself is compared, inclusive when the step value is.
  ICmpInst::Predicate Pred = continuePredicate(
      InductionIsSigned, InductionIsDescending, /*Inclusive=*/!LatchUsesInductionPHI);
  Value *NewCmp = B.CreateICmp(Pred, IndOp, Bound, "itr.chk");
  B.SetInsertPoint(Term);
  B.CreateCondBr(NewCmp, PL->getHeader(), Exit);
  Term->eraseFromParent();
  if (Cmp->use_empty())
    Cmp->eraseFromParent();
}

// Pass 2: emit each partition's guard branch, clamp its latch, wire the
// partitions into a chain, and update the dominator tree.
void LoopSplitUtils::chainPartitions(SplitState &S) {
  const ICmpInst::Predicate GuardPred =
      guardPredicate(InductionIsSigned, InductionIsDescending);

  // Emit each guard, clamp each latch, and chain partitions; a skipped
  // partition falls through to the next guard.
  const unsigned N = getNumPartitions();

  // Enters unconditionally when the caller opted out of the guard and the
  // partition is not provably empty; a proven-empty partition always skips.
  auto EntersUnconditionally = [](const PartitionInfo &P) {
    return !P.Empty && !P.Guarded;
  };

  // Where control goes when partition Idx is skipped or after it finishes: the
  // next partition's guard, or the final merge block for the last partition.
  auto MergeTargetAfter = [&](unsigned Idx) -> BasicBlock * {
    bool IsLastPartition = Idx + 1 == N;
    return IsLastPartition ? S.FinalExit : Partitions[Idx + 1].GuardBlock;
  };

  for (unsigned I = 0; I < N; ++I) {
    PartitionInfo &P = Partitions[I];
    BasicBlock *MergeAfter = MergeTargetAfter(I);

    Instruction *GuardTerm = P.GuardBlock->getTerminator();
    if (P.Empty) {
      // Provably empty: skip to the next partition. The unreachable loop body
      // is removed by later passes.
      IRBuilder<>(GuardTerm).CreateBr(MergeAfter);
    } else if (!P.Guarded) {
      // Caller guaranteed at least one iteration: enter unconditionally. The
      // skip edge to MergeAfter is omitted (see DT update below).
      IRBuilder<>(GuardTerm).CreateBr(P.Preheader);
    } else {
      IRBuilder<> B(GuardTerm);
      Value *Enter = B.CreateICmp(GuardPred, P.StartVal, P.SelEnd, "itr.chk");
      B.CreateCondBr(Enter, P.Preheader, MergeAfter);
    }
    GuardTerm->eraseFromParent();

    rewriteLatch(P.SubLoop, P.LatchIndOp, P.SelEnd, P.Exit);
    P.Exit->getTerminator()->setSuccessor(0, MergeAfter);
  }

  // Patch the dominator tree directly: a merge target is dominated by the prior
  // partition's exit when it enters unconditionally, otherwise by its guard.
  auto MergeTargetIDom = [&](const PartitionInfo &P) {
    return EntersUnconditionally(P) ? P.Exit : P.GuardBlock;
  };

  for (unsigned I = 1; I < N; ++I) {
    PartitionInfo &Prev = Partitions[I - 1];
    PartitionInfo &Cur = Partitions[I];
    DT->addNewBlock(Cur.GuardBlock, MergeTargetIDom(Prev));
    DT->changeImmediateDominator(Cur.Preheader, Cur.GuardBlock);
    DT->addNewBlock(Cur.Exit, Cur.SubLoop->getLoopLatch());
  }
  // The final exit is the last partition's merge target.
  DT->changeImmediateDominator(S.FinalExit, MergeTargetIDom(Partitions.back()));
}

// Rebuild SSA for every escaping value, repairing outside uses and seeding each
// later partition's carried PHI, using one SSAUpdater per value.
void LoopSplitUtils::reconstructSSA(SplitState &S) {
  const unsigned N = getNumPartitions();
  for (auto &EV : S.Escaping) {
    SSAUpdater Updater;
    Updater.Initialize(EV.Def->getType(), EV.Def->getName());

    // Value before any partition runs: carried PHI's initial value, else
    // poison.
    Value *Init =
        EV.CarriedHeaderPHI
            ? EV.CarriedHeaderPHI->getIncomingValueForBlock(S.OrigPreheader)
            : PoisonValue::get(EV.Def->getType());
    Updater.AddAvailableValue(S.EntryGuard, Init);
    for (unsigned I = 0; I < N; ++I)
      Updater.AddAvailableValue(Partitions[I].Exit, EV.PerPartitionDef[I]);

    // Repair outside uses before the carried-PHI seeds add new in-clone uses.
    if (EV.EscapesOutside) {
      SmallVector<Use *, 8> OutsideUses;
      for (Use &U : EV.Def->uses())
        if (auto *User = dyn_cast<Instruction>(U.getUser()))
          if (!L->contains(User))
            OutsideUses.push_back(&U);
      for (Use *U : OutsideUses)
        Updater.RewriteUse(*U);
    }

    // Seed each later partition's carried PHI from the preceding partitions.
    if (EV.CarriedHeaderPHI)
      for (unsigned I = 1; I < N; ++I) {
        PHINode *CarriedPHI = EV.PerPartitionPHI[I];
        int PreheaderEntryIdx =
            CarriedPHI->getBasicBlockIndex(Partitions[I].Preheader);
        assert(PreheaderEntryIdx >= 0 && "cloned preheader edge missing");
        Updater.RewriteUse(CarriedPHI->getOperandUse(PreheaderEntryIdx));
      }
  }
}
