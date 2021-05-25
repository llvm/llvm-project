//===- LoopDeletion.cpp - Dead Loop Deletion Pass ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Dead Loop Deletion Pass. This pass is responsible
// for eliminating loops with non-infinite computable trip counts that have no
// side effects or volatile instructions, and do not contribute to the
// computation of the function's return value.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/LoopDeletion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

using namespace llvm;

#define DEBUG_TYPE "loop-delete"

STATISTIC(NumDeleted, "Number of loops deleted");

enum class LoopDeletionResult {
  Unmodified,
  Modified,
  Deleted,
};

static LoopDeletionResult merge(LoopDeletionResult A, LoopDeletionResult B) {
  if (A == LoopDeletionResult::Deleted || B == LoopDeletionResult::Deleted)
    return LoopDeletionResult::Deleted;
  if (A == LoopDeletionResult::Modified || B == LoopDeletionResult::Modified)
    return LoopDeletionResult::Modified;
  return LoopDeletionResult::Unmodified;
}

/// Determines if a loop is dead.
///
/// This assumes that we've already checked for unique exit and exiting blocks,
/// and that the code is in LCSSA form.
static bool isLoopDead(Loop *L, ScalarEvolution &SE,
                       SmallVectorImpl<BasicBlock *> &ExitingBlocks,
                       BasicBlock *ExitBlock, bool &Changed,
                       BasicBlock *Preheader) {
  // Make sure that all PHI entries coming from the loop are loop invariant.
  // Because the code is in LCSSA form, any values used outside of the loop
  // must pass through a PHI in the exit block, meaning that this check is
  // sufficient to guarantee that no loop-variant values are used outside
  // of the loop.
  bool AllEntriesInvariant = true;
  bool AllOutgoingValuesSame = true;
  if (!L->hasNoExitBlocks()) {
    for (PHINode &P : ExitBlock->phis()) {
      Value *incoming = P.getIncomingValueForBlock(ExitingBlocks[0]);

      // Make sure all exiting blocks produce the same incoming value for the
      // block. If there are different incoming values for different exiting
      // blocks, then it is impossible to statically determine which value
      // should be used.
      AllOutgoingValuesSame =
          all_of(makeArrayRef(ExitingBlocks).slice(1), [&](BasicBlock *BB) {
            return incoming == P.getIncomingValueForBlock(BB);
          });

      if (!AllOutgoingValuesSame)
        break;

      if (Instruction *I = dyn_cast<Instruction>(incoming))
        if (!L->makeLoopInvariant(I, Changed, Preheader->getTerminator())) {
          AllEntriesInvariant = false;
          break;
        }
    }
  }

  if (Changed)
    SE.forgetLoopDispositions(L);

  if (!AllEntriesInvariant || !AllOutgoingValuesSame)
    return false;

  // Make sure that no instructions in the block have potential side-effects.
  // This includes instructions that could write to memory, and loads that are
  // marked volatile.
  for (auto &I : L->blocks())
    if (any_of(*I, [](Instruction &I) {
          return I.mayHaveSideEffects() && !I.isDroppable();
        }))
      return false;
  return true;
}

/// This function returns true if there is no viable path from the
/// entry block to the header of \p L. Right now, it only does
/// a local search to save compile time.
static bool isLoopNeverExecuted(Loop *L) {
  using namespace PatternMatch;

  auto *Preheader = L->getLoopPreheader();
  // TODO: We can relax this constraint, since we just need a loop
  // predecessor.
  assert(Preheader && "Needs preheader!");

  if (Preheader->isEntryBlock())
    return false;
  // All predecessors of the preheader should have a constant conditional
  // branch, with the loop's preheader as not-taken.
  for (auto *Pred: predecessors(Preheader)) {
    BasicBlock *Taken, *NotTaken;
    ConstantInt *Cond;
    if (!match(Pred->getTerminator(),
               m_Br(m_ConstantInt(Cond), Taken, NotTaken)))
      return false;
    if (!Cond->getZExtValue())
      std::swap(Taken, NotTaken);
    if (Taken == Preheader)
      return false;
  }
  assert(!pred_empty(Preheader) &&
         "Preheader should have predecessors at this point!");
  // All the predecessors have the loop preheader as not-taken target.
  return true;
}

static const SCEV *
getSCEVOnFirstIteration(Value *V, Loop *L, ScalarEvolution &SE,
                        DenseMap<Value *, const SCEV *> &FirstIterSCEV) {
  // Fist, check in cache.
  auto Existing = FirstIterSCEV.find(V);
  if (Existing != FirstIterSCEV.end())
    return Existing->second;
  const SCEV *S = nullptr;
  // TODO: Once ScalarEvolution supports getValueOnNthIteration for anything
  // else but AddRecs, it's a good use case for it. So far, just consider some
  // simple cases, like arithmetic operations.
  Value *LHS, *RHS;
  using namespace PatternMatch;
  if (match(V, m_Add(m_Value(LHS), m_Value(RHS)))) {
    const SCEV *LHSS = getSCEVOnFirstIteration(LHS, L, SE, FirstIterSCEV);
    const SCEV *RHSS = getSCEVOnFirstIteration(RHS, L, SE, FirstIterSCEV);
    S = SE.getAddExpr(LHSS, RHSS);
  } else if (match(V, m_Sub(m_Value(LHS), m_Value(RHS)))) {
    const SCEV *LHSS = getSCEVOnFirstIteration(LHS, L, SE, FirstIterSCEV);
    const SCEV *RHSS = getSCEVOnFirstIteration(RHS, L, SE, FirstIterSCEV);
    S = SE.getMinusSCEV(LHSS, RHSS);
  } else if (match(V, m_Mul(m_Value(LHS), m_Value(RHS)))) {
    const SCEV *LHSS = getSCEVOnFirstIteration(LHS, L, SE, FirstIterSCEV);
    const SCEV *RHSS = getSCEVOnFirstIteration(RHS, L, SE, FirstIterSCEV);
    S = SE.getMulExpr(LHSS, RHSS);
  } else
    S = SE.getSCEV(V);
  assert(S && "Case not handled?");
  FirstIterSCEV[V] = S;
  return S;
}

// Try to prove that one of conditions that dominates the latch must exit on 1st
// iteration.
static bool canProveExitOnFirstIteration(Loop *L, DominatorTree &DT,
                                         ScalarEvolution &SE, LoopInfo &LI) {
  BasicBlock *Latch = L->getLoopLatch();

  if (!Latch)
    return false;

  LoopBlocksRPO RPOT(L);
  RPOT.perform(&LI);

  BasicBlock *Header = L->getHeader();
  // Blocks that are reachable on the 1st iteration.
  SmallPtrSet<BasicBlock *, 4> LiveBlocks;
  // Edges that are reachable on the 1st iteration.
  DenseSet<BasicBlockEdge> LiveEdges;
  LiveBlocks.insert(L->getHeader());

  auto MarkLiveEdge = [&](BasicBlock *From, BasicBlock *To) {
    assert(LiveBlocks.count(From) && "Must be live!");
    LiveBlocks.insert(To);
    LiveEdges.insert({ From, To });
  };

  auto MarkAllSuccessorsLive = [&](BasicBlock *BB) {
    for (auto *Succ : successors(BB))
      MarkLiveEdge(BB, Succ);
  };

  // Check if there is only one predecessor on 1st iteration. Note that because
  // we iterate in RPOT, we have already visited all its (non-latch)
  // predecessors.
  auto GetSolePredecessorOnFirstIteration = [&](BasicBlock * BB)->BasicBlock * {
    if (BB == Header)
      return L->getLoopPredecessor();
    BasicBlock *OnlyPred = nullptr;
    for (auto *Pred : predecessors(BB))
      if (OnlyPred != Pred && LiveEdges.count({ Pred, BB })) {
        // 2 live preds.
        if (OnlyPred)
          return nullptr;
        OnlyPred = Pred;
      }

    assert(OnlyPred && "No live predecessors?");
    return OnlyPred;
  };
  DenseMap<Value *, const SCEV *> FirstIterSCEV;

  // Use the following algorithm to prove we never take the latch on the 1st
  // iteration:
  // 1. Traverse in topological order, so that whenever we visit a block, all
  //    its predecessors are already visited.
  // 2. If we can prove that the block may have only 1 predecessor on the 1st
  //    iteration, map all its phis onto input from this predecessor.
  // 3a. If we can prove which successor of out block is taken on the 1st
  //     iteration, mark this successor live.
  // 3b. If we cannot prove it, conservatively assume that all successors are
  //     live.
  for (auto *BB : RPOT) {
    // This block is not reachable on the 1st iterations.
    if (!LiveBlocks.count(BB))
      continue;

    // Skip inner loops.
    if (LI.getLoopFor(BB) != L) {
      MarkAllSuccessorsLive(BB);
      continue;
    }

    // If this block has only one live pred, map its phis onto their SCEVs.
    if (auto *OnlyPred = GetSolePredecessorOnFirstIteration(BB))
      for (auto &PN : BB->phis()) {
        if (!SE.isSCEVable(PN.getType()))
          continue;
        auto *Incoming = PN.getIncomingValueForBlock(OnlyPred);
        if (DT.dominates(Incoming, BB->getTerminator())) {
          const SCEV *IncSCEV =
              getSCEVOnFirstIteration(Incoming, L, SE, FirstIterSCEV);
          FirstIterSCEV[&PN] = IncSCEV;
        }
      }

    using namespace PatternMatch;
    ICmpInst::Predicate Pred;
    Value *LHS, *RHS;
    const BasicBlock *IfTrue, *IfFalse;
    // TODO: Handle switches.
    if (!match(BB->getTerminator(),
               m_Br(m_ICmp(Pred, m_Value(LHS), m_Value(RHS)),
                    m_BasicBlock(IfTrue), m_BasicBlock(IfFalse)))) {
      MarkAllSuccessorsLive(BB);
      continue;
    }

    if (!SE.isSCEVable(LHS->getType())) {
      MarkAllSuccessorsLive(BB);
      continue;
    }

    // Can we prove constant true or false for this condition?
    const SCEV *LHSS = getSCEVOnFirstIteration(LHS, L, SE, FirstIterSCEV);
    const SCEV *RHSS = getSCEVOnFirstIteration(RHS, L, SE, FirstIterSCEV);
    if (SE.isKnownPredicateAt(Pred, LHSS, RHSS, BB->getTerminator()))
      MarkLiveEdge(BB, BB->getTerminator()->getSuccessor(0));
    else if (SE.isKnownPredicateAt(ICmpInst::getInversePredicate(Pred), LHSS,
                                   RHSS, BB->getTerminator()))
      MarkLiveEdge(BB, BB->getTerminator()->getSuccessor(1));
    else
      MarkAllSuccessorsLive(BB);
  }

  // We can break the latch if it wasn't live.
  return !LiveEdges.count({ Latch, Header });
}

/// If we can prove the backedge is untaken, remove it.  This destroys the
/// loop, but leaves the (now trivially loop invariant) control flow and
/// side effects (if any) in place.
static LoopDeletionResult
breakBackedgeIfNotTaken(Loop *L, DominatorTree &DT, ScalarEvolution &SE,
                        LoopInfo &LI, MemorySSA *MSSA,
                        OptimizationRemarkEmitter &ORE) {
  assert(L->isLCSSAForm(DT) && "Expected LCSSA!");

  if (!L->getLoopLatch())
    return LoopDeletionResult::Unmodified;

  auto *BTC = SE.getBackedgeTakenCount(L);
  if (!BTC->isZero() && !canProveExitOnFirstIteration(L, DT, SE, LI))
    return LoopDeletionResult::Unmodified;

  breakLoopBackedge(L, DT, SE, LI, MSSA);
  return LoopDeletionResult::Deleted;
}

/// Remove a loop if it is dead.
///
/// A loop is considered dead either if it does not impact the observable
/// behavior of the program other than finite running time, or if it is
/// required to make progress by an attribute such as 'mustprogress' or
/// 'llvm.loop.mustprogress' and does not make any. This may remove
/// infinite loops that have been required to make progress.
///
/// This entire process relies pretty heavily on LoopSimplify form and LCSSA in
/// order to make various safety checks work.
///
/// \returns true if any changes were made. This may mutate the loop even if it
/// is unable to delete it due to hoisting trivially loop invariant
/// instructions out of the loop.
static LoopDeletionResult deleteLoopIfDead(Loop *L, DominatorTree &DT,
                                           ScalarEvolution &SE, LoopInfo &LI,
                                           MemorySSA *MSSA,
                                           OptimizationRemarkEmitter &ORE) {
  assert(L->isLCSSAForm(DT) && "Expected LCSSA!");

  // We can only remove the loop if there is a preheader that we can branch from
  // after removing it. Also, if LoopSimplify form is not available, stay out
  // of trouble.
  BasicBlock *Preheader = L->getLoopPreheader();
  if (!Preheader || !L->hasDedicatedExits()) {
    LLVM_DEBUG(
        dbgs()
        << "Deletion requires Loop with preheader and dedicated exits.\n");
    return LoopDeletionResult::Unmodified;
  }

  BasicBlock *ExitBlock = L->getUniqueExitBlock();

  if (ExitBlock && isLoopNeverExecuted(L)) {
    LLVM_DEBUG(dbgs() << "Loop is proven to never execute, delete it!");
    // We need to forget the loop before setting the incoming values of the exit
    // phis to undef, so we properly invalidate the SCEV expressions for those
    // phis.
    SE.forgetLoop(L);
    // Set incoming value to undef for phi nodes in the exit block.
    for (PHINode &P : ExitBlock->phis()) {
      std::fill(P.incoming_values().begin(), P.incoming_values().end(),
                UndefValue::get(P.getType()));
    }
    ORE.emit([&]() {
      return OptimizationRemark(DEBUG_TYPE, "NeverExecutes", L->getStartLoc(),
                                L->getHeader())
             << "Loop deleted because it never executes";
    });
    deleteDeadLoop(L, &DT, &SE, &LI, MSSA);
    ++NumDeleted;
    return LoopDeletionResult::Deleted;
  }

  // The remaining checks below are for a loop being dead because all statements
  // in the loop are invariant.
  SmallVector<BasicBlock *, 4> ExitingBlocks;
  L->getExitingBlocks(ExitingBlocks);

  // We require that the loop has at most one exit block. Otherwise, we'd be in
  // the situation of needing to be able to solve statically which exit block
  // will be branched to, or trying to preserve the branching logic in a loop
  // invariant manner.
  if (!ExitBlock && !L->hasNoExitBlocks()) {
    LLVM_DEBUG(dbgs() << "Deletion requires at most one exit block.\n");
    return LoopDeletionResult::Unmodified;
  }
  // Finally, we have to check that the loop really is dead.
  bool Changed = false;
  if (!isLoopDead(L, SE, ExitingBlocks, ExitBlock, Changed, Preheader)) {
    LLVM_DEBUG(dbgs() << "Loop is not invariant, cannot delete.\n");
    return Changed ? LoopDeletionResult::Modified
                   : LoopDeletionResult::Unmodified;
  }

  // Don't remove loops for which we can't solve the trip count unless the loop
  // was required to make progress but has been determined to be dead.
  const SCEV *S = SE.getConstantMaxBackedgeTakenCount(L);
  if (isa<SCEVCouldNotCompute>(S) &&
      !L->getHeader()->getParent()->mustProgress() && !hasMustProgress(L)) {
    LLVM_DEBUG(dbgs() << "Could not compute SCEV MaxBackedgeTakenCount and was "
                         "not required to make progress.\n");
    return Changed ? LoopDeletionResult::Modified
                   : LoopDeletionResult::Unmodified;
  }

  LLVM_DEBUG(dbgs() << "Loop is invariant, delete it!");
  ORE.emit([&]() {
    return OptimizationRemark(DEBUG_TYPE, "Invariant", L->getStartLoc(),
                              L->getHeader())
           << "Loop deleted because it is invariant";
  });
  deleteDeadLoop(L, &DT, &SE, &LI, MSSA);
  ++NumDeleted;

  return LoopDeletionResult::Deleted;
}

PreservedAnalyses LoopDeletionPass::run(Loop &L, LoopAnalysisManager &AM,
                                        LoopStandardAnalysisResults &AR,
                                        LPMUpdater &Updater) {

  LLVM_DEBUG(dbgs() << "Analyzing Loop for deletion: ");
  LLVM_DEBUG(L.dump());
  std::string LoopName = std::string(L.getName());
  // For the new PM, we can't use OptimizationRemarkEmitter as an analysis
  // pass. Function analyses need to be preserved across loop transformations
  // but ORE cannot be preserved (see comment before the pass definition).
  OptimizationRemarkEmitter ORE(L.getHeader()->getParent());
  auto Result = deleteLoopIfDead(&L, AR.DT, AR.SE, AR.LI, AR.MSSA, ORE);

  // If we can prove the backedge isn't taken, just break it and be done.  This
  // leaves the loop structure in place which means it can handle dispatching
  // to the right exit based on whatever loop invariant structure remains.
  if (Result != LoopDeletionResult::Deleted)
    Result = merge(Result, breakBackedgeIfNotTaken(&L, AR.DT, AR.SE, AR.LI,
                                                   AR.MSSA, ORE));

  if (Result == LoopDeletionResult::Unmodified)
    return PreservedAnalyses::all();

  if (Result == LoopDeletionResult::Deleted)
    Updater.markLoopAsDeleted(L, LoopName);

  auto PA = getLoopPassPreservedAnalyses();
  if (AR.MSSA)
    PA.preserve<MemorySSAAnalysis>();
  return PA;
}

namespace {
class LoopDeletionLegacyPass : public LoopPass {
public:
  static char ID; // Pass ID, replacement for typeid
  LoopDeletionLegacyPass() : LoopPass(ID) {
    initializeLoopDeletionLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  // Possibly eliminate loop L if it is dead.
  bool runOnLoop(Loop *L, LPPassManager &) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addPreserved<MemorySSAWrapperPass>();
    getLoopAnalysisUsage(AU);
  }
};
}

char LoopDeletionLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(LoopDeletionLegacyPass, "loop-deletion",
                      "Delete dead loops", false, false)
INITIALIZE_PASS_DEPENDENCY(LoopPass)
INITIALIZE_PASS_END(LoopDeletionLegacyPass, "loop-deletion",
                    "Delete dead loops", false, false)

Pass *llvm::createLoopDeletionPass() { return new LoopDeletionLegacyPass(); }

bool LoopDeletionLegacyPass::runOnLoop(Loop *L, LPPassManager &LPM) {
  if (skipLoop(L))
    return false;
  DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  ScalarEvolution &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  auto *MSSAAnalysis = getAnalysisIfAvailable<MemorySSAWrapperPass>();
  MemorySSA *MSSA = nullptr;
  if (MSSAAnalysis)
    MSSA = &MSSAAnalysis->getMSSA();
  // For the old PM, we can't use OptimizationRemarkEmitter as an analysis
  // pass.  Function analyses need to be preserved across loop transformations
  // but ORE cannot be preserved (see comment before the pass definition).
  OptimizationRemarkEmitter ORE(L->getHeader()->getParent());

  LLVM_DEBUG(dbgs() << "Analyzing Loop for deletion: ");
  LLVM_DEBUG(L->dump());

  LoopDeletionResult Result = deleteLoopIfDead(L, DT, SE, LI, MSSA, ORE);

  // If we can prove the backedge isn't taken, just break it and be done.  This
  // leaves the loop structure in place which means it can handle dispatching
  // to the right exit based on whatever loop invariant structure remains.
  if (Result != LoopDeletionResult::Deleted)
    Result = merge(Result, breakBackedgeIfNotTaken(L, DT, SE, LI, MSSA, ORE));

  if (Result == LoopDeletionResult::Deleted)
    LPM.markLoopAsDeleted(*L);

  return Result != LoopDeletionResult::Unmodified;
}
