//===- GuardWidening.cpp - ---- Guard widening ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the guard widening pass.  The semantics of the
// @llvm.experimental.guard intrinsic lets LLVM transform it so that it fails
// more often that it did before the transform.  This optimization is called
// "widening" and can be used hoist and common runtime checks in situations like
// these:
//
//    %cmp0 = 7 u< Length
//    call @llvm.experimental.guard(i1 %cmp0) [ "deopt"(...) ]
//    call @unknown_side_effects()
//    %cmp1 = 9 u< Length
//    call @llvm.experimental.guard(i1 %cmp1) [ "deopt"(...) ]
//    ...
//
// =>
//
//    %cmp0 = 9 u< Length
//    call @llvm.experimental.guard(i1 %cmp0) [ "deopt"(...) ]
//    call @unknown_side_effects()
//    ...
//
// If %cmp0 is false, @llvm.experimental.guard will "deoptimize" back to a
// generic implementation of the same function, which will have the correct
// semantics from that point onward.  It is always _legal_ to deoptimize (so
// replacing %cmp0 with false is "correct"), though it may not always be
// profitable to do so.
//
// NB! This pass is a work in progress.  It hasn't been tuned to be "production
// ready" yet.  It is known to have quadriatic running time and will not scale
// to large numbers of guards
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/GuardWidening.h"
#include "llvm/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Scalar.h"

using namespace llvm;

#define DEBUG_TYPE "guard-widening"

namespace {

class GuardWideningImpl {
  DominatorTree &DT;
  PostDominatorTree &PDT;
  LoopInfo &LI;

  /// The set of guards whose conditions have been widened into dominating
  /// guards.
  SmallVector<IntrinsicInst *, 16> EliminatedGuards;

  /// The set of guards which have been widened to include conditions to other
  /// guards.
  DenseSet<IntrinsicInst *> WidenedGuards;

  /// Try to eliminate guard \p Guard by widening it into an earlier dominating
  /// guard.  \p DFSI is the DFS iterator on the dominator tree that is
  /// currently visiting the block containing \p Guard, and \p GuardsPerBlock
  /// maps BasicBlocks to the set of guards seen in that block.
  bool eliminateGuardViaWidening(
      IntrinsicInst *Guard, const df_iterator<DomTreeNode *> &DFSI,
      const DenseMap<BasicBlock *, SmallVector<IntrinsicInst *, 8>> &
          GuardsPerBlock);

  /// Used to keep track of which widening potential is more effective.
  enum WideningScore {
    /// Don't widen.
    WS_IllegalOrNegative,

    /// Widening is performance neutral as far as the cycles spent in check
    /// conditions goes (but can still help, e.g., code layout, having less
    /// deopt state).
    WS_Neutral,

    /// Widening is profitable.
    WS_Positive,

    /// Widening is very profitable.  Not significantly different from \c
    /// WS_Positive, except by the order.
    WS_VeryPositive
  };

  static StringRef scoreTypeToString(WideningScore WS);

  /// Compute the score for widening the condition in \p DominatedGuard
  /// (contained in \p DominatedGuardLoop) into \p DominatingGuard (contained in
  /// \p DominatingGuardLoop).
  WideningScore computeWideningScore(IntrinsicInst *DominatedGuard,
                                     Loop *DominatedGuardLoop,
                                     IntrinsicInst *DominatingGuard,
                                     Loop *DominatingGuardLoop);

  /// Helper to check if \p V can be hoisted to \p InsertPos.
  bool isAvailableAt(Value *V, Instruction *InsertPos) {
    SmallPtrSet<Instruction *, 8> Visited;
    return isAvailableAt(V, InsertPos, Visited);
  }

  bool isAvailableAt(Value *V, Instruction *InsertPos,
                     SmallPtrSetImpl<Instruction *> &Visited);

  /// Helper to hoist \p V to \p InsertPos.  Guaranteed to succeed if \c
  /// isAvailableAt returned true.
  void makeAvailableAt(Value *V, Instruction *InsertPos);

  /// Common helper used by \c widenGuard and \c isWideningCondProfitable.  Try
  /// to generate an expression computing the logical AND of \p Cond0 and \p
  /// Cond1.  Return true if the expression computing the AND is only as
  /// expensive as computing one of the two. If \p InsertPt is true then
  /// actually generate the resulting expression, make it available at \p
  /// InsertPt and return it in \p Result (else no change to the IR is made).
  bool widenCondCommon(Value *Cond0, Value *Cond1, Instruction *InsertPt,
                       Value *&Result);

  /// Can we compute the logical AND of \p Cond0 and \p Cond1 for the price of
  /// computing only one of the two expressions?
  bool isWideningCondProfitable(Value *Cond0, Value *Cond1) {
    Value *ResultUnused;
    return widenCondCommon(Cond0, Cond1, /*InsertPt=*/nullptr, ResultUnused);
  }

  /// Widen \p ToWiden to fail if \p NewCondition is false (in addition to
  /// whatever it is already checking).
  void widenGuard(IntrinsicInst *ToWiden, Value *NewCondition) {
    Value *Result;
    widenCondCommon(ToWiden->getArgOperand(0), NewCondition, ToWiden, Result);
    ToWiden->setArgOperand(0, Result);
  }

public:
  explicit GuardWideningImpl(DominatorTree &DT, PostDominatorTree &PDT,
                             LoopInfo &LI)
      : DT(DT), PDT(PDT), LI(LI) {}

  /// The entry point for this pass.
  bool run();
};

struct GuardWideningLegacyPass : public FunctionPass {
  static char ID;
  GuardWideningPass Impl;

  GuardWideningLegacyPass() : FunctionPass(ID) {
    initializeGuardWideningLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;
    return GuardWideningImpl(
               getAnalysis<DominatorTreeWrapperPass>().getDomTree(),
               getAnalysis<PostDominatorTreeWrapperPass>().getPostDomTree(),
               getAnalysis<LoopInfoWrapperPass>().getLoopInfo()).run();
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<PostDominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
  }
};

}

bool GuardWideningImpl::run() {
  using namespace llvm::PatternMatch;

  DenseMap<BasicBlock *, SmallVector<IntrinsicInst *, 8>> GuardsInBlock;
  bool Changed = false;

  for (auto DFI = df_begin(DT.getRootNode()), DFE = df_end(DT.getRootNode());
       DFI != DFE; ++DFI) {
    auto *BB = (*DFI)->getBlock();
    auto &CurrentList = GuardsInBlock[BB];

    for (auto &I : *BB)
      if (match(&I, m_Intrinsic<Intrinsic::experimental_guard>()))
        CurrentList.push_back(cast<IntrinsicInst>(&I));

    for (auto *II : CurrentList)
      Changed |= eliminateGuardViaWidening(II, DFI, GuardsInBlock);
  }

  for (auto *II : EliminatedGuards)
    if (!WidenedGuards.count(II))
      II->eraseFromParent();

  return Changed;
}

bool GuardWideningImpl::eliminateGuardViaWidening(
    IntrinsicInst *GuardInst, const df_iterator<DomTreeNode *> &DFSI,
    const DenseMap<BasicBlock *, SmallVector<IntrinsicInst *, 8>> &
        GuardsInBlock) {
  IntrinsicInst *BestSoFar = nullptr;
  auto BestScoreSoFar = WS_IllegalOrNegative;
  auto *GuardInstLoop = LI.getLoopFor(GuardInst->getParent());

  // In the set of dominating guards, find the one we can merge GuardInst with
  // for the most profit.
  for (unsigned i = 0, e = DFSI.getPathLength(); i != e; ++i) {
    auto *CurBB = DFSI.getPath(i)->getBlock();
    auto *CurLoop = LI.getLoopFor(CurBB);
    assert(GuardsInBlock.count(CurBB) && "Must have been populated by now!");
    const auto &GuardsInCurBB = GuardsInBlock.find(CurBB)->second;

    auto I = GuardsInCurBB.begin();
    auto E = GuardsInCurBB.end();

#ifndef NDEBUG
    {
      unsigned Index = 0;
      for (auto &I : *CurBB) {
        if (Index == GuardsInCurBB.size())
          break;
        if (GuardsInCurBB[Index] == &I)
          Index++;
      }
      assert(Index == GuardsInCurBB.size() &&
             "Guards expected to be in order!");
    }
#endif

    assert((i == (e - 1)) == (GuardInst->getParent() == CurBB) && "Bad DFS?");

    if (i == (e - 1)) {
      // Corner case: make sure we're only looking at guards strictly dominating
      // GuardInst when visiting GuardInst->getParent().
      auto NewEnd = std::find(I, E, GuardInst);
      assert(NewEnd != E && "GuardInst not in its own block?");
      E = NewEnd;
    }

    for (auto *Candidate : make_range(I, E)) {
      auto Score =
          computeWideningScore(GuardInst, GuardInstLoop, Candidate, CurLoop);
      DEBUG(dbgs() << "Score between " << *GuardInst->getArgOperand(0)
                   << " and " << *Candidate->getArgOperand(0) << " is "
                   << scoreTypeToString(Score) << "\n");
      if (Score > BestScoreSoFar) {
        BestScoreSoFar = Score;
        BestSoFar = Candidate;
      }
    }
  }

  if (BestScoreSoFar == WS_IllegalOrNegative) {
    DEBUG(dbgs() << "Did not eliminate guard " << *GuardInst << "\n");
    return false;
  }

  assert(BestSoFar != GuardInst && "Should have never visited same guard!");
  assert(DT.dominates(BestSoFar, GuardInst) && "Should be!");

  DEBUG(dbgs() << "Widening " << *GuardInst << " into " << *BestSoFar
               << " with score " << scoreTypeToString(BestScoreSoFar) << "\n");
  widenGuard(BestSoFar, GuardInst->getArgOperand(0));
  GuardInst->setArgOperand(0, ConstantInt::getTrue(GuardInst->getContext()));
  EliminatedGuards.push_back(GuardInst);
  WidenedGuards.insert(BestSoFar);
  return true;
}

GuardWideningImpl::WideningScore GuardWideningImpl::computeWideningScore(
    IntrinsicInst *DominatedGuard, Loop *DominatedGuardLoop,
    IntrinsicInst *DominatingGuard, Loop *DominatingGuardLoop) {
  bool HoistingOutOfLoop = false;

  if (DominatingGuardLoop != DominatedGuardLoop) {
    if (DominatingGuardLoop &&
        !DominatingGuardLoop->contains(DominatedGuardLoop))
      return WS_IllegalOrNegative;

    HoistingOutOfLoop = true;
  }

  if (!isAvailableAt(DominatedGuard->getArgOperand(0), DominatingGuard))
    return WS_IllegalOrNegative;

  bool HoistingOutOfIf =
      !PDT.dominates(DominatedGuard->getParent(), DominatingGuard->getParent());

  if (isWideningCondProfitable(DominatedGuard->getArgOperand(0),
                               DominatingGuard->getArgOperand(0)))
    return HoistingOutOfLoop ? WS_VeryPositive : WS_Positive;

  if (HoistingOutOfLoop)
    return WS_Positive;

  return HoistingOutOfIf ? WS_IllegalOrNegative : WS_Neutral;
}

bool GuardWideningImpl::isAvailableAt(Value *V, Instruction *Loc,
                                      SmallPtrSetImpl<Instruction *> &Visited) {
  auto *Inst = dyn_cast<Instruction>(V);
  if (!Inst || DT.dominates(Inst, Loc) || Visited.count(Inst))
    return true;

  if (!isSafeToSpeculativelyExecute(Inst, Loc, &DT) ||
      Inst->mayReadFromMemory())
    return false;

  Visited.insert(Inst);

  // We only want to go _up_ the dominance chain when recursing.
  assert(!isa<PHINode>(Loc) &&
         "PHIs should return false for isSafeToSpeculativelyExecute");
  assert(DT.isReachableFromEntry(Inst->getParent()) &&
         "We did a DFS from the block entry!");
  return all_of(Inst->operands(),
                [&](Value *Op) { return isAvailableAt(Op, Loc, Visited); });
}

void GuardWideningImpl::makeAvailableAt(Value *V, Instruction *Loc) {
  auto *Inst = dyn_cast<Instruction>(V);
  if (!Inst || DT.dominates(Inst, Loc))
    return;

  assert(isSafeToSpeculativelyExecute(Inst, Loc, &DT) &&
         !Inst->mayReadFromMemory() && "Should've checked with isAvailableAt!");

  for (Value *Op : Inst->operands())
    makeAvailableAt(Op, Loc);

  Inst->moveBefore(Loc);
}

bool GuardWideningImpl::widenCondCommon(Value *Cond0, Value *Cond1,
                                        Instruction *InsertPt, Value *&Result) {
  using namespace llvm::PatternMatch;

  {
    // L >u C0 && L >u C1  ->  L >u max(C0, C1)
    ConstantInt *RHS0, *RHS1;
    Value *LHS;
    ICmpInst::Predicate Pred0, Pred1;
    if (match(Cond0, m_ICmp(Pred0, m_Value(LHS), m_ConstantInt(RHS0))) &&
        match(Cond1, m_ICmp(Pred1, m_Specific(LHS), m_ConstantInt(RHS1)))) {

      ConstantRange CR0 =
          ConstantRange::makeExactICmpRegion(Pred0, RHS0->getValue());
      ConstantRange CR1 =
          ConstantRange::makeExactICmpRegion(Pred1, RHS1->getValue());

      // SubsetIntersect is a subset of the actual mathematical intersection of
      // CR0 and CR1, while SupersetIntersect is a superset of the the actual
      // mathematical intersection.  If these two ConstantRanges are equal, then
      // we know we were able to represent the actual mathematical intersection
      // of CR0 and CR1, and can use the same to generate an icmp instruction.
      //
      // Given what we're doing here and the semantics of guards, it would
      // actually be correct to just use SubsetIntersect, but that may be too
      // aggressive in cases we care about.
      auto SubsetIntersect = CR0.inverse().unionWith(CR1.inverse()).inverse();
      auto SupersetIntersect = CR0.intersectWith(CR1);

      APInt NewRHSAP;
      CmpInst::Predicate Pred;
      if (SubsetIntersect == SupersetIntersect &&
          SubsetIntersect.getEquivalentICmp(Pred, NewRHSAP)) {
        if (InsertPt) {
          ConstantInt *NewRHS = ConstantInt::get(Cond0->getContext(), NewRHSAP);
          Result = new ICmpInst(InsertPt, Pred, LHS, NewRHS, "wide.chk");
        }
        return true;
      }
    }
  }

  // Base case -- just logical-and the two conditions together.

  if (InsertPt) {
    makeAvailableAt(Cond0, InsertPt);
    makeAvailableAt(Cond1, InsertPt);

    Result = BinaryOperator::CreateAnd(Cond0, Cond1, "wide.chk", InsertPt);
  }

  // We were not able to compute Cond0 AND Cond1 for the price of one.
  return false;
}

PreservedAnalyses GuardWideningPass::run(Function &F,
                                         AnalysisManager<Function> &AM) {
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto &LI = AM.getResult<LoopAnalysis>(F);
  auto &PDT = AM.getResult<PostDominatorTreeAnalysis>(F);
  bool Changed = GuardWideningImpl(DT, PDT, LI).run();
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

StringRef GuardWideningImpl::scoreTypeToString(WideningScore WS) {
  switch (WS) {
  case WS_IllegalOrNegative:
    return "IllegalOrNegative";
  case WS_Neutral:
    return "Neutral";
  case WS_Positive:
    return "Positive";
  case WS_VeryPositive:
    return "VeryPositive";
  }

  llvm_unreachable("Fully covered switch above!");
}

char GuardWideningLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(GuardWideningLegacyPass, "guard-widening", "Widen guards",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(PostDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_END(GuardWideningLegacyPass, "guard-widening", "Widen guards",
                    false, false)

FunctionPass *llvm::createGuardWideningPass() {
  return new GuardWideningLegacyPass();
}
