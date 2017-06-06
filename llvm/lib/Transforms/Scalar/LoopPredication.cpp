//===-- LoopPredication.cpp - Guard based loop predication pass -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The LoopPredication pass tries to convert loop variant range checks to loop
// invariant by widening checks across loop iterations. For example, it will
// convert
//
//   for (i = 0; i < n; i++) {
//     guard(i < len);
//     ...
//   }
//
// to
//
//   for (i = 0; i < n; i++) {
//     guard(n - 1 < len);
//     ...
//   }
//
// After this transformation the condition of the guard is loop invariant, so
// loop-unswitch can later unswitch the loop by this condition which basically
// predicates the loop by the widened condition:
//
//   if (n - 1 < len)
//     for (i = 0; i < n; i++) {
//       ...
//     }
//   else
//     deoptimize
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/LoopPredication.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

#define DEBUG_TYPE "loop-predication"

using namespace llvm;

namespace {
class LoopPredication {
  /// Represents an induction variable check:
  ///   icmp Pred, <induction variable>, <loop invariant limit>
  struct LoopICmp {
    ICmpInst::Predicate Pred;
    const SCEVAddRecExpr *IV;
    const SCEV *Limit;
    LoopICmp(ICmpInst::Predicate Pred, const SCEVAddRecExpr *IV,
             const SCEV *Limit)
        : Pred(Pred), IV(IV), Limit(Limit) {}
    LoopICmp() {}
  };

  ScalarEvolution *SE;

  Loop *L;
  const DataLayout *DL;
  BasicBlock *Preheader;

  Optional<LoopICmp> parseLoopICmp(ICmpInst *ICI);

  Value *expandCheck(SCEVExpander &Expander, IRBuilder<> &Builder,
                     ICmpInst::Predicate Pred, const SCEV *LHS, const SCEV *RHS,
                     Instruction *InsertAt);

  Optional<Value *> widenICmpRangeCheck(ICmpInst *ICI, SCEVExpander &Expander,
                                        IRBuilder<> &Builder);
  bool widenGuardConditions(IntrinsicInst *II, SCEVExpander &Expander);

public:
  LoopPredication(ScalarEvolution *SE) : SE(SE){};
  bool runOnLoop(Loop *L);
};

class LoopPredicationLegacyPass : public LoopPass {
public:
  static char ID;
  LoopPredicationLegacyPass() : LoopPass(ID) {
    initializeLoopPredicationLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    getLoopAnalysisUsage(AU);
  }

  bool runOnLoop(Loop *L, LPPassManager &LPM) override {
    if (skipLoop(L))
      return false;
    auto *SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();
    LoopPredication LP(SE);
    return LP.runOnLoop(L);
  }
};

char LoopPredicationLegacyPass::ID = 0;
} // end namespace llvm

INITIALIZE_PASS_BEGIN(LoopPredicationLegacyPass, "loop-predication",
                      "Loop predication", false, false)
INITIALIZE_PASS_DEPENDENCY(LoopPass)
INITIALIZE_PASS_END(LoopPredicationLegacyPass, "loop-predication",
                    "Loop predication", false, false)

Pass *llvm::createLoopPredicationPass() {
  return new LoopPredicationLegacyPass();
}

PreservedAnalyses LoopPredicationPass::run(Loop &L, LoopAnalysisManager &AM,
                                           LoopStandardAnalysisResults &AR,
                                           LPMUpdater &U) {
  LoopPredication LP(&AR.SE);
  if (!LP.runOnLoop(&L))
    return PreservedAnalyses::all();

  return getLoopPassPreservedAnalyses();
}

Optional<LoopPredication::LoopICmp>
LoopPredication::parseLoopICmp(ICmpInst *ICI) {
  ICmpInst::Predicate Pred = ICI->getPredicate();

  Value *LHS = ICI->getOperand(0);
  Value *RHS = ICI->getOperand(1);
  const SCEV *LHSS = SE->getSCEV(LHS);
  if (isa<SCEVCouldNotCompute>(LHSS))
    return None;
  const SCEV *RHSS = SE->getSCEV(RHS);
  if (isa<SCEVCouldNotCompute>(RHSS))
    return None;

  // Canonicalize RHS to be loop invariant bound, LHS - a loop computable IV
  if (SE->isLoopInvariant(LHSS, L)) {
    std::swap(LHS, RHS);
    std::swap(LHSS, RHSS);
    Pred = ICmpInst::getSwappedPredicate(Pred);
  }

  const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(LHSS);
  if (!AR || AR->getLoop() != L)
    return None;

  return LoopICmp(Pred, AR, RHSS);
}

Value *LoopPredication::expandCheck(SCEVExpander &Expander,
                                    IRBuilder<> &Builder,
                                    ICmpInst::Predicate Pred, const SCEV *LHS,
                                    const SCEV *RHS, Instruction *InsertAt) {
  Type *Ty = LHS->getType();
  assert(Ty == RHS->getType() && "expandCheck operands have different types?");
  Value *LHSV = Expander.expandCodeFor(LHS, Ty, InsertAt);
  Value *RHSV = Expander.expandCodeFor(RHS, Ty, InsertAt);
  return Builder.CreateICmp(Pred, LHSV, RHSV);
}

/// If ICI can be widened to a loop invariant condition emits the loop
/// invariant condition in the loop preheader and return it, otherwise
/// returns None.
Optional<Value *> LoopPredication::widenICmpRangeCheck(ICmpInst *ICI,
                                                       SCEVExpander &Expander,
                                                       IRBuilder<> &Builder) {
  DEBUG(dbgs() << "Analyzing ICmpInst condition:\n");
  DEBUG(ICI->dump());

  auto RangeCheck = parseLoopICmp(ICI);
  if (!RangeCheck) {
    DEBUG(dbgs() << "Failed to parse the loop latch condition!\n");
    return None;
  }

  ICmpInst::Predicate Pred = RangeCheck->Pred;
  const SCEVAddRecExpr *IndexAR = RangeCheck->IV;
  const SCEV *RHSS = RangeCheck->Limit;

  auto CanExpand = [this](const SCEV *S) {
    return SE->isLoopInvariant(S, L) && isSafeToExpand(S, *SE);
  };
  if (!CanExpand(RHSS))
    return None;

  DEBUG(dbgs() << "IndexAR: ");
  DEBUG(IndexAR->dump());

  bool IsIncreasing = false;
  if (!SE->isMonotonicPredicate(IndexAR, Pred, IsIncreasing))
    return None;

  // If the predicate is increasing the condition can change from false to true
  // as the loop progresses, in this case take the value on the first iteration
  // for the widened check. Otherwise the condition can change from true to
  // false as the loop progresses, so take the value on the last iteration.
  const SCEV *NewLHSS = IsIncreasing
                            ? IndexAR->getStart()
                            : SE->getSCEVAtScope(IndexAR, L->getParentLoop());
  if (NewLHSS == IndexAR) {
    DEBUG(dbgs() << "Can't compute NewLHSS!\n");
    return None;
  }

  DEBUG(dbgs() << "NewLHSS: ");
  DEBUG(NewLHSS->dump());

  if (!CanExpand(NewLHSS))
    return None;

  DEBUG(dbgs() << "NewLHSS is loop invariant and safe to expand. Expand!\n");

  Instruction *InsertAt = Preheader->getTerminator();
  return expandCheck(Expander, Builder, Pred, NewLHSS, RHSS, InsertAt);
}

bool LoopPredication::widenGuardConditions(IntrinsicInst *Guard,
                                           SCEVExpander &Expander) {
  DEBUG(dbgs() << "Processing guard:\n");
  DEBUG(Guard->dump());

  IRBuilder<> Builder(cast<Instruction>(Preheader->getTerminator()));

  // The guard condition is expected to be in form of:
  //   cond1 && cond2 && cond3 ...
  // Iterate over subconditions looking for for icmp conditions which can be
  // widened across loop iterations. Widening these conditions remember the
  // resulting list of subconditions in Checks vector.
  SmallVector<Value *, 4> Worklist(1, Guard->getOperand(0));
  SmallPtrSet<Value *, 4> Visited;

  SmallVector<Value *, 4> Checks;

  unsigned NumWidened = 0;
  do {
    Value *Condition = Worklist.pop_back_val();
    if (!Visited.insert(Condition).second)
      continue;

    Value *LHS, *RHS;
    using namespace llvm::PatternMatch;
    if (match(Condition, m_And(m_Value(LHS), m_Value(RHS)))) {
      Worklist.push_back(LHS);
      Worklist.push_back(RHS);
      continue;
    }

    if (ICmpInst *ICI = dyn_cast<ICmpInst>(Condition)) {
      if (auto NewRangeCheck = widenICmpRangeCheck(ICI, Expander, Builder)) {
        Checks.push_back(NewRangeCheck.getValue());
        NumWidened++;
        continue;
      }
    }

    // Save the condition as is if we can't widen it
    Checks.push_back(Condition);
  } while (Worklist.size() != 0);

  if (NumWidened == 0)
    return false;

  // Emit the new guard condition
  Builder.SetInsertPoint(Guard);
  Value *LastCheck = nullptr;
  for (auto *Check : Checks)
    if (!LastCheck)
      LastCheck = Check;
    else
      LastCheck = Builder.CreateAnd(LastCheck, Check);
  Guard->setOperand(0, LastCheck);

  DEBUG(dbgs() << "Widened checks = " << NumWidened << "\n");
  return true;
}

bool LoopPredication::runOnLoop(Loop *Loop) {
  L = Loop;

  DEBUG(dbgs() << "Analyzing ");
  DEBUG(L->dump());

  Module *M = L->getHeader()->getModule();

  // There is nothing to do if the module doesn't use guards
  auto *GuardDecl =
      M->getFunction(Intrinsic::getName(Intrinsic::experimental_guard));
  if (!GuardDecl || GuardDecl->use_empty())
    return false;

  DL = &M->getDataLayout();

  Preheader = L->getLoopPreheader();
  if (!Preheader)
    return false;

  // Collect all the guards into a vector and process later, so as not
  // to invalidate the instruction iterator.
  SmallVector<IntrinsicInst *, 4> Guards;
  for (const auto BB : L->blocks())
    for (auto &I : *BB)
      if (auto *II = dyn_cast<IntrinsicInst>(&I))
        if (II->getIntrinsicID() == Intrinsic::experimental_guard)
          Guards.push_back(II);

  if (Guards.empty())
    return false;

  SCEVExpander Expander(*SE, *DL, "loop-predication");

  bool Changed = false;
  for (auto *Guard : Guards)
    Changed |= widenGuardConditions(Guard, Expander);

  return Changed;
}
