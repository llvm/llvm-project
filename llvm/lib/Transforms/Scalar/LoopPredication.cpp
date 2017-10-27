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
// It's tempting to rely on SCEV here, but it has proven to be problematic.
// Generally the facts SCEV provides about the increment step of add
// recurrences are true if the backedge of the loop is taken, which implicitly
// assumes that the guard doesn't fail. Using these facts to optimize the
// guard results in a circular logic where the guard is optimized under the
// assumption that it never fails.
//
// For example, in the loop below the induction variable will be marked as nuw
// basing on the guard. Basing on nuw the guard predicate will be considered
// monotonic. Given a monotonic condition it's tempting to replace the induction
// variable in the condition with its value on the last iteration. But this
// transformation is not correct, e.g. e = 4, b = 5 breaks the loop.
//
//   for (int i = b; i != e; i++)
//     guard(i u< len)
//
// One of the ways to reason about this problem is to use an inductive proof
// approach. Given the loop:
//
//   if (B(0)) {
//     do {
//       I = PHI(0, I.INC)
//       I.INC = I + Step
//       guard(G(I));
//     } while (B(I));
//   }
//
// where B(x) and G(x) are predicates that map integers to booleans, we want a
// loop invariant expression M such the following program has the same semantics
// as the above:
//
//   if (B(0)) {
//     do {
//       I = PHI(0, I.INC)
//       I.INC = I + Step
//       guard(G(0) && M);
//     } while (B(I));
//   }
//
// One solution for M is M = forall X . (G(X) && B(X)) => G(X + Step)
// 
// Informal proof that the transformation above is correct:
//
//   By the definition of guards we can rewrite the guard condition to:
//     G(I) && G(0) && M
//
//   Let's prove that for each iteration of the loop:
//     G(0) && M => G(I)
//   And the condition above can be simplified to G(Start) && M.
// 
//   Induction base.
//     G(0) && M => G(0)
//
//   Induction step. Assuming G(0) && M => G(I) on the subsequent
//   iteration:
//
//     B(I) is true because it's the backedge condition.
//     G(I) is true because the backedge is guarded by this condition.
//
//   So M = forall X . (G(X) && B(X)) => G(X + Step) implies G(I + Step).
//
// Note that we can use anything stronger than M, i.e. any condition which
// implies M.
//
// For now the transformation is limited to the following case:
//   * The loop has a single latch with the condition of the form:
//     B(X) = latchStart + X <pred> latchLimit,
//     where <pred> is u<, u<=, s<, or s<=.
//   * The step of the IV used in the latch condition is 1.
//   * The guard condition is of the form
//     G(X) = guardStart + X u< guardLimit
//
// For the ult latch comparison case M is:
//   forall X . guardStart + X u< guardLimit && latchStart + X <u latchLimit =>
//      guardStart + X + 1 u< guardLimit
//
// The only way the antecedent can be true and the consequent can be false is
// if
//   X == guardLimit - 1 - guardStart
// (and guardLimit is non-zero, but we won't use this latter fact).
// If X == guardLimit - 1 - guardStart then the second half of the antecedent is
//   latchStart + guardLimit - 1 - guardStart u< latchLimit
// and its negation is
//   latchStart + guardLimit - 1 - guardStart u>= latchLimit
//
// In other words, if
//   latchLimit u<= latchStart + guardLimit - 1 - guardStart
// then:
// (the ranges below are written in ConstantRange notation, where [A, B) is the
// set for (I = A; I != B; I++ /*maywrap*/) yield(I);)
//
//    forall X . guardStart + X u< guardLimit &&
//               latchStart + X u< latchLimit =>
//      guardStart + X + 1 u< guardLimit
// == forall X . guardStart + X u< guardLimit &&
//               latchStart + X u< latchStart + guardLimit - 1 - guardStart =>
//      guardStart + X + 1 u< guardLimit
// == forall X . (guardStart + X) in [0, guardLimit) &&
//               (latchStart + X) in [0, latchStart + guardLimit - 1 - guardStart) =>
//      (guardStart + X + 1) in [0, guardLimit)
// == forall X . X in [-guardStart, guardLimit - guardStart) &&
//               X in [-latchStart, guardLimit - 1 - guardStart) =>
//       X in [-guardStart - 1, guardLimit - guardStart - 1)
// == true
//
// So the widened condition is:
//   guardStart u< guardLimit &&
//   latchStart + guardLimit - 1 - guardStart u>= latchLimit
// Similarly for ule condition the widened condition is:
//   guardStart u< guardLimit &&
//   latchStart + guardLimit - 1 - guardStart u> latchLimit
// For slt condition the widened condition is:
//   guardStart u< guardLimit &&
//   latchStart + guardLimit - 1 - guardStart s>= latchLimit
// For sle condition the widened condition is:
//   guardStart u< guardLimit &&
//   latchStart + guardLimit - 1 - guardStart s> latchLimit
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
  LoopICmp LatchCheck;

  Optional<LoopICmp> parseLoopICmp(ICmpInst *ICI) {
    return parseLoopICmp(ICI->getPredicate(), ICI->getOperand(0),
                         ICI->getOperand(1));
  }
  Optional<LoopICmp> parseLoopICmp(ICmpInst::Predicate Pred, Value *LHS,
                                   Value *RHS);

  Optional<LoopICmp> parseLoopLatchICmp();

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
LoopPredication::parseLoopICmp(ICmpInst::Predicate Pred, Value *LHS,
                               Value *RHS) {
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
  // TODO: we can check isLoopEntryGuardedByCond before emitting the check
 
  Type *Ty = LHS->getType();
  assert(Ty == RHS->getType() && "expandCheck operands have different types?");

  if (SE->isLoopEntryGuardedByCond(L, Pred, LHS, RHS))
    return Builder.getTrue();

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

  // parseLoopStructure guarantees that the latch condition is:
  //   ++i <pred> latchLimit, where <pred> is u<, u<=, s<, or s<=.
  // We are looking for the range checks of the form:
  //   i u< guardLimit
  auto RangeCheck = parseLoopICmp(ICI);
  if (!RangeCheck) {
    DEBUG(dbgs() << "Failed to parse the loop latch condition!\n");
    return None;
  }
  if (RangeCheck->Pred != ICmpInst::ICMP_ULT) {
    DEBUG(dbgs() << "Unsupported range check predicate(" << RangeCheck->Pred
                 << ")!\n");
    return None;
  }
  auto *RangeCheckIV = RangeCheck->IV;
  auto *Ty = RangeCheckIV->getType();
  if (Ty != LatchCheck.IV->getType()) {
    DEBUG(dbgs() << "Type mismatch between range check and latch IVs!\n");
    return None;
  }
  if (!RangeCheckIV->isAffine()) {
    DEBUG(dbgs() << "Range check IV is not affine!\n");
    return None;
  }
  auto *Step = RangeCheckIV->getStepRecurrence(*SE);
  if (Step != LatchCheck.IV->getStepRecurrence(*SE)) {
    DEBUG(dbgs() << "Range check and latch have IVs different steps!\n");
    return None;
  }
  assert(Step->isOne() && "must be one");

  // Generate the widened condition:
  //   guardStart u< guardLimit &&
  //   latchLimit <pred> guardLimit - 1 - guardStart + latchStart
  // where <pred> depends on the latch condition predicate. See the file
  // header comment for the reasoning.
  const SCEV *GuardStart = RangeCheckIV->getStart();
  const SCEV *GuardLimit = RangeCheck->Limit;
  const SCEV *LatchStart = LatchCheck.IV->getStart();
  const SCEV *LatchLimit = LatchCheck.Limit;

  // guardLimit - guardStart + latchStart - 1
  const SCEV *RHS =
      SE->getAddExpr(SE->getMinusSCEV(GuardLimit, GuardStart),
                     SE->getMinusSCEV(LatchStart, SE->getOne(Ty)));

  ICmpInst::Predicate LimitCheckPred;
  switch (LatchCheck.Pred) {
  case ICmpInst::ICMP_ULT:
    LimitCheckPred = ICmpInst::ICMP_ULE;
    break;
  case ICmpInst::ICMP_ULE:
    LimitCheckPred = ICmpInst::ICMP_ULT;
    break;
  case ICmpInst::ICMP_SLT:
    LimitCheckPred = ICmpInst::ICMP_SLE;
    break;
  case ICmpInst::ICMP_SLE:
    LimitCheckPred = ICmpInst::ICMP_SLT;
    break;
  default:
    llvm_unreachable("Unsupported loop latch!");
  }

  DEBUG(dbgs() << "LHS: " << *LatchLimit << "\n");
  DEBUG(dbgs() << "RHS: " << *RHS << "\n");
  DEBUG(dbgs() << "Pred: " << LimitCheckPred << "\n");

  auto CanExpand = [this](const SCEV *S) {
    return SE->isLoopInvariant(S, L) && isSafeToExpand(S, *SE);
  };
  if (!CanExpand(GuardStart) || !CanExpand(GuardLimit) ||
      !CanExpand(LatchLimit) || !CanExpand(RHS)) {
    DEBUG(dbgs() << "Can't expand limit check!\n");
    return None;
  }

  Instruction *InsertAt = Preheader->getTerminator();
  auto *LimitCheck =
      expandCheck(Expander, Builder, LimitCheckPred, LatchLimit, RHS, InsertAt);
  auto *FirstIterationCheck = expandCheck(Expander, Builder, RangeCheck->Pred,
                                          GuardStart, GuardLimit, InsertAt);
  return Builder.CreateAnd(FirstIterationCheck, LimitCheck);
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

Optional<LoopPredication::LoopICmp> LoopPredication::parseLoopLatchICmp() {
  using namespace PatternMatch;

  BasicBlock *LoopLatch = L->getLoopLatch();
  if (!LoopLatch) {
    DEBUG(dbgs() << "The loop doesn't have a single latch!\n");
    return None;
  }

  ICmpInst::Predicate Pred;
  Value *LHS, *RHS;
  BasicBlock *TrueDest, *FalseDest;

  if (!match(LoopLatch->getTerminator(),
             m_Br(m_ICmp(Pred, m_Value(LHS), m_Value(RHS)), TrueDest,
                  FalseDest))) {
    DEBUG(dbgs() << "Failed to match the latch terminator!\n");
    return None;
  }
  assert((TrueDest == L->getHeader() || FalseDest == L->getHeader()) &&
         "One of the latch's destinations must be the header");
  if (TrueDest != L->getHeader())
    Pred = ICmpInst::getInversePredicate(Pred);

  auto Result = parseLoopICmp(Pred, LHS, RHS);
  if (!Result) {
    DEBUG(dbgs() << "Failed to parse the loop latch condition!\n");
    return None;
  }

  if (Result->Pred != ICmpInst::ICMP_ULT &&
      Result->Pred != ICmpInst::ICMP_SLT &&
      Result->Pred != ICmpInst::ICMP_ULE &&
      Result->Pred != ICmpInst::ICMP_SLE) {
    DEBUG(dbgs() << "Unsupported loop latch predicate(" << Result->Pred
                 << ")!\n");
    return None;
  }

  // Check affine first, so if it's not we don't try to compute the step
  // recurrence.
  if (!Result->IV->isAffine()) {
    DEBUG(dbgs() << "The induction variable is not affine!\n");
    return None;
  }

  auto *Step = Result->IV->getStepRecurrence(*SE);
  if (!Step->isOne()) {
    DEBUG(dbgs() << "Unsupported loop stride(" << *Step << ")!\n");
    return None;
  }

  return Result;
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

  auto LatchCheckOpt = parseLoopLatchICmp();
  if (!LatchCheckOpt)
    return false;
  LatchCheck = *LatchCheckOpt;

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
