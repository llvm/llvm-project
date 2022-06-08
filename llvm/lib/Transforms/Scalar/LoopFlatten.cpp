//===- LoopFlatten.cpp - Loop flattening pass------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass flattens pairs nested loops into a single loop.
//
// The intention is to optimise loop nests like this, which together access an
// array linearly:
//
//   for (int i = 0; i < N; ++i)
//     for (int j = 0; j < M; ++j)
//       f(A[i*M+j]);
//
// into one loop:
//
//   for (int i = 0; i < (N*M); ++i)
//     f(A[i]);
//
// It can also flatten loops where the induction variables are not used in the
// loop. This is only worth doing if the induction variables are only used in an
// expression like i*M+j. If they had any other uses, we would have to insert a
// div/mod to reconstruct the original values, so this wouldn't be profitable.
//
// We also need to prove that N*M will not overflow. The preferred solution is
// to widen the IV, which avoids overflow checks, so that is tried first. If
// the IV cannot be widened, then we try to determine that this new tripcount
// expression won't overflow.
//
// Q: Does LoopFlatten use SCEV?
// Short answer: Yes and no.
//
// Long answer:
// For this transformation to be valid, we require all uses of the induction
// variables to be linear expressions of the form i*M+j. The different Loop
// APIs are used to get some loop components like the induction variable,
// compare statement, etc. In addition, we do some pattern matching to find the
// linear expressions and other loop components like the loop increment. The
// latter are examples of expressions that do use the induction variable, but
// are safe to ignore when we check all uses to be of the form i*M+j. We keep
// track of all of this in bookkeeping struct FlattenInfo.
// We assume the loops to be canonical, i.e. starting at 0 and increment with
// 1. This makes RHS of the compare the loop tripcount (with the right
// predicate). We use SCEV to then sanity check that this tripcount matches
// with the tripcount as computed by SCEV.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/LoopFlatten.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopNestAnalysis.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include "llvm/Transforms/Utils/SimplifyIndVar.h"

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "loop-flatten"

STATISTIC(NumFlattened, "Number of loops flattened");

static cl::opt<unsigned> RepeatedInstructionThreshold(
    "loop-flatten-cost-threshold", cl::Hidden, cl::init(2),
    cl::desc("Limit on the cost of instructions that can be repeated due to "
             "loop flattening"));

static cl::opt<bool>
    AssumeNoOverflow("loop-flatten-assume-no-overflow", cl::Hidden,
                     cl::init(false),
                     cl::desc("Assume that the product of the two iteration "
                              "trip counts will never overflow"));

static cl::opt<bool>
    WidenIV("loop-flatten-widen-iv", cl::Hidden, cl::init(true),
            cl::desc("Widen the loop induction variables, if possible, so "
                     "overflow checks won't reject flattening"));

// We require all uses of both induction variables to match this pattern:
//
//   (OuterPHI * InnerTripCount) + InnerPHI
//
// I.e., it needs to be a linear expression of the induction variables and the
// inner loop trip count. We keep track of all different expressions on which
// checks will be performed in this bookkeeping struct.
//
struct FlattenInfo {
  Loop *OuterLoop = nullptr;  // The loop pair to be flattened.
  Loop *InnerLoop = nullptr;

  PHINode *InnerInductionPHI = nullptr; // These PHINodes correspond to loop
  PHINode *OuterInductionPHI = nullptr; // induction variables, which are
                                        // expected to start at zero and
                                        // increment by one on each loop.

  Value *InnerTripCount = nullptr; // The product of these two tripcounts
  Value *OuterTripCount = nullptr; // will be the new flattened loop
                                   // tripcount. Also used to recognise a
                                   // linear expression that will be replaced.

  SmallPtrSet<Value *, 4> LinearIVUses;  // Contains the linear expressions
                                         // of the form i*M+j that will be
                                         // replaced.

  BinaryOperator *InnerIncrement = nullptr;  // Uses of induction variables in
  BinaryOperator *OuterIncrement = nullptr;  // loop control statements that
  BranchInst *InnerBranch = nullptr;         // are safe to ignore.

  BranchInst *OuterBranch = nullptr; // The instruction that needs to be
                                     // updated with new tripcount.

  SmallPtrSet<PHINode *, 4> InnerPHIsToTransform;

  bool Widened = false; // Whether this holds the flatten info before or after
                        // widening.

  PHINode *NarrowInnerInductionPHI = nullptr; // Holds the old/narrow induction
  PHINode *NarrowOuterInductionPHI = nullptr; // phis, i.e. the Phis before IV
                                              // has been apllied. Used to skip
                                              // checks on phi nodes.

  FlattenInfo(Loop *OL, Loop *IL) : OuterLoop(OL), InnerLoop(IL){};

  bool isNarrowInductionPhi(PHINode *Phi) {
    // This can't be the narrow phi if we haven't widened the IV first.
    if (!Widened)
      return false;
    return NarrowInnerInductionPHI == Phi || NarrowOuterInductionPHI == Phi;
  }
  bool isInnerLoopIncrement(User *U) {
    return InnerIncrement == U;
  }
  bool isOuterLoopIncrement(User *U) {
    return OuterIncrement == U;
  }
  bool isInnerLoopTest(User *U) {
    return InnerBranch->getCondition() == U;
  }

  bool checkOuterInductionPhiUsers(SmallPtrSet<Value *, 4> &ValidOuterPHIUses) {
    for (User *U : OuterInductionPHI->users()) {
      if (isOuterLoopIncrement(U))
        continue;

      auto IsValidOuterPHIUses = [&] (User *U) -> bool {
        LLVM_DEBUG(dbgs() << "Found use of outer induction variable: "; U->dump());
        if (!ValidOuterPHIUses.count(U)) {
          LLVM_DEBUG(dbgs() << "Did not match expected pattern, bailing\n");
          return false;
        }
        LLVM_DEBUG(dbgs() << "Use is optimisable\n");
        return true;
      };

      if (auto *V = dyn_cast<TruncInst>(U)) {
        for (auto *K : V->users()) {
          if (!IsValidOuterPHIUses(K))
            return false;
        }
        continue;
      }

      if (!IsValidOuterPHIUses(U))
        return false;
    }
    return true;
  }

  bool matchLinearIVUser(User *U, Value *InnerTripCount,
                         SmallPtrSet<Value *, 4> &ValidOuterPHIUses) {
    LLVM_DEBUG(dbgs() << "Found use of inner induction variable: "; U->dump());
    Value *MatchedMul = nullptr;
    Value *MatchedItCount = nullptr;

    bool IsAdd = match(U, m_c_Add(m_Specific(InnerInductionPHI),
                                  m_Value(MatchedMul))) &&
                 match(MatchedMul, m_c_Mul(m_Specific(OuterInductionPHI),
                                           m_Value(MatchedItCount)));

    // Matches the same pattern as above, except it also looks for truncs
    // on the phi, which can be the result of widening the induction variables.
    bool IsAddTrunc =
        match(U, m_c_Add(m_Trunc(m_Specific(InnerInductionPHI)),
                         m_Value(MatchedMul))) &&
        match(MatchedMul, m_c_Mul(m_Trunc(m_Specific(OuterInductionPHI)),
                                  m_Value(MatchedItCount)));

    if (!MatchedItCount)
      return false;

    // Look through extends if the IV has been widened. Don't look through
    // extends if we already looked through a trunc.
    if (Widened && IsAdd &&
        (isa<SExtInst>(MatchedItCount) || isa<ZExtInst>(MatchedItCount))) {
      assert(MatchedItCount->getType() == InnerInductionPHI->getType() &&
             "Unexpected type mismatch in types after widening");
      MatchedItCount = isa<SExtInst>(MatchedItCount)
                           ? dyn_cast<SExtInst>(MatchedItCount)->getOperand(0)
                           : dyn_cast<ZExtInst>(MatchedItCount)->getOperand(0);
    }

    if ((IsAdd || IsAddTrunc) && MatchedItCount == InnerTripCount) {
      LLVM_DEBUG(dbgs() << "Use is optimisable\n");
      ValidOuterPHIUses.insert(MatchedMul);
      LinearIVUses.insert(U);
      return true;
    }

    LLVM_DEBUG(dbgs() << "Did not match expected pattern, bailing\n");
    return false;
  }

  bool checkInnerInductionPhiUsers(SmallPtrSet<Value *, 4> &ValidOuterPHIUses) {
    Value *SExtInnerTripCount = InnerTripCount;
    if (Widened &&
        (isa<SExtInst>(InnerTripCount) || isa<ZExtInst>(InnerTripCount)))
      SExtInnerTripCount = cast<Instruction>(InnerTripCount)->getOperand(0);

    for (User *U : InnerInductionPHI->users()) {
      if (isInnerLoopIncrement(U))
        continue;

      // After widening the IVs, a trunc instruction might have been introduced,
      // so look through truncs.
      if (isa<TruncInst>(U)) {
        if (!U->hasOneUse())
          return false;
        U = *U->user_begin();
      }

      // If the use is in the compare (which is also the condition of the inner
      // branch) then the compare has been altered by another transformation e.g
      // icmp ult %inc, tripcount -> icmp ult %j, tripcount-1, where tripcount is
      // a constant. Ignore this use as the compare gets removed later anyway.
      if (isInnerLoopTest(U))
        continue;

      if (!matchLinearIVUser(U, SExtInnerTripCount, ValidOuterPHIUses))
        return false;
    }
    return true;
  }
};

static bool
setLoopComponents(Value *&TC, Value *&TripCount, BinaryOperator *&Increment,
                  SmallPtrSetImpl<Instruction *> &IterationInstructions) {
  TripCount = TC;
  IterationInstructions.insert(Increment);
  LLVM_DEBUG(dbgs() << "Found Increment: "; Increment->dump());
  LLVM_DEBUG(dbgs() << "Found trip count: "; TripCount->dump());
  LLVM_DEBUG(dbgs() << "Successfully found all loop components\n");
  return true;
}

// Given the RHS of the loop latch compare instruction, verify with SCEV
// that this is indeed the loop tripcount.
// TODO: This used to be a straightforward check but has grown to be quite
// complicated now. It is therefore worth revisiting what the additional
// benefits are of this (compared to relying on canonical loops and pattern
// matching).
static bool verifyTripCount(Value *RHS, Loop *L,
     SmallPtrSetImpl<Instruction *> &IterationInstructions,
    PHINode *&InductionPHI, Value *&TripCount, BinaryOperator *&Increment,
    BranchInst *&BackBranch, ScalarEvolution *SE, bool IsWidened) {
  const SCEV *BackedgeTakenCount = SE->getBackedgeTakenCount(L);
  if (isa<SCEVCouldNotCompute>(BackedgeTakenCount)) {
    LLVM_DEBUG(dbgs() << "Backedge-taken count is not predictable\n");
    return false;
  }

  // The Extend=false flag is used for getTripCountFromExitCount as we want
  // to verify and match it with the pattern matched tripcount. Please note
  // that overflow checks are performed in checkOverflow, but are first tried
  // to avoid by widening the IV.
  const SCEV *SCEVTripCount =
      SE->getTripCountFromExitCount(BackedgeTakenCount, /*Extend=*/false);

  const SCEV *SCEVRHS = SE->getSCEV(RHS);
  if (SCEVRHS == SCEVTripCount)
    return setLoopComponents(RHS, TripCount, Increment, IterationInstructions);
  ConstantInt *ConstantRHS = dyn_cast<ConstantInt>(RHS);
  if (ConstantRHS) {
    const SCEV *BackedgeTCExt = nullptr;
    if (IsWidened) {
      const SCEV *SCEVTripCountExt;
      // Find the extended backedge taken count and extended trip count using
      // SCEV. One of these should now match the RHS of the compare.
      BackedgeTCExt = SE->getZeroExtendExpr(BackedgeTakenCount, RHS->getType());
      SCEVTripCountExt = SE->getTripCountFromExitCount(BackedgeTCExt, false);
      if (SCEVRHS != BackedgeTCExt && SCEVRHS != SCEVTripCountExt) {
        LLVM_DEBUG(dbgs() << "Could not find valid trip count\n");
        return false;
      }
    }
    // If the RHS of the compare is equal to the backedge taken count we need
    // to add one to get the trip count.
    if (SCEVRHS == BackedgeTCExt || SCEVRHS == BackedgeTakenCount) {
      ConstantInt *One = ConstantInt::get(ConstantRHS->getType(), 1);
      Value *NewRHS = ConstantInt::get(
          ConstantRHS->getContext(), ConstantRHS->getValue() + One->getValue());
      return setLoopComponents(NewRHS, TripCount, Increment,
                               IterationInstructions);
    }
    return setLoopComponents(RHS, TripCount, Increment, IterationInstructions);
  }
  // If the RHS isn't a constant then check that the reason it doesn't match
  // the SCEV trip count is because the RHS is a ZExt or SExt instruction
  // (and take the trip count to be the RHS).
  if (!IsWidened) {
    LLVM_DEBUG(dbgs() << "Could not find valid trip count\n");
    return false;
  }
  auto *TripCountInst = dyn_cast<Instruction>(RHS);
  if (!TripCountInst) {
    LLVM_DEBUG(dbgs() << "Could not find valid trip count\n");
    return false;
  }
  if ((!isa<ZExtInst>(TripCountInst) && !isa<SExtInst>(TripCountInst)) ||
      SE->getSCEV(TripCountInst->getOperand(0)) != SCEVTripCount) {
    LLVM_DEBUG(dbgs() << "Could not find valid extended trip count\n");
    return false;
  }
  return setLoopComponents(RHS, TripCount, Increment, IterationInstructions);
}

// Finds the induction variable, increment and trip count for a simple loop that
// we can flatten.
static bool findLoopComponents(
    Loop *L, SmallPtrSetImpl<Instruction *> &IterationInstructions,
    PHINode *&InductionPHI, Value *&TripCount, BinaryOperator *&Increment,
    BranchInst *&BackBranch, ScalarEvolution *SE, bool IsWidened) {
  LLVM_DEBUG(dbgs() << "Finding components of loop: " << L->getName() << "\n");

  if (!L->isLoopSimplifyForm()) {
    LLVM_DEBUG(dbgs() << "Loop is not in normal form\n");
    return false;
  }

  // Currently, to simplify the implementation, the Loop induction variable must
  // start at zero and increment with a step size of one.
  if (!L->isCanonical(*SE)) {
    LLVM_DEBUG(dbgs() << "Loop is not canonical\n");
    return false;
  }

  // There must be exactly one exiting block, and it must be the same at the
  // latch.
  BasicBlock *Latch = L->getLoopLatch();
  if (L->getExitingBlock() != Latch) {
    LLVM_DEBUG(dbgs() << "Exiting and latch block are different\n");
    return false;
  }

  // Find the induction PHI. If there is no induction PHI, we can't do the
  // transformation. TODO: could other variables trigger this? Do we have to
  // search for the best one?
  InductionPHI = L->getInductionVariable(*SE);
  if (!InductionPHI) {
    LLVM_DEBUG(dbgs() << "Could not find induction PHI\n");
    return false;
  }
  LLVM_DEBUG(dbgs() << "Found induction PHI: "; InductionPHI->dump());

  bool ContinueOnTrue = L->contains(Latch->getTerminator()->getSuccessor(0));
  auto IsValidPredicate = [&](ICmpInst::Predicate Pred) {
    if (ContinueOnTrue)
      return Pred == CmpInst::ICMP_NE || Pred == CmpInst::ICMP_ULT;
    else
      return Pred == CmpInst::ICMP_EQ;
  };

  // Find Compare and make sure it is valid. getLatchCmpInst checks that the
  // back branch of the latch is conditional.
  ICmpInst *Compare = L->getLatchCmpInst();
  if (!Compare || !IsValidPredicate(Compare->getUnsignedPredicate()) ||
      Compare->hasNUsesOrMore(2)) {
    LLVM_DEBUG(dbgs() << "Could not find valid comparison\n");
    return false;
  }
  BackBranch = cast<BranchInst>(Latch->getTerminator());
  IterationInstructions.insert(BackBranch);
  LLVM_DEBUG(dbgs() << "Found back branch: "; BackBranch->dump());
  IterationInstructions.insert(Compare);
  LLVM_DEBUG(dbgs() << "Found comparison: "; Compare->dump());

  // Find increment and trip count.
  // There are exactly 2 incoming values to the induction phi; one from the
  // pre-header and one from the latch. The incoming latch value is the
  // increment variable.
  Increment =
      cast<BinaryOperator>(InductionPHI->getIncomingValueForBlock(Latch));
  if (Increment->hasNUsesOrMore(3)) {
    LLVM_DEBUG(dbgs() << "Could not find valid increment\n");
    return false;
  }
  // The trip count is the RHS of the compare. If this doesn't match the trip
  // count computed by SCEV then this is because the trip count variable
  // has been widened so the types don't match, or because it is a constant and
  // another transformation has changed the compare (e.g. icmp ult %inc,
  // tripcount -> icmp ult %j, tripcount-1), or both.
  Value *RHS = Compare->getOperand(1);

  return verifyTripCount(RHS, L, IterationInstructions, InductionPHI, TripCount,
                         Increment, BackBranch, SE, IsWidened);
}

static bool checkPHIs(FlattenInfo &FI, const TargetTransformInfo *TTI) {
  // All PHIs in the inner and outer headers must either be:
  // - The induction PHI, which we are going to rewrite as one induction in
  //   the new loop. This is already checked by findLoopComponents.
  // - An outer header PHI with all incoming values from outside the loop.
  //   LoopSimplify guarantees we have a pre-header, so we don't need to
  //   worry about that here.
  // - Pairs of PHIs in the inner and outer headers, which implement a
  //   loop-carried dependency that will still be valid in the new loop. To
  //   be valid, this variable must be modified only in the inner loop.

  // The set of PHI nodes in the outer loop header that we know will still be
  // valid after the transformation. These will not need to be modified (with
  // the exception of the induction variable), but we do need to check that
  // there are no unsafe PHI nodes.
  SmallPtrSet<PHINode *, 4> SafeOuterPHIs;
  SafeOuterPHIs.insert(FI.OuterInductionPHI);

  // Check that all PHI nodes in the inner loop header match one of the valid
  // patterns.
  for (PHINode &InnerPHI : FI.InnerLoop->getHeader()->phis()) {
    // The induction PHIs break these rules, and that's OK because we treat
    // them specially when doing the transformation.
    if (&InnerPHI == FI.InnerInductionPHI)
      continue;
    if (FI.isNarrowInductionPhi(&InnerPHI))
      continue;

    // Each inner loop PHI node must have two incoming values/blocks - one
    // from the pre-header, and one from the latch.
    assert(InnerPHI.getNumIncomingValues() == 2);
    Value *PreHeaderValue =
        InnerPHI.getIncomingValueForBlock(FI.InnerLoop->getLoopPreheader());
    Value *LatchValue =
        InnerPHI.getIncomingValueForBlock(FI.InnerLoop->getLoopLatch());

    // The incoming value from the outer loop must be the PHI node in the
    // outer loop header, with no modifications made in the top of the outer
    // loop.
    PHINode *OuterPHI = dyn_cast<PHINode>(PreHeaderValue);
    if (!OuterPHI || OuterPHI->getParent() != FI.OuterLoop->getHeader()) {
      LLVM_DEBUG(dbgs() << "value modified in top of outer loop\n");
      return false;
    }

    // The other incoming value must come from the inner loop, without any
    // modifications in the tail end of the outer loop. We are in LCSSA form,
    // so this will actually be a PHI in the inner loop's exit block, which
    // only uses values from inside the inner loop.
    PHINode *LCSSAPHI = dyn_cast<PHINode>(
        OuterPHI->getIncomingValueForBlock(FI.OuterLoop->getLoopLatch()));
    if (!LCSSAPHI) {
      LLVM_DEBUG(dbgs() << "could not find LCSSA PHI\n");
      return false;
    }

    // The value used by the LCSSA PHI must be the same one that the inner
    // loop's PHI uses.
    if (LCSSAPHI->hasConstantValue() != LatchValue) {
      LLVM_DEBUG(
          dbgs() << "LCSSA PHI incoming value does not match latch value\n");
      return false;
    }

    LLVM_DEBUG(dbgs() << "PHI pair is safe:\n");
    LLVM_DEBUG(dbgs() << "  Inner: "; InnerPHI.dump());
    LLVM_DEBUG(dbgs() << "  Outer: "; OuterPHI->dump());
    SafeOuterPHIs.insert(OuterPHI);
    FI.InnerPHIsToTransform.insert(&InnerPHI);
  }

  for (PHINode &OuterPHI : FI.OuterLoop->getHeader()->phis()) {
    if (FI.isNarrowInductionPhi(&OuterPHI))
      continue;
    if (!SafeOuterPHIs.count(&OuterPHI)) {
      LLVM_DEBUG(dbgs() << "found unsafe PHI in outer loop: "; OuterPHI.dump());
      return false;
    }
  }

  LLVM_DEBUG(dbgs() << "checkPHIs: OK\n");
  return true;
}

static bool
checkOuterLoopInsts(FlattenInfo &FI,
                    SmallPtrSetImpl<Instruction *> &IterationInstructions,
                    const TargetTransformInfo *TTI) {
  // Check for instructions in the outer but not inner loop. If any of these
  // have side-effects then this transformation is not legal, and if there is
  // a significant amount of code here which can't be optimised out that it's
  // not profitable (as these instructions would get executed for each
  // iteration of the inner loop).
  InstructionCost RepeatedInstrCost = 0;
  for (auto *B : FI.OuterLoop->getBlocks()) {
    if (FI.InnerLoop->contains(B))
      continue;

    for (auto &I : *B) {
      if (!isa<PHINode>(&I) && !I.isTerminator() &&
          !isSafeToSpeculativelyExecute(&I)) {
        LLVM_DEBUG(dbgs() << "Cannot flatten because instruction may have "
                             "side effects: ";
                   I.dump());
        return false;
      }
      // The execution count of the outer loop's iteration instructions
      // (increment, compare and branch) will be increased, but the
      // equivalent instructions will be removed from the inner loop, so
      // they make a net difference of zero.
      if (IterationInstructions.count(&I))
        continue;
      // The uncoditional branch to the inner loop's header will turn into
      // a fall-through, so adds no cost.
      BranchInst *Br = dyn_cast<BranchInst>(&I);
      if (Br && Br->isUnconditional() &&
          Br->getSuccessor(0) == FI.InnerLoop->getHeader())
        continue;
      // Multiplies of the outer iteration variable and inner iteration
      // count will be optimised out.
      if (match(&I, m_c_Mul(m_Specific(FI.OuterInductionPHI),
                            m_Specific(FI.InnerTripCount))))
        continue;
      InstructionCost Cost =
          TTI->getUserCost(&I, TargetTransformInfo::TCK_SizeAndLatency);
      LLVM_DEBUG(dbgs() << "Cost " << Cost << ": "; I.dump());
      RepeatedInstrCost += Cost;
    }
  }

  LLVM_DEBUG(dbgs() << "Cost of instructions that will be repeated: "
                    << RepeatedInstrCost << "\n");
  // Bail out if flattening the loops would cause instructions in the outer
  // loop but not in the inner loop to be executed extra times.
  if (RepeatedInstrCost > RepeatedInstructionThreshold) {
    LLVM_DEBUG(dbgs() << "checkOuterLoopInsts: not profitable, bailing.\n");
    return false;
  }

  LLVM_DEBUG(dbgs() << "checkOuterLoopInsts: OK\n");
  return true;
}



// We require all uses of both induction variables to match this pattern:
//
//   (OuterPHI * InnerTripCount) + InnerPHI
//
// Any uses of the induction variables not matching that pattern would
// require a div/mod to reconstruct in the flattened loop, so the
// transformation wouldn't be profitable.
static bool checkIVUsers(FlattenInfo &FI) {
  // Check that all uses of the inner loop's induction variable match the
  // expected pattern, recording the uses of the outer IV.
  SmallPtrSet<Value *, 4> ValidOuterPHIUses;
  if (!FI.checkInnerInductionPhiUsers(ValidOuterPHIUses))
    return false;

  // Check that there are no uses of the outer IV other than the ones found
  // as part of the pattern above.
  if (!FI.checkOuterInductionPhiUsers(ValidOuterPHIUses))
    return false;

  LLVM_DEBUG(dbgs() << "checkIVUsers: OK\n";
             dbgs() << "Found " << FI.LinearIVUses.size()
                    << " value(s) that can be replaced:\n";
             for (Value *V : FI.LinearIVUses) {
               dbgs() << "  ";
               V->dump();
             });
  return true;
}

// Return an OverflowResult dependant on if overflow of the multiplication of
// InnerTripCount and OuterTripCount can be assumed not to happen.
static OverflowResult checkOverflow(FlattenInfo &FI, DominatorTree *DT,
                                    AssumptionCache *AC) {
  Function *F = FI.OuterLoop->getHeader()->getParent();
  const DataLayout &DL = F->getParent()->getDataLayout();

  // For debugging/testing.
  if (AssumeNoOverflow)
    return OverflowResult::NeverOverflows;

  // Check if the multiply could not overflow due to known ranges of the
  // input values.
  OverflowResult OR = computeOverflowForUnsignedMul(
      FI.InnerTripCount, FI.OuterTripCount, DL, AC,
      FI.OuterLoop->getLoopPreheader()->getTerminator(), DT);
  if (OR != OverflowResult::MayOverflow)
    return OR;

  for (Value *V : FI.LinearIVUses) {
    for (Value *U : V->users()) {
      if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
        for (Value *GEPUser : U->users()) {
          auto *GEPUserInst = cast<Instruction>(GEPUser);
          if (!isa<LoadInst>(GEPUserInst) &&
              !(isa<StoreInst>(GEPUserInst) &&
                GEP == GEPUserInst->getOperand(1)))
            continue;
          if (!isGuaranteedToExecuteForEveryIteration(GEPUserInst,
                                                      FI.InnerLoop))
            continue;
          // The IV is used as the operand of a GEP which dominates the loop
          // latch, and the IV is at least as wide as the address space of the
          // GEP. In this case, the GEP would wrap around the address space
          // before the IV increment wraps, which would be UB.
          if (GEP->isInBounds() &&
              V->getType()->getIntegerBitWidth() >=
                  DL.getPointerTypeSizeInBits(GEP->getType())) {
            LLVM_DEBUG(
                dbgs() << "use of linear IV would be UB if overflow occurred: ";
                GEP->dump());
            return OverflowResult::NeverOverflows;
          }
        }
      }
    }
  }

  return OverflowResult::MayOverflow;
}

static bool CanFlattenLoopPair(FlattenInfo &FI, DominatorTree *DT, LoopInfo *LI,
                               ScalarEvolution *SE, AssumptionCache *AC,
                               const TargetTransformInfo *TTI) {
  SmallPtrSet<Instruction *, 8> IterationInstructions;
  if (!findLoopComponents(FI.InnerLoop, IterationInstructions,
                          FI.InnerInductionPHI, FI.InnerTripCount,
                          FI.InnerIncrement, FI.InnerBranch, SE, FI.Widened))
    return false;
  if (!findLoopComponents(FI.OuterLoop, IterationInstructions,
                          FI.OuterInductionPHI, FI.OuterTripCount,
                          FI.OuterIncrement, FI.OuterBranch, SE, FI.Widened))
    return false;

  // Both of the loop trip count values must be invariant in the outer loop
  // (non-instructions are all inherently invariant).
  if (!FI.OuterLoop->isLoopInvariant(FI.InnerTripCount)) {
    LLVM_DEBUG(dbgs() << "inner loop trip count not invariant\n");
    return false;
  }
  if (!FI.OuterLoop->isLoopInvariant(FI.OuterTripCount)) {
    LLVM_DEBUG(dbgs() << "outer loop trip count not invariant\n");
    return false;
  }

  if (!checkPHIs(FI, TTI))
    return false;

  // FIXME: it should be possible to handle different types correctly.
  if (FI.InnerInductionPHI->getType() != FI.OuterInductionPHI->getType())
    return false;

  if (!checkOuterLoopInsts(FI, IterationInstructions, TTI))
    return false;

  // Find the values in the loop that can be replaced with the linearized
  // induction variable, and check that there are no other uses of the inner
  // or outer induction variable. If there were, we could still do this
  // transformation, but we'd have to insert a div/mod to calculate the
  // original IVs, so it wouldn't be profitable.
  if (!checkIVUsers(FI))
    return false;

  LLVM_DEBUG(dbgs() << "CanFlattenLoopPair: OK\n");
  return true;
}

static bool DoFlattenLoopPair(FlattenInfo &FI, DominatorTree *DT, LoopInfo *LI,
                              ScalarEvolution *SE, AssumptionCache *AC,
                              const TargetTransformInfo *TTI, LPMUpdater *U,
                              MemorySSAUpdater *MSSAU) {
  Function *F = FI.OuterLoop->getHeader()->getParent();
  LLVM_DEBUG(dbgs() << "Checks all passed, doing the transformation\n");
  {
    using namespace ore;
    OptimizationRemark Remark(DEBUG_TYPE, "Flattened", FI.InnerLoop->getStartLoc(),
                              FI.InnerLoop->getHeader());
    OptimizationRemarkEmitter ORE(F);
    Remark << "Flattened into outer loop";
    ORE.emit(Remark);
  }

  Value *NewTripCount = BinaryOperator::CreateMul(
      FI.InnerTripCount, FI.OuterTripCount, "flatten.tripcount",
      FI.OuterLoop->getLoopPreheader()->getTerminator());
  LLVM_DEBUG(dbgs() << "Created new trip count in preheader: ";
             NewTripCount->dump());

  // Fix up PHI nodes that take values from the inner loop back-edge, which
  // we are about to remove.
  FI.InnerInductionPHI->removeIncomingValue(FI.InnerLoop->getLoopLatch());

  // The old Phi will be optimised away later, but for now we can't leave
  // leave it in an invalid state, so are updating them too.
  for (PHINode *PHI : FI.InnerPHIsToTransform)
    PHI->removeIncomingValue(FI.InnerLoop->getLoopLatch());

  // Modify the trip count of the outer loop to be the product of the two
  // trip counts.
  cast<User>(FI.OuterBranch->getCondition())->setOperand(1, NewTripCount);

  // Replace the inner loop backedge with an unconditional branch to the exit.
  BasicBlock *InnerExitBlock = FI.InnerLoop->getExitBlock();
  BasicBlock *InnerExitingBlock = FI.InnerLoop->getExitingBlock();
  InnerExitingBlock->getTerminator()->eraseFromParent();
  BranchInst::Create(InnerExitBlock, InnerExitingBlock);

  // Update the DomTree and MemorySSA.
  DT->deleteEdge(InnerExitingBlock, FI.InnerLoop->getHeader());
  if (MSSAU)
    MSSAU->removeEdge(InnerExitingBlock, FI.InnerLoop->getHeader());

  // Replace all uses of the polynomial calculated from the two induction
  // variables with the one new one.
  IRBuilder<> Builder(FI.OuterInductionPHI->getParent()->getTerminator());
  for (Value *V : FI.LinearIVUses) {
    Value *OuterValue = FI.OuterInductionPHI;
    if (FI.Widened)
      OuterValue = Builder.CreateTrunc(FI.OuterInductionPHI, V->getType(),
                                       "flatten.trunciv");

    LLVM_DEBUG(dbgs() << "Replacing: "; V->dump(); dbgs() << "with:      ";
               OuterValue->dump());
    V->replaceAllUsesWith(OuterValue);
  }

  // Tell LoopInfo, SCEV and the pass manager that the inner loop has been
  // deleted, and any information that have about the outer loop invalidated.
  SE->forgetLoop(FI.OuterLoop);
  SE->forgetLoop(FI.InnerLoop);
  if (U)
    U->markLoopAsDeleted(*FI.InnerLoop, FI.InnerLoop->getName());
  LI->erase(FI.InnerLoop);

  // Increment statistic value.
  NumFlattened++;

  return true;
}

static bool CanWidenIV(FlattenInfo &FI, DominatorTree *DT, LoopInfo *LI,
                       ScalarEvolution *SE, AssumptionCache *AC,
                       const TargetTransformInfo *TTI) {
  if (!WidenIV) {
    LLVM_DEBUG(dbgs() << "Widening the IVs is disabled\n");
    return false;
  }

  LLVM_DEBUG(dbgs() << "Try widening the IVs\n");
  Module *M = FI.InnerLoop->getHeader()->getParent()->getParent();
  auto &DL = M->getDataLayout();
  auto *InnerType = FI.InnerInductionPHI->getType();
  auto *OuterType = FI.OuterInductionPHI->getType();
  unsigned MaxLegalSize = DL.getLargestLegalIntTypeSizeInBits();
  auto *MaxLegalType = DL.getLargestLegalIntType(M->getContext());

  // If both induction types are less than the maximum legal integer width,
  // promote both to the widest type available so we know calculating
  // (OuterTripCount * InnerTripCount) as the new trip count is safe.
  if (InnerType != OuterType ||
      InnerType->getScalarSizeInBits() >= MaxLegalSize ||
      MaxLegalType->getScalarSizeInBits() <
          InnerType->getScalarSizeInBits() * 2) {
    LLVM_DEBUG(dbgs() << "Can't widen the IV\n");
    return false;
  }

  SCEVExpander Rewriter(*SE, DL, "loopflatten");
  SmallVector<WeakTrackingVH, 4> DeadInsts;
  unsigned ElimExt = 0;
  unsigned Widened = 0;

  auto CreateWideIV = [&](WideIVInfo WideIV, bool &Deleted) -> bool {
    PHINode *WidePhi =
        createWideIV(WideIV, LI, SE, Rewriter, DT, DeadInsts, ElimExt, Widened,
                     true /* HasGuards */, true /* UsePostIncrementRanges */);
    if (!WidePhi)
      return false;
    LLVM_DEBUG(dbgs() << "Created wide phi: "; WidePhi->dump());
    LLVM_DEBUG(dbgs() << "Deleting old phi: "; WideIV.NarrowIV->dump());
    Deleted = RecursivelyDeleteDeadPHINode(WideIV.NarrowIV);
    return true;
  };

  bool Deleted;
  if (!CreateWideIV({FI.InnerInductionPHI, MaxLegalType, false}, Deleted))
    return false;
  // Add the narrow phi to list, so that it will be adjusted later when the
  // the transformation is performed.
  if (!Deleted)
    FI.InnerPHIsToTransform.insert(FI.InnerInductionPHI);

  if (!CreateWideIV({FI.OuterInductionPHI, MaxLegalType, false}, Deleted))
    return false;

  assert(Widened && "Widened IV expected");
  FI.Widened = true;

  // Save the old/narrow induction phis, which we need to ignore in CheckPHIs.
  FI.NarrowInnerInductionPHI = FI.InnerInductionPHI;
  FI.NarrowOuterInductionPHI = FI.OuterInductionPHI;

  // After widening, rediscover all the loop components.
  return CanFlattenLoopPair(FI, DT, LI, SE, AC, TTI);
}

static bool FlattenLoopPair(FlattenInfo &FI, DominatorTree *DT, LoopInfo *LI,
                            ScalarEvolution *SE, AssumptionCache *AC,
                            const TargetTransformInfo *TTI, LPMUpdater *U,
                            MemorySSAUpdater *MSSAU) {
  LLVM_DEBUG(
      dbgs() << "Loop flattening running on outer loop "
             << FI.OuterLoop->getHeader()->getName() << " and inner loop "
             << FI.InnerLoop->getHeader()->getName() << " in "
             << FI.OuterLoop->getHeader()->getParent()->getName() << "\n");

  if (!CanFlattenLoopPair(FI, DT, LI, SE, AC, TTI))
    return false;

  // Check if we can widen the induction variables to avoid overflow checks.
  bool CanFlatten = CanWidenIV(FI, DT, LI, SE, AC, TTI);

  // It can happen that after widening of the IV, flattening may not be
  // possible/happening, e.g. when it is deemed unprofitable. So bail here if
  // that is the case.
  // TODO: IV widening without performing the actual flattening transformation
  // is not ideal. While this codegen change should not matter much, it is an
  // unnecessary change which is better to avoid. It's unlikely this happens
  // often, because if it's unprofitibale after widening, it should be
  // unprofitabe before widening as checked in the first round of checks. But
  // 'RepeatedInstructionThreshold' is set to only 2, which can probably be
  // relaxed. Because this is making a code change (the IV widening, but not
  // the flattening), we return true here.
  if (FI.Widened && !CanFlatten)
    return true;

  // If we have widened and can perform the transformation, do that here.
  if (CanFlatten)
    return DoFlattenLoopPair(FI, DT, LI, SE, AC, TTI, U, MSSAU);

  // Otherwise, if we haven't widened the IV, check if the new iteration
  // variable might overflow. In this case, we need to version the loop, and
  // select the original version at runtime if the iteration space is too
  // large.
  // TODO: We currently don't version the loop.
  OverflowResult OR = checkOverflow(FI, DT, AC);
  if (OR == OverflowResult::AlwaysOverflowsHigh ||
      OR == OverflowResult::AlwaysOverflowsLow) {
    LLVM_DEBUG(dbgs() << "Multiply would always overflow, so not profitable\n");
    return false;
  } else if (OR == OverflowResult::MayOverflow) {
    LLVM_DEBUG(dbgs() << "Multiply might overflow, not flattening\n");
    return false;
  }

  LLVM_DEBUG(dbgs() << "Multiply cannot overflow, modifying loop in-place\n");
  return DoFlattenLoopPair(FI, DT, LI, SE, AC, TTI, U, MSSAU);
}

bool Flatten(LoopNest &LN, DominatorTree *DT, LoopInfo *LI, ScalarEvolution *SE,
             AssumptionCache *AC, TargetTransformInfo *TTI, LPMUpdater *U,
             MemorySSAUpdater *MSSAU) {
  bool Changed = false;
  for (Loop *InnerLoop : LN.getLoops()) {
    auto *OuterLoop = InnerLoop->getParentLoop();
    if (!OuterLoop)
      continue;
    FlattenInfo FI(OuterLoop, InnerLoop);
    Changed |= FlattenLoopPair(FI, DT, LI, SE, AC, TTI, U, MSSAU);
  }
  return Changed;
}

PreservedAnalyses LoopFlattenPass::run(LoopNest &LN, LoopAnalysisManager &LAM,
                                       LoopStandardAnalysisResults &AR,
                                       LPMUpdater &U) {

  bool Changed = false;

  Optional<MemorySSAUpdater> MSSAU;
  if (AR.MSSA) {
    MSSAU = MemorySSAUpdater(AR.MSSA);
    if (VerifyMemorySSA)
      AR.MSSA->verifyMemorySSA();
  }

  // The loop flattening pass requires loops to be
  // in simplified form, and also needs LCSSA. Running
  // this pass will simplify all loops that contain inner loops,
  // regardless of whether anything ends up being flattened.
  Changed |= Flatten(LN, &AR.DT, &AR.LI, &AR.SE, &AR.AC, &AR.TTI, &U,
                     MSSAU.hasValue() ? MSSAU.getPointer() : nullptr);

  if (!Changed)
    return PreservedAnalyses::all();

  if (AR.MSSA && VerifyMemorySSA)
    AR.MSSA->verifyMemorySSA();

  auto PA = getLoopPassPreservedAnalyses();
  if (AR.MSSA)
    PA.preserve<MemorySSAAnalysis>();
  return PA;
}

namespace {
class LoopFlattenLegacyPass : public FunctionPass {
public:
  static char ID; // Pass ID, replacement for typeid
  LoopFlattenLegacyPass() : FunctionPass(ID) {
    initializeLoopFlattenLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  // Possibly flatten loop L into its child.
  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    getLoopAnalysisUsage(AU);
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.addPreserved<TargetTransformInfoWrapperPass>();
    AU.addRequired<AssumptionCacheTracker>();
    AU.addPreserved<AssumptionCacheTracker>();
    AU.addPreserved<MemorySSAWrapperPass>();
  }
};
} // namespace

char LoopFlattenLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(LoopFlattenLegacyPass, "loop-flatten", "Flattens loops",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_END(LoopFlattenLegacyPass, "loop-flatten", "Flattens loops",
                    false, false)

FunctionPass *llvm::createLoopFlattenPass() {
  return new LoopFlattenLegacyPass();
}

bool LoopFlattenLegacyPass::runOnFunction(Function &F) {
  ScalarEvolution *SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  LoopInfo *LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  auto *DTWP = getAnalysisIfAvailable<DominatorTreeWrapperPass>();
  DominatorTree *DT = DTWP ? &DTWP->getDomTree() : nullptr;
  auto &TTIP = getAnalysis<TargetTransformInfoWrapperPass>();
  auto *TTI = &TTIP.getTTI(F);
  auto *AC = &getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
  auto *MSSA = getAnalysisIfAvailable<MemorySSAWrapperPass>();

  Optional<MemorySSAUpdater> MSSAU;
  if (MSSA)
    MSSAU = MemorySSAUpdater(&MSSA->getMSSA());

  bool Changed = false;
  for (Loop *L : *LI) {
    auto LN = LoopNest::getLoopNest(*L, *SE);
    Changed |= Flatten(*LN, DT, LI, SE, AC, TTI, nullptr,
                       MSSAU.hasValue() ? MSSAU.getPointer() : nullptr);
  }
  return Changed;
}
