#include "llvm/Transforms/Utils/LoopConstrainer.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"

using namespace llvm;

static const char *ClonedLoopTag = "loop_constrainer.loop.clone";

#define DEBUG_TYPE "loop-constrainer"

/// Given a loop with an deccreasing induction variable, is it possible to
/// safely calculate the bounds of a new loop using the given Predicate.
static bool isSafeDecreasingBound(const SCEV *Start, const SCEV *BoundSCEV,
                                  const SCEV *Step, ICmpInst::Predicate Pred,
                                  unsigned LatchBrExitIdx, Loop *L,
                                  ScalarEvolution &SE) {
  if (Pred != ICmpInst::ICMP_SLT && Pred != ICmpInst::ICMP_SGT &&
      Pred != ICmpInst::ICMP_ULT && Pred != ICmpInst::ICMP_UGT)
    return false;

  if (!SE.isAvailableAtLoopEntry(BoundSCEV, L))
    return false;

  assert(SE.isKnownNegative(Step) && "expecting negative step");

  LLVM_DEBUG(dbgs() << "isSafeDecreasingBound with:\n");
  LLVM_DEBUG(dbgs() << "Start: " << *Start << "\n");
  LLVM_DEBUG(dbgs() << "Step: " << *Step << "\n");
  LLVM_DEBUG(dbgs() << "BoundSCEV: " << *BoundSCEV << "\n");
  LLVM_DEBUG(dbgs() << "Pred: " << Pred << "\n");
  LLVM_DEBUG(dbgs() << "LatchExitBrIdx: " << LatchBrExitIdx << "\n");

  bool IsSigned = ICmpInst::isSigned(Pred);
  // The predicate that we need to check that the induction variable lies
  // within bounds.
  ICmpInst::Predicate BoundPred =
      IsSigned ? CmpInst::ICMP_SGT : CmpInst::ICMP_UGT;

  if (LatchBrExitIdx == 1)
    return SE.isLoopEntryGuardedByCond(L, BoundPred, Start, BoundSCEV);

  assert(LatchBrExitIdx == 0 && "LatchBrExitIdx should be either 0 or 1");

  const SCEV *StepPlusOne = SE.getAddExpr(Step, SE.getOne(Step->getType()));
  unsigned BitWidth = cast<IntegerType>(BoundSCEV->getType())->getBitWidth();
  APInt Min = IsSigned ? APInt::getSignedMinValue(BitWidth)
                       : APInt::getMinValue(BitWidth);
  const SCEV *Limit = SE.getMinusSCEV(SE.getConstant(Min), StepPlusOne);

  const SCEV *MinusOne =
      SE.getMinusSCEV(BoundSCEV, SE.getOne(BoundSCEV->getType()));

  return SE.isLoopEntryGuardedByCond(L, BoundPred, Start, MinusOne) &&
         SE.isLoopEntryGuardedByCond(L, BoundPred, BoundSCEV, Limit);
}

/// Given a loop with an increasing induction variable, is it possible to
/// safely calculate the bounds of a new loop using the given Predicate.
static bool isSafeIncreasingBound(const SCEV *Start, const SCEV *BoundSCEV,
                                  const SCEV *Step, ICmpInst::Predicate Pred,
                                  unsigned LatchBrExitIdx, Loop *L,
                                  ScalarEvolution &SE) {
  if (Pred != ICmpInst::ICMP_SLT && Pred != ICmpInst::ICMP_SGT &&
      Pred != ICmpInst::ICMP_ULT && Pred != ICmpInst::ICMP_UGT)
    return false;

  if (!SE.isAvailableAtLoopEntry(BoundSCEV, L))
    return false;

  LLVM_DEBUG(dbgs() << "isSafeIncreasingBound with:\n");
  LLVM_DEBUG(dbgs() << "Start: " << *Start << "\n");
  LLVM_DEBUG(dbgs() << "Step: " << *Step << "\n");
  LLVM_DEBUG(dbgs() << "BoundSCEV: " << *BoundSCEV << "\n");
  LLVM_DEBUG(dbgs() << "Pred: " << Pred << "\n");
  LLVM_DEBUG(dbgs() << "LatchExitBrIdx: " << LatchBrExitIdx << "\n");

  bool IsSigned = ICmpInst::isSigned(Pred);
  // The predicate that we need to check that the induction variable lies
  // within bounds.
  ICmpInst::Predicate BoundPred =
      IsSigned ? CmpInst::ICMP_SLT : CmpInst::ICMP_ULT;

  if (LatchBrExitIdx == 1)
    return SE.isLoopEntryGuardedByCond(L, BoundPred, Start, BoundSCEV);

  assert(LatchBrExitIdx == 0 && "LatchBrExitIdx should be 0 or 1");

  const SCEV *StepMinusOne = SE.getMinusSCEV(Step, SE.getOne(Step->getType()));
  unsigned BitWidth = cast<IntegerType>(BoundSCEV->getType())->getBitWidth();
  APInt Max = IsSigned ? APInt::getSignedMaxValue(BitWidth)
                       : APInt::getMaxValue(BitWidth);
  const SCEV *Limit = SE.getMinusSCEV(SE.getConstant(Max), StepMinusOne);

  return (SE.isLoopEntryGuardedByCond(L, BoundPred, Start,
                                      SE.getAddExpr(BoundSCEV, Step)) &&
          SE.isLoopEntryGuardedByCond(L, BoundPred, BoundSCEV, Limit));
}

/// Returns estimate for max latch taken count of the loop of the narrowest
/// available type. If the latch block has such estimate, it is returned.
/// Otherwise, we use max exit count of whole loop (that is potentially of wider
/// type than latch check itself), which is still better than no estimate.
static const SCEV *getNarrowestLatchMaxTakenCountEstimate(ScalarEvolution &SE,
                                                          const Loop &L) {
  const SCEV *FromBlock =
      SE.getExitCount(&L, L.getLoopLatch(), ScalarEvolution::SymbolicMaximum);
  if (isa<SCEVCouldNotCompute>(FromBlock))
    return SE.getSymbolicMaxBackedgeTakenCount(&L);
  return FromBlock;
}

std::optional<LoopStructure>
LoopStructure::parseLoopStructure(ScalarEvolution &SE, Loop &L,
                                  bool AllowUnsignedLatchCond,
                                  const char *&FailureReason) {
  if (!L.isLoopSimplifyForm()) {
    FailureReason = "loop not in LoopSimplify form";
    return std::nullopt;
  }

  BasicBlock *Latch = L.getLoopLatch();
  assert(Latch && "Simplified loops only have one latch!");

  if (Latch->getTerminator()->getMetadata(ClonedLoopTag)) {
    FailureReason = "loop has already been cloned";
    return std::nullopt;
  }

  if (!L.isLoopExiting(Latch)) {
    FailureReason = "no loop latch";
    return std::nullopt;
  }

  BasicBlock *Header = L.getHeader();
  BasicBlock *Preheader = L.getLoopPreheader();
  if (!Preheader) {
    FailureReason = "no preheader";
    return std::nullopt;
  }

  BranchInst *LatchBr = dyn_cast<BranchInst>(Latch->getTerminator());
  if (!LatchBr || LatchBr->isUnconditional()) {
    FailureReason = "latch terminator not conditional branch";
    return std::nullopt;
  }

  unsigned LatchBrExitIdx = LatchBr->getSuccessor(0) == Header ? 1 : 0;

  ICmpInst *ICI = dyn_cast<ICmpInst>(LatchBr->getCondition());
  if (!ICI || !isa<IntegerType>(ICI->getOperand(0)->getType())) {
    FailureReason = "latch terminator branch not conditional on integral icmp";
    return std::nullopt;
  }

  const SCEV *MaxBETakenCount = getNarrowestLatchMaxTakenCountEstimate(SE, L);
  if (isa<SCEVCouldNotCompute>(MaxBETakenCount)) {
    FailureReason = "could not compute latch count";
    return std::nullopt;
  }
  assert(SE.getLoopDisposition(MaxBETakenCount, &L) ==
             ScalarEvolution::LoopInvariant &&
         "loop variant exit count doesn't make sense!");

  ICmpInst::Predicate Pred = ICI->getPredicate();
  Value *LeftValue = ICI->getOperand(0);
  const SCEV *LeftSCEV = SE.getSCEV(LeftValue);
  IntegerType *IndVarTy = cast<IntegerType>(LeftValue->getType());

  Value *RightValue = ICI->getOperand(1);
  const SCEV *RightSCEV = SE.getSCEV(RightValue);

  // We canonicalize `ICI` such that `LeftSCEV` is an add recurrence.
  if (!isa<SCEVAddRecExpr>(LeftSCEV)) {
    if (isa<SCEVAddRecExpr>(RightSCEV)) {
      std::swap(LeftSCEV, RightSCEV);
      std::swap(LeftValue, RightValue);
      Pred = ICmpInst::getSwappedPredicate(Pred);
    } else {
      FailureReason = "no add recurrences in the icmp";
      return std::nullopt;
    }
  }

  auto HasNoSignedWrap = [&](const SCEVAddRecExpr *AR) {
    if (AR->getNoWrapFlags(SCEV::FlagNSW))
      return true;

    IntegerType *Ty = cast<IntegerType>(AR->getType());
    IntegerType *WideTy =
        IntegerType::get(Ty->getContext(), Ty->getBitWidth() * 2);

    const SCEVAddRecExpr *ExtendAfterOp =
        dyn_cast<SCEVAddRecExpr>(SE.getSignExtendExpr(AR, WideTy));
    if (ExtendAfterOp) {
      const SCEV *ExtendedStart = SE.getSignExtendExpr(AR->getStart(), WideTy);
      const SCEV *ExtendedStep =
          SE.getSignExtendExpr(AR->getStepRecurrence(SE), WideTy);

      bool NoSignedWrap = ExtendAfterOp->getStart() == ExtendedStart &&
                          ExtendAfterOp->getStepRecurrence(SE) == ExtendedStep;

      if (NoSignedWrap)
        return true;
    }

    // We may have proved this when computing the sign extension above.
    return AR->getNoWrapFlags(SCEV::FlagNSW) != SCEV::FlagAnyWrap;
  };

  // `ICI` is interpreted as taking the backedge if the *next* value of the
  // induction variable satisfies some constraint.

  const SCEVAddRecExpr *IndVarBase = cast<SCEVAddRecExpr>(LeftSCEV);
  if (IndVarBase->getLoop() != &L) {
    FailureReason = "LHS in cmp is not an AddRec for this loop";
    return std::nullopt;
  }
  if (!IndVarBase->isAffine()) {
    FailureReason = "LHS in icmp not induction variable";
    return std::nullopt;
  }
  const SCEV *StepRec = IndVarBase->getStepRecurrence(SE);
  if (!isa<SCEVConstant>(StepRec)) {
    FailureReason = "LHS in icmp not induction variable";
    return std::nullopt;
  }
  ConstantInt *StepCI = cast<SCEVConstant>(StepRec)->getValue();

  if (ICI->isEquality() && !HasNoSignedWrap(IndVarBase)) {
    FailureReason = "LHS in icmp needs nsw for equality predicates";
    return std::nullopt;
  }

  assert(!StepCI->isZero() && "Zero step?");
  bool IsIncreasing = !StepCI->isNegative();
  bool IsSignedPredicate;
  const SCEV *StartNext = IndVarBase->getStart();
  const SCEV *Addend = SE.getNegativeSCEV(IndVarBase->getStepRecurrence(SE));
  const SCEV *IndVarStart = SE.getAddExpr(StartNext, Addend);
  const SCEV *Step = SE.getSCEV(StepCI);

  const SCEV *FixedRightSCEV = nullptr;

  // If RightValue resides within loop (but still being loop invariant),
  // regenerate it as preheader.
  if (auto *I = dyn_cast<Instruction>(RightValue))
    if (L.contains(I->getParent()))
      FixedRightSCEV = RightSCEV;

  if (IsIncreasing) {
    bool DecreasedRightValueByOne = false;
    if (StepCI->isOne()) {
      // Try to turn eq/ne predicates to those we can work with.
      if (Pred == ICmpInst::ICMP_NE && LatchBrExitIdx == 1)
        // while (++i != len) {         while (++i < len) {
        //   ...                 --->     ...
        // }                            }
        // If both parts are known non-negative, it is profitable to use
        // unsigned comparison in increasing loop. This allows us to make the
        // comparison check against "RightSCEV + 1" more optimistic.
        if (isKnownNonNegativeInLoop(IndVarStart, &L, SE) &&
            isKnownNonNegativeInLoop(RightSCEV, &L, SE))
          Pred = ICmpInst::ICMP_ULT;
        else
          Pred = ICmpInst::ICMP_SLT;
      else if (Pred == ICmpInst::ICMP_EQ && LatchBrExitIdx == 0) {
        // while (true) {               while (true) {
        //   if (++i == len)     --->     if (++i > len - 1)
        //     break;                       break;
        //   ...                          ...
        // }                            }
        if (IndVarBase->getNoWrapFlags(SCEV::FlagNUW) &&
            cannotBeMinInLoop(RightSCEV, &L, SE, /*Signed*/ false)) {
          Pred = ICmpInst::ICMP_UGT;
          RightSCEV =
              SE.getMinusSCEV(RightSCEV, SE.getOne(RightSCEV->getType()));
          DecreasedRightValueByOne = true;
        } else if (cannotBeMinInLoop(RightSCEV, &L, SE, /*Signed*/ true)) {
          Pred = ICmpInst::ICMP_SGT;
          RightSCEV =
              SE.getMinusSCEV(RightSCEV, SE.getOne(RightSCEV->getType()));
          DecreasedRightValueByOne = true;
        }
      }
    }

    bool LTPred = (Pred == ICmpInst::ICMP_SLT || Pred == ICmpInst::ICMP_ULT);
    bool GTPred = (Pred == ICmpInst::ICMP_SGT || Pred == ICmpInst::ICMP_UGT);
    bool FoundExpectedPred =
        (LTPred && LatchBrExitIdx == 1) || (GTPred && LatchBrExitIdx == 0);

    if (!FoundExpectedPred) {
      FailureReason = "expected icmp slt semantically, found something else";
      return std::nullopt;
    }

    IsSignedPredicate = ICmpInst::isSigned(Pred);
    if (!IsSignedPredicate && !AllowUnsignedLatchCond) {
      FailureReason = "unsigned latch conditions are explicitly prohibited";
      return std::nullopt;
    }

    if (!isSafeIncreasingBound(IndVarStart, RightSCEV, Step, Pred,
                               LatchBrExitIdx, &L, SE)) {
      FailureReason = "Unsafe loop bounds";
      return std::nullopt;
    }
    if (LatchBrExitIdx == 0) {
      // We need to increase the right value unless we have already decreased
      // it virtually when we replaced EQ with SGT.
      if (!DecreasedRightValueByOne)
        FixedRightSCEV =
            SE.getAddExpr(RightSCEV, SE.getOne(RightSCEV->getType()));
    } else {
      assert(!DecreasedRightValueByOne &&
             "Right value can be decreased only for LatchBrExitIdx == 0!");
    }
  } else {
    bool IncreasedRightValueByOne = false;
    if (StepCI->isMinusOne()) {
      // Try to turn eq/ne predicates to those we can work with.
      if (Pred == ICmpInst::ICMP_NE && LatchBrExitIdx == 1)
        // while (--i != len) {         while (--i > len) {
        //   ...                 --->     ...
        // }                            }
        // We intentionally don't turn the predicate into UGT even if we know
        // that both operands are non-negative, because it will only pessimize
        // our check against "RightSCEV - 1".
        Pred = ICmpInst::ICMP_SGT;
      else if (Pred == ICmpInst::ICMP_EQ && LatchBrExitIdx == 0) {
        // while (true) {               while (true) {
        //   if (--i == len)     --->     if (--i < len + 1)
        //     break;                       break;
        //   ...                          ...
        // }                            }
        if (IndVarBase->getNoWrapFlags(SCEV::FlagNUW) &&
            cannotBeMaxInLoop(RightSCEV, &L, SE, /* Signed */ false)) {
          Pred = ICmpInst::ICMP_ULT;
          RightSCEV = SE.getAddExpr(RightSCEV, SE.getOne(RightSCEV->getType()));
          IncreasedRightValueByOne = true;
        } else if (cannotBeMaxInLoop(RightSCEV, &L, SE, /* Signed */ true)) {
          Pred = ICmpInst::ICMP_SLT;
          RightSCEV = SE.getAddExpr(RightSCEV, SE.getOne(RightSCEV->getType()));
          IncreasedRightValueByOne = true;
        }
      }
    }

    bool LTPred = (Pred == ICmpInst::ICMP_SLT || Pred == ICmpInst::ICMP_ULT);
    bool GTPred = (Pred == ICmpInst::ICMP_SGT || Pred == ICmpInst::ICMP_UGT);

    bool FoundExpectedPred =
        (GTPred && LatchBrExitIdx == 1) || (LTPred && LatchBrExitIdx == 0);

    if (!FoundExpectedPred) {
      FailureReason = "expected icmp sgt semantically, found something else";
      return std::nullopt;
    }

    IsSignedPredicate =
        Pred == ICmpInst::ICMP_SLT || Pred == ICmpInst::ICMP_SGT;

    if (!IsSignedPredicate && !AllowUnsignedLatchCond) {
      FailureReason = "unsigned latch conditions are explicitly prohibited";
      return std::nullopt;
    }

    if (!isSafeDecreasingBound(IndVarStart, RightSCEV, Step, Pred,
                               LatchBrExitIdx, &L, SE)) {
      FailureReason = "Unsafe bounds";
      return std::nullopt;
    }

    if (LatchBrExitIdx == 0) {
      // We need to decrease the right value unless we have already increased
      // it virtually when we replaced EQ with SLT.
      if (!IncreasedRightValueByOne)
        FixedRightSCEV =
            SE.getMinusSCEV(RightSCEV, SE.getOne(RightSCEV->getType()));
    } else {
      assert(!IncreasedRightValueByOne &&
             "Right value can be increased only for LatchBrExitIdx == 0!");
    }
  }
  BasicBlock *LatchExit = LatchBr->getSuccessor(LatchBrExitIdx);

  assert(!L.contains(LatchExit) && "expected an exit block!");
  const DataLayout &DL = Preheader->getModule()->getDataLayout();
  SCEVExpander Expander(SE, DL, "loop-constrainer");
  Instruction *Ins = Preheader->getTerminator();

  if (FixedRightSCEV)
    RightValue =
        Expander.expandCodeFor(FixedRightSCEV, FixedRightSCEV->getType(), Ins);

  Value *IndVarStartV = Expander.expandCodeFor(IndVarStart, IndVarTy, Ins);
  IndVarStartV->setName("indvar.start");

  LoopStructure Result;

  Result.Tag = "main";
  Result.Header = Header;
  Result.Latch = Latch;
  Result.LatchBr = LatchBr;
  Result.LatchExit = LatchExit;
  Result.LatchBrExitIdx = LatchBrExitIdx;
  Result.IndVarStart = IndVarStartV;
  Result.IndVarStep = StepCI;
  Result.IndVarBase = LeftValue;
  Result.IndVarIncreasing = IsIncreasing;
  Result.LoopExitAt = RightValue;
  Result.IsSignedPredicate = IsSignedPredicate;
  Result.ExitCountTy = cast<IntegerType>(MaxBETakenCount->getType());

  FailureReason = nullptr;

  return Result;
}

