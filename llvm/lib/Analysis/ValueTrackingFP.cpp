//===- ValueTrackingFP.cpp - Floating-point value tracking ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains routines that help analyze properties that chains of
// floating-point computations have.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/DomConditionCache.h"
#include "llvm/Analysis/FloatingPointPredicateUtils.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/KnownFPClass.h"
#include <cassert>
#include <optional>

using namespace llvm;
using namespace llvm::PatternMatch;

static void computeKnownFPClassFromCond(const Value *V, Value *Cond,
                                        bool CondIsTrue,
                                        const Instruction *CtxI,
                                        KnownFPClass &KnownFromContext,
                                        unsigned Depth = 0) {
  Value *A, *B;
  if (Depth < MaxAnalysisRecursionDepth &&
      (CondIsTrue ? match(Cond, m_LogicalAnd(m_Value(A), m_Value(B)))
                  : match(Cond, m_LogicalOr(m_Value(A), m_Value(B))))) {
    computeKnownFPClassFromCond(V, A, CondIsTrue, CtxI, KnownFromContext,
                                Depth + 1);
    computeKnownFPClassFromCond(V, B, CondIsTrue, CtxI, KnownFromContext,
                                Depth + 1);
    return;
  }
  if (Depth < MaxAnalysisRecursionDepth && match(Cond, m_Not(m_Value(A)))) {
    computeKnownFPClassFromCond(V, A, !CondIsTrue, CtxI, KnownFromContext,
                                Depth + 1);
    return;
  }
  CmpPredicate Pred;
  Value *LHS;
  uint64_t ClassVal = 0;
  const APFloat *CRHS;
  const APInt *RHS;
  if (match(Cond, m_FCmp(Pred, m_Value(LHS), m_APFloat(CRHS)))) {
    auto [CmpVal, MaskIfTrue, MaskIfFalse] = fcmpImpliesClass(
        Pred, *cast<Instruction>(Cond)->getParent()->getParent(), LHS, *CRHS,
        LHS != V);
    if (CmpVal == V)
      KnownFromContext.knownNot(~(CondIsTrue ? MaskIfTrue : MaskIfFalse));
  } else if (match(Cond, m_Intrinsic<Intrinsic::is_fpclass>(
                             m_Specific(V), m_ConstantInt(ClassVal)))) {
    FPClassTest Mask = static_cast<FPClassTest>(ClassVal);
    KnownFromContext.knownNot(CondIsTrue ? ~Mask : Mask);
  } else if (match(Cond, m_ICmp(Pred, m_ElementWiseBitCast(m_Specific(V)),
                                m_APInt(RHS)))) {
    bool TrueIfSigned;
    if (!isSignBitCheck(Pred, *RHS, TrueIfSigned))
      return;
    if (TrueIfSigned == CondIsTrue)
      KnownFromContext.signBitMustBeOne();
    else
      KnownFromContext.signBitMustBeZero();
  }
}

static KnownFPClass computeKnownFPClassFromContext(const Value *V,
                                                   const SimplifyQuery &Q) {
  KnownFPClass KnownFromContext;

  if (Q.CC && Q.CC->AffectedValues.contains(V))
    computeKnownFPClassFromCond(V, Q.CC->Cond, !Q.CC->Invert, Q.CtxI,
                                KnownFromContext);

  if (!Q.CtxI)
    return KnownFromContext;

  if (Q.DC && Q.DT) {
    // Handle dominating conditions.
    for (CondBrInst *BI : Q.DC->conditionsFor(V)) {
      Value *Cond = BI->getCondition();

      BasicBlockEdge Edge0(BI->getParent(), BI->getSuccessor(0));
      if (Q.DT->dominates(Edge0, Q.CtxI->getParent()))
        computeKnownFPClassFromCond(V, Cond, /*CondIsTrue=*/true, Q.CtxI,
                                    KnownFromContext);

      BasicBlockEdge Edge1(BI->getParent(), BI->getSuccessor(1));
      if (Q.DT->dominates(Edge1, Q.CtxI->getParent()))
        computeKnownFPClassFromCond(V, Cond, /*CondIsTrue=*/false, Q.CtxI,
                                    KnownFromContext);
    }
  }

  if (!Q.AC)
    return KnownFromContext;

  // Try to restrict the floating-point classes based on information from
  // assumptions.
  for (auto &AssumeVH : Q.AC->assumptionsFor(V)) {
    if (!AssumeVH)
      continue;
    CallInst *I = cast<CallInst>(AssumeVH);

    assert(I->getFunction() == Q.CtxI->getParent()->getParent() &&
           "Got assumption for the wrong function!");
    assert(I->getIntrinsicID() == Intrinsic::assume &&
           "must be an assume intrinsic");

    if (!isValidAssumeForContext(I, Q))
      continue;

    computeKnownFPClassFromCond(V, I->getArgOperand(0),
                                /*CondIsTrue=*/true, Q.CtxI, KnownFromContext);
  }

  return KnownFromContext;
}

void llvm::adjustKnownFPClassForSelectArm(KnownFPClass &Known, Value *Cond,
                                          Value *Arm, bool Invert,
                                          const SimplifyQuery &SQ,
                                          unsigned Depth) {

  KnownFPClass KnownSrc;
  computeKnownFPClassFromCond(Arm, Cond,
                              /*CondIsTrue=*/!Invert, SQ.CtxI, KnownSrc,
                              Depth + 1);
  KnownSrc = KnownSrc.unionWith(Known);
  if (KnownSrc.isUnknown())
    return;

  if (isGuaranteedNotToBeUndef(Arm, SQ.AC, SQ.CtxI, SQ.DT, Depth + 1))
    Known = KnownSrc;
}

void computeKnownFPClass(const Value *V, const APInt &DemandedElts,
                         FPClassTest InterestedClasses, KnownFPClass &Known,
                         const SimplifyQuery &Q, unsigned Depth);

static void computeKnownFPClass(const Value *V, KnownFPClass &Known,
                                FPClassTest InterestedClasses,
                                const SimplifyQuery &Q, unsigned Depth) {
  auto *FVTy = dyn_cast<FixedVectorType>(V->getType());
  APInt DemandedElts =
      FVTy ? APInt::getAllOnes(FVTy->getNumElements()) : APInt(1, 1);
  computeKnownFPClass(V, DemandedElts, InterestedClasses, Known, Q, Depth);
}

static void computeKnownFPClassForFPTrunc(const Operator *Op, RoundingMode RM,
                                          const APInt &DemandedElts,
                                          FPClassTest InterestedClasses,
                                          KnownFPClass &Known,
                                          const SimplifyQuery &Q,
                                          unsigned Depth) {
  if ((InterestedClasses & (KnownFPClass::OrderedLessThanZeroMask | fcNan)) ==
      fcNone)
    return;

  KnownFPClass KnownSrc;
  computeKnownFPClass(Op->getOperand(0), DemandedElts, InterestedClasses,
                      KnownSrc, Q, Depth + 1);

  const fltSemantics &DstTy = Op->getType()->getScalarType()->getFltSemantics();
  const fltSemantics &SrcTy =
      Op->getOperand(0)->getType()->getScalarType()->getFltSemantics();

  const auto *I = dyn_cast<Instruction>(Op);
  const Function *F = I ? I->getFunction() : nullptr;
  DenormalMode DstMode =
      F ? F->getDenormalMode(DstTy) : DenormalMode::getDynamic();
  DenormalMode SrcMode =
      F ? F->getDenormalMode(SrcTy) : DenormalMode::getDynamic();

  Known = KnownFPClass::fptrunc(KnownSrc, DstTy, SrcTy, RM, DstMode, SrcMode);
}

static void
computeKnownFPClassForFAddSub(const Operator *Op, const APInt &DemandedElts,
                              FPClassTest InterestedClasses,
                              KnownFPClass &Known, const SimplifyQuery &Q,
                              unsigned Depth, bool IsAdd, bool IsStrict) {
  const Value *LHS = Op->getOperand(0);
  const Value *RHS = Op->getOperand(1);

  const RoundingMode RM =
      IsStrict ? cast<ConstrainedFPIntrinsic>(Op)->getRoundingMode().value_or(
                     RoundingMode::Dynamic)
               : RoundingMode::NearestTiesToEven;

  KnownFPClass KnownLHS, KnownRHS;
  bool WantNegative =
      IsAdd &&
      (InterestedClasses & KnownFPClass::OrderedLessThanZeroMask) != fcNone;
  bool WantNaN = (InterestedClasses & fcNan) != fcNone;
  bool WantNegZero = (InterestedClasses & fcNegZero) != fcNone;
  bool WantSubnormal = (InterestedClasses & fcSubnormal) != fcNone;

  if (!WantNaN && !WantNegative && !WantNegZero && !WantSubnormal)
    return;

  FPClassTest InterestedSrcs = InterestedClasses;
  if (WantNegative)
    InterestedSrcs |= KnownFPClass::OrderedLessThanZeroMask;
  if (InterestedClasses & fcNan)
    InterestedSrcs |= fcInf;
  computeKnownFPClass(RHS, DemandedElts, InterestedSrcs, KnownRHS, Q,
                      Depth + 1);

  // Special case fadd x, x, which is the canonical form of fmul x, 2.
  bool Self = LHS == RHS &&
              isGuaranteedNotToBeUndef(LHS, Q.AC, Q.CtxI, Q.DT, Depth + 1);
  if (Self)
    KnownLHS = KnownRHS;

  if ((WantNaN && KnownRHS.isKnownNeverNaN()) ||
      (WantNegative && KnownRHS.cannotBeOrderedLessThanZero()) || WantNegZero ||
      WantSubnormal || !IsAdd) {

    // FIXME: Context function should always be passed in separately
    const Function *F = cast<Instruction>(Op)->getFunction();
    const fltSemantics &FltSem =
        Op->getType()->getScalarType()->getFltSemantics();
    DenormalMode Mode =
        F ? F->getDenormalMode(FltSem) : DenormalMode::getDynamic();

    if (Self && IsAdd) {
      Known = KnownFPClass::fadd_self(KnownLHS, RM, Mode);
    } else {
      // RHS is canonically cheaper to compute. Skip inspecting the LHS if
      // there's no point.

      if (!Self) {
        computeKnownFPClass(LHS, DemandedElts, InterestedSrcs, KnownLHS, Q,
                            Depth + 1);
      }

      if (const APFloat *CRHS; IsAdd && match(RHS, m_APFloat(CRHS))) {
        Known = KnownFPClass::fadd(KnownLHS, *CRHS, RM, Mode);
        return;
      }

      Known = IsAdd ? KnownFPClass::fadd(KnownLHS, KnownRHS, RM, Mode)
                    : KnownFPClass::fsub(KnownLHS, KnownRHS, RM, Mode);
    }
  }
}

static void computeKnownFPClassForFRem(const Operator *Op,
                                       const APInt &DemandedElts,
                                       FPClassTest InterestedClasses,
                                       KnownFPClass &Known,
                                       const SimplifyQuery &Q, unsigned Depth) {
  const Value *LHS = Op->getOperand(0);
  const Value *RHS = Op->getOperand(1);

  FPClassTest InterestedLHS = fcNone;
  FPClassTest InterestedRHS = fcNone;

  // NaN is also generated for frem(Inf, x) and frem(x, 0.0).
  if (InterestedClasses & fcNan) {
    InterestedLHS |= fcNan | fcInf;
    InterestedRHS |= fcNan | fcZero | fcSubnormal;
  }

  // The sign for frem is the same as the first operand.
  if (InterestedClasses & (fcPosNormal | fcPosSubnormal))
    InterestedLHS |= fcPosNormal | fcPosSubnormal;
  if (InterestedClasses & (fcNegNormal | fcNegSubnormal))
    InterestedLHS |= fcNegNormal | fcNegSubnormal;

  // A negative zero result requires a negative finite first operand.
  if (InterestedClasses & fcNegZero)
    InterestedLHS |= fcNegFinite;

  // A positive zero result can additionally come from a negative finite
  // result being flushed to positive zero.
  if (InterestedClasses & fcPosZero)
    InterestedLHS |= fcPosFinite | fcNegNormal | fcNegSubnormal;

  const Function *F = cast<Instruction>(Op)->getFunction();
  DenormalMode Mode =
      F ? F->getDenormalMode(Op->getType()->getScalarType()->getFltSemantics())
        : DenormalMode::getDynamic();

  if (LHS == RHS && isGuaranteedNotToBeUndef(LHS, Q.AC, Q.CtxI, Q.DT)) {
    // X % X is always exactly [+-]0.0 or a NaN.
    FPClassTest InterestedSrcs = InterestedLHS | InterestedRHS;
    KnownFPClass KnownSrc;
    if (InterestedSrcs != fcNone)
      computeKnownFPClass(LHS, DemandedElts, InterestedSrcs, KnownSrc, Q,
                          Depth + 1);
    Known = KnownFPClass::frem_self(KnownSrc, Mode);
    return;
  }

  KnownFPClass KnownLHS;
  if (InterestedLHS != fcNone)
    computeKnownFPClass(LHS, DemandedElts, InterestedLHS, KnownLHS, Q,
                        Depth + 1);

  KnownFPClass KnownRHS;
  // RHS is only useful for refining NaN classes.
  if (InterestedRHS != fcNone && KnownLHS.isKnownNever(fcSNan))
    computeKnownFPClass(RHS, DemandedElts, InterestedRHS, KnownRHS, Q,
                        Depth + 1);

  Known = KnownFPClass::frem(KnownLHS, KnownRHS, Mode);
}

static constexpr KnownFPClass::MinMaxKind getMinMaxKind(Intrinsic::ID IID) {
  switch (IID) {
  case Intrinsic::minimum:
    return KnownFPClass::MinMaxKind::minimum;
  case Intrinsic::maximum:
    return KnownFPClass::MinMaxKind::maximum;
  case Intrinsic::minimumnum:
    return KnownFPClass::MinMaxKind::minimumnum;
  case Intrinsic::maximumnum:
    return KnownFPClass::MinMaxKind::maximumnum;
  case Intrinsic::minnum:
    return KnownFPClass::MinMaxKind::minnum;
  case Intrinsic::maxnum:
    return KnownFPClass::MinMaxKind::maxnum;
  default:
    llvm_unreachable("not a floating-point min-max intrinsic");
  }
}

/// \return true if this is a floating point value that is known to have a
/// magnitude smaller than 1. i.e., fabs(X) <= 1.0 or is nan.
static bool isAbsoluteValueULEOne(const Value *V) {
  // TODO: Handle frexp
  // TODO: Other rounding intrinsics?
  // TODO: Try computeKnownExponentRangeFromContext

  // fabs(x - floor(x)) <= 1
  const Value *SubFloorX;
  if (match(V, m_FSub(m_Value(SubFloorX),
                      m_Intrinsic<Intrinsic::floor>(m_Deferred(SubFloorX)))))
    return true;

  return match(V, m_Intrinsic<Intrinsic::amdgcn_trig_preop>(m_Value())) ||
         match(V, m_Intrinsic<Intrinsic::amdgcn_fract>(m_Value()));
}

void computeKnownFPClass(const Value *V, const APInt &DemandedElts,
                         FPClassTest InterestedClasses, KnownFPClass &Known,
                         const SimplifyQuery &Q, unsigned Depth) {
  assert(Known.isUnknown() && "should not be called with known information");

  if (!DemandedElts) {
    // No demanded elts, better to assume we don't know anything.
    Known.resetAll();
    return;
  }

  assert(Depth <= MaxAnalysisRecursionDepth && "Limit Search Depth");

  if (auto *CFP = dyn_cast<ConstantFP>(V)) {
    Known = KnownFPClass(CFP->getValueAPF());
    return;
  }

  if (isa<ConstantAggregateZero>(V)) {
    Known.setKnownFPClasses(fcPosZero);
    return;
  }

  if (isa<PoisonValue>(V)) {
    Known = KnownFPClass::getEmpty();
    return;
  }

  // Try to handle fixed width vector constants
  auto *VFVTy = dyn_cast<FixedVectorType>(V->getType());
  const Constant *CV = dyn_cast<Constant>(V);
  if (VFVTy && CV) {
    Known = KnownFPClass::getEmpty();

    // For vectors, verify that each element is not NaN.
    unsigned NumElts = VFVTy->getNumElements();
    for (unsigned i = 0; i != NumElts; ++i) {
      if (!DemandedElts[i])
        continue;

      Constant *Elt = CV->getAggregateElement(i);
      if (!Elt) {
        Known = KnownFPClass();
        return;
      }
      if (isa<PoisonValue>(Elt))
        continue;
      auto *CElt = dyn_cast<ConstantFP>(Elt);
      if (!CElt) {
        Known = KnownFPClass();
        return;
      }

      Known |= KnownFPClass(CElt->getValueAPF());
    }
    return;
  }

  if (const auto *CDS = dyn_cast<ConstantDataSequential>(V)) {
    Known = KnownFPClass::getEmpty();
    for (size_t I = 0, E = CDS->getNumElements(); I != E; ++I)
      Known |= KnownFPClass(CDS->getElementAsAPFloat(I));
    return;
  }

  if (const auto *CA = dyn_cast<ConstantAggregate>(V)) {
    // TODO: Handle complex aggregates
    Known = KnownFPClass::getEmpty();
    for (const Use &Op : CA->operands()) {
      auto *CFP = dyn_cast<ConstantFP>(Op.get());
      if (!CFP) {
        Known = KnownFPClass();
        return;
      }

      Known |= KnownFPClass(CFP->getValueAPF());
    }

    return;
  }

  FPClassTest KnownNotFromFlags = fcNone;
  if (const auto *CB = dyn_cast<CallBase>(V))
    KnownNotFromFlags |= CB->getRetNoFPClass();
  else if (const auto *Arg = dyn_cast<Argument>(V))
    KnownNotFromFlags |= Arg->getNoFPClass();

  const Operator *Op = dyn_cast<Operator>(V);
  if (const FPMathOperator *FPOp = dyn_cast_or_null<FPMathOperator>(Op)) {
    if (FPOp->hasNoNaNs())
      KnownNotFromFlags |= fcNan;
    if (FPOp->hasNoInfs())
      KnownNotFromFlags |= fcInf;
  }

  KnownFPClass AssumedClasses = computeKnownFPClassFromContext(V, Q);
  KnownNotFromFlags |= ~AssumedClasses.getKnownFPClasses();

  // We no longer need to find out about these bits from inputs if we can
  // assume this from flags/attributes.
  InterestedClasses &= ~KnownNotFromFlags;

  llvm::scope_exit ClearClassesFromFlags([=, &Known] {
    Known.knownNot(KnownNotFromFlags);
    if (!Known.getSignBit() && AssumedClasses.getSignBit()) {
      if (*AssumedClasses.getSignBit())
        Known.signBitMustBeOne();
      else
        Known.signBitMustBeZero();
    }
  });

  if (!Op)
    return;

  // All recursive calls that increase depth must come after this.
  if (Depth == MaxAnalysisRecursionDepth)
    return;

  const unsigned Opc = Op->getOpcode();
  switch (Opc) {
  case Instruction::FNeg: {
    computeKnownFPClass(Op->getOperand(0), DemandedElts, InterestedClasses,
                        Known, Q, Depth + 1);
    Known.fneg();
    break;
  }
  case Instruction::Select: {
    auto ComputeForArm = [&](Value *Arm, bool Invert) {
      KnownFPClass Res;
      computeKnownFPClass(Arm, DemandedElts, InterestedClasses, Res, Q,
                          Depth + 1);
      adjustKnownFPClassForSelectArm(Res, Op->getOperand(0), Arm, Invert, Q,
                                     Depth);
      return Res;
    };
    // Only known if known in both the LHS and RHS.
    Known =
        ComputeForArm(Op->getOperand(1), /*Invert=*/false)
            .intersectWith(ComputeForArm(Op->getOperand(2), /*Invert=*/true));
    break;
  }
  case Instruction::Load: {
    const MDNode *NoFPClass =
        cast<LoadInst>(Op)->getMetadata(LLVMContext::MD_nofpclass);
    if (!NoFPClass)
      break;

    ConstantInt *MaskVal =
        mdconst::extract<ConstantInt>(NoFPClass->getOperand(0));
    Known.knownNot(static_cast<FPClassTest>(MaskVal->getZExtValue()));
    break;
  }
  case Instruction::Call: {
    const CallInst *II = cast<CallInst>(Op);
    const Intrinsic::ID IID = II->getIntrinsicID();
    switch (IID) {
    case Intrinsic::fabs: {
      if ((InterestedClasses & (fcNan | fcPositive)) != fcNone) {
        // If we only care about the sign bit we don't need to inspect the
        // operand.
        computeKnownFPClass(II->getArgOperand(0), DemandedElts,
                            InterestedClasses, Known, Q, Depth + 1);
      }

      Known.fabs();
      break;
    }
    case Intrinsic::copysign: {
      KnownFPClass KnownSign;

      computeKnownFPClass(II->getArgOperand(0), DemandedElts, InterestedClasses,
                          Known, Q, Depth + 1);
      computeKnownFPClass(II->getArgOperand(1), DemandedElts, InterestedClasses,
                          KnownSign, Q, Depth + 1);
      Known.copysign(KnownSign);
      break;
    }
    case Intrinsic::fma:
    case Intrinsic::fmuladd: {
      if ((InterestedClasses & (fcNegative | fcNan)) == fcNone)
        break;

      const bool IsFMA = IID == Intrinsic::fma;

      // A NaN operand always gives a NaN result, so a result known not to be
      // NaN means the operands are not NaN. This does not work for inf:
      // fma produces NaN when multiplying zero by infinity, or when adding two
      // exact infinities of the opposite sign. So a result known not to be inf
      // (e.g. from nofpclass or a dominating condition) says nothing about the
      // operands. Only the ninf flag also applies to the operands.
      FPClassTest KnownNotSrc = fcNone;
      if ((KnownNotFromFlags & fcNan) == fcNan)
        KnownNotSrc |= fcNan;
      if (II->hasNoInfs())
        KnownNotSrc |= fcInf;

      // FIXME: This should check isGuaranteedNotToBeUndef
      if (II->getArgOperand(0) == II->getArgOperand(1)) {
        FPClassTest InterestedSquared = InterestedClasses;
        FPClassTest InterestedAddend = InterestedClasses;
        // Only fma(+inf, +inf, -inf) and fma(-inf, -inf, -inf) produce NaN when
        // squaring.
        if (InterestedClasses & fcQNan) {
          InterestedSquared |= fcSNan | fcInf;
          InterestedAddend |= fcSNan | fcNegInf;
        }

        KnownFPClass KnownSrc, KnownAddend;
        computeKnownFPClass(II->getArgOperand(2), DemandedElts,
                            InterestedAddend, KnownAddend, Q, Depth + 1);
        computeKnownFPClass(II->getArgOperand(0), DemandedElts,
                            InterestedSquared, KnownSrc, Q, Depth + 1);

        const Function *F = II->getFunction();
        const fltSemantics &FltSem =
            II->getType()->getScalarType()->getFltSemantics();
        DenormalMode Mode =
            F ? F->getDenormalMode(FltSem) : DenormalMode::getDynamic();

        KnownSrc.knownNot(KnownNotSrc);
        KnownAddend.knownNot(KnownNotSrc);

        Known = IsFMA
                    ? KnownFPClass::fma_square(KnownSrc, KnownAddend, Mode)
                    : KnownFPClass::fmuladd_square(KnownSrc, KnownAddend, Mode);
        break;
      }

      KnownFPClass KnownSrc[3];
      for (int I = 0; I != 3; ++I) {
        computeKnownFPClass(II->getArgOperand(I), DemandedElts,
                            InterestedClasses, KnownSrc[I], Q, Depth + 1);
        if (KnownSrc[I].isUnknown())
          return;

        KnownSrc[I].knownNot(KnownNotSrc);
      }

      const Function *F = II->getFunction();
      const fltSemantics &FltSem =
          II->getType()->getScalarType()->getFltSemantics();
      DenormalMode Mode =
          F ? F->getDenormalMode(FltSem) : DenormalMode::getDynamic();
      Known =
          IsFMA ? KnownFPClass::fma(KnownSrc[0], KnownSrc[1], KnownSrc[2], Mode)
                : KnownFPClass::fmuladd(KnownSrc[0], KnownSrc[1], KnownSrc[2],
                                        Mode);
      break;
    }
    case Intrinsic::experimental_constrained_fadd:
    case Intrinsic::experimental_constrained_fsub: {
      computeKnownFPClassForFAddSub(
          Op, DemandedElts, InterestedClasses, Known, Q, Depth,
          /*IsAdd=*/IID == Intrinsic::experimental_constrained_fadd,
          /*IsStrict=*/true);
      break;
    }
    case Intrinsic::experimental_constrained_frem: {
      computeKnownFPClassForFRem(Op, DemandedElts, InterestedClasses, Known, Q,
                                 Depth);
      break;
    }
    case Intrinsic::sqrt:
    case Intrinsic::experimental_constrained_sqrt: {
      KnownFPClass KnownSrc;
      FPClassTest InterestedSrcs = InterestedClasses;
      if (InterestedClasses & fcQNan)
        InterestedSrcs |= KnownFPClass::OrderedLessThanZeroMask | fcSNan;
      // Subnormals may flush to zero.
      if (InterestedClasses & fcZero)
        InterestedSrcs |= fcSubnormal;

      computeKnownFPClass(II->getArgOperand(0), DemandedElts, InterestedSrcs,
                          KnownSrc, Q, Depth + 1);

      DenormalMode Mode = DenormalMode::getDynamic();

      bool HasNSZ = Q.IIQ.hasNoSignedZeros(II);
      if (!HasNSZ) {
        const Function *F = II->getFunction();
        const fltSemantics &FltSem =
            II->getType()->getScalarType()->getFltSemantics();
        Mode = F ? F->getDenormalMode(FltSem) : DenormalMode::getDynamic();
      }

      Known = KnownFPClass::sqrt(KnownSrc, Mode);
      if (HasNSZ)
        Known.knownNot(fcNegZero);

      break;
    }
    case Intrinsic::sin: {
      KnownFPClass KnownSrc;
      computeKnownFPClass(II->getArgOperand(0), DemandedElts, InterestedClasses,
                          KnownSrc, Q, Depth + 1);
      Known = KnownFPClass::sin(KnownSrc);
      break;
    }
    case Intrinsic::cos: {
      KnownFPClass KnownSrc;
      computeKnownFPClass(II->getArgOperand(0), DemandedElts, InterestedClasses,
                          KnownSrc, Q, Depth + 1);
      Known = KnownFPClass::cos(KnownSrc);
      break;
    }
    case Intrinsic::tan: {
      KnownFPClass KnownSrc;
      computeKnownFPClass(II->getArgOperand(0), DemandedElts, InterestedClasses,
                          KnownSrc, Q, Depth + 1);
      Known = KnownFPClass::tan(KnownSrc);
      break;
    }
    case Intrinsic::sinh: {
      KnownFPClass KnownSrc;
      computeKnownFPClass(II->getArgOperand(0), DemandedElts, InterestedClasses,
                          KnownSrc, Q, Depth + 1);
      Known = KnownFPClass::sinh(KnownSrc);
      break;
    }
    case Intrinsic::cosh: {
      KnownFPClass KnownSrc;
      computeKnownFPClass(II->getArgOperand(0), DemandedElts, InterestedClasses,
                          KnownSrc, Q, Depth + 1);
      Known = KnownFPClass::cosh(KnownSrc);
      break;
    }
    case Intrinsic::tanh: {
      KnownFPClass KnownSrc;
      computeKnownFPClass(II->getArgOperand(0), DemandedElts, InterestedClasses,
                          KnownSrc, Q, Depth + 1);
      Known = KnownFPClass::tanh(KnownSrc);
      break;
    }
    case Intrinsic::asin: {
      KnownFPClass KnownSrc;
      computeKnownFPClass(II->getArgOperand(0), DemandedElts, InterestedClasses,
                          KnownSrc, Q, Depth + 1);
      Known = KnownFPClass::asin(KnownSrc);
      break;
    }
    case Intrinsic::acos: {
      KnownFPClass KnownSrc;
      computeKnownFPClass(II->getArgOperand(0), DemandedElts, InterestedClasses,
                          KnownSrc, Q, Depth + 1);
      Known = KnownFPClass::acos(KnownSrc);
      break;
    }
    case Intrinsic::atan: {
      KnownFPClass KnownSrc;
      computeKnownFPClass(II->getArgOperand(0), DemandedElts, InterestedClasses,
                          KnownSrc, Q, Depth + 1);
      Known = KnownFPClass::atan(KnownSrc);
      break;
    }
    case Intrinsic::atan2: {
      FPClassTest InterestedY = InterestedClasses;
      FPClassTest InterestedX = InterestedClasses;

      // We can rule out negative values if y cannot have a negative value.
      if ((InterestedClasses & fcNegFinite) != fcNone)
        InterestedY |= fcNegative;

      // We can rule out positive values if y cannot have a positive value.
      if ((InterestedClasses & fcPosFinite) != fcNone)
        InterestedY |= fcPositive | fcNegSubnormal;

      // We can rule out zero and subnormal if x cannot have a positive value.
      if ((InterestedClasses & (fcZero | fcSubnormal)) != fcNone)
        InterestedX |= fcPositive | fcNegSubnormal;

      KnownFPClass KnownY, KnownX;
      computeKnownFPClass(II->getArgOperand(0), DemandedElts, InterestedY,
                          KnownY, Q, Depth + 1);
      computeKnownFPClass(II->getArgOperand(1), DemandedElts, InterestedX,
                          KnownX, Q, Depth + 1);

      const Function *F = II->getFunction();
      DenormalMode Mode =
          F ? F->getDenormalMode(
                  II->getType()->getScalarType()->getFltSemantics())
            : DenormalMode::getDynamic();
      Known = KnownFPClass::atan2(KnownY, KnownX, Mode);
      break;
    }
    case Intrinsic::maxnum:
    case Intrinsic::minnum:
    case Intrinsic::minimum:
    case Intrinsic::maximum:
    case Intrinsic::minimumnum:
    case Intrinsic::maximumnum: {
      KnownFPClass KnownLHS, KnownRHS;
      computeKnownFPClass(II->getArgOperand(0), DemandedElts, InterestedClasses,
                          KnownLHS, Q, Depth + 1);
      computeKnownFPClass(II->getArgOperand(1), DemandedElts, InterestedClasses,
                          KnownRHS, Q, Depth + 1);

      const Function *F = II->getFunction();

      DenormalMode Mode =
          F ? F->getDenormalMode(
                  II->getType()->getScalarType()->getFltSemantics())
            : DenormalMode::getDynamic();

      Known = KnownFPClass::minMaxLike(KnownLHS, KnownRHS, getMinMaxKind(IID),
                                       Mode);
      break;
    }
    case Intrinsic::canonicalize: {
      KnownFPClass KnownSrc;
      computeKnownFPClass(II->getArgOperand(0), DemandedElts, InterestedClasses,
                          KnownSrc, Q, Depth + 1);

      const Function *F = II->getFunction();
      const fltSemantics &Flt =
          II->getType()->getScalarType()->getFltSemantics();
      DenormalMode Mode =
          F ? F->getDenormalMode(Flt) : DenormalMode::getDynamic();
      Known = KnownFPClass::canonicalize(KnownSrc, Flt, Mode);
      break;
    }
    case Intrinsic::vector_reduce_fmax:
    case Intrinsic::vector_reduce_fmin:
    case Intrinsic::vector_reduce_fmaximum:
    case Intrinsic::vector_reduce_fminimum:
    case Intrinsic::vector_reduce_fmaximumnum:
    case Intrinsic::vector_reduce_fminimumnum: {
      // reduce min/max will choose an element from one of the vector elements,
      // so we can infer and class information that is common to all elements.
      Known = computeKnownFPClass(II->getArgOperand(0), II->getFastMathFlags(),
                                  InterestedClasses, Q, Depth + 1);
      // Can only propagate sign if output is never NaN.
      if (!Known.isKnownNeverNaN())
        Known.setSignBit(std::nullopt);
      break;
    }
      // reverse preserves all characteristics of the input vec's element.
    case Intrinsic::vector_reverse:
      Known = computeKnownFPClass(
          II->getArgOperand(0), DemandedElts.reverseBits(),
          II->getFastMathFlags(), InterestedClasses, Q, Depth + 1);
      break;
    case Intrinsic::trunc:
    case Intrinsic::floor:
    case Intrinsic::ceil:
    case Intrinsic::rint:
    case Intrinsic::nearbyint:
    case Intrinsic::round:
    case Intrinsic::roundeven: {
      KnownFPClass KnownSrc;
      FPClassTest InterestedSrcs = InterestedClasses;

      // Negative round ups towards zero produce negative zero.
      if (InterestedSrcs & fcNegFinite)
        InterestedSrcs |= fcNegFinite;

      // Negative subnormals may flush to positive zero.
      if (InterestedSrcs & fcPosFinite)
        InterestedSrcs |= fcPosFinite | fcNegSubnormal;

      computeKnownFPClass(II->getArgOperand(0), DemandedElts, InterestedSrcs,
                          KnownSrc, Q, Depth + 1);

      const Function *F = II->getFunction();
      DenormalMode Mode =
          F ? F->getDenormalMode(
                  II->getType()->getScalarType()->getFltSemantics())
            : DenormalMode::getDynamic();
      const bool IsMultiUnitFPType =
          V->getType()->getScalarType()->isMultiUnitFPType();

      switch (IID) {
      case Intrinsic::trunc:
        Known = KnownFPClass::trunc(KnownSrc, Mode);
        break;
      case Intrinsic::floor:
        Known = KnownFPClass::floor(KnownSrc, IsMultiUnitFPType, Mode);
        break;
      case Intrinsic::ceil:
        Known = KnownFPClass::ceil(KnownSrc, IsMultiUnitFPType, Mode);
        break;
      default:
        Known =
            KnownFPClass::roundToIntegral(KnownSrc, IsMultiUnitFPType, Mode);
        break;
      }
      break;
    }
    case Intrinsic::exp:
    case Intrinsic::exp2:
    case Intrinsic::exp10:
    case Intrinsic::amdgcn_exp2: {
      KnownFPClass KnownSrc;
      computeKnownFPClass(II->getArgOperand(0), DemandedElts, InterestedClasses,
                          KnownSrc, Q, Depth + 1);

      Known = KnownFPClass::exp(KnownSrc);

      Type *EltTy = II->getType()->getScalarType();
      if (IID == Intrinsic::amdgcn_exp2 && EltTy->isFloatTy())
        Known.knownNot(fcSubnormal);

      break;
    }
    case Intrinsic::fptrunc_round: {
      Metadata *MD = cast<MetadataAsValue>(II->getArgOperand(1))->getMetadata();
      RoundingMode RM =
          *convertStrToRoundingMode(cast<MDString>(MD)->getString());
      computeKnownFPClassForFPTrunc(Op, RM, DemandedElts, InterestedClasses,
                                    Known, Q, Depth);
      break;
    }
    case Intrinsic::log:
    case Intrinsic::log10:
    case Intrinsic::log2:
    case Intrinsic::experimental_constrained_log:
    case Intrinsic::experimental_constrained_log10:
    case Intrinsic::experimental_constrained_log2:
    case Intrinsic::amdgcn_log: {
      FPClassTest InterestedSrcs = fcNone;

      // log(negative) produces NaN.
      if ((InterestedClasses & fcNan) != fcNone)
        InterestedSrcs |= fcNan | fcNegative;

      // log(logical-zero) produces negative infinity.
      if ((InterestedClasses & fcNegInf) != fcNone)
        InterestedSrcs |= fcZero | fcSubnormal;

      // log(x) < -0.0 if x < +1.0
      if ((InterestedClasses & fcNegNormal) != fcNone)
        InterestedSrcs |= fcPosSubnormal | fcPosNormal;

      // log(x) >= +0.0 if x >= +1.0
      if ((InterestedClasses & (fcPosZero | fcPosNormal)) != fcNone)
        InterestedSrcs |= fcPosNormal;

      // log(x) is positive infinity iff x is positive infinity.
      if ((InterestedClasses & fcPosInf) != fcNone)
        InterestedSrcs |= fcPosInf;

      KnownFPClass KnownSrc;
      if (InterestedSrcs != fcNone)
        computeKnownFPClass(II->getArgOperand(0), DemandedElts, InterestedSrcs,
                            KnownSrc, Q, Depth + 1);
      const Function *F = II->getFunction();
      DenormalMode Mode =
          F ? F->getDenormalMode(
                  II->getType()->getScalarType()->getFltSemantics())
            : DenormalMode::getDynamic();
      const bool IsMultiUnitFPType =
          V->getType()->getScalarType()->isMultiUnitFPType();
      Known = KnownFPClass::log(KnownSrc, Mode, IsMultiUnitFPType);
      break;
    }
    case Intrinsic::pow: {
      const bool WantNaN = (InterestedClasses & fcNan) != fcNone;
      const bool WantNegative = (InterestedClasses & fcNegative) != fcNone;
      if (!WantNaN && !WantNegative)
        break;

      FPClassTest InterestedLHS = fcNone;
      FPClassTest InterestedRHS = fcNone;
      if (WantNaN) {
        // pow may return NaN if one of the arguments is NaN. NaN may also be
        // produced from a negative, non-zero finite base and a non-integer
        // exponent.
        InterestedLHS |= fcNan | fcNegNormal | fcNegSubnormal;
        InterestedRHS |= fcNan;
      }
      if (WantNegative) {
        // A negative value is returned when a negative base is raised to an odd
        // integer power. Only normal values can be odd integers.
        InterestedLHS |= fcNegative;
        InterestedRHS |= fcNormal;
      }

      KnownFPClass KnownLHS;
      computeKnownFPClass(II->getArgOperand(0), DemandedElts, InterestedLHS,
                          KnownLHS, Q, Depth + 1);

      // If the LHS is unknown, then querying the RHS is only useful for rare
      // edge cases.
      if (KnownLHS.isUnknown())
        break;

      KnownFPClass KnownRHS;
      computeKnownFPClass(II->getArgOperand(1), DemandedElts, InterestedRHS,
                          KnownRHS, Q, Depth + 1);
      Known = KnownFPClass::pow(KnownLHS, KnownRHS);
      break;
    }
    case Intrinsic::powi: {
      if ((InterestedClasses & (fcNan | fcInf | fcNegative)) == fcNone)
        break;

      // The exponent is always a scalar, even when raising a vector to a power.
      const Value *Exp = II->getArgOperand(1);
      KnownBits ExponentKnownBits =
          computeKnownBits(Exp, APInt(1, 1), Q, Depth + 1);

      FPClassTest InterestedSrcs = fcNone;
      if (InterestedClasses & fcNan)
        InterestedSrcs |= fcNan;
      if (!ExponentKnownBits.isZero()) {
        if (InterestedClasses & fcInf)
          InterestedSrcs |= fcFinite | fcInf;
        if ((InterestedClasses & fcNegative) && !ExponentKnownBits.isEven())
          InterestedSrcs |= fcNegative;
      }

      KnownFPClass KnownSrc;
      if (InterestedSrcs != fcNone)
        computeKnownFPClass(II->getArgOperand(0), DemandedElts, InterestedSrcs,
                            KnownSrc, Q, Depth + 1);

      Known = KnownFPClass::powi(KnownSrc, ExponentKnownBits);
      break;
    }
    case Intrinsic::ldexp: {
      KnownFPClass KnownSrc;
      computeKnownFPClass(II->getArgOperand(0), DemandedElts, InterestedClasses,
                          KnownSrc, Q, Depth + 1);
      // Can refine inf/zero handling based on the exponent operand.
      const FPClassTest ExpInfoMask = fcZero | fcSubnormal | fcInf;

      const Value *ExpArg = II->getArgOperand(1);
      ConstantRange ExpKnownRange =
          ((KnownSrc.getKnownFPClasses() & ExpInfoMask) != fcNone)
              ? computeConstantRange(ExpArg, /*ForSigned=*/true, Q, Depth + 1)
              : ConstantRange::getFull(
                    ExpArg->getType()->getScalarSizeInBits());

      const fltSemantics &Flt =
          II->getType()->getScalarType()->getFltSemantics();

      const Function *F = II->getFunction();
      DenormalMode Mode =
          F ? F->getDenormalMode(Flt) : DenormalMode::getDynamic();

      Known = KnownFPClass::ldexp(KnownSrc, ExpKnownRange.getSignedMin(),
                                  ExpKnownRange.getSignedMax(), Flt, Mode);
      break;
    }
    case Intrinsic::arithmetic_fence: {
      computeKnownFPClass(II->getArgOperand(0), DemandedElts, InterestedClasses,
                          Known, Q, Depth + 1);
      break;
    }
    case Intrinsic::experimental_constrained_sitofp:
    case Intrinsic::experimental_constrained_uitofp:
      // Cannot produce nan
      Known.knownNot(fcNan);

      // sitofp and uitofp turn into +0.0 for zero.
      Known.knownNot(fcNegZero);

      // Integers cannot be subnormal
      Known.knownNot(fcSubnormal);

      if (IID == Intrinsic::experimental_constrained_uitofp)
        Known.signBitMustBeZero();

      // TODO: Copy inf handling from instructions
      break;

    case Intrinsic::amdgcn_fract: {
      Known.knownNot(fcInf);

      if (InterestedClasses & fcNan) {
        KnownFPClass KnownSrc;
        computeKnownFPClass(II->getArgOperand(0), DemandedElts,
                            InterestedClasses, KnownSrc, Q, Depth + 1);

        if (KnownSrc.isKnownNeverInfOrNaN())
          Known.knownNot(fcNan);
        else if (KnownSrc.isKnownNever(fcSNan))
          Known.knownNot(fcSNan);
      }

      break;
    }
    case Intrinsic::amdgcn_rcp: {
      KnownFPClass KnownSrc;
      computeKnownFPClass(II->getArgOperand(0), DemandedElts, InterestedClasses,
                          KnownSrc, Q, Depth + 1);

      Known.propagateNonNaN(KnownSrc);

      Type *EltTy = II->getType()->getScalarType();

      // f32 denormal always flushed.
      if (EltTy->isFloatTy()) {
        Known.knownNot(fcSubnormal);
        KnownSrc.knownNot(fcSubnormal);
      }

      if (KnownSrc.isKnownNever(fcNegative))
        Known.knownNot(fcNegative);
      if (KnownSrc.isKnownNever(fcPositive))
        Known.knownNot(fcPositive);

      if (const Function *F = II->getFunction()) {
        DenormalMode Mode = F->getDenormalMode(EltTy->getFltSemantics());
        if (KnownSrc.isKnownNeverLogicalPosZero(Mode))
          Known.knownNot(fcPosInf);
        if (KnownSrc.isKnownNeverLogicalNegZero(Mode))
          Known.knownNot(fcNegInf);
      }

      break;
    }
    case Intrinsic::amdgcn_rsq: {
      KnownFPClass KnownSrc;
      // The only negative value that can be returned is -inf for -0 inputs.
      Known.knownNot(fcNegZero | fcNegSubnormal | fcNegNormal);

      computeKnownFPClass(II->getArgOperand(0), DemandedElts, InterestedClasses,
                          KnownSrc, Q, Depth + 1);

      // Negative -> nan
      if (KnownSrc.isKnownNeverNaN() && KnownSrc.cannotBeOrderedLessThanZero())
        Known.knownNot(fcNan);
      else if (KnownSrc.isKnownNever(fcSNan))
        Known.knownNot(fcSNan);

      // +inf -> +0
      if (KnownSrc.isKnownNeverPosInfinity())
        Known.knownNot(fcPosZero);

      Type *EltTy = II->getType()->getScalarType();

      // f32 denormal always flushed.
      if (EltTy->isFloatTy())
        Known.knownNot(fcPosSubnormal);

      if (const Function *F = II->getFunction()) {
        DenormalMode Mode = F->getDenormalMode(EltTy->getFltSemantics());

        // -0 -> -inf
        if (KnownSrc.isKnownNeverLogicalNegZero(Mode))
          Known.knownNot(fcNegInf);

        // +0 -> +inf
        if (KnownSrc.isKnownNeverLogicalPosZero(Mode))
          Known.knownNot(fcPosInf);
      }

      break;
    }
    case Intrinsic::amdgcn_trig_preop: {
      // Always returns a value [0, 1)
      Known.knownNot(fcNan | fcInf | fcNegative);
      break;
    }
    case Intrinsic::convert_from_arbitrary_fp: {
      auto *MD = cast<MetadataAsValue>(II->getArgOperand(1))->getMetadata();
      StringRef FormatStr = cast<MDString>(MD)->getString();

      const fltSemantics *SrcSemantics =
          APFloat::getArbitraryFPSemantics(FormatStr);
      if (!SrcSemantics)
        break;

      const fltSemantics DstSemantics =
          II->getType()->getScalarType()->getFltSemantics();

      if (!APFloat::semanticsHasNaN(*SrcSemantics))
        Known.knownNot(fcNan);

      // fcInf can only be cleared if the source format has no Inf encoding
      // and the dst max exp can accommodate src max exp.
      if (!APFloat::semanticsHasInf(*SrcSemantics) &&
          APFloat::semanticsMaxExponent(*SrcSemantics) <=
              APFloat::semanticsMaxExponent(DstSemantics))
        Known.knownNot(fcInf);

      // Check and clear all neg flags for formats that do not have signed
      // representation.
      if (!APFloat::semanticsHasSignedRepr(*SrcSemantics))
        Known.knownNot(fcNegative);

      // Check if format has no zero at all (Float8E8M0FNU), or no negative
      // zero.
      if (!APFloat::semanticsHasZero(*SrcSemantics))
        Known.knownNot(fcZero);
      else if (SrcSemantics->nanEncoding == fltNanEncoding::NegativeZero)
        Known.knownNot(fcNegZero);

      // If src lands normally in dest, the result can never be subnormal.
      if (APFloat::isRepresentableAsNormalIn(*SrcSemantics, DstSemantics))
        Known.knownNot(fcSubnormal);
      break;
    }
    default:
      break;
    }

    break;
  }
  case Instruction::FAdd:
  case Instruction::FSub: {
    computeKnownFPClassForFAddSub(Op, DemandedElts, InterestedClasses, Known, Q,
                                  Depth, /*IsAdd=*/Opc == Instruction::FAdd,
                                  /*IsStrict=*/false);
    break;
  }
  case Instruction::FMul: {
    const Function *F = cast<Instruction>(Op)->getFunction();
    DenormalMode Mode =
        F ? F->getDenormalMode(
                Op->getType()->getScalarType()->getFltSemantics())
          : DenormalMode::getDynamic();

    Value *LHS = Op->getOperand(0);
    Value *RHS = Op->getOperand(1);
    // X * X is always non-negative or a NaN.
    // FIXME: Should check isGuaranteedNotToBeUndef
    if (LHS == RHS) {
      KnownFPClass KnownSrc;
      computeKnownFPClass(LHS, DemandedElts, fcAllFlags, KnownSrc, Q,
                          Depth + 1);
      Known = KnownFPClass::square(KnownSrc, Mode);
      break;
    }

    KnownFPClass KnownLHS, KnownRHS;

    const APFloat *CRHS;
    if (match(RHS, m_APFloat(CRHS))) {
      computeKnownFPClass(LHS, DemandedElts, fcAllFlags, KnownLHS, Q,
                          Depth + 1);
      Known = KnownFPClass::fmul(KnownLHS, *CRHS, Mode);
    } else {
      computeKnownFPClass(RHS, DemandedElts, fcAllFlags, KnownRHS, Q,
                          Depth + 1);
      // TODO: Improve accuracy in unfused FMA pattern. We can prove an
      // additional not-nan if the addend is known-not negative infinity if the
      // multiply is known-not infinity.

      computeKnownFPClass(LHS, DemandedElts, fcAllFlags, KnownLHS, Q,
                          Depth + 1);
      Known = KnownFPClass::fmul(KnownLHS, KnownRHS, Mode);
    }

    /// Propgate no-infs if the other source is known smaller than one, such
    /// that this cannot introduce overflow.
    if (KnownLHS.isKnownNever(fcInf) && isAbsoluteValueULEOne(RHS))
      Known.knownNot(fcInf);
    else if (KnownRHS.isKnownNever(fcInf) && isAbsoluteValueULEOne(LHS))
      Known.knownNot(fcInf);

    break;
  }
  case Instruction::FDiv: {
    const bool WantNan = (InterestedClasses & fcNan) != fcNone;

    const Function *F = cast<Instruction>(Op)->getFunction();
    const fltSemantics &FltSem =
        Op->getType()->getScalarType()->getFltSemantics();
    DenormalMode Mode =
        F ? F->getDenormalMode(FltSem) : DenormalMode::getDynamic();

    if (Op->getOperand(0) == Op->getOperand(1) &&
        isGuaranteedNotToBeUndef(Op->getOperand(0), Q.AC, Q.CtxI, Q.DT)) {
      // X / X is always exactly 1.0 or a NaN.
      Known.setKnownFPClasses(fcNan | fcPosNormal);

      if (!WantNan)
        break;

      KnownFPClass KnownSrc;
      computeKnownFPClass(Op->getOperand(0), DemandedElts,
                          fcNan | fcInf | fcZero | fcSubnormal, KnownSrc, Q,
                          Depth + 1);

      Known = KnownFPClass::fdiv_self(KnownSrc, Mode);
      break;
    }

    const bool WantNegative = (InterestedClasses & fcNegative) != fcNone;
    const bool WantPositive = (InterestedClasses & fcPositive) != fcNone;
    if (!WantNan && !WantNegative && !WantPositive)
      break;

    KnownFPClass KnownLHS, KnownRHS;
    computeKnownFPClass(Op->getOperand(1), DemandedElts, fcAllFlags, KnownRHS,
                        Q, Depth + 1);

    bool KnowSomethingUseful =
        KnownRHS.isKnownNever(fcSNan) ||
        KnownRHS.isKnownNever(fcNegNormal | fcNegSubnormal) ||
        KnownRHS.isKnownNever(fcPosNormal | fcPosSubnormal);

    if (KnowSomethingUseful)
      computeKnownFPClass(Op->getOperand(0), DemandedElts, fcAllFlags, KnownLHS,
                          Q, Depth + 1);

    Known = KnownFPClass::fdiv(KnownLHS, KnownRHS, Mode);
    break;
  }
  case Instruction::FRem: {
    computeKnownFPClassForFRem(Op, DemandedElts, InterestedClasses, Known, Q,
                               Depth);
    break;
  }
  case Instruction::FPExt: {
    KnownFPClass KnownSrc;
    computeKnownFPClass(Op->getOperand(0), DemandedElts, InterestedClasses,
                        KnownSrc, Q, Depth + 1);

    const fltSemantics &DstTy =
        Op->getType()->getScalarType()->getFltSemantics();
    const fltSemantics &SrcTy =
        Op->getOperand(0)->getType()->getScalarType()->getFltSemantics();

    const auto *I = dyn_cast<Instruction>(Op);
    const Function *F = I ? I->getFunction() : nullptr;
    DenormalMode DstMode =
        F ? F->getDenormalMode(DstTy) : DenormalMode::getDynamic();
    DenormalMode SrcMode =
        F ? F->getDenormalMode(SrcTy) : DenormalMode::getDynamic();

    Known = KnownFPClass::fpext(KnownSrc, DstTy, SrcTy, DstMode, SrcMode);
    break;
  }
  case Instruction::FPTrunc: {
    computeKnownFPClassForFPTrunc(Op, RoundingMode::NearestTiesToEven,
                                  DemandedElts, InterestedClasses, Known, Q,
                                  Depth);
    break;
  }
  case Instruction::SIToFP:
  case Instruction::UIToFP: {
    // Cannot produce nan
    Known.knownNot(fcNan);

    // Integers cannot be subnormal
    Known.knownNot(fcSubnormal);

    // sitofp and uitofp turn into +0.0 for zero.
    Known.knownNot(fcNegZero);

    // UIToFP is always non-negative regardless of known bits.
    if (Op->getOpcode() == Instruction::UIToFP)
      Known.signBitMustBeZero();

    // Only compute known bits if we can learn something useful from them.
    if (!(InterestedClasses & (fcPosZero | fcNormal | fcInf)))
      break;

    KnownBits IntKnown =
        computeKnownBits(Op->getOperand(0), DemandedElts, Q, Depth + 1);

    // If the integer is non-zero, the result cannot be +0.0
    if (IntKnown.isNonZero())
      Known.knownNot(fcPosZero);

    if (Op->getOpcode() == Instruction::SIToFP) {
      // If the signed integer is known non-negative, the result is
      // non-negative. If the signed integer is known negative, the result is
      // negative.
      if (IntKnown.isNonNegative()) {
        Known.signBitMustBeZero();
      } else if (IntKnown.isNegative()) {
        Known.signBitMustBeOne();
      }
    }

    // Guard kept for ilogb()
    if (InterestedClasses & fcInf) {
      // Get width of largest magnitude integer known.
      // This still works for a signed minimum value because the largest FP
      // value is scaled by some fraction close to 2.0 (1.0 + 0.xxxx).
      int IntSize = IntKnown.getBitWidth();
      if (Op->getOpcode() == Instruction::UIToFP)
        IntSize -= IntKnown.countMinLeadingZeros();
      else if (Op->getOpcode() == Instruction::SIToFP)
        IntSize -= IntKnown.countMinSignBits();

      // If the exponent of the largest finite FP value can hold the largest
      // integer, the result of the cast must be finite.
      Type *FPTy = Op->getType()->getScalarType();
      if (ilogb(APFloat::getLargest(FPTy->getFltSemantics())) >= IntSize)
        Known.knownNot(fcInf);
    }

    break;
  }
  case Instruction::ExtractElement: {
    // Look through extract element. If the index is non-constant or
    // out-of-range demand all elements, otherwise just the extracted element.
    const Value *Vec = Op->getOperand(0);

    APInt DemandedVecElts;
    if (auto *VecTy = dyn_cast<FixedVectorType>(Vec->getType())) {
      unsigned NumElts = VecTy->getNumElements();
      DemandedVecElts = APInt::getAllOnes(NumElts);
      auto *CIdx = dyn_cast<ConstantInt>(Op->getOperand(1));
      if (CIdx && CIdx->getValue().ult(NumElts))
        DemandedVecElts = APInt::getOneBitSet(NumElts, CIdx->getZExtValue());
    } else {
      DemandedVecElts = APInt(1, 1);
    }

    return computeKnownFPClass(Vec, DemandedVecElts, InterestedClasses, Known,
                               Q, Depth + 1);
  }
  case Instruction::InsertElement: {
    if (isa<ScalableVectorType>(Op->getType()))
      return;

    const Value *Vec = Op->getOperand(0);
    const Value *Elt = Op->getOperand(1);
    auto *CIdx = dyn_cast<ConstantInt>(Op->getOperand(2));
    unsigned NumElts = DemandedElts.getBitWidth();
    APInt DemandedVecElts = DemandedElts;
    bool NeedsElt = true;
    // If we know the index we are inserting to, clear it from Vec check.
    if (CIdx && CIdx->getValue().ult(NumElts)) {
      DemandedVecElts.clearBit(CIdx->getZExtValue());
      NeedsElt = DemandedElts[CIdx->getZExtValue()];
    }

    // Do we demand the inserted element?
    if (NeedsElt) {
      computeKnownFPClass(Elt, Known, InterestedClasses, Q, Depth + 1);
      // If we don't know any bits, early out.
      if (Known.isUnknown())
        break;
    } else {
      Known = KnownFPClass::getEmpty();
    }

    // Do we need anymore elements from Vec?
    if (!DemandedVecElts.isZero()) {
      KnownFPClass Known2;
      computeKnownFPClass(Vec, DemandedVecElts, InterestedClasses, Known2, Q,
                          Depth + 1);
      Known |= Known2;
    }

    break;
  }
  case Instruction::ShuffleVector: {
    // Handle vector splat idiom
    if (Value *Splat = getSplatValue(V)) {
      computeKnownFPClass(Splat, Known, InterestedClasses, Q, Depth + 1);
      break;
    }

    // For undef elements, we don't know anything about the common state of
    // the shuffle result.
    APInt DemandedLHS, DemandedRHS;
    auto *Shuf = dyn_cast<ShuffleVectorInst>(Op);
    if (!Shuf ||
        !getShuffleDemandedElts(Shuf, DemandedElts, DemandedLHS, DemandedRHS))
      return;

    if (!!DemandedLHS) {
      const Value *LHS = Shuf->getOperand(0);
      computeKnownFPClass(LHS, DemandedLHS, InterestedClasses, Known, Q,
                          Depth + 1);

      // If we don't know any bits, early out.
      if (Known.isUnknown())
        break;
    } else {
      Known = KnownFPClass::getEmpty();
    }

    if (!!DemandedRHS) {
      KnownFPClass Known2;
      const Value *RHS = Shuf->getOperand(1);
      computeKnownFPClass(RHS, DemandedRHS, InterestedClasses, Known2, Q,
                          Depth + 1);
      Known |= Known2;
    }

    break;
  }
  case Instruction::ExtractValue: {
    const ExtractValueInst *Extract = cast<ExtractValueInst>(Op);
    ArrayRef<unsigned> Indices = Extract->getIndices();
    const Value *Src = Extract->getAggregateOperand();
    if (isa<StructType>(Src->getType()) && Indices.size() == 1) {
      if (const auto *II = dyn_cast<IntrinsicInst>(Src)) {
        switch (II->getIntrinsicID()) {
        case Intrinsic::frexp: {
          if (Indices[0] != 0)
            break;

          FPClassTest InterestedSrcs = InterestedClasses;

          // Positive subnormals and negative subnormals could become positive
          // zero.
          if (InterestedClasses & fcPosZero)
            InterestedSrcs |= fcSubnormal;

          // Negative subnormals could become negative zero.
          if (InterestedClasses & fcNegZero)
            InterestedSrcs |= fcNegSubnormal;

          if (InterestedClasses & fcPosNormal)
            InterestedSrcs |= fcPosSubnormal;

          if (InterestedClasses & fcNegNormal)
            InterestedSrcs |= fcNegSubnormal;

          KnownFPClass KnownSrc;
          computeKnownFPClass(II->getArgOperand(0), DemandedElts,
                              InterestedSrcs, KnownSrc, Q, Depth + 1);

          const Function *F = cast<Instruction>(Op)->getFunction();
          const fltSemantics &FltSem =
              Op->getType()->getScalarType()->getFltSemantics();

          DenormalMode Mode =
              F ? F->getDenormalMode(FltSem) : DenormalMode::getDynamic();
          Known = KnownFPClass::frexp_mant(KnownSrc, Mode);
          return;
        }
        case Intrinsic::sincospi: {
          FPClassTest InterestedSrcs = InterestedClasses;

          // Infinite inputs become NaN.
          if (InterestedClasses & fcNan)
            InterestedSrcs |= fcInf;

          KnownFPClass KnownSrc;
          computeKnownFPClass(II->getArgOperand(0), DemandedElts,
                              InterestedSrcs, KnownSrc, Q, Depth + 1);

          const bool IsMultiUnitFPType =
              V->getType()->getScalarType()->isMultiUnitFPType();

          if (Indices[0] == 0)
            Known = KnownFPClass::sinpi(KnownSrc, IsMultiUnitFPType);
          else if (Indices[0] == 1)
            Known = KnownFPClass::cospi(KnownSrc, IsMultiUnitFPType);
          return;
        }
        default:
          break;
        }
      }
    }

    computeKnownFPClass(Src, DemandedElts, InterestedClasses, Known, Q,
                        Depth + 1);
    break;
  }
  case Instruction::PHI: {
    const PHINode *P = cast<PHINode>(Op);
    // Unreachable blocks may have zero-operand PHI nodes.
    if (P->getNumIncomingValues() == 0)
      break;

    // Otherwise take the unions of the known bit sets of the operands,
    // taking conservative care to avoid excessive recursion.
    const unsigned PhiRecursionLimit = MaxAnalysisRecursionDepth - 2;

    if (Depth < PhiRecursionLimit) {
      // Skip if every incoming value references to ourself.
      if (isa_and_nonnull<UndefValue>(P->hasConstantValue()))
        break;

      bool First = true;

      for (const Use &U : P->operands()) {
        Value *IncValue;
        Instruction *CtxI;
        breakSelfRecursivePHI(&U, P, IncValue, CtxI);
        // Skip direct self references.
        if (IncValue == P)
          continue;

        KnownFPClass KnownSrc;
        // Recurse, but cap the recursion to two levels, because we don't want
        // to waste time spinning around in loops. We need at least depth 2 to
        // detect known sign bits.
        computeKnownFPClass(IncValue, DemandedElts, InterestedClasses, KnownSrc,
                            Q.getWithoutCondContext().getWithInstruction(CtxI),
                            PhiRecursionLimit);

        if (First) {
          Known = KnownSrc;
          First = false;
        } else {
          Known |= KnownSrc;
        }

        if (Known.getKnownFPClasses() == fcAllFlags)
          break;
      }
    }

    // Look for the case of a for loop which has a positive
    // initial value and is incremented by a squared value.
    // This will propagate sign information out of such loops.
    if (P->getNumIncomingValues() != 2 || Known.cannotBeOrderedLessThanZero())
      break;
    for (unsigned I = 0; I < 2; I++) {
      Value *RecurValue = P->getIncomingValue(1 - I);
      IntrinsicInst *II = dyn_cast<IntrinsicInst>(RecurValue);
      if (!II)
        continue;
      Value *R, *L, *Init;
      PHINode *PN;
      if (matchSimpleTernaryIntrinsicRecurrence(II, PN, Init, L, R) &&
          PN == P) {
        switch (II->getIntrinsicID()) {
        case Intrinsic::fma:
        case Intrinsic::fmuladd: {
          KnownFPClass KnownStart;
          computeKnownFPClass(Init, DemandedElts, InterestedClasses, KnownStart,
                              Q, Depth + 1);
          if (KnownStart.cannotBeOrderedLessThanZero() && L == R &&
              isGuaranteedNotToBeUndef(L, Q.AC, Q.CtxI, Q.DT, Depth + 1))
            Known.knownNot(KnownFPClass::OrderedLessThanZeroMask);
          break;
        }
        }
      }
    }
    break;
  }
  case Instruction::BitCast: {
    const Value *Src;
    if (!match(Op, m_ElementWiseBitCast(m_Value(Src))) ||
        !Src->getType()->isIntOrIntVectorTy())
      break;

    const Type *Ty = Op->getType();

    Value *CastLHS, *CastRHS;

    // Match bitcast(umax(bitcast(a), bitcast(b)))
    if (match(Src, m_c_MaxOrMin(m_BitCast(m_Value(CastLHS)),
                                m_BitCast(m_Value(CastRHS)))) &&
        CastLHS->getType() == Ty && CastRHS->getType() == Ty) {
      KnownFPClass KnownLHS, KnownRHS;
      computeKnownFPClass(CastRHS, DemandedElts, InterestedClasses, KnownRHS, Q,
                          Depth + 1);
      if (!KnownRHS.isUnknown()) {
        computeKnownFPClass(CastLHS, DemandedElts, InterestedClasses, KnownLHS,
                            Q, Depth + 1);
        Known = KnownLHS | KnownRHS;
      }

      return;
    }

    const Type *EltTy = Ty->getScalarType();
    KnownBits Bits = computeKnownBits(Src, DemandedElts, Q, Depth + 1);

    Known = KnownFPClass::bitcast(EltTy->getFltSemantics(), Bits);
    break;
  }
  default:
    break;
  }
}

KnownFPClass llvm::computeKnownFPClass(const Value *V,
                                       const APInt &DemandedElts,
                                       FPClassTest InterestedClasses,
                                       const SimplifyQuery &SQ,
                                       unsigned Depth) {
  KnownFPClass KnownClasses;
  ::computeKnownFPClass(V, DemandedElts, InterestedClasses, KnownClasses, SQ,
                        Depth);
  return KnownClasses;
}

KnownFPClass llvm::computeKnownFPClass(const Value *V,
                                       FPClassTest InterestedClasses,
                                       const SimplifyQuery &SQ,
                                       unsigned Depth) {
  KnownFPClass Known;
  ::computeKnownFPClass(V, Known, InterestedClasses, SQ, Depth);
  return Known;
}

KnownFPClass llvm::computeKnownFPClass(
    const Value *V, const DataLayout &DL, FPClassTest InterestedClasses,
    const TargetLibraryInfo *TLI, AssumptionCache *AC, const Instruction *CtxI,
    const DominatorTree *DT, bool UseInstrInfo, unsigned Depth) {
  return computeKnownFPClass(V, InterestedClasses,
                             SimplifyQuery(DL, TLI, DT, AC, CtxI, UseInstrInfo),
                             Depth);
}

KnownFPClass
llvm::computeKnownFPClass(const Value *V, const APInt &DemandedElts,
                          FastMathFlags FMF, FPClassTest InterestedClasses,
                          const SimplifyQuery &SQ, unsigned Depth) {
  if (FMF.noNaNs())
    InterestedClasses &= ~fcNan;
  if (FMF.noInfs())
    InterestedClasses &= ~fcInf;

  KnownFPClass Result =
      computeKnownFPClass(V, DemandedElts, InterestedClasses, SQ, Depth);

  if (FMF.noNaNs())
    Result.setKnownFPClasses(Result.getKnownFPClasses() & ~fcNan);
  if (FMF.noInfs())
    Result.setKnownFPClasses(Result.getKnownFPClasses() & ~fcInf);
  return Result;
}

KnownFPClass llvm::computeKnownFPClass(const Value *V, FastMathFlags FMF,
                                       FPClassTest InterestedClasses,
                                       const SimplifyQuery &SQ,
                                       unsigned Depth) {
  auto *FVTy = dyn_cast<FixedVectorType>(V->getType());
  APInt DemandedElts =
      FVTy ? APInt::getAllOnes(FVTy->getNumElements()) : APInt(1, 1);
  return computeKnownFPClass(V, DemandedElts, FMF, InterestedClasses, SQ,
                             Depth);
}

bool llvm::cannotBeNegativeZero(const Value *V, const SimplifyQuery &SQ,
                                unsigned Depth) {
  KnownFPClass Known = computeKnownFPClass(V, fcNegZero, SQ, Depth);
  return Known.isKnownNever(fcNegZero);
}

bool llvm::cannotBeOrderedLessThanZero(const Value *V, const SimplifyQuery &SQ,
                                       unsigned Depth) {
  KnownFPClass Known =
      computeKnownFPClass(V, KnownFPClass::OrderedLessThanZeroMask, SQ, Depth);
  return Known.cannotBeOrderedLessThanZero();
}

bool llvm::isKnownNeverInfinity(const Value *V, const SimplifyQuery &SQ,
                                unsigned Depth) {
  KnownFPClass Known = computeKnownFPClass(V, fcInf, SQ, Depth);
  return Known.isKnownNeverInfinity();
}

/// Return true if the floating-point value can never contain a NaN or infinity.
bool llvm::isKnownNeverInfOrNaN(const Value *V, const SimplifyQuery &SQ,
                                unsigned Depth) {
  KnownFPClass Known = computeKnownFPClass(V, fcInf | fcNan, SQ, Depth);
  return Known.isKnownNeverNaN() && Known.isKnownNeverInfinity();
}

/// Return true if the floating-point scalar value is not a NaN or if the
/// floating-point vector value has no NaN elements. Return false if a value
/// could ever be NaN.
bool llvm::isKnownNeverNaN(const Value *V, const SimplifyQuery &SQ,
                           unsigned Depth) {
  KnownFPClass Known = computeKnownFPClass(V, fcNan, SQ, Depth);
  return Known.isKnownNeverNaN();
}

/// Return false if we can prove that the specified FP value's sign bit is 0.
/// Return true if we can prove that the specified FP value's sign bit is 1.
/// Otherwise return std::nullopt.
std::optional<bool> llvm::computeKnownFPSignBit(const Value *V,
                                                const SimplifyQuery &SQ,
                                                unsigned Depth) {
  KnownFPClass Known = computeKnownFPClass(V, fcAllFlags, SQ, Depth);
  return Known.getSignBit();
}

bool llvm::canIgnoreSignBitOfZero(const Use &U) {
  auto *User = cast<Instruction>(U.getUser());
  if (auto *FPOp = dyn_cast<FPMathOperator>(User)) {
    if (FPOp->hasNoSignedZeros())
      return true;
  }

  switch (User->getOpcode()) {
  case Instruction::FPToSI:
  case Instruction::FPToUI:
    return true;
  case Instruction::FRem:
    // frem(x, +-0.0) == NaN, and more generally frem(x, y) == frem(x, -y).
    return U.getOperandNo() == 1;
  case Instruction::FCmp:
    // fcmp treats both positive and negative zero as equal.
    return true;
  case Instruction::Call:
    if (auto *II = dyn_cast<IntrinsicInst>(User)) {
      switch (II->getIntrinsicID()) {
      case Intrinsic::fabs:
      case Intrinsic::exp:
      case Intrinsic::exp2:
      case Intrinsic::exp10:
      case Intrinsic::log:
      case Intrinsic::log2:
      case Intrinsic::log10:
      case Intrinsic::cos:
      case Intrinsic::acos:
      case Intrinsic::cosh:
      case Intrinsic::fptosi_sat:
      case Intrinsic::fptoui_sat:
      case Intrinsic::lround:
      case Intrinsic::llround:
      case Intrinsic::lrint:
      case Intrinsic::llrint:
        return true;
      case Intrinsic::copysign:
        return U.getOperandNo() == 0;
      case Intrinsic::pow:
        // pow(x, +-0.0) == +1.0 or NaN
        return U.getOperandNo() == 1;
      case Intrinsic::is_fpclass: {
        auto Test =
            static_cast<FPClassTest>(
                cast<ConstantInt>(II->getArgOperand(1))->getZExtValue()) &
            FPClassTest::fcZero;
        return Test == FPClassTest::fcZero || Test == FPClassTest::fcNone;
      }
      default:
        return false;
      }
    }
    return false;
  default:
    return false;
  }
}

bool llvm::canIgnoreSignBitOfNaN(const Use &U) {
  auto *User = cast<Instruction>(U.getUser());
  if (auto *FPOp = dyn_cast<FPMathOperator>(User)) {
    if (FPOp->hasNoNaNs())
      return true;
  }

  switch (User->getOpcode()) {
  case Instruction::FPToSI:
  case Instruction::FPToUI:
    return true;
  // Proper FP math operations ignore the sign bit of NaN.
  case Instruction::FAdd:
  case Instruction::FSub:
  case Instruction::FMul:
  case Instruction::FDiv:
  case Instruction::FRem:
  case Instruction::FPTrunc:
  case Instruction::FPExt:
  case Instruction::FCmp:
    return true;
  // Bitwise FP operations should preserve the sign bit of NaN.
  case Instruction::FNeg:
  case Instruction::Select:
  case Instruction::PHI:
    return false;
  case Instruction::Ret:
    return User->getFunction()->getAttributes().getRetNoFPClass() &
           FPClassTest::fcNan;
  case Instruction::Call:
  case Instruction::Invoke: {
    if (auto *II = dyn_cast<IntrinsicInst>(User)) {
      switch (II->getIntrinsicID()) {
      case Intrinsic::fabs:
        return true;
      case Intrinsic::copysign:
        return U.getOperandNo() == 0;
      // Other proper FP math intrinsics ignore the sign bit of NaN.
      case Intrinsic::maxnum:
      case Intrinsic::minnum:
      case Intrinsic::maximum:
      case Intrinsic::minimum:
      case Intrinsic::maximumnum:
      case Intrinsic::minimumnum:
      case Intrinsic::canonicalize:
      case Intrinsic::fptrunc_round:
      case Intrinsic::fma:
      case Intrinsic::fmuladd:
      case Intrinsic::ldexp:
      case Intrinsic::frexp:
      case Intrinsic::modf:
      case Intrinsic::sqrt:
      case Intrinsic::exp:
      case Intrinsic::exp2:
      case Intrinsic::exp10:
      case Intrinsic::log:
      case Intrinsic::log2:
      case Intrinsic::log10:
      case Intrinsic::pow:
      case Intrinsic::powi:
      case Intrinsic::sin:
      case Intrinsic::cos:
      case Intrinsic::sincos:
      case Intrinsic::sincospi:
      case Intrinsic::tan:
      case Intrinsic::asin:
      case Intrinsic::acos:
      case Intrinsic::atan:
      case Intrinsic::atan2:
      case Intrinsic::sinh:
      case Intrinsic::cosh:
      case Intrinsic::tanh:
      case Intrinsic::trunc:
      case Intrinsic::floor:
      case Intrinsic::ceil:
      case Intrinsic::round:
      case Intrinsic::roundeven:
      case Intrinsic::nearbyint:
      case Intrinsic::rint:
      case Intrinsic::fptoui_sat:
      case Intrinsic::fptosi_sat:
      case Intrinsic::lround:
      case Intrinsic::llround:
      case Intrinsic::lrint:
      case Intrinsic::llrint:
      case Intrinsic::is_fpclass:
        return true;
      default:
        return false;
      }
    }

    FPClassTest NoFPClass =
        cast<CallBase>(User)->getParamNoFPClass(U.getOperandNo());
    return NoFPClass & FPClassTest::fcNan;
  }
  default:
    return false;
  }
}

bool llvm::isKnownIntegral(const Value *V, const SimplifyQuery &SQ,
                           FastMathFlags FMF) {
  if (isa<PoisonValue>(V))
    return true;
  if (isa<UndefValue>(V))
    return false;

  if (match(V, m_CheckedFp([](const APFloat &Val) { return Val.isInteger(); })))
    return true;

  const Instruction *I = dyn_cast<Instruction>(V);
  if (!I)
    return false;

  switch (I->getOpcode()) {
  case Instruction::SIToFP:
  case Instruction::UIToFP:
    // TODO: Could check nofpclass(inf) on incoming argument
    if (FMF.noInfs())
      return true;

    // Need to check int size cannot produce infinity, which computeKnownFPClass
    // knows how to do already.
    return isKnownNeverInfinity(I, SQ);
  case Instruction::Call: {
    const CallInst *CI = cast<CallInst>(I);
    switch (CI->getIntrinsicID()) {
    case Intrinsic::trunc:
    case Intrinsic::floor:
    case Intrinsic::ceil:
    case Intrinsic::rint:
    case Intrinsic::nearbyint:
    case Intrinsic::round:
    case Intrinsic::roundeven:
      return (FMF.noInfs() && FMF.noNaNs()) || isKnownNeverInfOrNaN(I, SQ);
    default:
      break;
    }

    break;
  }
  default:
    break;
  }

  return false;
}
