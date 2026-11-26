//===- InstCombineSimplifyDemandedFP.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains logic for simplifying floating-point instructions based on
// information about how they are used.
//
//===----------------------------------------------------------------------===//

#include "InstCombineInternal.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/KnownFPClass.h"
#include "llvm/Transforms/InstCombine/InstCombiner.h"

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "instcombine"

bool InstCombinerImpl::SimplifyDemandedInstructionFPClass(Instruction &Inst) {
  KnownFPClass Known;

  Value *V = SimplifyDemandedUseFPClass(&Inst, fcAllFlags, Known,
                                        SQ.getWithInstruction(&Inst));
  if (!V)
    return false;
  if (V == &Inst)
    return true;
  replaceInstUsesWith(Inst, V);
  return true;
}

/// For floating-point classes that resolve to a single bit pattern, return that
/// value.
static Constant *getFPClassConstant(Type *Ty, FPClassTest Mask,
                                    bool IsCanonicalizing = false) {
  if (Mask == fcNone)
    return PoisonValue::get(Ty);

  if (Mask == fcPosZero)
    return Constant::getNullValue(Ty);

  // TODO: Support aggregate types that are allowed by FPMathOperator.
  if (Ty->isAggregateType())
    return nullptr;

  // Turn any possible snans into quiet if we can.
  if (Mask == fcNan && IsCanonicalizing)
    return ConstantFP::getQNaN(Ty);

  switch (Mask) {
  case fcNegZero:
    return ConstantFP::getZero(Ty, true);
  case fcPosInf:
    return ConstantFP::getInfinity(Ty);
  case fcNegInf:
    return ConstantFP::getInfinity(Ty, true);
  case fcQNan:
    // Payload bits cannot be dropped for pure signbit operations.
    return IsCanonicalizing ? ConstantFP::getQNaN(Ty) : nullptr;
  default:
    return nullptr;
  }
}

/// Perform multiple-use aware simplfications for fabs(\p Src). Returns a
/// replacement value if it's simplified, otherwise nullptr. Updates \p Known
/// with the known fpclass if not simplified.
static Value *simplifyDemandedFPClassFabs(KnownFPClass &Known, Value *Src,
                                          FPClassTest DemandedMask,
                                          KnownFPClass KnownSrc, bool NSZ) {
  if ((DemandedMask & fcNan) == fcNone)
    KnownSrc.knownNot(fcNan);
  if ((DemandedMask & fcInf) == fcNone)
    KnownSrc.knownNot(fcInf);

  if (KnownSrc.getSignBit() == false ||
      ((DemandedMask & fcNan) == fcNone && KnownSrc.isKnownNever(fcNegative)))
    return Src;

  // If the only sign bit difference is due to -0, ignore it with nsz
  if (NSZ &&
      KnownSrc.isKnownNever(KnownFPClass::OrderedLessThanZeroMask | fcNan))
    return Src;

  Known = KnownFPClass::fabs(KnownSrc);
  Known.knownNot(~DemandedMask);
  return nullptr;
}

/// Try to set an inferred no-nans or no-infs in \p FMF. \p ValidResults is a
/// mask of known valid results for the operator (already computed from the
/// result, and the known operand inputs in \p Known)
static FastMathFlags inferFastMathValueFlags(FastMathFlags FMF,
                                             FPClassTest ValidResults,
                                             ArrayRef<KnownFPClass> Known) {
  if (!FMF.noNaNs() && (ValidResults & fcNan) == fcNone) {
    if (all_of(Known, [](const KnownFPClass KnownSrc) {
          return KnownSrc.isKnownNeverNaN();
        }))
      FMF.setNoNaNs();
  }

  if (!FMF.noInfs() && (ValidResults & fcInf) == fcNone) {
    if (all_of(Known, [](const KnownFPClass KnownSrc) {
          return KnownSrc.isKnownNeverInfinity();
        }))
      FMF.setNoInfs();
  }

  return FMF;
}

static FPClassTest adjustDemandedMaskFromFlags(FPClassTest DemandedMask,
                                               FastMathFlags FMF) {
  if (FMF.noNaNs())
    DemandedMask &= ~fcNan;

  if (FMF.noInfs())
    DemandedMask &= ~fcInf;
  return DemandedMask;
}

/// Apply epilog fixups to a floating-point intrinsic. See if the result can
/// fold to a constant, or apply fast math flags.
static Value *simplifyDemandedFPClassResult(Instruction *FPOp,
                                            FastMathFlags FMF,
                                            FPClassTest DemandedMask,
                                            KnownFPClass &Known,
                                            ArrayRef<KnownFPClass> KnownSrcs) {
  FPClassTest ValidResults = DemandedMask & Known.getKnownFPClasses();
  Constant *SingleVal = getFPClassConstant(FPOp->getType(), ValidResults,
                                           /*IsCanonicalizing=*/true);
  if (SingleVal)
    return SingleVal;

  FastMathFlags InferredFMF =
      inferFastMathValueFlags(FMF, ValidResults, KnownSrcs);
  if (InferredFMF != FMF) {
    FPOp->dropUBImplyingAttrsAndMetadata();
    FPOp->setFastMathFlags(InferredFMF);
    return FPOp;
  }

  return nullptr;
}

/// Perform multiple-use aware simplfications for fneg(fabs(\p Src)). Returns a
/// replacement value if it's simplified, otherwise nullptr. Updates \p Known
/// with the known fpclass if not simplified.
static Value *simplifyDemandedFPClassFnegFabs(KnownFPClass &Known, Value *Src,
                                              FPClassTest DemandedMask,
                                              KnownFPClass KnownSrc, bool NSZ) {
  if ((DemandedMask & fcNan) == fcNone)
    KnownSrc.knownNot(fcNan);
  if ((DemandedMask & fcInf) == fcNone)
    KnownSrc.knownNot(fcInf);

  // If the source value is known negative, we can directly fold to it.
  if (KnownSrc.getSignBit() == true)
    return Src;

  // If the only sign bit difference is for 0, ignore it with nsz.
  if (NSZ &&
      KnownSrc.isKnownNever(KnownFPClass::OrderedGreaterThanZeroMask | fcNan))
    return Src;

  Known = KnownFPClass::fneg(KnownFPClass::fabs(KnownSrc));
  Known.knownNot(~DemandedMask);
  return nullptr;
}

static Value *simplifyDemandedFPClassCopysignMag(Value *MagSrc,
                                                 FPClassTest DemandedMask,
                                                 KnownFPClass KnownSrc,
                                                 bool NSZ) {
  if (NSZ) {
    constexpr FPClassTest NegOrZero = fcNegative | fcPosZero;
    constexpr FPClassTest PosOrZero = fcPositive | fcNegZero;

    if ((DemandedMask & ~NegOrZero) == fcNone &&
        KnownSrc.isKnownAlways(NegOrZero))
      return MagSrc;

    if ((DemandedMask & ~PosOrZero) == fcNone &&
        KnownSrc.isKnownAlways(PosOrZero))
      return MagSrc;
  } else {
    if ((DemandedMask & ~fcNegative) == fcNone && KnownSrc.getSignBit() == true)
      return MagSrc;

    if ((DemandedMask & ~fcPositive) == fcNone &&
        KnownSrc.getSignBit() == false)
      return MagSrc;
  }

  return nullptr;
}

static Value *
simplifyDemandedFPClassMinMax(KnownFPClass &Known, Intrinsic::ID IID,
                              const CallInst *CI, FPClassTest DemandedMask,
                              KnownFPClass KnownLHS, KnownFPClass KnownRHS,
                              const Function &F, bool NSZ) {
  bool OrderedZeroSign = !NSZ;

  KnownFPClass::MinMaxKind OpKind;
  switch (IID) {
  case Intrinsic::maximum: {
    OpKind = KnownFPClass::MinMaxKind::maximum;

    // If one operand is known greater than the other, it must be that
    // operand unless the other is a nan.
    if (cannotOrderStrictlyLess(KnownLHS.getKnownFPClasses(),
                                KnownRHS.getKnownFPClasses(),
                                OrderedZeroSign) &&
        KnownRHS.isKnownNever(fcNan))
      return CI->getArgOperand(0);

    if (cannotOrderStrictlyGreater(KnownLHS.getKnownFPClasses(),
                                   KnownRHS.getKnownFPClasses(),
                                   OrderedZeroSign) &&
        KnownLHS.isKnownNever(fcNan))
      return CI->getArgOperand(1);

    break;
  }
  case Intrinsic::minimum: {
    OpKind = KnownFPClass::MinMaxKind::minimum;

    // If one operand is known less than the other, it must be that operand
    // unless the other is a nan.
    if (cannotOrderStrictlyGreater(KnownLHS.getKnownFPClasses(),
                                   KnownRHS.getKnownFPClasses(),
                                   OrderedZeroSign) &&
        KnownRHS.isKnownNever(fcNan))
      return CI->getArgOperand(0);

    if (cannotOrderStrictlyLess(KnownLHS.getKnownFPClasses(),
                                KnownRHS.getKnownFPClasses(),
                                OrderedZeroSign) &&
        KnownLHS.isKnownNever(fcNan))
      return CI->getArgOperand(1);

    break;
  }
  case Intrinsic::maxnum:
  case Intrinsic::maximumnum: {
    OpKind = IID == Intrinsic::maxnum ? KnownFPClass::MinMaxKind::maxnum
                                      : KnownFPClass::MinMaxKind::maximumnum;

    if (cannotOrderStrictlyLess(KnownLHS.getKnownFPClasses(),
                                KnownRHS.getKnownFPClasses(),
                                OrderedZeroSign) &&
        KnownLHS.isKnownNever(fcNan))
      return CI->getArgOperand(0);

    if (cannotOrderStrictlyGreater(KnownLHS.getKnownFPClasses(),
                                   KnownRHS.getKnownFPClasses(),
                                   OrderedZeroSign) &&
        KnownRHS.isKnownNever(fcNan))
      return CI->getArgOperand(1);

    break;
  }
  case Intrinsic::minnum:
  case Intrinsic::minimumnum: {
    OpKind = IID == Intrinsic::minnum ? KnownFPClass::MinMaxKind::minnum
                                      : KnownFPClass::MinMaxKind::minimumnum;

    if (cannotOrderStrictlyGreater(KnownLHS.getKnownFPClasses(),
                                   KnownRHS.getKnownFPClasses(),
                                   OrderedZeroSign) &&
        KnownLHS.isKnownNever(fcNan))
      return CI->getArgOperand(0);

    if (cannotOrderStrictlyLess(KnownLHS.getKnownFPClasses(),
                                KnownRHS.getKnownFPClasses(),
                                OrderedZeroSign) &&
        KnownRHS.isKnownNever(fcNan))
      return CI->getArgOperand(1);

    break;
  }
  default:
    llvm_unreachable("not a min/max intrinsic");
  }

  Type *EltTy = CI->getType()->getScalarType();
  DenormalMode Mode = F.getDenormalMode(EltTy->getFltSemantics());
  Known = KnownFPClass::minMaxLike(KnownLHS, KnownRHS, OpKind, Mode);
  Known.knownNot(~DemandedMask);

  return getFPClassConstant(CI->getType(), Known.getKnownFPClasses(),
                            /*IsCanonicalizing=*/true);
}

static Value *
simplifyDemandedUseFPClassFPTrunc(InstCombinerImpl &IC, Instruction &I,
                                  RoundingMode RM, FastMathFlags FMF,
                                  FPClassTest DemandedMask, KnownFPClass &Known,
                                  const SimplifyQuery &SQ, unsigned Depth) {
  const fltSemantics &DstTy = I.getType()->getScalarType()->getFltSemantics();
  const fltSemantics &SrcTy =
      I.getOperand(0)->getType()->getScalarType()->getFltSemantics();

  const Function *Fn = I.getFunction();
  DenormalMode DstMode =
      Fn ? Fn->getDenormalMode(DstTy) : DenormalMode::getDynamic();
  DenormalMode SrcMode =
      Fn ? Fn->getDenormalMode(SrcTy) : DenormalMode::getDynamic();

  FPClassTest SrcDemandedMask = DemandedMask;
  if (DemandedMask & fcNan)
    SrcDemandedMask |= fcNan;

  // Zero results may have been rounded from subnormal or normal sources.
  if (DemandedMask & fcNegZero)
    SrcDemandedMask |= fcNegSubnormal | fcNegNormal;
  if (DemandedMask & fcPosZero)
    SrcDemandedMask |= fcPosSubnormal | fcPosNormal;

  // Subnormal results may have been normal in the source type
  if (DemandedMask & fcNegSubnormal)
    SrcDemandedMask |= fcNegNormal;
  if (DemandedMask & fcPosSubnormal)
    SrcDemandedMask |= fcPosNormal;

  const bool SubnormalNeverProducesNormal =
      DstTy.minExponent > SrcTy.minExponent;

  // A subnormal input may round to a normal value in the destination type.
  // For example: bf16 = fptrunc(f32) and f80 = fptrunc(f128).
  if (!SubnormalNeverProducesNormal) {
    // TODO: Refine based on the rounding mode.
    if (DemandedMask & fcNegNormal)
      SrcDemandedMask |= fcNegSubnormal;
    if (DemandedMask & fcPosNormal)
      SrcDemandedMask |= fcPosSubnormal;
  }

  // Overflow may saturate to max-normal depending on the rounding mode.
  const bool PosOverflowIsNeverPosInf =
      RM == RoundingMode::TowardZero || RM == RoundingMode::TowardNegative;
  const bool NegOverflowIsNeverNegInf =
      RM == RoundingMode::TowardZero || RM == RoundingMode::TowardPositive;

  if ((DemandedMask & fcPosInf) && !PosOverflowIsNeverPosInf)
    SrcDemandedMask |= fcPosNormal;
  if ((DemandedMask & fcNegInf) && !NegOverflowIsNeverNegInf)
    SrcDemandedMask |= fcNegNormal;

  // TODO: This is a conservative approach for determining the actual input and
  // output denormal mode used by fptrunc.
  if (DstMode != DenormalMode::getIEEE() ||
      SrcMode != DenormalMode::getIEEE()) {
    // Input subnormals may flush to zero, and output subnormals may also flush
    // to zero. The result can be subnormal (and flushed to zero) even if the
    // input is normal.
    if ((DemandedMask & fcZero) != fcNone)
      SrcDemandedMask |= fcSubnormal | fcNormal;
  }

  KnownFPClass KnownSrc;
  if (IC.SimplifyDemandedFPClass(&I, 0, SrcDemandedMask, KnownSrc, SQ,
                                 Depth + 1))
    return &I;

  Known = KnownFPClass::fptrunc(KnownSrc, DstTy, SrcTy, RM, DstMode, SrcMode);
  Known.knownNot(~DemandedMask);

  return simplifyDemandedFPClassResult(&I, FMF, DemandedMask, Known,
                                       {KnownSrc});
}

Value *InstCombinerImpl::SimplifyDemandedUseFPClass(Instruction *I,
                                                    FPClassTest DemandedMask,
                                                    KnownFPClass &Known,
                                                    const SimplifyQuery &SQ,
                                                    unsigned Depth) {
  assert(Depth <= MaxAnalysisRecursionDepth && "Limit Search Depth");
  assert(Known == KnownFPClass() && "expected uninitialized state");

  Type *VTy = I->getType();

  FastMathFlags FMF;
  if (auto *FPOp = dyn_cast<FPMathOperator>(I)) {
    FMF = FPOp->getFastMathFlags();
    DemandedMask = adjustDemandedMaskFromFlags(DemandedMask, FMF);
  }

  switch (I->getOpcode()) {
  case Instruction::FNeg: {
    // Special case fneg(fabs(x))

    Value *FNegSrc = I->getOperand(0);
    Value *FNegFAbsSrc;
    if (match(FNegSrc, m_OneUse(m_FAbs(m_Value(FNegFAbsSrc))))) {
      KnownFPClass KnownSrc;
      if (SimplifyDemandedFPClass(cast<Instruction>(FNegSrc), 0,
                                  llvm::unknown_sign(DemandedMask), KnownSrc,
                                  SQ, Depth + 1))
        return I;

      FastMathFlags FabsFMF = cast<FPMathOperator>(FNegSrc)->getFastMathFlags();
      FPClassTest ThisDemandedMask =
          adjustDemandedMaskFromFlags(DemandedMask, FabsFMF);

      bool IsNSZ = FMF.noSignedZeros() || FabsFMF.noSignedZeros();
      if (Value *Simplified = simplifyDemandedFPClassFnegFabs(
              Known, FNegFAbsSrc, ThisDemandedMask, KnownSrc, IsNSZ))
        return Simplified;

      if ((ThisDemandedMask & fcNan) == fcNone)
        KnownSrc.knownNot(fcNan);
      if ((ThisDemandedMask & fcInf) == fcNone)
        KnownSrc.knownNot(fcInf);

      // fneg(fabs(x)) => fneg(x)
      if (KnownSrc.getSignBit() == false)
        return replaceOperand(*I, 0, FNegFAbsSrc);

      // fneg(fabs(x)) => fneg(x), ignoring -0 if nsz.
      if (IsNSZ &&
          KnownSrc.isKnownNever(KnownFPClass::OrderedLessThanZeroMask | fcNan))
        return replaceOperand(*I, 0, FNegFAbsSrc);

      break;
    }

    if (SimplifyDemandedFPClass(I, 0, llvm::fneg(DemandedMask), Known, SQ,
                                Depth + 1))
      return I;
    Known.fneg();
    Known.knownNot(~DemandedMask);
    break;
  }
  case Instruction::FAdd:
  case Instruction::FSub: {
    KnownFPClass KnownLHS, KnownRHS;

    // fadd x, x can be handled more aggressively.
    if (I->getOperand(0) == I->getOperand(1) &&
        I->getOpcode() == Instruction::FAdd &&
        isGuaranteedNotToBeUndef(I->getOperand(0), SQ.AC, SQ.CtxI, SQ.DT,
                                 Depth + 1)) {
      Type *EltTy = VTy->getScalarType();
      DenormalMode Mode = F.getDenormalMode(EltTy->getFltSemantics());

      FPClassTest SrcDemandedMask = DemandedMask;
      if (DemandedMask & fcNan)
        SrcDemandedMask |= fcNan;

      // Doubling a subnormal could have resulted in a normal value.
      if (DemandedMask & fcPosNormal)
        SrcDemandedMask |= fcPosSubnormal;
      if (DemandedMask & fcNegNormal)
        SrcDemandedMask |= fcNegSubnormal;

      // Doubling a subnormal may produce 0 if FTZ/DAZ.
      if (Mode != DenormalMode::getIEEE()) {
        if (DemandedMask & fcPosZero) {
          SrcDemandedMask |= fcPosSubnormal;

          if (Mode.inputsMayBePositiveZero() || Mode.outputsMayBePositiveZero())
            SrcDemandedMask |= fcNegSubnormal;
        }

        if (DemandedMask & fcNegZero)
          SrcDemandedMask |= fcNegSubnormal;
      }

      // Doubling a normal could have resulted in an infinity.
      if (DemandedMask & fcPosInf)
        SrcDemandedMask |= fcPosNormal;
      if (DemandedMask & fcNegInf)
        SrcDemandedMask |= fcNegNormal;

      if (SimplifyDemandedFPClass(I, 0, SrcDemandedMask, KnownLHS, SQ,
                                  Depth + 1))
        return I;

      Known = KnownFPClass::fadd_self(KnownLHS, RoundingMode::NearestTiesToEven,
                                      Mode);
      KnownRHS = KnownLHS;
    } else {
      FPClassTest SrcDemandedMask = fcFinite;

      // inf + (-inf) = nan
      if (DemandedMask & fcNan)
        SrcDemandedMask |= fcNan | fcInf;

      if (DemandedMask & fcInf)
        SrcDemandedMask |= fcInf;

      if (SimplifyDemandedFPClass(I, 1, SrcDemandedMask, KnownRHS, SQ,
                                  Depth + 1) ||
          SimplifyDemandedFPClass(I, 0, SrcDemandedMask, KnownLHS, SQ,
                                  Depth + 1))
        return I;

      Type *EltTy = VTy->getScalarType();
      DenormalMode Mode = F.getDenormalMode(EltTy->getFltSemantics());

      Known = I->getOpcode() == Instruction::FAdd
                  ? KnownFPClass::fadd(KnownLHS, KnownRHS,
                                       RoundingMode::NearestTiesToEven, Mode)
                  : KnownFPClass::fsub(KnownLHS, KnownRHS,
                                       RoundingMode::NearestTiesToEven, Mode);
    }

    Known.knownNot(~DemandedMask);

    if (Constant *SingleVal = getFPClassConstant(VTy, Known.getKnownFPClasses(),
                                                 /*IsCanonicalizing=*/true))
      return SingleVal;

    // Propagate known result to simplify edge case checks.
    bool ResultNotNan = (DemandedMask & fcNan) == fcNone;

    // With nnan: X + {+/-}Inf --> {+/-}Inf
    if (ResultNotNan && I->getOpcode() == Instruction::FAdd &&
        KnownRHS.isKnownAlways(fcInf | fcNan) && KnownLHS.isKnownNever(fcNan))
      return I->getOperand(1);

    // With nnan: {+/-}Inf + X --> {+/-}Inf
    // With nnan: {+/-}Inf - X --> {+/-}Inf
    if (ResultNotNan && KnownLHS.isKnownAlways(fcInf | fcNan) &&
        KnownRHS.isKnownNever(fcNan))
      return I->getOperand(0);

    FastMathFlags InferredFMF = inferFastMathValueFlags(
        FMF, Known.getKnownFPClasses(), {KnownLHS, KnownRHS});
    if (InferredFMF != FMF) {
      I->setFastMathFlags(InferredFMF);
      return I;
    }

    return nullptr;
  }
  case Instruction::FMul: {
    KnownFPClass KnownLHS, KnownRHS;

    Value *X = I->getOperand(0);
    Value *Y = I->getOperand(1);

    FPClassTest SrcDemandedMask =
        DemandedMask & (fcNan | fcZero | fcSubnormal | fcNormal);

    if (DemandedMask & fcInf) {
      // mul x, inf = inf
      // mul large_x, large_y = inf
      SrcDemandedMask |= fcSubnormal | fcNormal | fcInf;
    }

    if (DemandedMask & fcNan) {
      // mul +/-inf, 0 => nan
      SrcDemandedMask |= fcZero | fcInf | fcNan;

      // TODO: Mode check
      // mul +/-inf, sub => nan if daz
      SrcDemandedMask |= fcSubnormal;
    }

    // mul normal, subnormal = normal
    // Normal inputs may result in underflow.
    if (DemandedMask & (fcNormal | fcSubnormal))
      SrcDemandedMask |= fcNormal | fcSubnormal;

    if (DemandedMask & fcZero)
      SrcDemandedMask |= fcNormal | fcSubnormal;

    if (X == Y &&
        isGuaranteedNotToBeUndef(X, SQ.AC, SQ.CtxI, SQ.DT, Depth + 1)) {
      if (SimplifyDemandedFPClass(I, 0, SrcDemandedMask, KnownLHS, SQ,
                                  Depth + 1))
        return I;
      Type *EltTy = VTy->getScalarType();

      DenormalMode Mode = F.getDenormalMode(EltTy->getFltSemantics());
      Known = KnownFPClass::square(KnownLHS, Mode);
      Known.knownNot(~DemandedMask);

      if (Constant *Folded = getFPClassConstant(VTy, Known.getKnownFPClasses(),
                                                /*IsCanonicalizing=*/true))
        return Folded;

      if (Known.isKnownAlways(fcPosZero | fcPosInf | fcNan) &&
          KnownLHS.isKnownNever(fcSubnormal | fcNormal)) {
        // We can skip the fabs if the source was already known positive.
        if (KnownLHS.isKnownAlways(fcPositive))
          return X;

        // => fabs(x), in case this was a -inf or -0.
        // Note: Dropping canonicalize.
        IRBuilderBase::InsertPointGuard Guard(Builder);
        Builder.SetInsertPoint(I);
        Value *Fabs = Builder.CreateFAbs(X, FMF);
        Fabs->takeName(I);
        return Fabs;
      }

      return nullptr;
    }

    if (SimplifyDemandedFPClass(I, 1, SrcDemandedMask, KnownRHS, SQ,
                                Depth + 1) ||
        SimplifyDemandedFPClass(I, 0, SrcDemandedMask, KnownLHS, SQ, Depth + 1))
      return I;

    if (FMF.noInfs()) {
      // Flag implies inputs cannot be infinity.
      KnownLHS.knownNot(fcInf);
      KnownRHS.knownNot(fcInf);
    }

    bool NonNanResult = (DemandedMask & fcNan) == fcNone;

    // With no-nans/no-infs:
    // X * 0.0 --> copysign(0.0, X)
    // X * -0.0 --> copysign(0.0, -X)
    if ((NonNanResult || KnownLHS.isKnownNeverInfOrNaN()) &&
        KnownRHS.isKnownAlways(fcPosZero | fcNan)) {
      IRBuilderBase::InsertPointGuard Guard(Builder);
      Builder.SetInsertPoint(I);

      // => copysign(+0, lhs)
      // Note: Dropping canonicalize
      Value *Copysign = Builder.CreateCopySign(Y, X, FMF);
      Copysign->takeName(I);
      return Copysign;
    }

    if (KnownLHS.isKnownAlways(fcPosZero | fcNan) &&
        (NonNanResult || KnownRHS.isKnownNeverInfOrNaN())) {
      IRBuilderBase::InsertPointGuard Guard(Builder);
      Builder.SetInsertPoint(I);

      // => copysign(+0, rhs)
      // Note: Dropping canonicalize
      Value *Copysign = Builder.CreateCopySign(X, Y, FMF);
      Copysign->takeName(I);
      return Copysign;
    }

    if ((NonNanResult || KnownLHS.isKnownNeverInfOrNaN()) &&
        KnownRHS.isKnownAlways(fcNegZero | fcNan)) {
      IRBuilderBase::InsertPointGuard Guard(Builder);
      Builder.SetInsertPoint(I);

      // => copysign(0, fneg(lhs))
      // Note: Dropping canonicalize
      Value *Copysign =
          Builder.CreateCopySign(Y, Builder.CreateFNegFMF(X, FMF), FMF);
      Copysign->takeName(I);
      return Copysign;
    }

    if (KnownLHS.isKnownAlways(fcNegZero | fcNan) &&
        (NonNanResult || KnownRHS.isKnownNeverInfOrNaN())) {
      IRBuilderBase::InsertPointGuard Guard(Builder);
      Builder.SetInsertPoint(I);

      // => copysign(+0, fneg(rhs))
      // Note: Dropping canonicalize
      Value *Copysign =
          Builder.CreateCopySign(X, Builder.CreateFNegFMF(Y, FMF), FMF);
      Copysign->takeName(I);
      return Copysign;
    }

    Type *EltTy = VTy->getScalarType();
    DenormalMode Mode = F.getDenormalMode(EltTy->getFltSemantics());

    if (KnownLHS.isKnownAlways(fcInf | fcNan) &&
        (KnownRHS.isKnownNeverNaN() &&
         KnownRHS.cannotBeOrderedGreaterEqZero(Mode))) {
      IRBuilderBase::InsertPointGuard Guard(Builder);
      Builder.SetInsertPoint(I);

      // Note: Dropping canonicalize
      Value *Neg = Builder.CreateFNegFMF(X, FMF);
      Neg->takeName(I);
      return Neg;
    }

    if (KnownRHS.isKnownAlways(fcInf | fcNan) &&
        (KnownLHS.isKnownNeverNaN() &&
         KnownLHS.cannotBeOrderedGreaterEqZero(Mode))) {
      IRBuilderBase::InsertPointGuard Guard(Builder);
      Builder.SetInsertPoint(I);

      // Note: Dropping canonicalize
      Value *Neg = Builder.CreateFNegFMF(Y, FMF);
      Neg->takeName(I);
      return Neg;
    }

    Known = KnownFPClass::fmul(KnownLHS, KnownRHS, Mode);
    Known.knownNot(~DemandedMask);

    if (Constant *SingleVal = getFPClassConstant(VTy, Known.getKnownFPClasses(),
                                                 /*IsCanonicalizing=*/true))
      return SingleVal;

    FastMathFlags InferredFMF = inferFastMathValueFlags(
        FMF, Known.getKnownFPClasses(), {KnownLHS, KnownRHS});
    if (InferredFMF != FMF) {
      I->setFastMathFlags(InferredFMF);
      return I;
    }

    return nullptr;
  }
  case Instruction::FDiv: {
    Value *X = I->getOperand(0);
    Value *Y = I->getOperand(1);
    if (X == Y &&
        isGuaranteedNotToBeUndef(X, SQ.AC, SQ.CtxI, SQ.DT, Depth + 1)) {
      // If the source is 0, inf or nan, the result is a nan
      IRBuilderBase::InsertPointGuard Guard(Builder);
      Builder.SetInsertPoint(I);

      Value *IsZeroOrNan = Builder.CreateFCmpFMF(
          FCmpInst::FCMP_UEQ, I->getOperand(0), ConstantFP::getZero(VTy), FMF);

      Value *Fabs = Builder.CreateFAbs(I->getOperand(0), FMF);
      Value *IsInfOrNan = Builder.CreateFCmpFMF(
          FCmpInst::FCMP_UEQ, Fabs, ConstantFP::getInfinity(VTy), FMF);

      Value *IsInfOrZeroOrNan = Builder.CreateOr(IsInfOrNan, IsZeroOrNan);

      return Builder.CreateSelectFMFWithUnknownProfile(
          IsInfOrZeroOrNan, ConstantFP::getQNaN(VTy),
          ConstantFP::get(
              VTy, APFloat::getOne(VTy->getScalarType()->getFltSemantics())),
          FMF, DEBUG_TYPE);
    }

    Type *EltTy = VTy->getScalarType();
    DenormalMode Mode = F.getDenormalMode(EltTy->getFltSemantics());

    // Every output class could require denormal inputs (except for the
    // degenerate case of only-nan results, without DAZ).
    FPClassTest SrcDemandedMask = (DemandedMask & fcNan) | fcSubnormal;

    // Normal inputs may result in underflow.
    // x / x = 1.0 for non0/inf/nan
    // -x = +y / -z
    // -x = -y / +z
    if (DemandedMask & (fcSubnormal | fcNormal))
      SrcDemandedMask |= fcNormal;

    if (DemandedMask & fcNan) {
      // 0 / 0 = nan
      // inf / inf = nan

      // Subnormal is added in case of DAZ, but this isn't strictly
      // necessary. Every other input class implies a possible subnormal source,
      // so this only could matter in the degenerate case of only-nan results.
      SrcDemandedMask |= fcZero | fcInf | fcNan;
    }

    // Zero outputs may be the result of underflow.
    if (DemandedMask & fcZero)
      SrcDemandedMask |= fcNormal | fcSubnormal;

    FPClassTest LHSDemandedMask = SrcDemandedMask;
    FPClassTest RHSDemandedMask = SrcDemandedMask;

    // 0 / inf = 0
    if (DemandedMask & fcZero) {
      assert((LHSDemandedMask & fcSubnormal) &&
             "should not have to worry about daz here");
      LHSDemandedMask |= fcZero;
      RHSDemandedMask |= fcInf;
    }

    // x / 0 = inf
    // large_normal / small_normal = inf
    // inf / 1 = inf
    // large_normal / subnormal = inf
    if (DemandedMask & fcInf) {
      LHSDemandedMask |= fcInf | fcNormal | fcSubnormal;
      RHSDemandedMask |= fcZero | fcSubnormal | fcNormal;
    }

    KnownFPClass KnownLHS, KnownRHS;
    if (SimplifyDemandedFPClass(I, 0, LHSDemandedMask, KnownLHS, SQ,
                                Depth + 1) ||
        SimplifyDemandedFPClass(I, 1, RHSDemandedMask, KnownRHS, SQ, Depth + 1))
      return I;

    bool ResultNotNan = (DemandedMask & fcNan) == fcNone;
    bool ResultNotInf = (DemandedMask & fcInf) == fcNone;

    // Replacing 0/x with a zero is only valid when the divisor can't be
    // (logical) zero, since 0/0 is NaN -- unless NaN results aren't demanded. A
    // subnormal divisor can flush to zero under a flushing denormal mode.
    bool CanIgnoreZeroByZeroNan =
        ResultNotNan || KnownRHS.isKnownNeverLogicalZero(Mode);

    // nsz [+-]0 / x -> 0
    if (FMF.noSignedZeros() && KnownLHS.isKnownAlways(fcZero) &&
        KnownRHS.isKnownNeverNaN() && CanIgnoreZeroByZeroNan)
      return ConstantFP::getZero(VTy);

    if (KnownLHS.isKnownAlways(fcPosZero) && KnownRHS.isKnownNeverNaN() &&
        CanIgnoreZeroByZeroNan) {
      IRBuilderBase::InsertPointGuard Guard(Builder);
      Builder.SetInsertPoint(I);

      // nnan +0 / x -> copysign(0, rhs)
      // TODO: -0 / x => copysign(0, fneg(rhs))
      Value *Copysign = Builder.CreateCopySign(X, Y, FMF);
      Copysign->takeName(I);
      return Copysign;
    }

    if (!ResultNotInf &&
        ((ResultNotNan || (KnownLHS.isKnownNeverNaN() &&
                           KnownLHS.isKnownNeverLogicalZero(Mode))) &&
         (KnownRHS.isKnownAlways(fcPosZero) ||
          (FMF.noSignedZeros() && KnownRHS.isKnownAlways(fcZero))))) {
      IRBuilderBase::InsertPointGuard Guard(Builder);
      Builder.SetInsertPoint(I);

      // nnan x / 0 => copysign(inf, x);
      // nnan nsz x / -0 => copysign(inf, x);
      Value *Copysign =
          Builder.CreateCopySign(ConstantFP::getInfinity(VTy), X, FMF);
      Copysign->takeName(I);
      return Copysign;
    }

    // nnan ninf X / [-]0.0 -> poison
    if (ResultNotNan && ResultNotInf && KnownRHS.isKnownAlways(fcZero))
      return PoisonValue::get(VTy);

    Known = KnownFPClass::fdiv(KnownLHS, KnownRHS, Mode);
    Known.knownNot(~DemandedMask);

    if (Constant *SingleVal = getFPClassConstant(VTy, Known.getKnownFPClasses(),
                                                 /*IsCanonicalizing=*/true))
      return SingleVal;

    FastMathFlags InferredFMF = inferFastMathValueFlags(
        FMF, Known.getKnownFPClasses(), {KnownLHS, KnownRHS});
    if (InferredFMF != FMF) {
      I->setFastMathFlags(InferredFMF);
      return I;
    }

    return nullptr;
  }
  case Instruction::FPTrunc:
    return simplifyDemandedUseFPClassFPTrunc(
        *this, *I, RoundingMode::NearestTiesToEven, FMF, DemandedMask, Known,
        SQ, Depth);
  case Instruction::FPExt: {
    const fltSemantics &DstTy = VTy->getScalarType()->getFltSemantics();
    const fltSemantics &SrcTy =
        I->getOperand(0)->getType()->getScalarType()->getFltSemantics();

    const DenormalMode DstMode = F.getDenormalMode(DstTy);
    const DenormalMode SrcMode = F.getDenormalMode(SrcTy);

    FPClassTest SrcDemandedMask = DemandedMask;
    if (DemandedMask & fcNan)
      SrcDemandedMask |= fcNan;

    // No subnormal result does not imply not-subnormal in the source type.
    if ((DemandedMask & fcNegNormal) != fcNone)
      SrcDemandedMask |= fcNegSubnormal;
    if ((DemandedMask & fcPosNormal) != fcNone)
      SrcDemandedMask |= fcPosSubnormal;

    // TODO: This is a conservative approach for determining the actual input
    // and output denormal mode used by fpext.
    if (DstMode != DenormalMode::getIEEE() ||
        SrcMode != DenormalMode::getIEEE()) {
      // We are assuming that normal values will never produce subnormal results
      // for fpext. This assumption is incorrect for ppcf128 = fpext(f80).
      if ((DemandedMask & fcZero) != fcNone)
        SrcDemandedMask |= fcSubnormal;
    }

    // TODO: Properly handle ppcf128 = fpext(f80) (since f80 has a larger
    // exponent range than ppcf128).

    KnownFPClass KnownSrc;
    if (SimplifyDemandedFPClass(I, 0, SrcDemandedMask, KnownSrc, SQ, Depth + 1))
      return I;

    Known = KnownFPClass::fpext(KnownSrc, DstTy, SrcTy, DstMode, SrcMode);
    Known.knownNot(~DemandedMask);

    return simplifyDemandedFPClassResult(I, FMF, DemandedMask, Known,
                                         {KnownSrc});
  }
  case Instruction::Call: {
    CallInst *CI = cast<CallInst>(I);
    const Intrinsic::ID IID = CI->getIntrinsicID();
    switch (IID) {
    case Intrinsic::fabs: {
      KnownFPClass KnownSrc;
      if (SimplifyDemandedFPClass(I, 0, llvm::inverse_fabs(DemandedMask),
                                  KnownSrc, SQ, Depth + 1))
        return I;

      if (Value *Simplified = simplifyDemandedFPClassFabs(
              Known, CI->getArgOperand(0), DemandedMask, KnownSrc,
              FMF.noSignedZeros()))
        return Simplified;
      break;
    }
    case Intrinsic::arithmetic_fence:
      if (SimplifyDemandedFPClass(I, 0, DemandedMask, Known, SQ, Depth + 1))
        return I;
      break;
    case Intrinsic::copysign: {
      // Flip on more potentially demanded classes
      const FPClassTest DemandedMaskAnySign = llvm::unknown_sign(DemandedMask);
      KnownFPClass KnownMag;
      if (SimplifyDemandedFPClass(CI, 0, DemandedMaskAnySign, KnownMag, SQ,
                                  Depth + 1))
        return I;

      if ((DemandedMask & fcNegative) == DemandedMask &&
          CI->getArgOperand(1) != ConstantFP::get(VTy, -1.0)) {
        // Roundabout way of replacing with fneg(fabs)
        CI->setOperand(1, ConstantFP::get(VTy, -1.0));
        return I;
      }

      if ((DemandedMask & fcPositive) == DemandedMask &&
          CI->getArgOperand(1) != ConstantFP::getZero(VTy)) {
        // Roundabout way of replacing with fabs
        CI->setOperand(1, ConstantFP::getZero(VTy));
        return I;
      }

      if (Value *Simplified = simplifyDemandedFPClassCopysignMag(
              CI->getArgOperand(0), DemandedMask, KnownMag,
              FMF.noSignedZeros()))
        return Simplified;

      KnownFPClass KnownSign =
          computeKnownFPClass(CI->getArgOperand(1), fcAllFlags, SQ, Depth + 1);
      if (KnownMag.getSignBit() && KnownSign.getSignBit() &&
          *KnownMag.getSignBit() == *KnownSign.getSignBit())
        return CI->getOperand(0);

      // TODO: Call argument attribute not considered
      // Input implied not-nan from flag.
      if (FMF.noNaNs())
        KnownSign.knownNot(fcNan);

      if (KnownSign.getSignBit() == false &&
          CI->getArgOperand(1) != ConstantFP::getZero(VTy)) {
        CI->dropUBImplyingAttrsAndMetadata();
        CI->setOperand(1, ConstantFP::getZero(VTy));
        return I;
      }

      if (KnownSign.getSignBit() == true &&
          CI->getArgOperand(1) != ConstantFP::get(VTy, -1.0)) {
        CI->dropUBImplyingAttrsAndMetadata();
        CI->setOperand(1, ConstantFP::get(VTy, -1.0));
        return I;
      }

      Known = KnownFPClass::copysign(KnownMag, KnownSign);
      Known.knownNot(~DemandedMask);
      break;
    }
    case Intrinsic::fma:
    case Intrinsic::fmuladd: {
      // Keep in mind that fma can rule out NaN in places that fmuladd cannot.

      // We can't do any simplification on the source besides stripping out
      // unneeded nans.
      FPClassTest SrcDemandedMask = DemandedMask | ~fcNan;
      if (DemandedMask & fcNan)
        SrcDemandedMask |= fcNan;

      KnownFPClass KnownSrc[3];

      const bool IsFMA = IID == Intrinsic::fma;

      Type *EltTy = VTy->getScalarType();
      if (CI->getArgOperand(0) == CI->getArgOperand(1) &&
          isGuaranteedNotToBeUndef(CI->getArgOperand(0), SQ.AC, SQ.CtxI, SQ.DT,
                                   Depth + 1)) {
        if (SimplifyDemandedFPClass(CI, 0, SrcDemandedMask, KnownSrc[0], SQ,
                                    Depth + 1) ||
            SimplifyDemandedFPClass(CI, 2, SrcDemandedMask, KnownSrc[2], SQ,
                                    Depth + 1))
          return I;

        KnownSrc[1] = KnownSrc[0];
        DenormalMode Mode = F.getDenormalMode(EltTy->getFltSemantics());
        Known = IsFMA ? KnownFPClass::fma_square(KnownSrc[0], KnownSrc[2], Mode)
                      : KnownFPClass::fmuladd_square(KnownSrc[0], KnownSrc[2],
                                                     Mode);
      } else {
        for (int OpIdx = 0; OpIdx != 3; ++OpIdx) {
          if (SimplifyDemandedFPClass(CI, OpIdx, SrcDemandedMask,
                                      KnownSrc[OpIdx], SQ, Depth + 1))
            return CI;
        }

        DenormalMode Mode = F.getDenormalMode(EltTy->getFltSemantics());
        Known = IsFMA ? KnownFPClass::fma(KnownSrc[0], KnownSrc[1], KnownSrc[2],
                                          Mode)
                      : KnownFPClass::fmuladd(KnownSrc[0], KnownSrc[1],
                                              KnownSrc[2], Mode);
      }

      return simplifyDemandedFPClassResult(CI, FMF, DemandedMask, Known,
                                           {KnownSrc});
    }
    case Intrinsic::maximum:
    case Intrinsic::minimum:
    case Intrinsic::maximumnum:
    case Intrinsic::minimumnum:
    case Intrinsic::maxnum:
    case Intrinsic::minnum: {
      const bool PropagateNaN =
          IID == Intrinsic::maximum || IID == Intrinsic::minimum;

      // We can't tell much based on the demanded result without inspecting the
      // operands (e.g., a known-positive result could have been clamped), but
      // we can still prune known-nan inputs.
      FPClassTest SrcDemandedMask =
          PropagateNaN && ((DemandedMask & fcNan) == fcNone)
              ? DemandedMask | ~fcNan
              : fcAllFlags;

      KnownFPClass KnownLHS, KnownRHS;
      if (SimplifyDemandedFPClass(CI, 1, SrcDemandedMask, KnownRHS, SQ,
                                  Depth + 1) ||
          SimplifyDemandedFPClass(CI, 0, SrcDemandedMask, KnownLHS, SQ,
                                  Depth + 1))
        return I;

      Value *Simplified =
          simplifyDemandedFPClassMinMax(Known, IID, CI, DemandedMask, KnownLHS,
                                        KnownRHS, F, FMF.noSignedZeros());
      if (Simplified)
        return Simplified;

      auto *FPOp = cast<FPMathOperator>(CI);

      FPClassTest ValidResults = DemandedMask & Known.getKnownFPClasses();
      FastMathFlags InferredFMF = FMF;

      if (!FMF.noSignedZeros()) {
        // Add NSZ flag if we know the result will not be sensitive to the sign
        // of 0.
        FPClassTest ZeroMask = fcZero;

        Type *EltTy = VTy->getScalarType();
        DenormalMode Mode = F.getDenormalMode(EltTy->getFltSemantics());
        if (Mode != DenormalMode::getIEEE())
          ZeroMask |= fcSubnormal;

        bool ResultNotLogical0 = (ValidResults & ZeroMask) == fcNone;
        if (ResultNotLogical0 || ((KnownLHS.isKnownNeverLogicalNegZero(Mode) ||
                                   KnownRHS.isKnownNeverLogicalPosZero(Mode)) &&
                                  (KnownLHS.isKnownNeverLogicalPosZero(Mode) ||
                                   KnownRHS.isKnownNeverLogicalNegZero(Mode))))
          InferredFMF.setNoSignedZeros(true);
      }

      if (!FMF.noNaNs() &&
          ((PropagateNaN && (ValidResults & fcNan) == fcNone) ||
           (KnownLHS.isKnownNeverNaN() && KnownRHS.isKnownNeverNaN()))) {
        CI->dropUBImplyingAttrsAndMetadata();
        InferredFMF.setNoNaNs(true);
      }

      if (InferredFMF != FMF) {
        CI->setFastMathFlags(InferredFMF);
        return FPOp;
      }

      return nullptr;
    }
    case Intrinsic::exp:
    case Intrinsic::exp2:
    case Intrinsic::exp10: {
      if ((DemandedMask & fcPositive) == fcNone) {
        // Only returns positive values or nans.
        if ((DemandedMask & fcNan) == fcNone)
          return PoisonValue::get(VTy);

        // Only need nan propagation.
        if ((DemandedMask & ~fcNan) == fcNone)
          return ConstantFP::getQNaN(VTy);

        return CI->getArgOperand(0);
      }

      FPClassTest SrcDemandedMask = fcAllFlags;

      if ((DemandedMask & fcNan) == fcNone)
        SrcDemandedMask &= ~fcSNan;
      if ((DemandedMask & fcQNan) == fcNone)
        SrcDemandedMask &= ~fcQNan;

      // exp(-inf) == +0.0
      if ((DemandedMask & fcPosZero) == fcNone)
        SrcDemandedMask &= ~fcNegInf;
      // exp(nnorm) == [+0.0, +1.0]
      if ((DemandedMask & fcPosFinite) == fcNone)
        SrcDemandedMask &= ~fcNegNormal;
      // exp(zero | sub) == +1.0
      if ((DemandedMask & fcPosNormal) == fcNone)
        SrcDemandedMask &= ~(fcZero | fcSubnormal);
      // exp(pnorm) == [+1.0, +inf]
      if ((DemandedMask & (fcPosNormal | fcPosInf)) == fcNone)
        SrcDemandedMask &= ~fcPosNormal;
      // exp(+inf) == +inf
      if ((DemandedMask & fcPosInf) == fcNone)
        SrcDemandedMask &= ~fcPosInf;

      KnownFPClass KnownSrc;

      // TODO: This could really make use of KnownFPClass of specific value
      // range, (i.e., close enough to 1)
      if (SimplifyDemandedFPClass(I, 0, SrcDemandedMask, KnownSrc, SQ,
                                  Depth + 1))
        return I;

      // exp(+/-0 | subnormal) = 1
      if (KnownSrc.isKnownAlways(fcZero | fcSubnormal))
        return ConstantFP::get(VTy, 1.0);

      // Only perform nan propagation.
      // Note: Dropping canonicalize / quiet of signaling nan.
      if (KnownSrc.isKnownAlways(fcNan))
        return CI->getArgOperand(0);

      // exp(0 | subnormal | +inf | nan) => x + 1.0
      if (KnownSrc.isKnownAlways(fcZero | fcSubnormal | fcPosInf | fcNan)) {
        IRBuilderBase::InsertPointGuard Guard(Builder);
        Builder.SetInsertPoint(CI);

        // fadd +/-0, 1.0 => 1.0
        // fadd subnormal, 1.0 => 1.0
        // fadd +inf, 1.0 => +inf
        // fadd nan, 1.0 => nan
        return Builder.CreateFAddFMF(CI->getArgOperand(0),
                                     ConstantFP::get(VTy, 1.0), FMF);
      }

      if (KnownSrc.isKnownAlways(fcInf | fcNan)) {
        // exp(-inf) = 0
        // exp(+inf) = +inf
        IRBuilderBase::InsertPointGuard Guard(Builder);
        Builder.SetInsertPoint(CI);

        // Note: Dropping canonicalize / quiet of signaling nan.
        Value *X = CI->getArgOperand(0);
        Value *IsPosInfOrNan = Builder.CreateFCmpFMF(
            FCmpInst::FCMP_UEQ, X, ConstantFP::getInfinity(VTy), FMF);
        // We do not know whether an infinity or a NaN is more likely here,
        // so mark the branch weights as unkown.
        Value *ZeroOrInf = Builder.CreateSelectFMFWithUnknownProfile(
            IsPosInfOrNan, X, ConstantFP::getZero(VTy), FMF, DEBUG_TYPE);
        return ZeroOrInf;
      }

      Known = KnownFPClass::exp(KnownSrc);
      Known.knownNot(~DemandedMask);

      return simplifyDemandedFPClassResult(CI, FMF, DemandedMask, Known,
                                           KnownSrc);
    }
    case Intrinsic::log:
    case Intrinsic::log2:
    case Intrinsic::log10: {
      Type *EltTy = VTy->getScalarType();
      bool IsMultiUnitFPType = EltTy->isMultiUnitFPType();

      FPClassTest DemandedSrcMask = DemandedMask & (fcNan | fcPosInf);
      if (DemandedMask & fcNan)
        DemandedSrcMask |= fcNan;

      DenormalMode Mode = F.getDenormalMode(EltTy->getFltSemantics());

      // log(x) can be negative zero or subnormal if x is close to +1.0 for the
      // PPCDoubleDouble type.
      if (IsMultiUnitFPType && (DemandedMask & (fcNegZero | fcSubnormal))) {
        DemandedSrcMask |= fcPosNormal;
        // Note that DoubleAPFloat::isDenormal considers 1.0 + DBL_TRUE_MIN to
        // be denormal, which APFloat::classify treats as fcSubnormal.
        DemandedSrcMask |= fcPosSubnormal;
      }

      // log(x < 0) = nan
      if (DemandedMask & fcNan)
        DemandedSrcMask |= (fcNegative & ~fcNegZero);

      // log(0) = -inf
      if (DemandedMask & fcNegInf) {
        DemandedSrcMask |= fcZero;

        // Subnormals may be treated as logical zero.
        if (Mode.inputsMayBeZero())
          DemandedSrcMask |= fcSubnormal;
      }

      if (DemandedMask & fcNormal)
        DemandedSrcMask |= fcNormal | fcSubnormal;

      // log(1) = 0
      if (DemandedMask & fcZero)
        DemandedSrcMask |= fcPosNormal;

      KnownFPClass KnownSrc;
      if (SimplifyDemandedFPClass(I, 0, DemandedSrcMask, KnownSrc, SQ,
                                  Depth + 1))
        return I;

      Known = KnownFPClass::log(KnownSrc, Mode, IsMultiUnitFPType);
      Known.knownNot(~DemandedMask);

      return simplifyDemandedFPClassResult(CI, FMF, DemandedMask, Known,
                                           KnownSrc);
    }
    case Intrinsic::sqrt: {
      Type *EltTy = VTy->getScalarType();
      DenormalMode Mode = F.getDenormalMode(EltTy->getFltSemantics());

      // Be explicit about the DemandedSrcMask so we do not miss anything.
      FPClassTest DemandedSrcMask = fcNone;
      if (DemandedMask & fcSNan)
        DemandedSrcMask |= fcSNan;
      if (DemandedMask & fcQNan)
        DemandedSrcMask |=
            fcQNan | fcSNan | fcNegInf | fcNegNormal | fcNegSubnormal;
      if (DemandedMask & fcNegZero) {
        DemandedSrcMask |= fcNegZero;
        if (Mode.Input == DenormalMode::PreserveSign ||
            Mode.Input == DenormalMode::Dynamic)
          DemandedSrcMask |= fcNegSubnormal;
      }
      if (DemandedMask & fcPosZero) {
        DemandedSrcMask |= fcPosZero;
        if (Mode.inputsMayBeZero())
          DemandedSrcMask |= fcPosSubnormal;
        if (Mode.inputsMayBePositiveZero())
          DemandedSrcMask |= fcNegSubnormal;
      }
      if (DemandedMask & fcPosNormal)
        DemandedSrcMask |= fcPosSubnormal | fcPosNormal;
      if (DemandedMask & fcPosInf)
        DemandedSrcMask |= fcPosInf;

      KnownFPClass KnownSrc;
      if (SimplifyDemandedFPClass(I, 0, DemandedSrcMask, KnownSrc, SQ,
                                  Depth + 1))
        return I;

      KnownSrc.knownNot(~DemandedSrcMask);

      // sqrt(-x) = nan, but be careful of negative subnormals flushed to 0.
      // TODO: This becomes redundant if KnownFPClass::sqrt rules out
      // fcPosNormal.
      if (KnownSrc.isKnownNever(fcPositive) &&
          KnownSrc.isKnownNeverLogicalZero(Mode))
        return ConstantFP::getQNaN(VTy);

      Known = KnownFPClass::sqrt(KnownSrc, Mode);
      Known.knownNot(~DemandedMask);

      if (Known.getKnownFPClasses() == fcZero) {
        if (FMF.noSignedZeros())
          return ConstantFP::getZero(VTy);
        IRBuilderBase::InsertPointGuard Guard(Builder);
        Builder.SetInsertPoint(CI);

        Value *Copysign = Builder.CreateCopySign(ConstantFP::getZero(VTy),
                                                 CI->getArgOperand(0), FMF);
        Copysign->takeName(CI);
        return Copysign;
      }

      // TODO: fold sqrt(x) -> x when x is nan | pinf | zero

      return simplifyDemandedFPClassResult(CI, FMF, DemandedMask, Known,
                                           {KnownSrc});
    }
    case Intrinsic::ldexp: {
      Type *EltTy = VTy->getScalarType();
      const fltSemantics &FltSem = EltTy->getFltSemantics();
      DenormalMode Mode = F.getDenormalMode(FltSem);

      FPClassTest SrcDemandedMask = fcNone;
      if (DemandedMask & fcNan)
        SrcDemandedMask |= fcNan;

      if (DemandedMask & fcPosInf)
        SrcDemandedMask |= fcPosInf | fcPosNormal | fcPosSubnormal;
      if (DemandedMask & fcNegInf)
        SrcDemandedMask |= fcNegInf | fcNegNormal | fcNegSubnormal;

      if (DemandedMask & (fcPosNormal | fcPosSubnormal))
        SrcDemandedMask |= fcPosNormal | fcPosSubnormal;
      if (DemandedMask & (fcNegNormal | fcNegSubnormal))
        SrcDemandedMask |= fcNegNormal | fcNegSubnormal;

      // Note: If we refine based on KnownExpBits in the future, then this
      // denormal handling code will need to be updated.
      if (DemandedMask & fcPosZero) {
        SrcDemandedMask |= fcPosFinite;
        if (Mode.inputsMayBePositiveZero())
          SrcDemandedMask |= fcNegSubnormal;
        if (Mode.outputsMayBePositiveZero())
          SrcDemandedMask |= fcNegNormal | fcNegSubnormal;
      }
      if (DemandedMask & fcNegZero)
        SrcDemandedMask |= fcNegFinite;

      KnownFPClass KnownSrc;
      if (SimplifyDemandedFPClass(CI, 0, SrcDemandedMask, KnownSrc, SQ,
                                  Depth + 1))
        return CI;

      KnownBits KnownExpBits =
          ::computeKnownBits(CI->getArgOperand(1), SQ, Depth + 1);

      Known = KnownFPClass::ldexp(KnownSrc, KnownExpBits, FltSem, Mode);
      Known.knownNot(~DemandedMask);

      return simplifyDemandedFPClassResult(CI, FMF, DemandedMask, Known,
                                           {KnownSrc});
    }
    case Intrinsic::trunc:
    case Intrinsic::floor:
    case Intrinsic::ceil:
    case Intrinsic::rint:
    case Intrinsic::nearbyint:
    case Intrinsic::round:
    case Intrinsic::roundeven: {
      Type *EltTy = VTy->getScalarType();
      DenormalMode Mode = F.getDenormalMode(EltTy->getFltSemantics());

      FPClassTest DemandedSrcMask = DemandedMask;
      if (DemandedMask & fcNan)
        DemandedSrcMask |= fcNan;

      const bool IsMultiUnitFPType = EltTy->isMultiUnitFPType();
      // Rounding a finite multi-unit value may overflow to an infinity.
      if (IsMultiUnitFPType) {
        if (IID != Intrinsic::trunc && IID != Intrinsic::floor &&
            (DemandedMask & fcPosInf))
          DemandedSrcMask |= fcPosNormal;
        if (IID != Intrinsic::trunc && IID != Intrinsic::ceil &&
            (DemandedMask & fcNegInf))
          DemandedSrcMask |= fcNegNormal;
      }

      // Zero results imply valid subnormal sources.
      if (DemandedMask & fcNegZero)
        DemandedSrcMask |= fcNegSubnormal | fcNegNormal;

      if (DemandedMask & fcPosZero) {
        DemandedSrcMask |= fcPosSubnormal | fcPosNormal;
        if (Mode.inputsMayBePositiveZero())
          DemandedSrcMask |= fcNegSubnormal;
      }

      // Rounding a subnormal away from zero may produce a normal value.
      if (DemandedMask & fcNegNormal)
        DemandedSrcMask |= fcNegSubnormal;
      if (DemandedMask & fcPosNormal)
        DemandedSrcMask |= fcPosSubnormal;

      KnownFPClass KnownSrc;
      if (SimplifyDemandedFPClass(CI, 0, DemandedSrcMask, KnownSrc, SQ,
                                  Depth + 1))
        return I;

      // Note: Possibly dropping snan quiet.
      if (KnownSrc.isKnownAlways(fcInf | fcNan | fcZero))
        return CI->getArgOperand(0);

      bool IsRoundNearestOrTrunc =
          IID == Intrinsic::round || IID == Intrinsic::roundeven ||
          IID == Intrinsic::nearbyint || IID == Intrinsic::rint ||
          IID == Intrinsic::trunc;

      if ((IID == Intrinsic::floor || IsRoundNearestOrTrunc) &&
          KnownSrc.isKnownAlways(fcPosZero | fcPosSubnormal))
        return ConstantFP::getZero(VTy);

      if ((IID == Intrinsic::ceil || IsRoundNearestOrTrunc) &&
          KnownSrc.isKnownAlways(fcNegZero | fcNegSubnormal) &&
          !Mode.inputsMayBePositiveZero())
        return ConstantFP::getZero(VTy, true);

      if (IID == Intrinsic::floor && KnownSrc.isKnownAlways(fcNegSubnormal) &&
          !Mode.inputsMayBeZero())
        return ConstantFP::get(VTy, -1.0);

      if (IID == Intrinsic::ceil && KnownSrc.isKnownAlways(fcPosSubnormal) &&
          !Mode.inputsMayBeZero())
        return ConstantFP::get(VTy, 1.0);

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

      Known.knownNot(~DemandedMask);

      if (Constant *SingleVal =
              getFPClassConstant(VTy, Known.getKnownFPClasses(),
                                 /*IsCanonicalizing=*/true))
        return SingleVal;

      if ((IID == Intrinsic::trunc || IsRoundNearestOrTrunc) &&
          KnownSrc.isKnownAlways(fcZero | fcSubnormal) &&
          (!Mode.inputsMayBePositiveZero() ||
           KnownSrc.isKnownNever(fcNegSubnormal))) {
        IRBuilderBase::InsertPointGuard Guard(Builder);
        Builder.SetInsertPoint(CI);

        Value *Copysign = Builder.CreateCopySign(ConstantFP::getZero(VTy),
                                                 CI->getArgOperand(0));
        Copysign->takeName(CI);
        return Copysign;
      }

      FastMathFlags InferredFMF =
          inferFastMathValueFlags(FMF, Known.getKnownFPClasses(), KnownSrc);
      if (InferredFMF != FMF) {
        CI->dropUBImplyingAttrsAndMetadata();
        CI->setFastMathFlags(InferredFMF);
        return CI;
      }

      return nullptr;
    }
    case Intrinsic::fptrunc_round: {
      Metadata *MD = cast<MetadataAsValue>(CI->getArgOperand(1))->getMetadata();
      RoundingMode RM =
          *convertStrToRoundingMode(cast<MDString>(MD)->getString());
      return simplifyDemandedUseFPClassFPTrunc(*this, *CI, RM, FMF,
                                               DemandedMask, Known, SQ, Depth);
    }
    case Intrinsic::canonicalize: {
      Type *EltTy = VTy->getScalarType();

      if (EltTy->isIEEELikeFPTy()) {
        const fltSemantics &Flt = EltTy->getFltSemantics();
        DenormalMode Mode = F.getDenormalMode(Flt);

        FPClassTest SrcDemandedMask = DemandedMask;

        // A demanded quiet nan result may have come from a signaling nan, so we
        // need to expand the demanded mask.
        if ((DemandedMask & fcQNan) != fcNone)
          SrcDemandedMask |= fcSNan;

        if (Mode != DenormalMode::getIEEE()) {
          // Any zero results may have come from flushed denormals.
          if (DemandedMask & fcPosZero) {
            SrcDemandedMask |= fcPosSubnormal;
            if (Mode.inputsMayBePositiveZero() ||
                Mode.outputsMayBePositiveZero())
              SrcDemandedMask |= fcNegSubnormal;
          }
          if (DemandedMask & fcNegZero)
            SrcDemandedMask |= fcNegSubnormal;
        }

        if (Mode == DenormalMode::getPreserveSign()) {
          // If a denormal input will be flushed, and we don't need zeros, we
          // don't need denormals either.
          if ((DemandedMask & fcPosZero) == fcNone)
            SrcDemandedMask &= ~fcPosSubnormal;

          if ((DemandedMask & fcNegZero) == fcNone)
            SrcDemandedMask &= ~fcNegSubnormal;
        }

        KnownFPClass KnownSrc;

        // Simplify upstream operations before trying to simplify this call.
        if (SimplifyDemandedFPClass(I, 0, SrcDemandedMask, KnownSrc, SQ,
                                    Depth + 1))
          return I;

        // Perform the canonicalization to see if this folded to a constant.
        Known = KnownFPClass::canonicalize(KnownSrc, Flt, Mode);
        Known.knownNot(~DemandedMask);

        if (Constant *SingleVal =
                getFPClassConstant(VTy, Known.getKnownFPClasses()))
          return SingleVal;

        // For IEEE handling, there is only a bit change for nan inputs, so we
        // can drop it if we do not demand nan results or we know the input
        // isn't a nan.
        // Otherwise, we also need to avoid denormal inputs to drop the
        // canonicalize.
        if (KnownSrc.isKnownNeverNaN() && (Mode == DenormalMode::getIEEE() ||
                                           KnownSrc.isKnownNeverSubnormal()))
          return CI->getArgOperand(0);

        FastMathFlags InferredFMF =
            inferFastMathValueFlags(FMF, Known.getKnownFPClasses(), KnownSrc);
        if (InferredFMF != FMF) {
          CI->dropUBImplyingAttrsAndMetadata();
          CI->setFastMathFlags(InferredFMF);
          return CI;
        }

        return nullptr;
      }

      [[fallthrough]];
    }
    default:
      Known = computeKnownFPClass(I, DemandedMask, SQ, Depth + 1);
      Known.knownNot(~DemandedMask);
      break;
    }

    break;
  }
  case Instruction::Select: {
    KnownFPClass KnownLHS, KnownRHS;
    if (SimplifyDemandedFPClass(I, 2, DemandedMask, KnownRHS, SQ, Depth + 1) ||
        SimplifyDemandedFPClass(I, 1, DemandedMask, KnownLHS, SQ, Depth + 1))
      return I;

    if (KnownLHS.isKnownNever(DemandedMask))
      return I->getOperand(2);
    if (KnownRHS.isKnownNever(DemandedMask))
      return I->getOperand(1);

    adjustKnownFPClassForSelectArm(KnownLHS, I->getOperand(0), I->getOperand(1),
                                   /*Invert=*/false, SQ, Depth);
    adjustKnownFPClassForSelectArm(KnownRHS, I->getOperand(0), I->getOperand(2),
                                   /*Invert=*/true, SQ, Depth);
    Known = KnownLHS.intersectWith(KnownRHS);
    Known.knownNot(~DemandedMask);
    break;
  }
  case Instruction::ExtractElement: {
    // TODO: Handle demanded element mask
    if (SimplifyDemandedFPClass(I, 0, DemandedMask, Known, SQ, Depth + 1))
      return I;
    Known.knownNot(~DemandedMask);
    break;
  }
  case Instruction::InsertElement: {
    KnownFPClass KnownInserted, KnownVec;
    if (SimplifyDemandedFPClass(I, 1, DemandedMask, KnownInserted, SQ,
                                Depth + 1) ||
        SimplifyDemandedFPClass(I, 0, DemandedMask, KnownVec, SQ, Depth + 1))
      return I;

    // TODO: Use demanded elements logic from computeKnownFPClass
    Known = KnownVec | KnownInserted;
    Known.knownNot(~DemandedMask);
    break;
  }
  case Instruction::ShuffleVector: {
    KnownFPClass KnownLHS, KnownRHS;
    if (SimplifyDemandedFPClass(I, 1, DemandedMask, KnownRHS, SQ, Depth + 1) ||
        SimplifyDemandedFPClass(I, 0, DemandedMask, KnownLHS, SQ, Depth + 1))
      return I;

    // TODO: This is overly conservative and should consider demanded elements,
    // and splats.
    Known = KnownLHS | KnownRHS;
    Known.knownNot(~DemandedMask);
    break;
  }
  case Instruction::InsertValue: {
    KnownFPClass KnownAgg, KnownElt;
    if (SimplifyDemandedFPClass(I, 0, DemandedMask, KnownAgg, SQ, Depth + 1) ||
        SimplifyDemandedFPClass(I, 1, DemandedMask, KnownElt, SQ, Depth + 1))
      return I;

    Known = KnownAgg | KnownElt;
    break;
  }
  case Instruction::ExtractValue: {
    Value *ExtractSrc;
    if (match(I, m_ExtractValue<0>(m_OneUse(m_Value(ExtractSrc))))) {
      if (auto *II = dyn_cast<IntrinsicInst>(ExtractSrc)) {
        const Intrinsic::ID IID = II->getIntrinsicID();
        switch (IID) {
        case Intrinsic::frexp: {
          FPClassTest SrcDemandedMask = fcNone;

          if (DemandedMask & fcNan)
            SrcDemandedMask |= fcNan;

          // Positive subnormals and negative subnormals could become positive
          // zero.
          if (DemandedMask & fcPosZero)
            SrcDemandedMask |= fcPosZero | fcSubnormal;

          // Negative subnormals could become negative zero.
          if (DemandedMask & fcNegZero)
            SrcDemandedMask |= fcNegZero | fcNegSubnormal;

          if (DemandedMask & (fcNegNormal | fcNegSubnormal))
            SrcDemandedMask |= fcNegNormal | fcNegSubnormal;
          if (DemandedMask & (fcPosNormal | fcPosSubnormal))
            SrcDemandedMask |= fcPosNormal | fcPosSubnormal;

          if (DemandedMask & fcPosInf)
            SrcDemandedMask |= fcPosInf;
          if (DemandedMask & fcNegInf)
            SrcDemandedMask |= fcNegInf;

          KnownFPClass KnownSrc;
          if (SimplifyDemandedFPClass(II, 0, SrcDemandedMask, KnownSrc, SQ,
                                      Depth + 1))
            return I;

          Type *EltTy = VTy->getScalarType();
          DenormalMode Mode = F.getDenormalMode(EltTy->getFltSemantics());

          Known = KnownFPClass::frexp_mant(KnownSrc, Mode);
          Known.setKnownFPClasses(Known.getKnownFPClasses() & DemandedMask);

          if (Constant *SingleVal =
                  getFPClassConstant(VTy, Known.getKnownFPClasses(),
                                     /*IsCanonicalizing=*/true))
            return SingleVal;

          // frexp returns zero, infinity, and NaN inputs unchanged.
          if (KnownSrc.isKnownAlways(fcZero | fcInf | fcNan))
            return II->getArgOperand(0);

          return nullptr;
        }
        default:
          break;
        }
      }
    }

    KnownFPClass KnownSrc;
    if (SimplifyDemandedFPClass(I, 0, DemandedMask, KnownSrc, SQ, Depth + 1))
      return I;
    Known = KnownSrc;
    break;
  }
  case Instruction::PHI: {
    const unsigned PhiRecursionLimit = MaxAnalysisRecursionDepth - 2;
    if (Depth >= PhiRecursionLimit)
      break;

    PHINode *P = cast<PHINode>(I);
    SimplifyQuery ContextSQ = SQ.getWithoutCondContext();

    bool First = true;
    bool Changed = false;
    for (unsigned I = 0, E = P->getNumIncomingValues(); I != E; ++I) {
      // TODO: Better support for self recursive phi
      BasicBlock *PredBB = P->getIncomingBlock(I);
      const Instruction *CtxI = PredBB->getTerminator();

      // Attempt to simplify all incoming edges at a time. If we simplify one
      // incoming edge, the phi may fold away, losing information on a later
      // visit.
      KnownFPClass KnownSrc;
      if (SimplifyDemandedFPClass(
              P, P->getOperandNumForIncomingValue(I), DemandedMask, KnownSrc,
              ContextSQ.getWithInstruction(CtxI), Depth + 1)) {
        // Fixup the other block references to the simplified value.
        P->setIncomingValueForBlock(PredBB, P->getIncomingValue(I));
        Changed = true;
      }

      if (First) {
        Known = KnownSrc;
        First = false;
      } else {
        Known |= KnownSrc;
      }
    }

    if (Changed)
      return P;

    Known.knownNot(~DemandedMask);
    break;
  }
  default:
    Known = computeKnownFPClass(I, DemandedMask, SQ, Depth + 1);
    Known.knownNot(~DemandedMask);
    break;
  }

  return getFPClassConstant(VTy, Known.getKnownFPClasses());
}

/// Helper routine of SimplifyDemandedUseFPClass. It computes Known
/// floating-point classes. It also tries to handle simplifications that can be
/// done based on DemandedMask, but without modifying the Instruction.
Value *InstCombinerImpl::SimplifyMultipleUseDemandedFPClass(
    Instruction *I, FPClassTest DemandedMask, KnownFPClass &Known,
    const SimplifyQuery &SQ, unsigned Depth) {
  FastMathFlags FMF;
  if (auto *FPOp = dyn_cast<FPMathOperator>(I)) {
    FMF = FPOp->getFastMathFlags();
    DemandedMask = adjustDemandedMaskFromFlags(DemandedMask, FMF);
  }

  switch (I->getOpcode()) {
  case Instruction::Select: {
    // TODO: Can we infer which side it came from based on adjusted result
    // class?
    KnownFPClass KnownRHS =
        computeKnownFPClass(I->getOperand(2), DemandedMask, SQ, Depth + 1);
    if (KnownRHS.isKnownNever(DemandedMask))
      return I->getOperand(1);

    KnownFPClass KnownLHS =
        computeKnownFPClass(I->getOperand(1), DemandedMask, SQ, Depth + 1);
    if (KnownLHS.isKnownNever(DemandedMask))
      return I->getOperand(2);

    adjustKnownFPClassForSelectArm(KnownLHS, I->getOperand(0), I->getOperand(1),
                                   /*Invert=*/false, SQ, Depth);
    adjustKnownFPClassForSelectArm(KnownRHS, I->getOperand(0), I->getOperand(2),
                                   /*Invert=*/true, SQ, Depth);
    Known = KnownLHS.intersectWith(KnownRHS);
    Known.knownNot(~DemandedMask);
    break;
  }
  case Instruction::FNeg: {
    // Special case fneg(fabs(x))
    Value *Src;

    Value *FNegSrc = I->getOperand(0);
    if (!match(FNegSrc, m_FAbs(m_Value(Src)))) {
      Known = computeKnownFPClass(I, DemandedMask, SQ, Depth + 1);
      break;
    }

    KnownFPClass KnownSrc = computeKnownFPClass(Src, fcAllFlags, SQ, Depth + 1);

    FastMathFlags FabsFMF = cast<FPMathOperator>(FNegSrc)->getFastMathFlags();
    FPClassTest ThisDemandedMask =
        adjustDemandedMaskFromFlags(DemandedMask, FabsFMF);

    // We cannot apply the NSZ logic with multiple uses. We can apply it if the
    // inner fabs has it and this is the only use.
    if (Value *Simplified = simplifyDemandedFPClassFnegFabs(
            Known, Src, ThisDemandedMask, KnownSrc, /*NSZ=*/false))
      return Simplified;
    break;
  }
  case Instruction::Call: {
    const CallInst *CI = cast<CallInst>(I);
    const Intrinsic::ID IID = CI->getIntrinsicID();
    switch (IID) {
    case Intrinsic::fabs: {
      Value *Src = CI->getArgOperand(0);
      KnownFPClass KnownSrc =
          computeKnownFPClass(Src, fcAllFlags, SQ, Depth + 1);

      // NSZ cannot be applied in multiple use case (maybe it could if all uses
      // were known nsz)
      if (Value *Simplified = simplifyDemandedFPClassFabs(
              Known, CI->getArgOperand(0), DemandedMask, KnownSrc,
              /*NSZ=*/false))
        return Simplified;
      break;
    }
    case Intrinsic::copysign: {
      Value *Mag = CI->getArgOperand(0);
      Value *Sign = CI->getArgOperand(1);
      KnownFPClass KnownMag =
          computeKnownFPClass(Mag, fcAllFlags, SQ, Depth + 1);

      // Rule out some cases by magnitude, which may help prove the sign bit is
      // one direction or the other.
      KnownMag.knownNot(~llvm::unknown_sign(DemandedMask));

      // Cannot use nsz in the multiple use case.
      if (Value *Simplified = simplifyDemandedFPClassCopysignMag(
              Mag, DemandedMask, KnownMag, /*NSZ=*/false))
        return Simplified;

      KnownFPClass KnownSign =
          computeKnownFPClass(Sign, fcAllFlags, SQ, Depth + 1);

      if (FMF.noInfs())
        KnownSign.knownNot(fcInf);
      if (FMF.noNaNs())
        KnownSign.knownNot(fcNan);

      if (KnownSign.getSignBit() && KnownMag.getSignBit() &&
          *KnownSign.getSignBit() == *KnownMag.getSignBit())
        return Mag;

      Known = KnownFPClass::copysign(KnownMag, KnownSign);
      break;
    }
    case Intrinsic::maxnum:
    case Intrinsic::minnum:
    case Intrinsic::maximum:
    case Intrinsic::minimum:
    case Intrinsic::maximumnum:
    case Intrinsic::minimumnum: {
      KnownFPClass KnownRHS = computeKnownFPClass(CI->getArgOperand(1),
                                                  DemandedMask, SQ, Depth + 1);
      if (KnownRHS.isUnknown())
        return nullptr;

      KnownFPClass KnownLHS = computeKnownFPClass(CI->getArgOperand(0),
                                                  DemandedMask, SQ, Depth + 1);

      // Cannot use NSZ in the multiple use case.
      return simplifyDemandedFPClassMinMax(Known, IID, CI, DemandedMask,
                                           KnownLHS, KnownRHS, F,
                                           /*NSZ=*/false);
    }
    default:
      break;
    }

    [[fallthrough]];
  }
  default:
    Known = computeKnownFPClass(I, DemandedMask, SQ, Depth + 1);
    Known.knownNot(~DemandedMask);
    break;
  }

  return getFPClassConstant(I->getType(), Known.getKnownFPClasses());
}

bool InstCombinerImpl::SimplifyDemandedFPClass(Instruction *I, unsigned OpNo,
                                               FPClassTest DemandedMask,
                                               KnownFPClass &Known,
                                               const SimplifyQuery &SQ,
                                               unsigned Depth) {
  Use &U = I->getOperandUse(OpNo);
  Value *V = U.get();
  Type *VTy = V->getType();

  if (DemandedMask == fcNone) {
    if (isa<PoisonValue>(V))
      return false;
    replaceUse(U, PoisonValue::get(VTy));
    return true;
  }

  // Handle constant
  Instruction *VInst = dyn_cast<Instruction>(V);
  if (!VInst) {
    // Handle constants and arguments
    Known = computeKnownFPClass(V, fcAllFlags, SQ, Depth);
    Known.knownNot(~DemandedMask);

    if (Known.getKnownFPClasses() == fcNone) {
      if (isa<PoisonValue>(V))
        return false;
      replaceUse(U, PoisonValue::get(VTy));
      return true;
    }

    // Do not try to replace values which are already constants (unless we are
    // folding to poison). Doing so could promote poison elements to non-poison
    // constants.
    if (isa<Constant>(V))
      return false;

    Value *FoldedToConst = getFPClassConstant(VTy, Known.getKnownFPClasses());
    if (!FoldedToConst || FoldedToConst == V)
      return false;

    replaceUse(U, FoldedToConst);
    return true;
  }

  if (Depth == MaxAnalysisRecursionDepth) {
    Known.knownNot(~DemandedMask);
    return false;
  }

  Value *NewVal;

  if (VInst->hasOneUse()) {
    // If the instruction has one use, we can directly simplify it.
    NewVal = SimplifyDemandedUseFPClass(VInst, DemandedMask, Known, SQ, Depth);
  } else {
    // If there are multiple uses of this instruction, then we can simplify
    // VInst to some other value, but not modify the instruction.
    NewVal = SimplifyMultipleUseDemandedFPClass(VInst, DemandedMask, Known, SQ,
                                                Depth);
  }

  if (!NewVal)
    return false;
  // Recursion through a PHI may have already replaced this use. If the
  // recursive simplification only changed V in place, do not restore it.
  if (NewVal == V && U.get() != V)
    return true;
  if (Instruction *OpInst = dyn_cast<Instruction>(U))
    salvageDebugInfo(*OpInst);

  replaceUse(U, NewVal);
  return true;
}
