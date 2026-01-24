//===- ValueTracking.cpp - Walk computations to compute properties --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains routines that help analyze properties that chains of
// computations have.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ValueTracking.h"
#include "ValueTrackingUtils.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumeBundleQueries.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/DomConditionCache.h"
#include "llvm/Analysis/FloatingPointPredicateUtils.h"
#include "llvm/Analysis/GuardUtils.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/Analysis/WithCache.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/ConstantFPRange.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/EHPersonalities.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/IR/IntrinsicsX86.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/KnownFPClass.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/TargetParser/RISCVTargetParser.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <optional>
#include <utility>

using namespace llvm;
using namespace llvm::vtutils;
using namespace llvm::PatternMatch;

static void computeKnownBits(const Value *V, const APInt &DemandedElts,
                             KnownBits &Known, const SimplifyQuery &Q,
                             unsigned Depth);

void llvm::computeKnownBits(const Value *V, KnownBits &Known,
                            const SimplifyQuery &Q, unsigned Depth) {
  // Since the number of lanes in a scalable vector is unknown at compile time,
  // we track one bit which is implicitly broadcast to all lanes.  This means
  // that all lanes in a scalable vector are considered demanded.
  auto *FVTy = dyn_cast<FixedVectorType>(V->getType());
  APInt DemandedElts =
      FVTy ? APInt::getAllOnes(FVTy->getNumElements()) : APInt(1, 1);
  ::computeKnownBits(V, DemandedElts, Known, Q, Depth);
}

void llvm::computeKnownBits(const Value *V, KnownBits &Known,
                            const DataLayout &DL, AssumptionCache *AC,
                            const Instruction *CxtI, const DominatorTree *DT,
                            bool UseInstrInfo, unsigned Depth) {
  computeKnownBits(V, Known,
                   SimplifyQuery(DL, DT, AC, safeCxtI(V, CxtI), UseInstrInfo),
                   Depth);
}

KnownBits llvm::computeKnownBits(const Value *V, const DataLayout &DL,
                                 AssumptionCache *AC, const Instruction *CxtI,
                                 const DominatorTree *DT, bool UseInstrInfo,
                                 unsigned Depth) {
  return computeKnownBits(
      V, SimplifyQuery(DL, DT, AC, safeCxtI(V, CxtI), UseInstrInfo), Depth);
}

KnownBits llvm::computeKnownBits(const Value *V, const APInt &DemandedElts,
                                 const DataLayout &DL, AssumptionCache *AC,
                                 const Instruction *CxtI,
                                 const DominatorTree *DT, bool UseInstrInfo,
                                 unsigned Depth) {
  return computeKnownBits(
      V, DemandedElts,
      SimplifyQuery(DL, DT, AC, safeCxtI(V, CxtI), UseInstrInfo), Depth);
}

static bool haveNoCommonBitsSetSpecialCases(const Value *LHS, const Value *RHS,
                                            const SimplifyQuery &SQ) {
  // Look for an inverted mask: (X & ~M) op (Y & M).
  {
    Value *M;
    if (match(LHS, m_c_And(m_Not(m_Value(M)), m_Value())) &&
        match(RHS, m_c_And(m_Specific(M), m_Value())) &&
        isGuaranteedNotToBeUndef(M, SQ.AC, SQ.CxtI, SQ.DT))
      return true;
  }

  // X op (Y & ~X)
  if (match(RHS, m_c_And(m_Not(m_Specific(LHS)), m_Value())) &&
      isGuaranteedNotToBeUndef(LHS, SQ.AC, SQ.CxtI, SQ.DT))
    return true;

  // X op ((X & Y) ^ Y) -- this is the canonical form of the previous pattern
  // for constant Y.
  Value *Y;
  if (match(RHS,
            m_c_Xor(m_c_And(m_Specific(LHS), m_Value(Y)), m_Deferred(Y))) &&
      isGuaranteedNotToBeUndef(LHS, SQ.AC, SQ.CxtI, SQ.DT) &&
      isGuaranteedNotToBeUndef(Y, SQ.AC, SQ.CxtI, SQ.DT))
    return true;

  // Peek through extends to find a 'not' of the other side:
  // (ext Y) op ext(~Y)
  if (match(LHS, m_ZExtOrSExt(m_Value(Y))) &&
      match(RHS, m_ZExtOrSExt(m_Not(m_Specific(Y)))) &&
      isGuaranteedNotToBeUndef(Y, SQ.AC, SQ.CxtI, SQ.DT))
    return true;

  // Look for: (A & B) op ~(A | B)
  {
    Value *A, *B;
    if (match(LHS, m_And(m_Value(A), m_Value(B))) &&
        match(RHS, m_Not(m_c_Or(m_Specific(A), m_Specific(B)))) &&
        isGuaranteedNotToBeUndef(A, SQ.AC, SQ.CxtI, SQ.DT) &&
        isGuaranteedNotToBeUndef(B, SQ.AC, SQ.CxtI, SQ.DT))
      return true;
  }

  // Look for: (X << V) op (Y >> (BitWidth - V))
  // or        (X >> V) op (Y << (BitWidth - V))
  {
    const Value *V;
    const APInt *R;
    if (((match(RHS, m_Shl(m_Value(), m_Sub(m_APInt(R), m_Value(V)))) &&
          match(LHS, m_LShr(m_Value(), m_Specific(V)))) ||
         (match(RHS, m_LShr(m_Value(), m_Sub(m_APInt(R), m_Value(V)))) &&
          match(LHS, m_Shl(m_Value(), m_Specific(V))))) &&
        R->uge(LHS->getType()->getScalarSizeInBits()))
      return true;
  }

  return false;
}

bool llvm::haveNoCommonBitsSet(const WithCache<const Value *> &LHSCache,
                               const WithCache<const Value *> &RHSCache,
                               const SimplifyQuery &SQ) {
  const Value *LHS = LHSCache.getValue();
  const Value *RHS = RHSCache.getValue();

  assert(LHS->getType() == RHS->getType() &&
         "LHS and RHS should have the same type");
  assert(LHS->getType()->isIntOrIntVectorTy() &&
         "LHS and RHS should be integers");

  if (haveNoCommonBitsSetSpecialCases(LHS, RHS, SQ) ||
      haveNoCommonBitsSetSpecialCases(RHS, LHS, SQ))
    return true;

  return KnownBits::haveNoCommonBitsSet(LHSCache.getKnownBits(SQ),
                                        RHSCache.getKnownBits(SQ));
}

bool llvm::isOnlyUsedInZeroComparison(const Instruction *I) {
  return !I->user_empty() &&
         all_of(I->users(), match_fn(m_ICmp(m_Value(), m_Zero())));
}

bool llvm::isOnlyUsedInZeroEqualityComparison(const Instruction *I) {
  return !I->user_empty() && all_of(I->users(), [](const User *U) {
    CmpPredicate P;
    return match(U, m_ICmp(P, m_Value(), m_Zero())) && ICmpInst::isEquality(P);
  });
}

bool llvm::isKnownToBeAPowerOfTwo(const Value *V, const DataLayout &DL,
                                  bool OrZero, AssumptionCache *AC,
                                  const Instruction *CxtI,
                                  const DominatorTree *DT, bool UseInstrInfo,
                                  unsigned Depth) {
  return ::isKnownToBeAPowerOfTwo(
      V, OrZero, SimplifyQuery(DL, DT, AC, safeCxtI(V, CxtI), UseInstrInfo),
      Depth);
}

static bool isKnownNonZero(const Value *V, const APInt &DemandedElts,
                           const SimplifyQuery &Q, unsigned Depth);

bool llvm::isKnownNonNegative(const Value *V, const SimplifyQuery &SQ,
                              unsigned Depth) {
  return computeKnownBits(V, SQ, Depth).isNonNegative();
}

bool llvm::isKnownPositive(const Value *V, const SimplifyQuery &SQ,
                           unsigned Depth) {
  if (auto *CI = dyn_cast<ConstantInt>(V))
    return CI->getValue().isStrictlyPositive();

  // If `isKnownNonNegative` ever becomes more sophisticated, make sure to keep
  // this updated.
  KnownBits Known = computeKnownBits(V, SQ, Depth);
  return Known.isNonNegative() &&
         (Known.isNonZero() || isKnownNonZero(V, SQ, Depth));
}

bool llvm::isKnownNegative(const Value *V, const SimplifyQuery &SQ,
                           unsigned Depth) {
  return computeKnownBits(V, SQ, Depth).isNegative();
}

static bool isKnownNonEqual(const Value *V1, const Value *V2,
                            const APInt &DemandedElts, const SimplifyQuery &Q,
                            unsigned Depth);

bool llvm::isKnownNonEqual(const Value *V1, const Value *V2,
                           const SimplifyQuery &Q, unsigned Depth) {
  // We don't support looking through casts.
  if (V1 == V2 || V1->getType() != V2->getType())
    return false;
  auto *FVTy = dyn_cast<FixedVectorType>(V1->getType());
  APInt DemandedElts =
      FVTy ? APInt::getAllOnes(FVTy->getNumElements()) : APInt(1, 1);
  return ::isKnownNonEqual(V1, V2, DemandedElts, Q, Depth);
}

bool llvm::MaskedValueIsZero(const Value *V, const APInt &Mask,
                             const SimplifyQuery &SQ, unsigned Depth) {
  KnownBits Known(Mask.getBitWidth());
  computeKnownBits(V, Known, SQ, Depth);
  return Mask.isSubsetOf(Known.Zero);
}

static unsigned ComputeNumSignBits(const Value *V, const APInt &DemandedElts,
                                   const SimplifyQuery &Q, unsigned Depth);

static unsigned ComputeNumSignBits(const Value *V, const SimplifyQuery &Q,
                                   unsigned Depth = 0) {
  auto *FVTy = dyn_cast<FixedVectorType>(V->getType());
  APInt DemandedElts =
      FVTy ? APInt::getAllOnes(FVTy->getNumElements()) : APInt(1, 1);
  return ComputeNumSignBits(V, DemandedElts, Q, Depth);
}

unsigned llvm::ComputeNumSignBits(const Value *V, const DataLayout &DL,
                                  AssumptionCache *AC, const Instruction *CxtI,
                                  const DominatorTree *DT, bool UseInstrInfo,
                                  unsigned Depth) {
  return ::ComputeNumSignBits(
      V, SimplifyQuery(DL, DT, AC, safeCxtI(V, CxtI), UseInstrInfo), Depth);
}

unsigned llvm::ComputeMaxSignificantBits(const Value *V, const DataLayout &DL,
                                         AssumptionCache *AC,
                                         const Instruction *CxtI,
                                         const DominatorTree *DT,
                                         unsigned Depth) {
  unsigned SignBits = ComputeNumSignBits(V, DL, AC, CxtI, DT, Depth);
  return V->getType()->getScalarSizeInBits() - SignBits + 1;
}

/// Try to detect the lerp pattern: a * (b - c) + c * d
/// where a >= 0, b >= 0, c >= 0, d >= 0, and b >= c.
///
/// In that particular case, we can use the following chain of reasoning:
///
///   a * (b - c) + c * d <= a' * (b - c) + a' * c = a' * b where a' = max(a, d)
///
/// Since that is true for arbitrary a, b, c and d within our constraints, we
/// can conclude that:
///
///   max(a * (b - c) + c * d) <= max(max(a), max(d)) * max(b) = U
///
/// Considering that any result of the lerp would be less or equal to U, it
/// would have at least the number of leading 0s as in U.
///
/// While being quite a specific situation, it is fairly common in computer
/// graphics in the shape of alpha blending.
///
/// Modifies given KnownOut in-place with the inferred information.
static void computeKnownBitsFromLerpPattern(const Value *Op0, const Value *Op1,
                                            const APInt &DemandedElts,
                                            KnownBits &KnownOut,
                                            const SimplifyQuery &Q,
                                            unsigned Depth) {

  Type *Ty = Op0->getType();
  const unsigned BitWidth = Ty->getScalarSizeInBits();

  // Only handle scalar types for now
  if (Ty->isVectorTy())
    return;

  // Try to match: a * (b - c) + c * d.
  // When a == 1 => A == nullptr, the same applies to d/D as well.
  const Value *A = nullptr, *B = nullptr, *C = nullptr, *D = nullptr;
  const Instruction *SubBC = nullptr;

  const auto MatchSubBC = [&]() {
    // (b - c) can have two forms that interest us:
    //
    //   1. sub nuw %b, %c
    //   2. xor %c, %b
    //
    // For the first case, nuw flag guarantees our requirement b >= c.
    //
    // The second case might happen when the analysis can infer that b is a mask
    // for c and we can transform sub operation into xor (that is usually true
    // for constant b's). Even though xor is symmetrical, canonicalization
    // ensures that the constant will be the RHS. We have additional checks
    // later on to ensure that this xor operation is equivalent to subtraction.
    return m_Instruction(SubBC, m_CombineOr(m_NUWSub(m_Value(B), m_Value(C)),
                                            m_Xor(m_Value(C), m_Value(B))));
  };

  const auto MatchASubBC = [&]() {
    // Cases:
    //   - a * (b - c)
    //   - (b - c) * a
    //   - (b - c) <- a implicitly equals 1
    return m_CombineOr(m_c_Mul(m_Value(A), MatchSubBC()), MatchSubBC());
  };

  const auto MatchCD = [&]() {
    // Cases:
    //   - d * c
    //   - c * d
    //   - c <- d implicitly equals 1
    return m_CombineOr(m_c_Mul(m_Value(D), m_Specific(C)), m_Specific(C));
  };

  const auto Match = [&](const Value *LHS, const Value *RHS) {
    // We do use m_Specific(C) in MatchCD, so we have to make sure that
    // it's bound to anything and match(LHS, MatchASubBC()) absolutely
    // has to evaluate first and return true.
    //
    // If Match returns true, it is guaranteed that B != nullptr, C != nullptr.
    return match(LHS, MatchASubBC()) && match(RHS, MatchCD());
  };

  if (!Match(Op0, Op1) && !Match(Op1, Op0))
    return;

  const auto ComputeKnownBitsOrOne = [&](const Value *V) {
    // For some of the values we use the convention of leaving
    // it nullptr to signify an implicit constant 1.
    return V ? computeKnownBits(V, DemandedElts, Q, Depth + 1)
             : KnownBits::makeConstant(APInt(BitWidth, 1));
  };

  // Check that all operands are non-negative
  const KnownBits KnownA = ComputeKnownBitsOrOne(A);
  if (!KnownA.isNonNegative())
    return;

  const KnownBits KnownD = ComputeKnownBitsOrOne(D);
  if (!KnownD.isNonNegative())
    return;

  const KnownBits KnownB = computeKnownBits(B, DemandedElts, Q, Depth + 1);
  if (!KnownB.isNonNegative())
    return;

  const KnownBits KnownC = computeKnownBits(C, DemandedElts, Q, Depth + 1);
  if (!KnownC.isNonNegative())
    return;

  // If we matched subtraction as xor, we need to actually check that xor
  // is semantically equivalent to subtraction.
  //
  // For that to be true, b has to be a mask for c or that b's known
  // ones cover all known and possible ones of c.
  if (SubBC->getOpcode() == Instruction::Xor &&
      !KnownC.getMaxValue().isSubsetOf(KnownB.getMinValue()))
    return;

  const APInt MaxA = KnownA.getMaxValue();
  const APInt MaxD = KnownD.getMaxValue();
  const APInt MaxAD = APIntOps::umax(MaxA, MaxD);
  const APInt MaxB = KnownB.getMaxValue();

  // We can't infer leading zeros info if the upper-bound estimate wraps.
  bool Overflow;
  const APInt UpperBound = MaxAD.umul_ov(MaxB, Overflow);

  if (Overflow)
    return;

  // If we know that x <= y and both are positive than x has at least the same
  // number of leading zeros as y.
  const unsigned MinimumNumberOfLeadingZeros = UpperBound.countl_zero();
  KnownOut.Zero.setHighBits(MinimumNumberOfLeadingZeros);
}

static void computeKnownBitsAddSub(bool Add, const Value *Op0, const Value *Op1,
                                   bool NSW, bool NUW,
                                   const APInt &DemandedElts,
                                   KnownBits &KnownOut, KnownBits &Known2,
                                   const SimplifyQuery &Q, unsigned Depth) {
  computeKnownBits(Op1, DemandedElts, KnownOut, Q, Depth + 1);

  // If one operand is unknown and we have no nowrap information,
  // the result will be unknown independently of the second operand.
  if (KnownOut.isUnknown() && !NSW && !NUW)
    return;

  computeKnownBits(Op0, DemandedElts, Known2, Q, Depth + 1);
  KnownOut = KnownBits::computeForAddSub(Add, NSW, NUW, Known2, KnownOut);

  if (!Add && NSW && !KnownOut.isNonNegative() &&
      isImpliedByDomCondition(ICmpInst::ICMP_SLE, Op1, Op0, Q.CxtI, Q.DL)
          .value_or(false))
    KnownOut.makeNonNegative();

  if (Add)
    // Try to match lerp pattern and combine results
    computeKnownBitsFromLerpPattern(Op0, Op1, DemandedElts, KnownOut, Q, Depth);
}

static void computeKnownBitsMul(const Value *Op0, const Value *Op1, bool NSW,
                                bool NUW, const APInt &DemandedElts,
                                KnownBits &Known, KnownBits &Known2,
                                const SimplifyQuery &Q, unsigned Depth) {
  computeKnownBits(Op1, DemandedElts, Known, Q, Depth + 1);
  computeKnownBits(Op0, DemandedElts, Known2, Q, Depth + 1);

  bool isKnownNegative = false;
  bool isKnownNonNegative = false;
  // If the multiplication is known not to overflow, compute the sign bit.
  if (NSW) {
    if (Op0 == Op1) {
      // The product of a number with itself is non-negative.
      isKnownNonNegative = true;
    } else {
      bool isKnownNonNegativeOp1 = Known.isNonNegative();
      bool isKnownNonNegativeOp0 = Known2.isNonNegative();
      bool isKnownNegativeOp1 = Known.isNegative();
      bool isKnownNegativeOp0 = Known2.isNegative();
      // The product of two numbers with the same sign is non-negative.
      isKnownNonNegative = (isKnownNegativeOp1 && isKnownNegativeOp0) ||
                           (isKnownNonNegativeOp1 && isKnownNonNegativeOp0);
      if (!isKnownNonNegative && NUW) {
        // mul nuw nsw with a factor > 1 is non-negative.
        KnownBits One = KnownBits::makeConstant(APInt(Known.getBitWidth(), 1));
        isKnownNonNegative = KnownBits::sgt(Known, One).value_or(false) ||
                             KnownBits::sgt(Known2, One).value_or(false);
      }

      // The product of a negative number and a non-negative number is either
      // negative or zero.
      if (!isKnownNonNegative)
        isKnownNegative =
            (isKnownNegativeOp1 && isKnownNonNegativeOp0 &&
             Known2.isNonZero()) ||
            (isKnownNegativeOp0 && isKnownNonNegativeOp1 && Known.isNonZero());
    }
  }

  bool SelfMultiply = Op0 == Op1;
  if (SelfMultiply)
    SelfMultiply &=
        isGuaranteedNotToBeUndef(Op0, Q.AC, Q.CxtI, Q.DT, Depth + 1);
  Known = KnownBits::mul(Known, Known2, SelfMultiply);

  if (SelfMultiply) {
    unsigned SignBits = ComputeNumSignBits(Op0, DemandedElts, Q, Depth + 1);
    unsigned TyBits = Op0->getType()->getScalarSizeInBits();
    unsigned OutValidBits = 2 * (TyBits - SignBits + 1);

    if (OutValidBits < TyBits) {
      APInt KnownZeroMask =
          APInt::getHighBitsSet(TyBits, TyBits - OutValidBits + 1);
      Known.Zero |= KnownZeroMask;
    }
  }

  // Only make use of no-wrap flags if we failed to compute the sign bit
  // directly.  This matters if the multiplication always overflows, in
  // which case we prefer to follow the result of the direct computation,
  // though as the program is invoking undefined behaviour we can choose
  // whatever we like here.
  if (isKnownNonNegative && !Known.isNegative())
    Known.makeNonNegative();
  else if (isKnownNegative && !Known.isNonNegative())
    Known.makeNegative();
}

static void computeKnownBitsFromCond(const Value *V, Value *Cond,
                                     KnownBits &Known, const SimplifyQuery &SQ,
                                     bool Invert, unsigned Depth) {
  Value *A, *B;
  if (Depth < MaxAnalysisRecursionDepth &&
      match(Cond, m_LogicalOp(m_Value(A), m_Value(B)))) {
    KnownBits Known2(Known.getBitWidth());
    KnownBits Known3(Known.getBitWidth());
    computeKnownBitsFromCond(V, A, Known2, SQ, Invert, Depth + 1);
    computeKnownBitsFromCond(V, B, Known3, SQ, Invert, Depth + 1);
    if (Invert ? match(Cond, m_LogicalOr(m_Value(), m_Value()))
               : match(Cond, m_LogicalAnd(m_Value(), m_Value())))
      Known2 = Known2.unionWith(Known3);
    else
      Known2 = Known2.intersectWith(Known3);
    Known = Known.unionWith(Known2);
    return;
  }

  if (auto *Cmp = dyn_cast<ICmpInst>(Cond)) {
    computeKnownBitsFromICmpCond(V, Cmp, Known, SQ, Invert);
    return;
  }

  if (match(Cond, m_Trunc(m_Specific(V)))) {
    KnownBits DstKnown(1);
    if (Invert) {
      DstKnown.setAllZero();
    } else {
      DstKnown.setAllOnes();
    }
    if (cast<TruncInst>(Cond)->hasNoUnsignedWrap()) {
      Known = Known.unionWith(DstKnown.zext(Known.getBitWidth()));
      return;
    }
    Known = Known.unionWith(DstKnown.anyext(Known.getBitWidth()));
    return;
  }

  if (Depth < MaxAnalysisRecursionDepth && match(Cond, m_Not(m_Value(A))))
    computeKnownBitsFromCond(V, A, Known, SQ, !Invert, Depth + 1);
}

void llvm::computeKnownBitsFromContext(const Value *V, KnownBits &Known,
                                       const SimplifyQuery &Q, unsigned Depth) {
  // Handle injected condition.
  if (Q.CC && Q.CC->AffectedValues.contains(V))
    computeKnownBitsFromCond(V, Q.CC->Cond, Known, Q, Q.CC->Invert, Depth);

  if (!Q.CxtI)
    return;

  if (Q.DC && Q.DT) {
    // Handle dominating conditions.
    for (BranchInst *BI : Q.DC->conditionsFor(V)) {
      BasicBlockEdge Edge0(BI->getParent(), BI->getSuccessor(0));
      if (Q.DT->dominates(Edge0, Q.CxtI->getParent()))
        computeKnownBitsFromCond(V, BI->getCondition(), Known, Q,
                                 /*Invert*/ false, Depth);

      BasicBlockEdge Edge1(BI->getParent(), BI->getSuccessor(1));
      if (Q.DT->dominates(Edge1, Q.CxtI->getParent()))
        computeKnownBitsFromCond(V, BI->getCondition(), Known, Q,
                                 /*Invert*/ true, Depth);
    }

    if (Known.hasConflict())
      Known.resetAll();
  }

  if (!Q.AC)
    return;

  unsigned BitWidth = Known.getBitWidth();

  // Note that the patterns below need to be kept in sync with the code
  // in AssumptionCache::updateAffectedValues.

  for (AssumptionCache::ResultElem &Elem : Q.AC->assumptionsFor(V)) {
    if (!Elem.Assume)
      continue;

    AssumeInst *I = cast<AssumeInst>(Elem.Assume);
    assert(I->getParent()->getParent() == Q.CxtI->getParent()->getParent() &&
           "Got assumption for the wrong function!");

    if (Elem.Index != AssumptionCache::ExprResultIdx) {
      if (!V->getType()->isPointerTy())
        continue;
      if (RetainedKnowledge RK = getKnowledgeFromBundle(
              *I, I->bundle_op_info_begin()[Elem.Index])) {
        // Allow AllowEphemerals in isValidAssumeForContext, as the CxtI might
        // be the producer of the pointer in the bundle. At the moment, align
        // assumptions aren't optimized away.
        if (RK.WasOn == V && RK.AttrKind == Attribute::Alignment &&
            isPowerOf2_64(RK.ArgValue) &&
            isValidAssumeForContext(I, Q.CxtI, Q.DT, /*AllowEphemerals*/ true))
          Known.Zero.setLowBits(Log2_64(RK.ArgValue));
      }
      continue;
    }

    // Warning: This loop can end up being somewhat performance sensitive.
    // We're running this loop for once for each value queried resulting in a
    // runtime of ~O(#assumes * #values).

    Value *Arg = I->getArgOperand(0);

    if (Arg == V && isValidAssumeForContext(I, Q.CxtI, Q.DT)) {
      assert(BitWidth == 1 && "assume operand is not i1?");
      (void)BitWidth;
      Known.setAllOnes();
      return;
    }
    if (match(Arg, m_Not(m_Specific(V))) &&
        isValidAssumeForContext(I, Q.CxtI, Q.DT)) {
      assert(BitWidth == 1 && "assume operand is not i1?");
      (void)BitWidth;
      Known.setAllZero();
      return;
    }
    auto *Trunc = dyn_cast<TruncInst>(Arg);
    if (Trunc && Trunc->getOperand(0) == V &&
        isValidAssumeForContext(I, Q.CxtI, Q.DT)) {
      if (Trunc->hasNoUnsignedWrap()) {
        Known = KnownBits::makeConstant(APInt(BitWidth, 1));
        return;
      }
      Known.One.setBit(0);
      return;
    }

    // The remaining tests are all recursive, so bail out if we hit the limit.
    if (Depth == MaxAnalysisRecursionDepth)
      continue;

    ICmpInst *Cmp = dyn_cast<ICmpInst>(Arg);
    if (!Cmp)
      continue;

    if (!isValidAssumeForContext(I, Q.CxtI, Q.DT))
      continue;

    computeKnownBitsFromICmpCond(V, Cmp, Known, Q, /*Invert=*/false);
  }

  // Conflicting assumption: Undefined behavior will occur on this execution
  // path.
  if (Known.hasConflict())
    Known.resetAll();
}

/// Compute known bits from a shift operator, including those with a
/// non-constant shift amount. Known is the output of this function. Known2 is a
/// pre-allocated temporary with the same bit width as Known and on return
/// contains the known bit of the shift value source. KF is an
/// operator-specific function that, given the known-bits and a shift amount,
/// compute the implied known-bits of the shift operator's result respectively
/// for that shift amount. The results from calling KF are conservatively
/// combined for all permitted shift amounts.
static void computeKnownBitsFromShiftOperator(
    const Operator *I, const APInt &DemandedElts, KnownBits &Known,
    KnownBits &Known2, const SimplifyQuery &Q, unsigned Depth,
    function_ref<KnownBits(const KnownBits &, const KnownBits &, bool)> KF) {
  computeKnownBits(I->getOperand(0), DemandedElts, Known2, Q, Depth + 1);
  computeKnownBits(I->getOperand(1), DemandedElts, Known, Q, Depth + 1);
  // To limit compile-time impact, only query isKnownNonZero() if we know at
  // least something about the shift amount.
  bool ShAmtNonZero =
      Known.isNonZero() ||
      (Known.getMaxValue().ult(Known.getBitWidth()) &&
       isKnownNonZero(I->getOperand(1), DemandedElts, Q, Depth + 1));
  Known = KF(Known2, Known, ShAmtNonZero);
}

static KnownBits
getKnownBitsFromAndXorOr(const Operator *I, const APInt &DemandedElts,
                         const KnownBits &KnownLHS, const KnownBits &KnownRHS,
                         const SimplifyQuery &Q, unsigned Depth) {
  unsigned BitWidth = KnownLHS.getBitWidth();
  KnownBits KnownOut(BitWidth);
  bool IsAnd = false;
  bool HasKnownOne = !KnownLHS.One.isZero() || !KnownRHS.One.isZero();
  Value *X = nullptr, *Y = nullptr;

  switch (I->getOpcode()) {
  case Instruction::And:
    KnownOut = KnownLHS & KnownRHS;
    IsAnd = true;
    // and(x, -x) is common idioms that will clear all but lowest set
    // bit. If we have a single known bit in x, we can clear all bits
    // above it.
    // TODO: instcombine often reassociates independent `and` which can hide
    // this pattern. Try to match and(x, and(-x, y)) / and(and(x, y), -x).
    if (HasKnownOne && match(I, m_c_And(m_Value(X), m_Neg(m_Deferred(X))))) {
      // -(-x) == x so using whichever (LHS/RHS) gets us a better result.
      if (KnownLHS.countMaxTrailingZeros() <= KnownRHS.countMaxTrailingZeros())
        KnownOut = KnownLHS.blsi();
      else
        KnownOut = KnownRHS.blsi();
    }
    break;
  case Instruction::Or:
    KnownOut = KnownLHS | KnownRHS;
    break;
  case Instruction::Xor:
    KnownOut = KnownLHS ^ KnownRHS;
    // xor(x, x-1) is common idioms that will clear all but lowest set
    // bit. If we have a single known bit in x, we can clear all bits
    // above it.
    // TODO: xor(x, x-1) is often rewritting as xor(x, x-C) where C !=
    // -1 but for the purpose of demanded bits (xor(x, x-C) &
    // Demanded) == (xor(x, x-1) & Demanded). Extend the xor pattern
    // to use arbitrary C if xor(x, x-C) as the same as xor(x, x-1).
    if (HasKnownOne &&
        match(I, m_c_Xor(m_Value(X), m_Add(m_Deferred(X), m_AllOnes())))) {
      const KnownBits &XBits = I->getOperand(0) == X ? KnownLHS : KnownRHS;
      KnownOut = XBits.blsmsk();
    }
    break;
  default:
    llvm_unreachable("Invalid Op used in 'analyzeKnownBitsFromAndXorOr'");
  }

  // and(x, add (x, -1)) is a common idiom that always clears the low bit;
  // xor/or(x, add (x, -1)) is an idiom that will always set the low bit.
  // here we handle the more general case of adding any odd number by
  // matching the form and/xor/or(x, add(x, y)) where y is odd.
  // TODO: This could be generalized to clearing any bit set in y where the
  // following bit is known to be unset in y.
  if (!KnownOut.Zero[0] && !KnownOut.One[0] &&
      (match(I, m_c_BinOp(m_Value(X), m_c_Add(m_Deferred(X), m_Value(Y)))) ||
       match(I, m_c_BinOp(m_Value(X), m_Sub(m_Deferred(X), m_Value(Y)))) ||
       match(I, m_c_BinOp(m_Value(X), m_Sub(m_Value(Y), m_Deferred(X)))))) {
    KnownBits KnownY(BitWidth);
    computeKnownBits(Y, DemandedElts, KnownY, Q, Depth + 1);
    if (KnownY.countMinTrailingOnes() > 0) {
      if (IsAnd)
        KnownOut.Zero.setBit(0);
      else
        KnownOut.One.setBit(0);
    }
  }
  return KnownOut;
}

static KnownBits computeKnownBitsForHorizontalOperation(
    const Operator *I, const APInt &DemandedElts, const SimplifyQuery &Q,
    unsigned Depth,
    const function_ref<KnownBits(const KnownBits &, const KnownBits &)>
        KnownBitsFunc) {
  APInt DemandedEltsLHS, DemandedEltsRHS;
  getHorizDemandedEltsForFirstOperand(Q.DL.getTypeSizeInBits(I->getType()),
                                      DemandedElts, DemandedEltsLHS,
                                      DemandedEltsRHS);

  const auto ComputeForSingleOpFunc =
      [Depth, &Q, KnownBitsFunc](const Value *Op, APInt &DemandedEltsOp) {
        return KnownBitsFunc(
            computeKnownBits(Op, DemandedEltsOp, Q, Depth + 1),
            computeKnownBits(Op, DemandedEltsOp << 1, Q, Depth + 1));
      };

  if (DemandedEltsRHS.isZero())
    return ComputeForSingleOpFunc(I->getOperand(0), DemandedEltsLHS);
  if (DemandedEltsLHS.isZero())
    return ComputeForSingleOpFunc(I->getOperand(1), DemandedEltsRHS);

  return ComputeForSingleOpFunc(I->getOperand(0), DemandedEltsLHS)
      .intersectWith(ComputeForSingleOpFunc(I->getOperand(1), DemandedEltsRHS));
}

// Public so this can be used in `SimplifyDemandedUseBits`.
KnownBits llvm::analyzeKnownBitsFromAndXorOr(const Operator *I,
                                             const KnownBits &KnownLHS,
                                             const KnownBits &KnownRHS,
                                             const SimplifyQuery &SQ,
                                             unsigned Depth) {
  auto *FVTy = dyn_cast<FixedVectorType>(I->getType());
  APInt DemandedElts =
      FVTy ? APInt::getAllOnes(FVTy->getNumElements()) : APInt(1, 1);

  return getKnownBitsFromAndXorOr(I, DemandedElts, KnownLHS, KnownRHS, SQ,
                                  Depth);
}

ConstantRange llvm::getVScaleRange(const Function *F, unsigned BitWidth) {
  Attribute Attr = F->getFnAttribute(Attribute::VScaleRange);
  // Without vscale_range, we only know that vscale is non-zero.
  if (!Attr.isValid())
    return ConstantRange(APInt(BitWidth, 1), APInt::getZero(BitWidth));

  unsigned AttrMin = Attr.getVScaleRangeMin();
  // Minimum is larger than vscale width, result is always poison.
  if ((unsigned)llvm::bit_width(AttrMin) > BitWidth)
    return ConstantRange::getEmpty(BitWidth);

  APInt Min(BitWidth, AttrMin);
  std::optional<unsigned> AttrMax = Attr.getVScaleRangeMax();
  if (!AttrMax || (unsigned)llvm::bit_width(*AttrMax) > BitWidth)
    return ConstantRange(Min, APInt::getZero(BitWidth));

  return ConstantRange(Min, APInt(BitWidth, *AttrMax) + 1);
}

void llvm::adjustKnownBitsForSelectArm(KnownBits &Known, Value *Cond,
                                       Value *Arm, bool Invert,
                                       const SimplifyQuery &Q, unsigned Depth) {
  // If we have a constant arm, we are done.
  if (Known.isConstant())
    return;

  // See what condition implies about the bits of the select arm.
  KnownBits CondRes(Known.getBitWidth());
  computeKnownBitsFromCond(Arm, Cond, CondRes, Q, Invert, Depth + 1);
  // If we don't get any information from the condition, no reason to
  // proceed.
  if (CondRes.isUnknown())
    return;

  // We can have conflict if the condition is dead. I.e if we have
  // (x | 64) < 32 ? (x | 64) : y
  // we will have conflict at bit 6 from the condition/the `or`.
  // In that case just return. Its not particularly important
  // what we do, as this select is going to be simplified soon.
  CondRes = CondRes.unionWith(Known);
  if (CondRes.hasConflict())
    return;

  // Finally make sure the information we found is valid. This is relatively
  // expensive so it's left for the very end.
  if (!isGuaranteedNotToBeUndef(Arm, Q.AC, Q.CxtI, Q.DT, Depth + 1))
    return;

  // Finally, we know we get information from the condition and its valid,
  // so return it.
  Known = CondRes;
}

static void computeKnownBitsFromOperator(const Operator *I,
                                         const APInt &DemandedElts,
                                         KnownBits &Known,
                                         const SimplifyQuery &Q,
                                         unsigned Depth) {
  unsigned BitWidth = Known.getBitWidth();

  KnownBits Known2(BitWidth);
  switch (I->getOpcode()) {
  default: break;
  case Instruction::Load:
    if (MDNode *MD =
            Q.IIQ.getMetadata(cast<LoadInst>(I), LLVMContext::MD_range))
      computeKnownBitsFromRangeMetadata(*MD, Known);
    break;
  case Instruction::And:
    computeKnownBits(I->getOperand(1), DemandedElts, Known, Q, Depth + 1);
    computeKnownBits(I->getOperand(0), DemandedElts, Known2, Q, Depth + 1);

    Known = getKnownBitsFromAndXorOr(I, DemandedElts, Known2, Known, Q, Depth);
    break;
  case Instruction::Or:
    computeKnownBits(I->getOperand(1), DemandedElts, Known, Q, Depth + 1);
    computeKnownBits(I->getOperand(0), DemandedElts, Known2, Q, Depth + 1);

    Known = getKnownBitsFromAndXorOr(I, DemandedElts, Known2, Known, Q, Depth);
    break;
  case Instruction::Xor:
    computeKnownBits(I->getOperand(1), DemandedElts, Known, Q, Depth + 1);
    computeKnownBits(I->getOperand(0), DemandedElts, Known2, Q, Depth + 1);

    Known = getKnownBitsFromAndXorOr(I, DemandedElts, Known2, Known, Q, Depth);
    break;
  case Instruction::Mul: {
    bool NSW = Q.IIQ.hasNoSignedWrap(cast<OverflowingBinaryOperator>(I));
    bool NUW = Q.IIQ.hasNoUnsignedWrap(cast<OverflowingBinaryOperator>(I));
    computeKnownBitsMul(I->getOperand(0), I->getOperand(1), NSW, NUW,
                        DemandedElts, Known, Known2, Q, Depth);
    break;
  }
  case Instruction::UDiv: {
    computeKnownBits(I->getOperand(0), DemandedElts, Known, Q, Depth + 1);
    computeKnownBits(I->getOperand(1), DemandedElts, Known2, Q, Depth + 1);
    Known =
        KnownBits::udiv(Known, Known2, Q.IIQ.isExact(cast<BinaryOperator>(I)));
    break;
  }
  case Instruction::SDiv: {
    computeKnownBits(I->getOperand(0), DemandedElts, Known, Q, Depth + 1);
    computeKnownBits(I->getOperand(1), DemandedElts, Known2, Q, Depth + 1);
    Known =
        KnownBits::sdiv(Known, Known2, Q.IIQ.isExact(cast<BinaryOperator>(I)));
    break;
  }
  case Instruction::Select: {
    auto ComputeForArm = [&](Value *Arm, bool Invert) {
      KnownBits Res(Known.getBitWidth());
      computeKnownBits(Arm, DemandedElts, Res, Q, Depth + 1);
      adjustKnownBitsForSelectArm(Res, I->getOperand(0), Arm, Invert, Q, Depth);
      return Res;
    };
    // Only known if known in both the LHS and RHS.
    Known =
        ComputeForArm(I->getOperand(1), /*Invert=*/false)
            .intersectWith(ComputeForArm(I->getOperand(2), /*Invert=*/true));
    break;
  }
  case Instruction::FPTrunc:
  case Instruction::FPExt:
  case Instruction::FPToUI:
  case Instruction::FPToSI:
  case Instruction::SIToFP:
  case Instruction::UIToFP:
    break; // Can't work with floating point.
  case Instruction::PtrToInt:
  case Instruction::PtrToAddr:
  case Instruction::IntToPtr:
    // Fall through and handle them the same as zext/trunc.
    [[fallthrough]];
  case Instruction::ZExt:
  case Instruction::Trunc: {
    Type *SrcTy = I->getOperand(0)->getType();

    unsigned SrcBitWidth;
    // Note that we handle pointer operands here because of inttoptr/ptrtoint
    // which fall through here.
    Type *ScalarTy = SrcTy->getScalarType();
    SrcBitWidth = ScalarTy->isPointerTy() ?
      Q.DL.getPointerTypeSizeInBits(ScalarTy) :
      Q.DL.getTypeSizeInBits(ScalarTy);

    assert(SrcBitWidth && "SrcBitWidth can't be zero");
    Known = Known.anyextOrTrunc(SrcBitWidth);
    computeKnownBits(I->getOperand(0), DemandedElts, Known, Q, Depth + 1);
    if (auto *Inst = dyn_cast<PossiblyNonNegInst>(I);
        Inst && Inst->hasNonNeg() && !Known.isNegative())
      Known.makeNonNegative();
    Known = Known.zextOrTrunc(BitWidth);
    break;
  }
  case Instruction::BitCast: {
    Type *SrcTy = I->getOperand(0)->getType();
    if (SrcTy->isIntOrPtrTy() &&
        // TODO: For now, not handling conversions like:
        // (bitcast i64 %x to <2 x i32>)
        !I->getType()->isVectorTy()) {
      computeKnownBits(I->getOperand(0), Known, Q, Depth + 1);
      break;
    }

    const Value *V;
    // Handle bitcast from floating point to integer.
    if (match(I, m_ElementWiseBitCast(m_Value(V))) &&
        V->getType()->isFPOrFPVectorTy()) {
      Type *FPType = V->getType()->getScalarType();
      KnownFPClass Result =
          computeKnownFPClass(V, DemandedElts, fcAllFlags, Q, Depth + 1);
      FPClassTest FPClasses = Result.KnownFPClasses;

      // TODO: Treat it as zero/poison if the use of I is unreachable.
      if (FPClasses == fcNone)
        break;

      if (Result.isKnownNever(fcNormal | fcSubnormal | fcNan)) {
        Known.setAllConflict();

        if (FPClasses & fcInf)
          Known = Known.intersectWith(KnownBits::makeConstant(
              APFloat::getInf(FPType->getFltSemantics()).bitcastToAPInt()));

        if (FPClasses & fcZero)
          Known = Known.intersectWith(KnownBits::makeConstant(
              APInt::getZero(FPType->getScalarSizeInBits())));

        Known.Zero.clearSignBit();
        Known.One.clearSignBit();
      }

      if (Result.SignBit) {
        if (*Result.SignBit)
          Known.makeNegative();
        else
          Known.makeNonNegative();
      }

      break;
    }

    // Handle cast from vector integer type to scalar or vector integer.
    auto *SrcVecTy = dyn_cast<FixedVectorType>(SrcTy);
    if (!SrcVecTy || !SrcVecTy->getElementType()->isIntegerTy() ||
        !I->getType()->isIntOrIntVectorTy() ||
        isa<ScalableVectorType>(I->getType()))
      break;

    unsigned NumElts = DemandedElts.getBitWidth();
    bool IsLE = Q.DL.isLittleEndian();
    // Look through a cast from narrow vector elements to wider type.
    // Examples: v4i32 -> v2i64, v3i8 -> v24
    unsigned SubBitWidth = SrcVecTy->getScalarSizeInBits();
    if (BitWidth % SubBitWidth == 0) {
      // Known bits are automatically intersected across demanded elements of a
      // vector. So for example, if a bit is computed as known zero, it must be
      // zero across all demanded elements of the vector.
      //
      // For this bitcast, each demanded element of the output is sub-divided
      // across a set of smaller vector elements in the source vector. To get
      // the known bits for an entire element of the output, compute the known
      // bits for each sub-element sequentially. This is done by shifting the
      // one-set-bit demanded elements parameter across the sub-elements for
      // consecutive calls to computeKnownBits. We are using the demanded
      // elements parameter as a mask operator.
      //
      // The known bits of each sub-element are then inserted into place
      // (dependent on endian) to form the full result of known bits.
      unsigned SubScale = BitWidth / SubBitWidth;
      APInt SubDemandedElts = APInt::getZero(NumElts * SubScale);
      for (unsigned i = 0; i != NumElts; ++i) {
        if (DemandedElts[i])
          SubDemandedElts.setBit(i * SubScale);
      }

      KnownBits KnownSrc(SubBitWidth);
      for (unsigned i = 0; i != SubScale; ++i) {
        computeKnownBits(I->getOperand(0), SubDemandedElts.shl(i), KnownSrc, Q,
                         Depth + 1);
        unsigned ShiftElt = IsLE ? i : SubScale - 1 - i;
        Known.insertBits(KnownSrc, ShiftElt * SubBitWidth);
      }
    }
    // Look through a cast from wider vector elements to narrow type.
    // Examples: v2i64 -> v4i32
    if (SubBitWidth % BitWidth == 0) {
      unsigned SubScale = SubBitWidth / BitWidth;
      KnownBits KnownSrc(SubBitWidth);
      APInt SubDemandedElts =
          APIntOps::ScaleBitMask(DemandedElts, NumElts / SubScale);
      computeKnownBits(I->getOperand(0), SubDemandedElts, KnownSrc, Q,
                       Depth + 1);

      Known.setAllConflict();
      for (unsigned i = 0; i != NumElts; ++i) {
        if (DemandedElts[i]) {
          unsigned Shifts = IsLE ? i : NumElts - 1 - i;
          unsigned Offset = (Shifts % SubScale) * BitWidth;
          Known = Known.intersectWith(KnownSrc.extractBits(BitWidth, Offset));
          if (Known.isUnknown())
            break;
        }
      }
    }
    break;
  }
  case Instruction::SExt: {
    // Compute the bits in the result that are not present in the input.
    unsigned SrcBitWidth = I->getOperand(0)->getType()->getScalarSizeInBits();

    Known = Known.trunc(SrcBitWidth);
    computeKnownBits(I->getOperand(0), DemandedElts, Known, Q, Depth + 1);
    // If the sign bit of the input is known set or clear, then we know the
    // top bits of the result.
    Known = Known.sext(BitWidth);
    break;
  }
  case Instruction::Shl: {
    bool NUW = Q.IIQ.hasNoUnsignedWrap(cast<OverflowingBinaryOperator>(I));
    bool NSW = Q.IIQ.hasNoSignedWrap(cast<OverflowingBinaryOperator>(I));
    auto KF = [NUW, NSW](const KnownBits &KnownVal, const KnownBits &KnownAmt,
                         bool ShAmtNonZero) {
      return KnownBits::shl(KnownVal, KnownAmt, NUW, NSW, ShAmtNonZero);
    };
    computeKnownBitsFromShiftOperator(I, DemandedElts, Known, Known2, Q, Depth,
                                      KF);
    // Trailing zeros of a right-shifted constant never decrease.
    const APInt *C;
    if (match(I->getOperand(0), m_APInt(C)))
      Known.Zero.setLowBits(C->countr_zero());
    break;
  }
  case Instruction::LShr: {
    bool Exact = Q.IIQ.isExact(cast<BinaryOperator>(I));
    auto KF = [Exact](const KnownBits &KnownVal, const KnownBits &KnownAmt,
                      bool ShAmtNonZero) {
      return KnownBits::lshr(KnownVal, KnownAmt, ShAmtNonZero, Exact);
    };
    computeKnownBitsFromShiftOperator(I, DemandedElts, Known, Known2, Q, Depth,
                                      KF);
    // Leading zeros of a left-shifted constant never decrease.
    const APInt *C;
    if (match(I->getOperand(0), m_APInt(C)))
      Known.Zero.setHighBits(C->countl_zero());
    break;
  }
  case Instruction::AShr: {
    bool Exact = Q.IIQ.isExact(cast<BinaryOperator>(I));
    auto KF = [Exact](const KnownBits &KnownVal, const KnownBits &KnownAmt,
                      bool ShAmtNonZero) {
      return KnownBits::ashr(KnownVal, KnownAmt, ShAmtNonZero, Exact);
    };
    computeKnownBitsFromShiftOperator(I, DemandedElts, Known, Known2, Q, Depth,
                                      KF);
    break;
  }
  case Instruction::Sub: {
    bool NSW = Q.IIQ.hasNoSignedWrap(cast<OverflowingBinaryOperator>(I));
    bool NUW = Q.IIQ.hasNoUnsignedWrap(cast<OverflowingBinaryOperator>(I));
    computeKnownBitsAddSub(false, I->getOperand(0), I->getOperand(1), NSW, NUW,
                           DemandedElts, Known, Known2, Q, Depth);
    break;
  }
  case Instruction::Add: {
    bool NSW = Q.IIQ.hasNoSignedWrap(cast<OverflowingBinaryOperator>(I));
    bool NUW = Q.IIQ.hasNoUnsignedWrap(cast<OverflowingBinaryOperator>(I));
    computeKnownBitsAddSub(true, I->getOperand(0), I->getOperand(1), NSW, NUW,
                           DemandedElts, Known, Known2, Q, Depth);
    break;
  }
  case Instruction::SRem:
    computeKnownBits(I->getOperand(0), DemandedElts, Known, Q, Depth + 1);
    computeKnownBits(I->getOperand(1), DemandedElts, Known2, Q, Depth + 1);
    Known = KnownBits::srem(Known, Known2);
    break;

  case Instruction::URem:
    computeKnownBits(I->getOperand(0), DemandedElts, Known, Q, Depth + 1);
    computeKnownBits(I->getOperand(1), DemandedElts, Known2, Q, Depth + 1);
    Known = KnownBits::urem(Known, Known2);
    break;
  case Instruction::Alloca:
    Known.Zero.setLowBits(Log2(cast<AllocaInst>(I)->getAlign()));
    break;
  case Instruction::GetElementPtr: {
    // Analyze all of the subscripts of this getelementptr instruction
    // to determine if we can prove known low zero bits.
    computeKnownBits(I->getOperand(0), Known, Q, Depth + 1);
    // Accumulate the constant indices in a separate variable
    // to minimize the number of calls to computeForAddSub.
    unsigned IndexWidth = Q.DL.getIndexTypeSizeInBits(I->getType());
    APInt AccConstIndices(IndexWidth, 0);

    auto AddIndexToKnown = [&](KnownBits IndexBits) {
      if (IndexWidth == BitWidth) {
        // Note that inbounds does *not* guarantee nsw for the addition, as only
        // the offset is signed, while the base address is unsigned.
        Known = KnownBits::add(Known, IndexBits);
      } else {
        // If the index width is smaller than the pointer width, only add the
        // value to the low bits.
        assert(IndexWidth < BitWidth &&
               "Index width can't be larger than pointer width");
        Known.insertBits(KnownBits::add(Known.trunc(IndexWidth), IndexBits), 0);
      }
    };

    gep_type_iterator GTI = gep_type_begin(I);
    for (unsigned i = 1, e = I->getNumOperands(); i != e; ++i, ++GTI) {
      // TrailZ can only become smaller, short-circuit if we hit zero.
      if (Known.isUnknown())
        break;

      Value *Index = I->getOperand(i);

      // Handle case when index is zero.
      Constant *CIndex = dyn_cast<Constant>(Index);
      if (CIndex && CIndex->isZeroValue())
        continue;

      if (StructType *STy = GTI.getStructTypeOrNull()) {
        // Handle struct member offset arithmetic.

        assert(CIndex &&
               "Access to structure field must be known at compile time");

        if (CIndex->getType()->isVectorTy())
          Index = CIndex->getSplatValue();

        unsigned Idx = cast<ConstantInt>(Index)->getZExtValue();
        const StructLayout *SL = Q.DL.getStructLayout(STy);
        uint64_t Offset = SL->getElementOffset(Idx);
        AccConstIndices += Offset;
        continue;
      }

      // Handle array index arithmetic.
      Type *IndexedTy = GTI.getIndexedType();
      if (!IndexedTy->isSized()) {
        Known.resetAll();
        break;
      }

      TypeSize Stride = GTI.getSequentialElementStride(Q.DL);
      uint64_t StrideInBytes = Stride.getKnownMinValue();
      if (!Stride.isScalable()) {
        // Fast path for constant offset.
        if (auto *CI = dyn_cast<ConstantInt>(Index)) {
          AccConstIndices +=
              CI->getValue().sextOrTrunc(IndexWidth) * StrideInBytes;
          continue;
        }
      }

      KnownBits IndexBits =
          computeKnownBits(Index, Q, Depth + 1).sextOrTrunc(IndexWidth);
      KnownBits ScalingFactor(IndexWidth);
      // Multiply by current sizeof type.
      // &A[i] == A + i * sizeof(*A[i]).
      if (Stride.isScalable()) {
        // For scalable types the only thing we know about sizeof is
        // that this is a multiple of the minimum size.
        ScalingFactor.Zero.setLowBits(llvm::countr_zero(StrideInBytes));
      } else {
        ScalingFactor =
            KnownBits::makeConstant(APInt(IndexWidth, StrideInBytes));
      }
      AddIndexToKnown(KnownBits::mul(IndexBits, ScalingFactor));
    }
    if (!Known.isUnknown() && !AccConstIndices.isZero())
      AddIndexToKnown(KnownBits::makeConstant(AccConstIndices));
    break;
  }
  case Instruction::PHI: {
    const PHINode *P = cast<PHINode>(I);
    BinaryOperator *BO = nullptr;
    Value *R = nullptr, *L = nullptr;
    if (matchSimpleRecurrence(P, BO, R, L)) {
      // Handle the case of a simple two-predecessor recurrence PHI.
      // There's a lot more that could theoretically be done here, but
      // this is sufficient to catch some interesting cases.
      unsigned Opcode = BO->getOpcode();

      switch (Opcode) {
      // If this is a shift recurrence, we know the bits being shifted in. We
      // can combine that with information about the start value of the
      // recurrence to conclude facts about the result. If this is a udiv
      // recurrence, we know that the result can never exceed either the
      // numerator or the start value, whichever is greater.
      case Instruction::LShr:
      case Instruction::AShr:
      case Instruction::Shl:
      case Instruction::UDiv:
        if (BO->getOperand(0) != I)
          break;
        [[fallthrough]];

      // For a urem recurrence, the result can never exceed the start value. The
      // phi could either be the numerator or the denominator.
      case Instruction::URem: {
        // We have matched a recurrence of the form:
        // %iv = [R, %entry], [%iv.next, %backedge]
        // %iv.next = shift_op %iv, L

        // Recurse with the phi context to avoid concern about whether facts
        // inferred hold at original context instruction.  TODO: It may be
        // correct to use the original context.  IF warranted, explore and
        // add sufficient tests to cover.
        SimplifyQuery RecQ = Q.getWithoutCondContext();
        RecQ.CxtI = P;
        computeKnownBits(R, DemandedElts, Known2, RecQ, Depth + 1);
        switch (Opcode) {
        case Instruction::Shl:
          // A shl recurrence will only increase the tailing zeros
          Known.Zero.setLowBits(Known2.countMinTrailingZeros());
          break;
        case Instruction::LShr:
        case Instruction::UDiv:
        case Instruction::URem:
          // lshr, udiv, and urem recurrences will preserve the leading zeros of
          // the start value.
          Known.Zero.setHighBits(Known2.countMinLeadingZeros());
          break;
        case Instruction::AShr:
          // An ashr recurrence will extend the initial sign bit
          Known.Zero.setHighBits(Known2.countMinLeadingZeros());
          Known.One.setHighBits(Known2.countMinLeadingOnes());
          break;
        }
        break;
      }

      // Check for operations that have the property that if
      // both their operands have low zero bits, the result
      // will have low zero bits.
      case Instruction::Add:
      case Instruction::Sub:
      case Instruction::And:
      case Instruction::Or:
      case Instruction::Mul: {
        // Change the context instruction to the "edge" that flows into the
        // phi. This is important because that is where the value is actually
        // "evaluated" even though it is used later somewhere else. (see also
        // D69571).
        SimplifyQuery RecQ = Q.getWithoutCondContext();

        unsigned OpNum = P->getOperand(0) == R ? 0 : 1;
        Instruction *RInst = P->getIncomingBlock(OpNum)->getTerminator();
        Instruction *LInst = P->getIncomingBlock(1 - OpNum)->getTerminator();

        // Ok, we have a PHI of the form L op= R. Check for low
        // zero bits.
        RecQ.CxtI = RInst;
        computeKnownBits(R, DemandedElts, Known2, RecQ, Depth + 1);

        // We need to take the minimum number of known bits
        KnownBits Known3(BitWidth);
        RecQ.CxtI = LInst;
        computeKnownBits(L, DemandedElts, Known3, RecQ, Depth + 1);

        Known.Zero.setLowBits(std::min(Known2.countMinTrailingZeros(),
                                       Known3.countMinTrailingZeros()));

        auto *OverflowOp = dyn_cast<OverflowingBinaryOperator>(BO);
        if (!OverflowOp || !Q.IIQ.hasNoSignedWrap(OverflowOp))
          break;

        switch (Opcode) {
        // If initial value of recurrence is nonnegative, and we are adding
        // a nonnegative number with nsw, the result can only be nonnegative
        // or poison value regardless of the number of times we execute the
        // add in phi recurrence. If initial value is negative and we are
        // adding a negative number with nsw, the result can only be
        // negative or poison value. Similar arguments apply to sub and mul.
        //
        // (add non-negative, non-negative) --> non-negative
        // (add negative, negative) --> negative
        case Instruction::Add: {
          if (Known2.isNonNegative() && Known3.isNonNegative())
            Known.makeNonNegative();
          else if (Known2.isNegative() && Known3.isNegative())
            Known.makeNegative();
          break;
        }

        // (sub nsw non-negative, negative) --> non-negative
        // (sub nsw negative, non-negative) --> negative
        case Instruction::Sub: {
          if (BO->getOperand(0) != I)
            break;
          if (Known2.isNonNegative() && Known3.isNegative())
            Known.makeNonNegative();
          else if (Known2.isNegative() && Known3.isNonNegative())
            Known.makeNegative();
          break;
        }

        // (mul nsw non-negative, non-negative) --> non-negative
        case Instruction::Mul:
          if (Known2.isNonNegative() && Known3.isNonNegative())
            Known.makeNonNegative();
          break;

        default:
          break;
        }
        break;
      }

      default:
        break;
      }
    }

    // Unreachable blocks may have zero-operand PHI nodes.
    if (P->getNumIncomingValues() == 0)
      break;

    // Otherwise take the unions of the known bit sets of the operands,
    // taking conservative care to avoid excessive recursion.
    if (Depth < MaxAnalysisRecursionDepth - 1 && Known.isUnknown()) {
      // Skip if every incoming value references to ourself.
      if (isa_and_nonnull<UndefValue>(P->hasConstantValue()))
        break;

      Known.setAllConflict();
      for (const Use &U : P->operands()) {
        Value *IncValue;
        const PHINode *CxtPhi;
        Instruction *CxtI;
        breakSelfRecursivePHI(&U, P, IncValue, CxtI, &CxtPhi);
        // Skip direct self references.
        if (IncValue == P)
          continue;

        // Change the context instruction to the "edge" that flows into the
        // phi. This is important because that is where the value is actually
        // "evaluated" even though it is used later somewhere else. (see also
        // D69571).
        SimplifyQuery RecQ = Q.getWithoutCondContext().getWithInstruction(CxtI);

        Known2 = KnownBits(BitWidth);

        // Recurse, but cap the recursion to one level, because we don't
        // want to waste time spinning around in loops.
        // TODO: See if we can base recursion limiter on number of incoming phi
        // edges so we don't overly clamp analysis.
        computeKnownBits(IncValue, DemandedElts, Known2, RecQ,
                         MaxAnalysisRecursionDepth - 1);

        // See if we can further use a conditional branch into the phi
        // to help us determine the range of the value.
        if (!Known2.isConstant()) {
          CmpPredicate Pred;
          const APInt *RHSC;
          BasicBlock *TrueSucc, *FalseSucc;
          // TODO: Use RHS Value and compute range from its known bits.
          if (match(RecQ.CxtI,
                    m_Br(m_c_ICmp(Pred, m_Specific(IncValue), m_APInt(RHSC)),
                         m_BasicBlock(TrueSucc), m_BasicBlock(FalseSucc)))) {
            // Check for cases of duplicate successors.
            if ((TrueSucc == CxtPhi->getParent()) !=
                (FalseSucc == CxtPhi->getParent())) {
              // If we're using the false successor, invert the predicate.
              if (FalseSucc == CxtPhi->getParent())
                Pred = CmpInst::getInversePredicate(Pred);
              // Get the knownbits implied by the incoming phi condition.
              auto CR = ConstantRange::makeExactICmpRegion(Pred, *RHSC);
              KnownBits KnownUnion = Known2.unionWith(CR.toKnownBits());
              // We can have conflicts here if we are analyzing deadcode (its
              // impossible for us reach this BB based the icmp).
              if (KnownUnion.hasConflict()) {
                // No reason to continue analyzing in a known dead region, so
                // just resetAll and break. This will cause us to also exit the
                // outer loop.
                Known.resetAll();
                break;
              }
              Known2 = KnownUnion;
            }
          }
        }

        Known = Known.intersectWith(Known2);
        // If all bits have been ruled out, there's no need to check
        // more operands.
        if (Known.isUnknown())
          break;
      }
    }
    break;
  }
  case Instruction::Call:
  case Instruction::Invoke: {
    // If range metadata is attached to this call, set known bits from that,
    // and then intersect with known bits based on other properties of the
    // function.
    if (MDNode *MD =
            Q.IIQ.getMetadata(cast<Instruction>(I), LLVMContext::MD_range))
      computeKnownBitsFromRangeMetadata(*MD, Known);

    const auto *CB = cast<CallBase>(I);

    if (std::optional<ConstantRange> Range = CB->getRange())
      Known = Known.unionWith(Range->toKnownBits());

    if (const Value *RV = CB->getReturnedArgOperand()) {
      if (RV->getType() == I->getType()) {
        computeKnownBits(RV, Known2, Q, Depth + 1);
        Known = Known.unionWith(Known2);
        // If the function doesn't return properly for all input values
        // (e.g. unreachable exits) then there might be conflicts between the
        // argument value and the range metadata. Simply discard the known bits
        // in case of conflicts.
        if (Known.hasConflict())
          Known.resetAll();
      }
    }
    if (const IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
      switch (II->getIntrinsicID()) {
      default:
        break;
      case Intrinsic::abs: {
        computeKnownBits(I->getOperand(0), DemandedElts, Known2, Q, Depth + 1);
        bool IntMinIsPoison = match(II->getArgOperand(1), m_One());
        Known = Known.unionWith(Known2.abs(IntMinIsPoison));
        break;
      }
      case Intrinsic::bitreverse:
        computeKnownBits(I->getOperand(0), DemandedElts, Known2, Q, Depth + 1);
        Known = Known.unionWith(Known2.reverseBits());
        break;
      case Intrinsic::bswap:
        computeKnownBits(I->getOperand(0), DemandedElts, Known2, Q, Depth + 1);
        Known = Known.unionWith(Known2.byteSwap());
        break;
      case Intrinsic::ctlz: {
        computeKnownBits(I->getOperand(0), DemandedElts, Known2, Q, Depth + 1);
        // If we have a known 1, its position is our upper bound.
        unsigned PossibleLZ = Known2.countMaxLeadingZeros();
        // If this call is poison for 0 input, the result will be less than 2^n.
        if (II->getArgOperand(1) == ConstantInt::getTrue(II->getContext()))
          PossibleLZ = std::min(PossibleLZ, BitWidth - 1);
        unsigned LowBits = llvm::bit_width(PossibleLZ);
        Known.Zero.setBitsFrom(LowBits);
        break;
      }
      case Intrinsic::cttz: {
        computeKnownBits(I->getOperand(0), DemandedElts, Known2, Q, Depth + 1);
        // If we have a known 1, its position is our upper bound.
        unsigned PossibleTZ = Known2.countMaxTrailingZeros();
        // If this call is poison for 0 input, the result will be less than 2^n.
        if (II->getArgOperand(1) == ConstantInt::getTrue(II->getContext()))
          PossibleTZ = std::min(PossibleTZ, BitWidth - 1);
        unsigned LowBits = llvm::bit_width(PossibleTZ);
        Known.Zero.setBitsFrom(LowBits);
        break;
      }
      case Intrinsic::ctpop: {
        computeKnownBits(I->getOperand(0), DemandedElts, Known2, Q, Depth + 1);
        // We can bound the space the count needs.  Also, bits known to be zero
        // can't contribute to the population.
        unsigned BitsPossiblySet = Known2.countMaxPopulation();
        unsigned LowBits = llvm::bit_width(BitsPossiblySet);
        Known.Zero.setBitsFrom(LowBits);
        // TODO: we could bound KnownOne using the lower bound on the number
        // of bits which might be set provided by popcnt KnownOne2.
        break;
      }
      case Intrinsic::fshr:
      case Intrinsic::fshl: {
        const APInt *SA;
        if (!match(I->getOperand(2), m_APInt(SA)))
          break;

        // Normalize to funnel shift left.
        uint64_t ShiftAmt = SA->urem(BitWidth);
        if (II->getIntrinsicID() == Intrinsic::fshr)
          ShiftAmt = BitWidth - ShiftAmt;

        KnownBits Known3(BitWidth);
        computeKnownBits(I->getOperand(0), DemandedElts, Known2, Q, Depth + 1);
        computeKnownBits(I->getOperand(1), DemandedElts, Known3, Q, Depth + 1);

        Known2 <<= ShiftAmt;
        Known3 >>= BitWidth - ShiftAmt;
        Known = Known2.unionWith(Known3);
        break;
      }
      case Intrinsic::uadd_sat:
        computeKnownBits(I->getOperand(0), DemandedElts, Known, Q, Depth + 1);
        computeKnownBits(I->getOperand(1), DemandedElts, Known2, Q, Depth + 1);
        Known = KnownBits::uadd_sat(Known, Known2);
        break;
      case Intrinsic::usub_sat:
        computeKnownBits(I->getOperand(0), DemandedElts, Known, Q, Depth + 1);
        computeKnownBits(I->getOperand(1), DemandedElts, Known2, Q, Depth + 1);
        Known = KnownBits::usub_sat(Known, Known2);
        break;
      case Intrinsic::sadd_sat:
        computeKnownBits(I->getOperand(0), DemandedElts, Known, Q, Depth + 1);
        computeKnownBits(I->getOperand(1), DemandedElts, Known2, Q, Depth + 1);
        Known = KnownBits::sadd_sat(Known, Known2);
        break;
      case Intrinsic::ssub_sat:
        computeKnownBits(I->getOperand(0), DemandedElts, Known, Q, Depth + 1);
        computeKnownBits(I->getOperand(1), DemandedElts, Known2, Q, Depth + 1);
        Known = KnownBits::ssub_sat(Known, Known2);
        break;
        // Vec reverse preserves bits from input vec.
      case Intrinsic::vector_reverse:
        computeKnownBits(I->getOperand(0), DemandedElts.reverseBits(), Known, Q,
                         Depth + 1);
        break;
        // for min/max/and/or reduce, any bit common to each element in the
        // input vec is set in the output.
      case Intrinsic::vector_reduce_and:
      case Intrinsic::vector_reduce_or:
      case Intrinsic::vector_reduce_umax:
      case Intrinsic::vector_reduce_umin:
      case Intrinsic::vector_reduce_smax:
      case Intrinsic::vector_reduce_smin:
        computeKnownBits(I->getOperand(0), Known, Q, Depth + 1);
        break;
      case Intrinsic::vector_reduce_xor: {
        computeKnownBits(I->getOperand(0), Known, Q, Depth + 1);
        // The zeros common to all vecs are zero in the output.
        // If the number of elements is odd, then the common ones remain. If the
        // number of elements is even, then the common ones becomes zeros.
        auto *VecTy = cast<VectorType>(I->getOperand(0)->getType());
        // Even, so the ones become zeros.
        bool EvenCnt = VecTy->getElementCount().isKnownEven();
        if (EvenCnt)
          Known.Zero |= Known.One;
        // Maybe even element count so need to clear ones.
        if (VecTy->isScalableTy() || EvenCnt)
          Known.One.clearAllBits();
        break;
      }
      case Intrinsic::vector_reduce_add: {
        auto *VecTy = dyn_cast<FixedVectorType>(I->getOperand(0)->getType());
        if (!VecTy)
          break;
        computeKnownBits(I->getOperand(0), Known, Q, Depth + 1);
        Known = Known.reduceAdd(VecTy->getNumElements());
        break;
      }
      case Intrinsic::umin:
        computeKnownBits(I->getOperand(0), DemandedElts, Known, Q, Depth + 1);
        computeKnownBits(I->getOperand(1), DemandedElts, Known2, Q, Depth + 1);
        Known = KnownBits::umin(Known, Known2);
        break;
      case Intrinsic::umax:
        computeKnownBits(I->getOperand(0), DemandedElts, Known, Q, Depth + 1);
        computeKnownBits(I->getOperand(1), DemandedElts, Known2, Q, Depth + 1);
        Known = KnownBits::umax(Known, Known2);
        break;
      case Intrinsic::smin:
        computeKnownBits(I->getOperand(0), DemandedElts, Known, Q, Depth + 1);
        computeKnownBits(I->getOperand(1), DemandedElts, Known2, Q, Depth + 1);
        Known = KnownBits::smin(Known, Known2);
        unionWithMinMaxIntrinsicClamp(II, Known);
        break;
      case Intrinsic::smax:
        computeKnownBits(I->getOperand(0), DemandedElts, Known, Q, Depth + 1);
        computeKnownBits(I->getOperand(1), DemandedElts, Known2, Q, Depth + 1);
        Known = KnownBits::smax(Known, Known2);
        unionWithMinMaxIntrinsicClamp(II, Known);
        break;
      case Intrinsic::ptrmask: {
        computeKnownBits(I->getOperand(0), DemandedElts, Known, Q, Depth + 1);

        const Value *Mask = I->getOperand(1);
        Known2 = KnownBits(Mask->getType()->getScalarSizeInBits());
        computeKnownBits(Mask, DemandedElts, Known2, Q, Depth + 1);
        // TODO: 1-extend would be more precise.
        Known &= Known2.anyextOrTrunc(BitWidth);
        break;
      }
      case Intrinsic::x86_sse2_pmulh_w:
      case Intrinsic::x86_avx2_pmulh_w:
      case Intrinsic::x86_avx512_pmulh_w_512:
        computeKnownBits(I->getOperand(0), DemandedElts, Known, Q, Depth + 1);
        computeKnownBits(I->getOperand(1), DemandedElts, Known2, Q, Depth + 1);
        Known = KnownBits::mulhs(Known, Known2);
        break;
      case Intrinsic::x86_sse2_pmulhu_w:
      case Intrinsic::x86_avx2_pmulhu_w:
      case Intrinsic::x86_avx512_pmulhu_w_512:
        computeKnownBits(I->getOperand(0), DemandedElts, Known, Q, Depth + 1);
        computeKnownBits(I->getOperand(1), DemandedElts, Known2, Q, Depth + 1);
        Known = KnownBits::mulhu(Known, Known2);
        break;
      case Intrinsic::x86_sse42_crc32_64_64:
        Known.Zero.setBitsFrom(32);
        break;
      case Intrinsic::x86_ssse3_phadd_d_128:
      case Intrinsic::x86_ssse3_phadd_w_128:
      case Intrinsic::x86_avx2_phadd_d:
      case Intrinsic::x86_avx2_phadd_w: {
        Known = computeKnownBitsForHorizontalOperation(
            I, DemandedElts, Q, Depth,
            [](const KnownBits &KnownLHS, const KnownBits &KnownRHS) {
              return KnownBits::add(KnownLHS, KnownRHS);
            });
        break;
      }
      case Intrinsic::x86_ssse3_phadd_sw_128:
      case Intrinsic::x86_avx2_phadd_sw: {
        Known = computeKnownBitsForHorizontalOperation(
            I, DemandedElts, Q, Depth, KnownBits::sadd_sat);
        break;
      }
      case Intrinsic::x86_ssse3_phsub_d_128:
      case Intrinsic::x86_ssse3_phsub_w_128:
      case Intrinsic::x86_avx2_phsub_d:
      case Intrinsic::x86_avx2_phsub_w: {
        Known = computeKnownBitsForHorizontalOperation(
            I, DemandedElts, Q, Depth,
            [](const KnownBits &KnownLHS, const KnownBits &KnownRHS) {
              return KnownBits::sub(KnownLHS, KnownRHS);
            });
        break;
      }
      case Intrinsic::x86_ssse3_phsub_sw_128:
      case Intrinsic::x86_avx2_phsub_sw: {
        Known = computeKnownBitsForHorizontalOperation(
            I, DemandedElts, Q, Depth, KnownBits::ssub_sat);
        break;
      }
      case Intrinsic::riscv_vsetvli:
      case Intrinsic::riscv_vsetvlimax: {
        bool HasAVL = II->getIntrinsicID() == Intrinsic::riscv_vsetvli;
        const ConstantRange Range = getVScaleRange(II->getFunction(), BitWidth);
        uint64_t SEW = RISCVVType::decodeVSEW(
            cast<ConstantInt>(II->getArgOperand(HasAVL))->getZExtValue());
        RISCVVType::VLMUL VLMUL = static_cast<RISCVVType::VLMUL>(
            cast<ConstantInt>(II->getArgOperand(1 + HasAVL))->getZExtValue());
        uint64_t MaxVLEN =
            Range.getUnsignedMax().getZExtValue() * RISCV::RVVBitsPerBlock;
        uint64_t MaxVL = MaxVLEN / RISCVVType::getSEWLMULRatio(SEW, VLMUL);

        // Result of vsetvli must be not larger than AVL.
        if (HasAVL)
          if (auto *CI = dyn_cast<ConstantInt>(II->getArgOperand(0)))
            MaxVL = std::min(MaxVL, CI->getZExtValue());

        unsigned KnownZeroFirstBit = Log2_32(MaxVL) + 1;
        if (BitWidth > KnownZeroFirstBit)
          Known.Zero.setBitsFrom(KnownZeroFirstBit);
        break;
      }
      case Intrinsic::vscale: {
        if (!II->getParent() || !II->getFunction())
          break;

        Known = getVScaleRange(II->getFunction(), BitWidth).toKnownBits();
        break;
      }
      }
    }
    break;
  }
  case Instruction::ShuffleVector: {
    if (auto *Splat = getSplatValue(I)) {
      computeKnownBits(Splat, Known, Q, Depth + 1);
      break;
    }

    auto *Shuf = dyn_cast<ShuffleVectorInst>(I);
    // FIXME: Do we need to handle ConstantExpr involving shufflevectors?
    if (!Shuf) {
      Known.resetAll();
      return;
    }
    // For undef elements, we don't know anything about the common state of
    // the shuffle result.
    APInt DemandedLHS, DemandedRHS;
    if (!getShuffleDemandedElts(Shuf, DemandedElts, DemandedLHS, DemandedRHS)) {
      Known.resetAll();
      return;
    }
    Known.setAllConflict();
    if (!!DemandedLHS) {
      const Value *LHS = Shuf->getOperand(0);
      computeKnownBits(LHS, DemandedLHS, Known, Q, Depth + 1);
      // If we don't know any bits, early out.
      if (Known.isUnknown())
        break;
    }
    if (!!DemandedRHS) {
      const Value *RHS = Shuf->getOperand(1);
      computeKnownBits(RHS, DemandedRHS, Known2, Q, Depth + 1);
      Known = Known.intersectWith(Known2);
    }
    break;
  }
  case Instruction::InsertElement: {
    if (isa<ScalableVectorType>(I->getType())) {
      Known.resetAll();
      return;
    }
    const Value *Vec = I->getOperand(0);
    const Value *Elt = I->getOperand(1);
    auto *CIdx = dyn_cast<ConstantInt>(I->getOperand(2));
    unsigned NumElts = DemandedElts.getBitWidth();
    APInt DemandedVecElts = DemandedElts;
    bool NeedsElt = true;
    // If we know the index we are inserting too, clear it from Vec check.
    if (CIdx && CIdx->getValue().ult(NumElts)) {
      DemandedVecElts.clearBit(CIdx->getZExtValue());
      NeedsElt = DemandedElts[CIdx->getZExtValue()];
    }

    Known.setAllConflict();
    if (NeedsElt) {
      computeKnownBits(Elt, Known, Q, Depth + 1);
      // If we don't know any bits, early out.
      if (Known.isUnknown())
        break;
    }

    if (!DemandedVecElts.isZero()) {
      computeKnownBits(Vec, DemandedVecElts, Known2, Q, Depth + 1);
      Known = Known.intersectWith(Known2);
    }
    break;
  }
  case Instruction::ExtractElement: {
    // Look through extract element. If the index is non-constant or
    // out-of-range demand all elements, otherwise just the extracted element.
    const Value *Vec = I->getOperand(0);
    const Value *Idx = I->getOperand(1);
    auto *CIdx = dyn_cast<ConstantInt>(Idx);
    if (isa<ScalableVectorType>(Vec->getType())) {
      // FIXME: there's probably *something* we can do with scalable vectors
      Known.resetAll();
      break;
    }
    unsigned NumElts = cast<FixedVectorType>(Vec->getType())->getNumElements();
    APInt DemandedVecElts = APInt::getAllOnes(NumElts);
    if (CIdx && CIdx->getValue().ult(NumElts))
      DemandedVecElts = APInt::getOneBitSet(NumElts, CIdx->getZExtValue());
    computeKnownBits(Vec, DemandedVecElts, Known, Q, Depth + 1);
    break;
  }
  case Instruction::ExtractValue:
    if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I->getOperand(0))) {
      const ExtractValueInst *EVI = cast<ExtractValueInst>(I);
      if (EVI->getNumIndices() != 1) break;
      if (EVI->getIndices()[0] == 0) {
        switch (II->getIntrinsicID()) {
        default: break;
        case Intrinsic::uadd_with_overflow:
        case Intrinsic::sadd_with_overflow:
          computeKnownBitsAddSub(
              true, II->getArgOperand(0), II->getArgOperand(1), /*NSW=*/false,
              /* NUW=*/false, DemandedElts, Known, Known2, Q, Depth);
          break;
        case Intrinsic::usub_with_overflow:
        case Intrinsic::ssub_with_overflow:
          computeKnownBitsAddSub(
              false, II->getArgOperand(0), II->getArgOperand(1), /*NSW=*/false,
              /* NUW=*/false, DemandedElts, Known, Known2, Q, Depth);
          break;
        case Intrinsic::umul_with_overflow:
        case Intrinsic::smul_with_overflow:
          computeKnownBitsMul(II->getArgOperand(0), II->getArgOperand(1), false,
                              false, DemandedElts, Known, Known2, Q, Depth);
          break;
        }
      }
    }
    break;
  case Instruction::Freeze:
    if (isGuaranteedNotToBePoison(I->getOperand(0), Q.AC, Q.CxtI, Q.DT,
                                  Depth + 1))
      computeKnownBits(I->getOperand(0), Known, Q, Depth + 1);
    break;
  }
}

/// Determine which bits of V are known to be either zero or one and return
/// them.
KnownBits llvm::computeKnownBits(const Value *V, const APInt &DemandedElts,
                                 const SimplifyQuery &Q, unsigned Depth) {
  KnownBits Known(getBitWidth(V->getType(), Q.DL));
  ::computeKnownBits(V, DemandedElts, Known, Q, Depth);
  return Known;
}

/// Determine which bits of V are known to be either zero or one and return
/// them.
KnownBits llvm::computeKnownBits(const Value *V, const SimplifyQuery &Q,
                                 unsigned Depth) {
  KnownBits Known(getBitWidth(V->getType(), Q.DL));
  computeKnownBits(V, Known, Q, Depth);
  return Known;
}

/// Determine which bits of V are known to be either zero or one and return
/// them in the Known bit set.
///
/// NOTE: we cannot consider 'undef' to be "IsZero" here.  The problem is that
/// we cannot optimize based on the assumption that it is zero without changing
/// it to be an explicit zero.  If we don't change it to zero, other code could
/// optimized based on the contradictory assumption that it is non-zero.
/// Because instcombine aggressively folds operations with undef args anyway,
/// this won't lose us code quality.
///
/// This function is defined on values with integer type, values with pointer
/// type, and vectors of integers.  In the case
/// where V is a vector, known zero, and known one values are the
/// same width as the vector element, and the bit is set only if it is true
/// for all of the demanded elements in the vector specified by DemandedElts.
void computeKnownBits(const Value *V, const APInt &DemandedElts,
                      KnownBits &Known, const SimplifyQuery &Q,
                      unsigned Depth) {
  if (!DemandedElts) {
    // No demanded elts, better to assume we don't know anything.
    Known.resetAll();
    return;
  }

  assert(V && "No Value?");
  assert(Depth <= MaxAnalysisRecursionDepth && "Limit Search Depth");

#ifndef NDEBUG
  Type *Ty = V->getType();
  unsigned BitWidth = Known.getBitWidth();

  assert((Ty->isIntOrIntVectorTy(BitWidth) || Ty->isPtrOrPtrVectorTy()) &&
         "Not integer or pointer type!");

  if (auto *FVTy = dyn_cast<FixedVectorType>(Ty)) {
    assert(
        FVTy->getNumElements() == DemandedElts.getBitWidth() &&
        "DemandedElt width should equal the fixed vector number of elements");
  } else {
    assert(DemandedElts == APInt(1, 1) &&
           "DemandedElt width should be 1 for scalars or scalable vectors");
  }

  Type *ScalarTy = Ty->getScalarType();
  if (ScalarTy->isPointerTy()) {
    assert(BitWidth == Q.DL.getPointerTypeSizeInBits(ScalarTy) &&
           "V and Known should have same BitWidth");
  } else {
    assert(BitWidth == Q.DL.getTypeSizeInBits(ScalarTy) &&
           "V and Known should have same BitWidth");
  }
#endif

  const APInt *C;
  if (match(V, m_APInt(C))) {
    // We know all of the bits for a scalar constant or a splat vector constant!
    Known = KnownBits::makeConstant(*C);
    return;
  }
  // Null and aggregate-zero are all-zeros.
  if (isa<ConstantPointerNull>(V) || isa<ConstantAggregateZero>(V)) {
    Known.setAllZero();
    return;
  }
  // Handle a constant vector by taking the intersection of the known bits of
  // each element.
  if (const ConstantDataVector *CDV = dyn_cast<ConstantDataVector>(V)) {
    assert(!isa<ScalableVectorType>(V->getType()));
    // We know that CDV must be a vector of integers. Take the intersection of
    // each element.
    Known.setAllConflict();
    for (unsigned i = 0, e = CDV->getNumElements(); i != e; ++i) {
      if (!DemandedElts[i])
        continue;
      APInt Elt = CDV->getElementAsAPInt(i);
      Known.Zero &= ~Elt;
      Known.One &= Elt;
    }
    if (Known.hasConflict())
      Known.resetAll();
    return;
  }

  if (const auto *CV = dyn_cast<ConstantVector>(V)) {
    assert(!isa<ScalableVectorType>(V->getType()));
    // We know that CV must be a vector of integers. Take the intersection of
    // each element.
    Known.setAllConflict();
    for (unsigned i = 0, e = CV->getNumOperands(); i != e; ++i) {
      if (!DemandedElts[i])
        continue;
      Constant *Element = CV->getAggregateElement(i);
      if (isa<PoisonValue>(Element))
        continue;
      auto *ElementCI = dyn_cast_or_null<ConstantInt>(Element);
      if (!ElementCI) {
        Known.resetAll();
        return;
      }
      const APInt &Elt = ElementCI->getValue();
      Known.Zero &= ~Elt;
      Known.One &= Elt;
    }
    if (Known.hasConflict())
      Known.resetAll();
    return;
  }

  // Start out not knowing anything.
  Known.resetAll();

  // We can't imply anything about undefs.
  if (isa<UndefValue>(V))
    return;

  // There's no point in looking through other users of ConstantData for
  // assumptions.  Confirm that we've handled them all.
  assert(!isa<ConstantData>(V) && "Unhandled constant data!");

  if (const auto *A = dyn_cast<Argument>(V))
    if (std::optional<ConstantRange> Range = A->getRange())
      Known = Range->toKnownBits();

  // All recursive calls that increase depth must come after this.
  if (Depth == MaxAnalysisRecursionDepth)
    return;

  // A weak GlobalAlias is totally unknown. A non-weak GlobalAlias has
  // the bits of its aliasee.
  if (const GlobalAlias *GA = dyn_cast<GlobalAlias>(V)) {
    if (!GA->isInterposable())
      computeKnownBits(GA->getAliasee(), Known, Q, Depth + 1);
    return;
  }

  if (const Operator *I = dyn_cast<Operator>(V))
    computeKnownBitsFromOperator(I, DemandedElts, Known, Q, Depth);
  else if (const GlobalValue *GV = dyn_cast<GlobalValue>(V)) {
    if (std::optional<ConstantRange> CR = GV->getAbsoluteSymbolRange())
      Known = CR->toKnownBits();
  }

  // Aligned pointers have trailing zeros - refine Known.Zero set
  if (isa<PointerType>(V->getType())) {
    Align Alignment = V->getPointerAlignment(Q.DL);
    Known.Zero.setLowBits(Log2(Alignment));
  }

  // computeKnownBitsFromContext strictly refines Known.
  // Therefore, we run them after computeKnownBitsFromOperator.

  // Check whether we can determine known bits from context such as assumes.
  computeKnownBitsFromContext(V, Known, Q, Depth);
}

/// Try to detect a recurrence that the value of the induction variable is
/// always a power of two (or zero).
static bool isPowerOfTwoRecurrence(const PHINode *PN, bool OrZero,
                                   SimplifyQuery &Q, unsigned Depth) {
  BinaryOperator *BO = nullptr;
  Value *Start = nullptr, *Step = nullptr;
  if (!matchSimpleRecurrence(PN, BO, Start, Step))
    return false;

  // Initial value must be a power of two.
  for (const Use &U : PN->operands()) {
    if (U.get() == Start) {
      // Initial value comes from a different BB, need to adjust context
      // instruction for analysis.
      Q.CxtI = PN->getIncomingBlock(U)->getTerminator();
      if (!isKnownToBeAPowerOfTwo(Start, OrZero, Q, Depth))
        return false;
    }
  }

  // Except for Mul, the induction variable must be on the left side of the
  // increment expression, otherwise its value can be arbitrary.
  if (BO->getOpcode() != Instruction::Mul && BO->getOperand(1) != Step)
    return false;

  Q.CxtI = BO->getParent()->getTerminator();
  switch (BO->getOpcode()) {
  case Instruction::Mul:
    // Power of two is closed under multiplication.
    return (OrZero || Q.IIQ.hasNoUnsignedWrap(BO) ||
            Q.IIQ.hasNoSignedWrap(BO)) &&
           isKnownToBeAPowerOfTwo(Step, OrZero, Q, Depth);
  case Instruction::SDiv:
    // Start value must not be signmask for signed division, so simply being a
    // power of two is not sufficient, and it has to be a constant.
    if (!match(Start, m_Power2()) || match(Start, m_SignMask()))
      return false;
    [[fallthrough]];
  case Instruction::UDiv:
    // Divisor must be a power of two.
    // If OrZero is false, cannot guarantee induction variable is non-zero after
    // division, same for Shr, unless it is exact division.
    return (OrZero || Q.IIQ.isExact(BO)) &&
           isKnownToBeAPowerOfTwo(Step, false, Q, Depth);
  case Instruction::Shl:
    return OrZero || Q.IIQ.hasNoUnsignedWrap(BO) || Q.IIQ.hasNoSignedWrap(BO);
  case Instruction::AShr:
    if (!match(Start, m_Power2()) || match(Start, m_SignMask()))
      return false;
    [[fallthrough]];
  case Instruction::LShr:
    return OrZero || Q.IIQ.isExact(BO);
  default:
    return false;
  }
}

/// Return true if the given value is known to have exactly one
/// bit set when defined. For vectors return true if every element is known to
/// be a power of two when defined. Supports values with integer or pointer
/// types and vectors of integers.
bool llvm::isKnownToBeAPowerOfTwo(const Value *V, bool OrZero,
                                  const SimplifyQuery &Q, unsigned Depth) {
  assert(Depth <= MaxAnalysisRecursionDepth && "Limit Search Depth");

  if (isa<Constant>(V))
    return OrZero ? match(V, m_Power2OrZero()) : match(V, m_Power2());

  // i1 is by definition a power of 2 or zero.
  if (OrZero && V->getType()->getScalarSizeInBits() == 1)
    return true;

  // Try to infer from assumptions.
  if (Q.AC && Q.CxtI) {
    for (auto &AssumeVH : Q.AC->assumptionsFor(V)) {
      if (!AssumeVH)
        continue;
      CallInst *I = cast<CallInst>(AssumeVH);
      if (isImpliedToBeAPowerOfTwoFromCond(V, OrZero, I->getArgOperand(0),
                                           /*CondIsTrue=*/true) &&
          isValidAssumeForContext(I, Q.CxtI, Q.DT))
        return true;
    }
  }

  // Handle dominating conditions.
  if (Q.DC && Q.CxtI && Q.DT) {
    for (BranchInst *BI : Q.DC->conditionsFor(V)) {
      Value *Cond = BI->getCondition();

      BasicBlockEdge Edge0(BI->getParent(), BI->getSuccessor(0));
      if (isImpliedToBeAPowerOfTwoFromCond(V, OrZero, Cond,
                                           /*CondIsTrue=*/true) &&
          Q.DT->dominates(Edge0, Q.CxtI->getParent()))
        return true;

      BasicBlockEdge Edge1(BI->getParent(), BI->getSuccessor(1));
      if (isImpliedToBeAPowerOfTwoFromCond(V, OrZero, Cond,
                                           /*CondIsTrue=*/false) &&
          Q.DT->dominates(Edge1, Q.CxtI->getParent()))
        return true;
    }
  }

  auto *I = dyn_cast<Instruction>(V);
  if (!I)
    return false;

  if (Q.CxtI && match(V, m_VScale())) {
    const Function *F = Q.CxtI->getFunction();
    // The vscale_range indicates vscale is a power-of-two.
    return F->hasFnAttribute(Attribute::VScaleRange);
  }

  // 1 << X is clearly a power of two if the one is not shifted off the end.  If
  // it is shifted off the end then the result is undefined.
  if (match(I, m_Shl(m_One(), m_Value())))
    return true;

  // (signmask) >>l X is clearly a power of two if the one is not shifted off
  // the bottom.  If it is shifted off the bottom then the result is undefined.
  if (match(I, m_LShr(m_SignMask(), m_Value())))
    return true;

  // The remaining tests are all recursive, so bail out if we hit the limit.
  if (Depth++ == MaxAnalysisRecursionDepth)
    return false;

  switch (I->getOpcode()) {
  case Instruction::ZExt:
    return isKnownToBeAPowerOfTwo(I->getOperand(0), OrZero, Q, Depth);
  case Instruction::Trunc:
    return OrZero && isKnownToBeAPowerOfTwo(I->getOperand(0), OrZero, Q, Depth);
  case Instruction::Shl:
    if (OrZero || Q.IIQ.hasNoUnsignedWrap(I) || Q.IIQ.hasNoSignedWrap(I))
      return isKnownToBeAPowerOfTwo(I->getOperand(0), OrZero, Q, Depth);
    return false;
  case Instruction::LShr:
    if (OrZero || Q.IIQ.isExact(cast<BinaryOperator>(I)))
      return isKnownToBeAPowerOfTwo(I->getOperand(0), OrZero, Q, Depth);
    return false;
  case Instruction::UDiv:
    if (Q.IIQ.isExact(cast<BinaryOperator>(I)))
      return isKnownToBeAPowerOfTwo(I->getOperand(0), OrZero, Q, Depth);
    return false;
  case Instruction::Mul:
    return isKnownToBeAPowerOfTwo(I->getOperand(1), OrZero, Q, Depth) &&
           isKnownToBeAPowerOfTwo(I->getOperand(0), OrZero, Q, Depth) &&
           (OrZero || isKnownNonZero(I, Q, Depth));
  case Instruction::And:
    // A power of two and'd with anything is a power of two or zero.
    if (OrZero &&
        (isKnownToBeAPowerOfTwo(I->getOperand(1), /*OrZero*/ true, Q, Depth) ||
         isKnownToBeAPowerOfTwo(I->getOperand(0), /*OrZero*/ true, Q, Depth)))
      return true;
    // X & (-X) is always a power of two or zero.
    if (match(I->getOperand(0), m_Neg(m_Specific(I->getOperand(1)))) ||
        match(I->getOperand(1), m_Neg(m_Specific(I->getOperand(0)))))
      return OrZero || isKnownNonZero(I->getOperand(0), Q, Depth);
    return false;
  case Instruction::Add: {
    // Adding a power-of-two or zero to the same power-of-two or zero yields
    // either the original power-of-two, a larger power-of-two or zero.
    const OverflowingBinaryOperator *VOBO = cast<OverflowingBinaryOperator>(V);
    if (OrZero || Q.IIQ.hasNoUnsignedWrap(VOBO) ||
        Q.IIQ.hasNoSignedWrap(VOBO)) {
      if (match(I->getOperand(0),
                m_c_And(m_Specific(I->getOperand(1)), m_Value())) &&
          isKnownToBeAPowerOfTwo(I->getOperand(1), OrZero, Q, Depth))
        return true;
      if (match(I->getOperand(1),
                m_c_And(m_Specific(I->getOperand(0)), m_Value())) &&
          isKnownToBeAPowerOfTwo(I->getOperand(0), OrZero, Q, Depth))
        return true;

      unsigned BitWidth = V->getType()->getScalarSizeInBits();
      KnownBits LHSBits(BitWidth);
      computeKnownBits(I->getOperand(0), LHSBits, Q, Depth);

      KnownBits RHSBits(BitWidth);
      computeKnownBits(I->getOperand(1), RHSBits, Q, Depth);
      // If i8 V is a power of two or zero:
      //  ZeroBits: 1 1 1 0 1 1 1 1
      // ~ZeroBits: 0 0 0 1 0 0 0 0
      if ((~(LHSBits.Zero & RHSBits.Zero)).isPowerOf2())
        // If OrZero isn't set, we cannot give back a zero result.
        // Make sure either the LHS or RHS has a bit set.
        if (OrZero || RHSBits.One.getBoolValue() || LHSBits.One.getBoolValue())
          return true;
    }

    // LShr(UINT_MAX, Y) + 1 is a power of two (if add is nuw) or zero.
    if (OrZero || Q.IIQ.hasNoUnsignedWrap(VOBO))
      if (match(I, m_Add(m_LShr(m_AllOnes(), m_Value()), m_One())))
        return true;
    return false;
  }
  case Instruction::Select:
    return isKnownToBeAPowerOfTwo(I->getOperand(1), OrZero, Q, Depth) &&
           isKnownToBeAPowerOfTwo(I->getOperand(2), OrZero, Q, Depth);
  case Instruction::PHI: {
    // A PHI node is power of two if all incoming values are power of two, or if
    // it is an induction variable where in each step its value is a power of
    // two.
    auto *PN = cast<PHINode>(I);
    SimplifyQuery RecQ = Q.getWithoutCondContext();

    // Check if it is an induction variable and always power of two.
    if (isPowerOfTwoRecurrence(PN, OrZero, RecQ, Depth))
      return true;

    // Recursively check all incoming values. Limit recursion to 2 levels, so
    // that search complexity is limited to number of operands^2.
    unsigned NewDepth = std::max(Depth, MaxAnalysisRecursionDepth - 1);
    return llvm::all_of(PN->operands(), [&](const Use &U) {
      // Value is power of 2 if it is coming from PHI node itself by induction.
      if (U.get() == PN)
        return true;

      // Change the context instruction to the incoming block where it is
      // evaluated.
      RecQ.CxtI = PN->getIncomingBlock(U)->getTerminator();
      return isKnownToBeAPowerOfTwo(U.get(), OrZero, RecQ, NewDepth);
    });
  }
  case Instruction::Invoke:
  case Instruction::Call: {
    if (auto *II = dyn_cast<IntrinsicInst>(I)) {
      switch (II->getIntrinsicID()) {
      case Intrinsic::umax:
      case Intrinsic::smax:
      case Intrinsic::umin:
      case Intrinsic::smin:
        return isKnownToBeAPowerOfTwo(II->getArgOperand(1), OrZero, Q, Depth) &&
               isKnownToBeAPowerOfTwo(II->getArgOperand(0), OrZero, Q, Depth);
      // bswap/bitreverse just move around bits, but don't change any 1s/0s
      // thus dont change pow2/non-pow2 status.
      case Intrinsic::bitreverse:
      case Intrinsic::bswap:
        return isKnownToBeAPowerOfTwo(II->getArgOperand(0), OrZero, Q, Depth);
      case Intrinsic::fshr:
      case Intrinsic::fshl:
        // If Op0 == Op1, this is a rotate. is_pow2(rotate(x, y)) == is_pow2(x)
        if (II->getArgOperand(0) == II->getArgOperand(1))
          return isKnownToBeAPowerOfTwo(II->getArgOperand(0), OrZero, Q, Depth);
        break;
      default:
        break;
      }
    }
    return false;
  }
  default:
    return false;
  }
}

/// Test whether a GEP's result is known to be non-null.
///
/// Uses properties inherent in a GEP to try to determine whether it is known
/// to be non-null.
///
/// Currently this routine does not support vector GEPs.
static bool isGEPKnownNonNull(const GEPOperator *GEP, const SimplifyQuery &Q,
                              unsigned Depth) {
  const Function *F = nullptr;
  if (const Instruction *I = dyn_cast<Instruction>(GEP))
    F = I->getFunction();

  // If the gep is nuw or inbounds with invalid null pointer, then the GEP
  // may be null iff the base pointer is null and the offset is zero.
  if (!GEP->hasNoUnsignedWrap() &&
      !(GEP->isInBounds() &&
        !NullPointerIsDefined(F, GEP->getPointerAddressSpace())))
    return false;

  // FIXME: Support vector-GEPs.
  assert(GEP->getType()->isPointerTy() && "We only support plain pointer GEP");

  // If the base pointer is non-null, we cannot walk to a null address with an
  // inbounds GEP in address space zero.
  if (isKnownNonZero(GEP->getPointerOperand(), Q, Depth))
    return true;

  // Walk the GEP operands and see if any operand introduces a non-zero offset.
  // If so, then the GEP cannot produce a null pointer, as doing so would
  // inherently violate the inbounds contract within address space zero.
  for (gep_type_iterator GTI = gep_type_begin(GEP), GTE = gep_type_end(GEP);
       GTI != GTE; ++GTI) {
    // Struct types are easy -- they must always be indexed by a constant.
    if (StructType *STy = GTI.getStructTypeOrNull()) {
      ConstantInt *OpC = cast<ConstantInt>(GTI.getOperand());
      unsigned ElementIdx = OpC->getZExtValue();
      const StructLayout *SL = Q.DL.getStructLayout(STy);
      uint64_t ElementOffset = SL->getElementOffset(ElementIdx);
      if (ElementOffset > 0)
        return true;
      continue;
    }

    // If we have a zero-sized type, the index doesn't matter. Keep looping.
    if (GTI.getSequentialElementStride(Q.DL).isZero())
      continue;

    // Fast path the constant operand case both for efficiency and so we don't
    // increment Depth when just zipping down an all-constant GEP.
    if (ConstantInt *OpC = dyn_cast<ConstantInt>(GTI.getOperand())) {
      if (!OpC->isZero())
        return true;
      continue;
    }

    // We post-increment Depth here because while isKnownNonZero increments it
    // as well, when we pop back up that increment won't persist. We don't want
    // to recurse 10k times just because we have 10k GEP operands. We don't
    // bail completely out because we want to handle constant GEPs regardless
    // of depth.
    if (Depth++ >= MaxAnalysisRecursionDepth)
      continue;

    if (isKnownNonZero(GTI.getOperand(), Q, Depth))
      return true;
  }

  return false;
}

/// Does the 'Range' metadata (which must be a valid MD_range operand list)
/// ensure that the value it's attached to is never Value?  'RangeType' is
/// is the type of the value described by the range.
static bool rangeMetadataExcludesValue(const MDNode* Ranges, const APInt& Value) {
  const unsigned NumRanges = Ranges->getNumOperands() / 2;
  assert(NumRanges >= 1);
  for (unsigned i = 0; i < NumRanges; ++i) {
    ConstantInt *Lower =
        mdconst::extract<ConstantInt>(Ranges->getOperand(2 * i + 0));
    ConstantInt *Upper =
        mdconst::extract<ConstantInt>(Ranges->getOperand(2 * i + 1));
    ConstantRange Range(Lower->getValue(), Upper->getValue());
    if (Range.contains(Value))
      return false;
  }
  return true;
}

static bool isNonZeroAdd(const APInt &DemandedElts, const SimplifyQuery &Q,
                         unsigned BitWidth, Value *X, Value *Y, bool NSW,
                         bool NUW, unsigned Depth) {
  // (X + (X != 0)) is non zero
  if (matchOpWithOpEqZero(X, Y))
    return true;

  if (NUW)
    return isKnownNonZero(Y, DemandedElts, Q, Depth) ||
           isKnownNonZero(X, DemandedElts, Q, Depth);

  KnownBits XKnown = computeKnownBits(X, DemandedElts, Q, Depth);
  KnownBits YKnown = computeKnownBits(Y, DemandedElts, Q, Depth);

  // If X and Y are both non-negative (as signed values) then their sum is not
  // zero unless both X and Y are zero.
  if (XKnown.isNonNegative() && YKnown.isNonNegative())
    if (isKnownNonZero(Y, DemandedElts, Q, Depth) ||
        isKnownNonZero(X, DemandedElts, Q, Depth))
      return true;

  // If X and Y are both negative (as signed values) then their sum is not
  // zero unless both X and Y equal INT_MIN.
  if (XKnown.isNegative() && YKnown.isNegative()) {
    APInt Mask = APInt::getSignedMaxValue(BitWidth);
    // The sign bit of X is set.  If some other bit is set then X is not equal
    // to INT_MIN.
    if (XKnown.One.intersects(Mask))
      return true;
    // The sign bit of Y is set.  If some other bit is set then Y is not equal
    // to INT_MIN.
    if (YKnown.One.intersects(Mask))
      return true;
  }

  // The sum of a non-negative number and a power of two is not zero.
  if (XKnown.isNonNegative() &&
      isKnownToBeAPowerOfTwo(Y, /*OrZero*/ false, Q, Depth))
    return true;
  if (YKnown.isNonNegative() &&
      isKnownToBeAPowerOfTwo(X, /*OrZero*/ false, Q, Depth))
    return true;

  return KnownBits::add(XKnown, YKnown, NSW, NUW).isNonZero();
}

static bool isNonZeroSub(const APInt &DemandedElts, const SimplifyQuery &Q,
                         unsigned BitWidth, Value *X, Value *Y,
                         unsigned Depth) {
  // (X - (X != 0)) is non zero
  // ((X != 0) - X) is non zero
  if (matchOpWithOpEqZero(X, Y))
    return true;

  // TODO: Move this case into isKnownNonEqual().
  if (auto *C = dyn_cast<Constant>(X))
    if (C->isNullValue() && isKnownNonZero(Y, DemandedElts, Q, Depth))
      return true;

  return ::isKnownNonEqual(X, Y, DemandedElts, Q, Depth);
}

static bool isNonZeroMul(const APInt &DemandedElts, const SimplifyQuery &Q,
                         unsigned BitWidth, Value *X, Value *Y, bool NSW,
                         bool NUW, unsigned Depth) {
  // If X and Y are non-zero then so is X * Y as long as the multiplication
  // does not overflow.
  if (NSW || NUW)
    return isKnownNonZero(X, DemandedElts, Q, Depth) &&
           isKnownNonZero(Y, DemandedElts, Q, Depth);

  // If either X or Y is odd, then if the other is non-zero the result can't
  // be zero.
  KnownBits XKnown = computeKnownBits(X, DemandedElts, Q, Depth);
  if (XKnown.One[0])
    return isKnownNonZero(Y, DemandedElts, Q, Depth);

  KnownBits YKnown = computeKnownBits(Y, DemandedElts, Q, Depth);
  if (YKnown.One[0])
    return XKnown.isNonZero() || isKnownNonZero(X, DemandedElts, Q, Depth);

  // If there exists any subset of X (sX) and subset of Y (sY) s.t sX * sY is
  // non-zero, then X * Y is non-zero. We can find sX and sY by just taking
  // the lowest known One of X and Y. If they are non-zero, the result
  // must be non-zero. We can check if LSB(X) * LSB(Y) != 0 by doing
  // X.CountLeadingZeros + Y.CountLeadingZeros < BitWidth.
  return (XKnown.countMaxTrailingZeros() + YKnown.countMaxTrailingZeros()) <
         BitWidth;
}

static bool isNonZeroShift(const Operator *I, const APInt &DemandedElts,
                           const SimplifyQuery &Q, const KnownBits &KnownVal,
                           unsigned Depth) {
  auto ShiftOp = [&](const APInt &Lhs, const APInt &Rhs) {
    switch (I->getOpcode()) {
    case Instruction::Shl:
      return Lhs.shl(Rhs);
    case Instruction::LShr:
      return Lhs.lshr(Rhs);
    case Instruction::AShr:
      return Lhs.ashr(Rhs);
    default:
      llvm_unreachable("Unknown Shift Opcode");
    }
  };

  auto InvShiftOp = [&](const APInt &Lhs, const APInt &Rhs) {
    switch (I->getOpcode()) {
    case Instruction::Shl:
      return Lhs.lshr(Rhs);
    case Instruction::LShr:
    case Instruction::AShr:
      return Lhs.shl(Rhs);
    default:
      llvm_unreachable("Unknown Shift Opcode");
    }
  };

  if (KnownVal.isUnknown())
    return false;

  KnownBits KnownCnt =
      computeKnownBits(I->getOperand(1), DemandedElts, Q, Depth);
  APInt MaxShift = KnownCnt.getMaxValue();
  unsigned NumBits = KnownVal.getBitWidth();
  if (MaxShift.uge(NumBits))
    return false;

  if (!ShiftOp(KnownVal.One, MaxShift).isZero())
    return true;

  // If all of the bits shifted out are known to be zero, and Val is known
  // non-zero then at least one non-zero bit must remain.
  if (InvShiftOp(KnownVal.Zero, NumBits - MaxShift)
          .eq(InvShiftOp(APInt::getAllOnes(NumBits), NumBits - MaxShift)) &&
      isKnownNonZero(I->getOperand(0), DemandedElts, Q, Depth))
    return true;

  return false;
}

static bool isKnownNonZeroFromOperator(const Operator *I,
                                       const APInt &DemandedElts,
                                       const SimplifyQuery &Q, unsigned Depth) {
  unsigned BitWidth = getBitWidth(I->getType()->getScalarType(), Q.DL);
  switch (I->getOpcode()) {
  case Instruction::Alloca:
    // Alloca never returns null, malloc might.
    return I->getType()->getPointerAddressSpace() == 0;
  case Instruction::GetElementPtr:
    if (I->getType()->isPointerTy())
      return isGEPKnownNonNull(cast<GEPOperator>(I), Q, Depth);
    break;
  case Instruction::BitCast: {
    // We need to be a bit careful here. We can only peek through the bitcast
    // if the scalar size of elements in the operand are smaller than and a
    // multiple of the size they are casting too. Take three cases:
    //
    // 1) Unsafe:
    //        bitcast <2 x i16> %NonZero to <4 x i8>
    //
    //    %NonZero can have 2 non-zero i16 elements, but isKnownNonZero on a
    //    <4 x i8> requires that all 4 i8 elements be non-zero which isn't
    //    guranteed (imagine just sign bit set in the 2 i16 elements).
    //
    // 2) Unsafe:
    //        bitcast <4 x i3> %NonZero to <3 x i4>
    //
    //    Even though the scalar size of the src (`i3`) is smaller than the
    //    scalar size of the dst `i4`, because `i3` is not a multiple of `i4`
    //    its possible for the `3 x i4` elements to be zero because there are
    //    some elements in the destination that don't contain any full src
    //    element.
    //
    // 3) Safe:
    //        bitcast <4 x i8> %NonZero to <2 x i16>
    //
    //    This is always safe as non-zero in the 4 i8 elements implies
    //    non-zero in the combination of any two adjacent ones. Since i8 is a
    //    multiple of i16, each i16 is guranteed to have 2 full i8 elements.
    //    This all implies the 2 i16 elements are non-zero.
    Type *FromTy = I->getOperand(0)->getType();
    if ((FromTy->isIntOrIntVectorTy() || FromTy->isPtrOrPtrVectorTy()) &&
        (BitWidth % getBitWidth(FromTy->getScalarType(), Q.DL)) == 0)
      return isKnownNonZero(I->getOperand(0), Q, Depth);
  } break;
  case Instruction::IntToPtr:
    // Note that we have to take special care to avoid looking through
    // truncating casts, e.g., int2ptr/ptr2int with appropriate sizes, as well
    // as casts that can alter the value, e.g., AddrSpaceCasts.
    if (!isa<ScalableVectorType>(I->getType()) &&
        Q.DL.getTypeSizeInBits(I->getOperand(0)->getType()).getFixedValue() <=
            Q.DL.getTypeSizeInBits(I->getType()).getFixedValue())
      return isKnownNonZero(I->getOperand(0), DemandedElts, Q, Depth);
    break;
  case Instruction::PtrToAddr:
    // isKnownNonZero() for pointers refers to the address bits being non-zero,
    // so we can directly forward.
    return isKnownNonZero(I->getOperand(0), DemandedElts, Q, Depth);
  case Instruction::PtrToInt:
    // For inttoptr, make sure the result size is >= the address size. If the
    // address is non-zero, any larger value is also non-zero.
    if (Q.DL.getAddressSizeInBits(I->getOperand(0)->getType()) <=
        I->getType()->getScalarSizeInBits())
      return isKnownNonZero(I->getOperand(0), DemandedElts, Q, Depth);
    break;
  case Instruction::Trunc:
    // nuw/nsw trunc preserves zero/non-zero status of input.
    if (auto *TI = dyn_cast<TruncInst>(I))
      if (TI->hasNoSignedWrap() || TI->hasNoUnsignedWrap())
        return isKnownNonZero(TI->getOperand(0), DemandedElts, Q, Depth);
    break;

  // Iff x - y != 0, then x ^ y != 0
  // Therefore we can do the same exact checks
  case Instruction::Xor:
  case Instruction::Sub:
    return isNonZeroSub(DemandedElts, Q, BitWidth, I->getOperand(0),
                        I->getOperand(1), Depth);
  case Instruction::Or:
    // (X | (X != 0)) is non zero
    if (matchOpWithOpEqZero(I->getOperand(0), I->getOperand(1)))
      return true;
    // X | Y != 0 if X != Y.
    if (isKnownNonEqual(I->getOperand(0), I->getOperand(1), DemandedElts, Q,
                        Depth))
      return true;
    // X | Y != 0 if X != 0 or Y != 0.
    return isKnownNonZero(I->getOperand(1), DemandedElts, Q, Depth) ||
           isKnownNonZero(I->getOperand(0), DemandedElts, Q, Depth);
  case Instruction::SExt:
  case Instruction::ZExt:
    // ext X != 0 if X != 0.
    return isKnownNonZero(I->getOperand(0), DemandedElts, Q, Depth);

  case Instruction::Shl: {
    // shl nsw/nuw can't remove any non-zero bits.
    const OverflowingBinaryOperator *BO = cast<OverflowingBinaryOperator>(I);
    if (Q.IIQ.hasNoUnsignedWrap(BO) || Q.IIQ.hasNoSignedWrap(BO))
      return isKnownNonZero(I->getOperand(0), DemandedElts, Q, Depth);

    // shl X, Y != 0 if X is odd.  Note that the value of the shift is undefined
    // if the lowest bit is shifted off the end.
    KnownBits Known(BitWidth);
    computeKnownBits(I->getOperand(0), DemandedElts, Known, Q, Depth);
    if (Known.One[0])
      return true;

    return isNonZeroShift(I, DemandedElts, Q, Known, Depth);
  }
  case Instruction::LShr:
  case Instruction::AShr: {
    // shr exact can only shift out zero bits.
    const PossiblyExactOperator *BO = cast<PossiblyExactOperator>(I);
    if (BO->isExact())
      return isKnownNonZero(I->getOperand(0), DemandedElts, Q, Depth);

    // shr X, Y != 0 if X is negative.  Note that the value of the shift is not
    // defined if the sign bit is shifted off the end.
    KnownBits Known =
        computeKnownBits(I->getOperand(0), DemandedElts, Q, Depth);
    if (Known.isNegative())
      return true;

    return isNonZeroShift(I, DemandedElts, Q, Known, Depth);
  }
  case Instruction::UDiv:
  case Instruction::SDiv: {
    // X / Y
    // div exact can only produce a zero if the dividend is zero.
    if (cast<PossiblyExactOperator>(I)->isExact())
      return isKnownNonZero(I->getOperand(0), DemandedElts, Q, Depth);

    KnownBits XKnown =
        computeKnownBits(I->getOperand(0), DemandedElts, Q, Depth);
    // If X is fully unknown we won't be able to figure anything out so don't
    // both computing knownbits for Y.
    if (XKnown.isUnknown())
      return false;

    KnownBits YKnown =
        computeKnownBits(I->getOperand(1), DemandedElts, Q, Depth);
    if (I->getOpcode() == Instruction::SDiv) {
      // For signed division need to compare abs value of the operands.
      XKnown = XKnown.abs(/*IntMinIsPoison*/ false);
      YKnown = YKnown.abs(/*IntMinIsPoison*/ false);
    }
    // If X u>= Y then div is non zero (0/0 is UB).
    std::optional<bool> XUgeY = KnownBits::uge(XKnown, YKnown);
    // If X is total unknown or X u< Y we won't be able to prove non-zero
    // with compute known bits so just return early.
    return XUgeY && *XUgeY;
  }
  case Instruction::Add: {
    // X + Y.

    // If Add has nuw wrap flag, then if either X or Y is non-zero the result is
    // non-zero.
    auto *BO = cast<OverflowingBinaryOperator>(I);
    return isNonZeroAdd(DemandedElts, Q, BitWidth, I->getOperand(0),
                        I->getOperand(1), Q.IIQ.hasNoSignedWrap(BO),
                        Q.IIQ.hasNoUnsignedWrap(BO), Depth);
  }
  case Instruction::Mul: {
    const OverflowingBinaryOperator *BO = cast<OverflowingBinaryOperator>(I);
    return isNonZeroMul(DemandedElts, Q, BitWidth, I->getOperand(0),
                        I->getOperand(1), Q.IIQ.hasNoSignedWrap(BO),
                        Q.IIQ.hasNoUnsignedWrap(BO), Depth);
  }
  case Instruction::Select: {
    // (C ? X : Y) != 0 if X != 0 and Y != 0.

    // First check if the arm is non-zero using `isKnownNonZero`. If that fails,
    // then see if the select condition implies the arm is non-zero. For example
    // (X != 0 ? X : Y), we know the true arm is non-zero as the `X` "return" is
    // dominated by `X != 0`.
    auto SelectArmIsNonZero = [&](bool IsTrueArm) {
      Value *Op;
      Op = IsTrueArm ? I->getOperand(1) : I->getOperand(2);
      // Op is trivially non-zero.
      if (isKnownNonZero(Op, DemandedElts, Q, Depth))
        return true;

      // The condition of the select dominates the true/false arm. Check if the
      // condition implies that a given arm is non-zero.
      Value *X;
      CmpPredicate Pred;
      if (!match(I->getOperand(0), m_c_ICmp(Pred, m_Specific(Op), m_Value(X))))
        return false;

      if (!IsTrueArm)
        Pred = ICmpInst::getInversePredicate(Pred);

      return cmpExcludesZero(Pred, X);
    };

    if (SelectArmIsNonZero(/* IsTrueArm */ true) &&
        SelectArmIsNonZero(/* IsTrueArm */ false))
      return true;
    break;
  }
  case Instruction::PHI: {
    auto *PN = cast<PHINode>(I);
    if (Q.IIQ.UseInstrInfo && isNonZeroRecurrence(PN))
      return true;

    // Check if all incoming values are non-zero using recursion.
    SimplifyQuery RecQ = Q.getWithoutCondContext();
    unsigned NewDepth = std::max(Depth, MaxAnalysisRecursionDepth - 1);
    return llvm::all_of(PN->operands(), [&](const Use &U) {
      if (U.get() == PN)
        return true;
      RecQ.CxtI = PN->getIncomingBlock(U)->getTerminator();
      // Check if the branch on the phi excludes zero.
      CmpPredicate Pred;
      Value *X;
      BasicBlock *TrueSucc, *FalseSucc;
      if (match(RecQ.CxtI,
                m_Br(m_c_ICmp(Pred, m_Specific(U.get()), m_Value(X)),
                     m_BasicBlock(TrueSucc), m_BasicBlock(FalseSucc)))) {
        // Check for cases of duplicate successors.
        if ((TrueSucc == PN->getParent()) != (FalseSucc == PN->getParent())) {
          // If we're using the false successor, invert the predicate.
          if (FalseSucc == PN->getParent())
            Pred = CmpInst::getInversePredicate(Pred);
          if (cmpExcludesZero(Pred, X))
            return true;
        }
      }
      // Finally recurse on the edge and check it directly.
      return isKnownNonZero(U.get(), DemandedElts, RecQ, NewDepth);
    });
  }
  case Instruction::InsertElement: {
    if (isa<ScalableVectorType>(I->getType()))
      break;

    const Value *Vec = I->getOperand(0);
    const Value *Elt = I->getOperand(1);
    auto *CIdx = dyn_cast<ConstantInt>(I->getOperand(2));

    unsigned NumElts = DemandedElts.getBitWidth();
    APInt DemandedVecElts = DemandedElts;
    bool SkipElt = false;
    // If we know the index we are inserting too, clear it from Vec check.
    if (CIdx && CIdx->getValue().ult(NumElts)) {
      DemandedVecElts.clearBit(CIdx->getZExtValue());
      SkipElt = !DemandedElts[CIdx->getZExtValue()];
    }

    // Result is zero if Elt is non-zero and rest of the demanded elts in Vec
    // are non-zero.
    return (SkipElt || isKnownNonZero(Elt, Q, Depth)) &&
           (DemandedVecElts.isZero() ||
            isKnownNonZero(Vec, DemandedVecElts, Q, Depth));
  }
  case Instruction::ExtractElement:
    if (const auto *EEI = dyn_cast<ExtractElementInst>(I)) {
      const Value *Vec = EEI->getVectorOperand();
      const Value *Idx = EEI->getIndexOperand();
      auto *CIdx = dyn_cast<ConstantInt>(Idx);
      if (auto *VecTy = dyn_cast<FixedVectorType>(Vec->getType())) {
        unsigned NumElts = VecTy->getNumElements();
        APInt DemandedVecElts = APInt::getAllOnes(NumElts);
        if (CIdx && CIdx->getValue().ult(NumElts))
          DemandedVecElts = APInt::getOneBitSet(NumElts, CIdx->getZExtValue());
        return isKnownNonZero(Vec, DemandedVecElts, Q, Depth);
      }
    }
    break;
  case Instruction::ShuffleVector: {
    auto *Shuf = dyn_cast<ShuffleVectorInst>(I);
    if (!Shuf)
      break;
    APInt DemandedLHS, DemandedRHS;
    // For undef elements, we don't know anything about the common state of
    // the shuffle result.
    if (!getShuffleDemandedElts(Shuf, DemandedElts, DemandedLHS, DemandedRHS))
      break;
    // If demanded elements for both vecs are non-zero, the shuffle is non-zero.
    return (DemandedRHS.isZero() ||
            isKnownNonZero(Shuf->getOperand(1), DemandedRHS, Q, Depth)) &&
           (DemandedLHS.isZero() ||
            isKnownNonZero(Shuf->getOperand(0), DemandedLHS, Q, Depth));
  }
  case Instruction::Freeze:
    return isKnownNonZero(I->getOperand(0), Q, Depth) &&
           isGuaranteedNotToBePoison(I->getOperand(0), Q.AC, Q.CxtI, Q.DT,
                                     Depth);
  case Instruction::Load: {
    auto *LI = cast<LoadInst>(I);
    // A Load tagged with nonnull or dereferenceable with null pointer undefined
    // is never null.
    if (auto *PtrT = dyn_cast<PointerType>(I->getType())) {
      if (Q.IIQ.getMetadata(LI, LLVMContext::MD_nonnull) ||
          (Q.IIQ.getMetadata(LI, LLVMContext::MD_dereferenceable) &&
           !NullPointerIsDefined(LI->getFunction(), PtrT->getAddressSpace())))
        return true;
    } else if (MDNode *Ranges = Q.IIQ.getMetadata(LI, LLVMContext::MD_range)) {
      return rangeMetadataExcludesValue(Ranges, APInt::getZero(BitWidth));
    }

    // No need to fall through to computeKnownBits as range metadata is already
    // handled in isKnownNonZero.
    return false;
  }
  case Instruction::ExtractValue: {
    const WithOverflowInst *WO;
    if (match(I, m_ExtractValue<0>(m_WithOverflowInst(WO)))) {
      switch (WO->getBinaryOp()) {
      default:
        break;
      case Instruction::Add:
        return isNonZeroAdd(DemandedElts, Q, BitWidth, WO->getArgOperand(0),
                            WO->getArgOperand(1),
                            /*NSW=*/false,
                            /*NUW=*/false, Depth);
      case Instruction::Sub:
        return isNonZeroSub(DemandedElts, Q, BitWidth, WO->getArgOperand(0),
                            WO->getArgOperand(1), Depth);
      case Instruction::Mul:
        return isNonZeroMul(DemandedElts, Q, BitWidth, WO->getArgOperand(0),
                            WO->getArgOperand(1),
                            /*NSW=*/false, /*NUW=*/false, Depth);
        break;
      }
    }
    break;
  }
  case Instruction::Call:
  case Instruction::Invoke: {
    const auto *Call = cast<CallBase>(I);
    if (I->getType()->isPointerTy()) {
      if (Call->isReturnNonNull())
        return true;
      if (const auto *RP = getArgumentAliasingToReturnedPointer(Call, true))
        return isKnownNonZero(RP, Q, Depth);
    } else {
      if (MDNode *Ranges = Q.IIQ.getMetadata(Call, LLVMContext::MD_range))
        return rangeMetadataExcludesValue(Ranges, APInt::getZero(BitWidth));
      if (std::optional<ConstantRange> Range = Call->getRange()) {
        const APInt ZeroValue(Range->getBitWidth(), 0);
        if (!Range->contains(ZeroValue))
          return true;
      }
      if (const Value *RV = Call->getReturnedArgOperand())
        if (RV->getType() == I->getType() && isKnownNonZero(RV, Q, Depth))
          return true;
    }

    if (auto *II = dyn_cast<IntrinsicInst>(I)) {
      switch (II->getIntrinsicID()) {
      case Intrinsic::sshl_sat:
      case Intrinsic::ushl_sat:
      case Intrinsic::abs:
      case Intrinsic::bitreverse:
      case Intrinsic::bswap:
      case Intrinsic::ctpop:
        return isKnownNonZero(II->getArgOperand(0), DemandedElts, Q, Depth);
        // NB: We don't do usub_sat here as in any case we can prove its
        // non-zero, we will fold it to `sub nuw` in InstCombine.
      case Intrinsic::ssub_sat:
        // For most types, if x != y then ssub.sat x, y != 0.  But
        // ssub.sat.i1 0, -1 = 0, because 1 saturates to 0.  This means
        // isNonZeroSub will do the wrong thing for ssub.sat.i1.
        if (BitWidth == 1)
          return false;
        return isNonZeroSub(DemandedElts, Q, BitWidth, II->getArgOperand(0),
                            II->getArgOperand(1), Depth);
      case Intrinsic::sadd_sat:
        return isNonZeroAdd(DemandedElts, Q, BitWidth, II->getArgOperand(0),
                            II->getArgOperand(1),
                            /*NSW=*/true, /* NUW=*/false, Depth);
        // Vec reverse preserves zero/non-zero status from input vec.
      case Intrinsic::vector_reverse:
        return isKnownNonZero(II->getArgOperand(0), DemandedElts.reverseBits(),
                              Q, Depth);
        // umin/smin/smax/smin/or of all non-zero elements is always non-zero.
      case Intrinsic::vector_reduce_or:
      case Intrinsic::vector_reduce_umax:
      case Intrinsic::vector_reduce_umin:
      case Intrinsic::vector_reduce_smax:
      case Intrinsic::vector_reduce_smin:
        return isKnownNonZero(II->getArgOperand(0), Q, Depth);
      case Intrinsic::umax:
      case Intrinsic::uadd_sat:
        // umax(X, (X != 0)) is non zero
        // X +usat (X != 0) is non zero
        if (matchOpWithOpEqZero(II->getArgOperand(0), II->getArgOperand(1)))
          return true;

        return isKnownNonZero(II->getArgOperand(1), DemandedElts, Q, Depth) ||
               isKnownNonZero(II->getArgOperand(0), DemandedElts, Q, Depth);
      case Intrinsic::smax: {
        // If either arg is strictly positive the result is non-zero. Otherwise
        // the result is non-zero if both ops are non-zero.
        auto IsNonZero = [&](Value *Op, std::optional<bool> &OpNonZero,
                             const KnownBits &OpKnown) {
          if (!OpNonZero.has_value())
            OpNonZero = OpKnown.isNonZero() ||
                        isKnownNonZero(Op, DemandedElts, Q, Depth);
          return *OpNonZero;
        };
        // Avoid re-computing isKnownNonZero.
        std::optional<bool> Op0NonZero, Op1NonZero;
        KnownBits Op1Known =
            computeKnownBits(II->getArgOperand(1), DemandedElts, Q, Depth);
        if (Op1Known.isNonNegative() &&
            IsNonZero(II->getArgOperand(1), Op1NonZero, Op1Known))
          return true;
        KnownBits Op0Known =
            computeKnownBits(II->getArgOperand(0), DemandedElts, Q, Depth);
        if (Op0Known.isNonNegative() &&
            IsNonZero(II->getArgOperand(0), Op0NonZero, Op0Known))
          return true;
        return IsNonZero(II->getArgOperand(1), Op1NonZero, Op1Known) &&
               IsNonZero(II->getArgOperand(0), Op0NonZero, Op0Known);
      }
      case Intrinsic::smin: {
        // If either arg is negative the result is non-zero. Otherwise
        // the result is non-zero if both ops are non-zero.
        KnownBits Op1Known =
            computeKnownBits(II->getArgOperand(1), DemandedElts, Q, Depth);
        if (Op1Known.isNegative())
          return true;
        KnownBits Op0Known =
            computeKnownBits(II->getArgOperand(0), DemandedElts, Q, Depth);
        if (Op0Known.isNegative())
          return true;

        if (Op1Known.isNonZero() && Op0Known.isNonZero())
          return true;
      }
        [[fallthrough]];
      case Intrinsic::umin:
        return isKnownNonZero(II->getArgOperand(0), DemandedElts, Q, Depth) &&
               isKnownNonZero(II->getArgOperand(1), DemandedElts, Q, Depth);
      case Intrinsic::cttz:
        return computeKnownBits(II->getArgOperand(0), DemandedElts, Q, Depth)
            .Zero[0];
      case Intrinsic::ctlz:
        return computeKnownBits(II->getArgOperand(0), DemandedElts, Q, Depth)
            .isNonNegative();
      case Intrinsic::fshr:
      case Intrinsic::fshl:
        // If Op0 == Op1, this is a rotate. rotate(x, y) != 0 iff x != 0.
        if (II->getArgOperand(0) == II->getArgOperand(1))
          return isKnownNonZero(II->getArgOperand(0), DemandedElts, Q, Depth);
        break;
      case Intrinsic::vscale:
        return true;
      case Intrinsic::experimental_get_vector_length:
        return isKnownNonZero(I->getOperand(0), Q, Depth);
      default:
        break;
      }
      break;
    }

    return false;
  }
  }

  KnownBits Known(BitWidth);
  computeKnownBits(I, DemandedElts, Known, Q, Depth);
  return Known.One != 0;
}

/// Return true if the given value is known to be non-zero when defined. For
/// vectors, return true if every demanded element is known to be non-zero when
/// defined. For pointers, if the context instruction and dominator tree are
/// specified, perform context-sensitive analysis and return true if the
/// pointer couldn't possibly be null at the specified instruction.
/// Supports values with integer or pointer type and vectors of integers.
bool isKnownNonZero(const Value *V, const APInt &DemandedElts,
                    const SimplifyQuery &Q, unsigned Depth) {
  Type *Ty = V->getType();

#ifndef NDEBUG
  assert(Depth <= MaxAnalysisRecursionDepth && "Limit Search Depth");

  if (auto *FVTy = dyn_cast<FixedVectorType>(Ty)) {
    assert(
        FVTy->getNumElements() == DemandedElts.getBitWidth() &&
        "DemandedElt width should equal the fixed vector number of elements");
  } else {
    assert(DemandedElts == APInt(1, 1) &&
           "DemandedElt width should be 1 for scalars");
  }
#endif

  if (auto *C = dyn_cast<Constant>(V)) {
    if (C->isNullValue())
      return false;
    if (isa<ConstantInt>(C))
      // Must be non-zero due to null test above.
      return true;

    // For constant vectors, check that all elements are poison or known
    // non-zero to determine that the whole vector is known non-zero.
    if (auto *VecTy = dyn_cast<FixedVectorType>(Ty)) {
      for (unsigned i = 0, e = VecTy->getNumElements(); i != e; ++i) {
        if (!DemandedElts[i])
          continue;
        Constant *Elt = C->getAggregateElement(i);
        if (!Elt || Elt->isNullValue())
          return false;
        if (!isa<PoisonValue>(Elt) && !isa<ConstantInt>(Elt))
          return false;
      }
      return true;
    }

    // Constant ptrauth can be null, iff the base pointer can be.
    if (auto *CPA = dyn_cast<ConstantPtrAuth>(V))
      return isKnownNonZero(CPA->getPointer(), DemandedElts, Q, Depth);

    // A global variable in address space 0 is non null unless extern weak
    // or an absolute symbol reference. Other address spaces may have null as a
    // valid address for a global, so we can't assume anything.
    if (const GlobalValue *GV = dyn_cast<GlobalValue>(V)) {
      if (!GV->isAbsoluteSymbolRef() && !GV->hasExternalWeakLinkage() &&
          GV->getType()->getAddressSpace() == 0)
        return true;
    }

    // For constant expressions, fall through to the Operator code below.
    if (!isa<ConstantExpr>(V))
      return false;
  }

  if (const auto *A = dyn_cast<Argument>(V))
    if (std::optional<ConstantRange> Range = A->getRange()) {
      const APInt ZeroValue(Range->getBitWidth(), 0);
      if (!Range->contains(ZeroValue))
        return true;
    }

  if (!isa<Constant>(V) && isKnownNonZeroFromAssume(V, Q))
    return true;

  // Some of the tests below are recursive, so bail out if we hit the limit.
  if (Depth++ >= MaxAnalysisRecursionDepth)
    return false;

  // Check for pointer simplifications.

  if (PointerType *PtrTy = dyn_cast<PointerType>(Ty)) {
    // A byval, inalloca may not be null in a non-default addres space. A
    // nonnull argument is assumed never 0.
    if (const Argument *A = dyn_cast<Argument>(V)) {
      if (((A->hasPassPointeeByValueCopyAttr() &&
            !NullPointerIsDefined(A->getParent(), PtrTy->getAddressSpace())) ||
           A->hasNonNullAttr()))
        return true;
    }
  }

  if (const auto *I = dyn_cast<Operator>(V))
    if (isKnownNonZeroFromOperator(I, DemandedElts, Q, Depth))
      return true;

  if (!isa<Constant>(V) &&
      isKnownNonNullFromDominatingCondition(V, Q.CxtI, Q.DT))
    return true;

  if (const Value *Stripped = stripNullTest(V))
    return isKnownNonZero(Stripped, DemandedElts, Q, Depth);

  return false;
}

bool llvm::isKnownNonZero(const Value *V, const SimplifyQuery &Q,
                          unsigned Depth) {
  auto *FVTy = dyn_cast<FixedVectorType>(V->getType());
  APInt DemandedElts =
      FVTy ? APInt::getAllOnes(FVTy->getNumElements()) : APInt(1, 1);
  return ::isKnownNonZero(V, DemandedElts, Q, Depth);
}

/// Return true if V1 == (binop V2, X), where X is known non-zero.
/// Only handle a small subset of binops where (binop V2, X) with non-zero X
/// implies V2 != V1.
static bool isModifyingBinopOfNonZero(const Value *V1, const Value *V2,
                                      const APInt &DemandedElts,
                                      const SimplifyQuery &Q, unsigned Depth) {
  const BinaryOperator *BO = dyn_cast<BinaryOperator>(V1);
  if (!BO)
    return false;
  switch (BO->getOpcode()) {
  default:
    break;
  case Instruction::Or:
    if (!cast<PossiblyDisjointInst>(V1)->isDisjoint())
      break;
    [[fallthrough]];
  case Instruction::Xor:
  case Instruction::Add:
    Value *Op = nullptr;
    if (V2 == BO->getOperand(0))
      Op = BO->getOperand(1);
    else if (V2 == BO->getOperand(1))
      Op = BO->getOperand(0);
    else
      return false;
    return isKnownNonZero(Op, DemandedElts, Q, Depth + 1);
  }
  return false;
}

/// Return true if V2 == V1 * C, where V1 is known non-zero, C is not 0/1 and
/// the multiplication is nuw or nsw.
static bool isNonEqualMul(const Value *V1, const Value *V2,
                          const APInt &DemandedElts, const SimplifyQuery &Q,
                          unsigned Depth) {
  if (auto *OBO = dyn_cast<OverflowingBinaryOperator>(V2)) {
    const APInt *C;
    return match(OBO, m_Mul(m_Specific(V1), m_APInt(C))) &&
           (OBO->hasNoUnsignedWrap() || OBO->hasNoSignedWrap()) &&
           !C->isZero() && !C->isOne() &&
           isKnownNonZero(V1, DemandedElts, Q, Depth + 1);
  }
  return false;
}

/// Return true if V2 == V1 << C, where V1 is known non-zero, C is not 0 and
/// the shift is nuw or nsw.
static bool isNonEqualShl(const Value *V1, const Value *V2,
                          const APInt &DemandedElts, const SimplifyQuery &Q,
                          unsigned Depth) {
  if (auto *OBO = dyn_cast<OverflowingBinaryOperator>(V2)) {
    const APInt *C;
    return match(OBO, m_Shl(m_Specific(V1), m_APInt(C))) &&
           (OBO->hasNoUnsignedWrap() || OBO->hasNoSignedWrap()) &&
           !C->isZero() && isKnownNonZero(V1, DemandedElts, Q, Depth + 1);
  }
  return false;
}

static bool isNonEqualPHIs(const PHINode *PN1, const PHINode *PN2,
                           const APInt &DemandedElts, const SimplifyQuery &Q,
                           unsigned Depth) {
  // Check two PHIs are in same block.
  if (PN1->getParent() != PN2->getParent())
    return false;

  SmallPtrSet<const BasicBlock *, 8> VisitedBBs;
  bool UsedFullRecursion = false;
  for (const BasicBlock *IncomBB : PN1->blocks()) {
    if (!VisitedBBs.insert(IncomBB).second)
      continue; // Don't reprocess blocks that we have dealt with already.
    const Value *IV1 = PN1->getIncomingValueForBlock(IncomBB);
    const Value *IV2 = PN2->getIncomingValueForBlock(IncomBB);
    const APInt *C1, *C2;
    if (match(IV1, m_APInt(C1)) && match(IV2, m_APInt(C2)) && *C1 != *C2)
      continue;

    // Only one pair of phi operands is allowed for full recursion.
    if (UsedFullRecursion)
      return false;

    SimplifyQuery RecQ = Q.getWithoutCondContext();
    RecQ.CxtI = IncomBB->getTerminator();
    if (!isKnownNonEqual(IV1, IV2, DemandedElts, RecQ, Depth + 1))
      return false;
    UsedFullRecursion = true;
  }
  return true;
}

static bool isNonEqualSelect(const Value *V1, const Value *V2,
                             const APInt &DemandedElts, const SimplifyQuery &Q,
                             unsigned Depth) {
  const SelectInst *SI1 = dyn_cast<SelectInst>(V1);
  if (!SI1)
    return false;

  if (const SelectInst *SI2 = dyn_cast<SelectInst>(V2)) {
    const Value *Cond1 = SI1->getCondition();
    const Value *Cond2 = SI2->getCondition();
    if (Cond1 == Cond2)
      return isKnownNonEqual(SI1->getTrueValue(), SI2->getTrueValue(),
                             DemandedElts, Q, Depth + 1) &&
             isKnownNonEqual(SI1->getFalseValue(), SI2->getFalseValue(),
                             DemandedElts, Q, Depth + 1);
  }
  return isKnownNonEqual(SI1->getTrueValue(), V2, DemandedElts, Q, Depth + 1) &&
         isKnownNonEqual(SI1->getFalseValue(), V2, DemandedElts, Q, Depth + 1);
}

static bool isKnownNonEqualFromContext(const Value *V1, const Value *V2,
                                       const SimplifyQuery &Q, unsigned Depth) {
  if (!Q.CxtI)
    return false;

  // Try to infer NonEqual based on information from dominating conditions.
  if (Q.DC && Q.DT) {
    auto IsKnownNonEqualFromDominatingCondition = [&](const Value *V) {
      for (BranchInst *BI : Q.DC->conditionsFor(V)) {
        Value *Cond = BI->getCondition();
        BasicBlockEdge Edge0(BI->getParent(), BI->getSuccessor(0));
        if (Q.DT->dominates(Edge0, Q.CxtI->getParent()) &&
            isImpliedCondition(Cond, ICmpInst::ICMP_NE, V1, V2, Q.DL,
                               /*LHSIsTrue=*/true, Depth)
                .value_or(false))
          return true;

        BasicBlockEdge Edge1(BI->getParent(), BI->getSuccessor(1));
        if (Q.DT->dominates(Edge1, Q.CxtI->getParent()) &&
            isImpliedCondition(Cond, ICmpInst::ICMP_NE, V1, V2, Q.DL,
                               /*LHSIsTrue=*/false, Depth)
                .value_or(false))
          return true;
      }

      return false;
    };

    if (IsKnownNonEqualFromDominatingCondition(V1) ||
        IsKnownNonEqualFromDominatingCondition(V2))
      return true;
  }

  if (!Q.AC)
    return false;

  // Try to infer NonEqual based on information from assumptions.
  for (auto &AssumeVH : Q.AC->assumptionsFor(V1)) {
    if (!AssumeVH)
      continue;
    CallInst *I = cast<CallInst>(AssumeVH);

    assert(I->getFunction() == Q.CxtI->getFunction() &&
           "Got assumption for the wrong function!");
    assert(I->getIntrinsicID() == Intrinsic::assume &&
           "must be an assume intrinsic");

    if (isImpliedCondition(I->getArgOperand(0), ICmpInst::ICMP_NE, V1, V2, Q.DL,
                           /*LHSIsTrue=*/true, Depth)
            .value_or(false) &&
        isValidAssumeForContext(I, Q.CxtI, Q.DT))
      return true;
  }

  return false;
}

/// Return true if it is known that V1 != V2.
static bool isKnownNonEqual(const Value *V1, const Value *V2,
                            const APInt &DemandedElts, const SimplifyQuery &Q,
                            unsigned Depth) {
  if (V1 == V2)
    return false;
  if (V1->getType() != V2->getType())
    // We can't look through casts yet.
    return false;

  if (Depth >= MaxAnalysisRecursionDepth)
    return false;

  // See if we can recurse through (exactly one of) our operands.  This
  // requires our operation be 1-to-1 and map every input value to exactly
  // one output value.  Such an operation is invertible.
  auto *O1 = dyn_cast<Operator>(V1);
  auto *O2 = dyn_cast<Operator>(V2);
  if (O1 && O2 && O1->getOpcode() == O2->getOpcode()) {
    if (auto Values = getInvertibleOperands(O1, O2))
      return isKnownNonEqual(Values->first, Values->second, DemandedElts, Q,
                             Depth + 1);

    if (const PHINode *PN1 = dyn_cast<PHINode>(V1)) {
      const PHINode *PN2 = cast<PHINode>(V2);
      // FIXME: This is missing a generalization to handle the case where one is
      // a PHI and another one isn't.
      if (isNonEqualPHIs(PN1, PN2, DemandedElts, Q, Depth))
        return true;
    };
  }

  if (isModifyingBinopOfNonZero(V1, V2, DemandedElts, Q, Depth) ||
      isModifyingBinopOfNonZero(V2, V1, DemandedElts, Q, Depth))
    return true;

  if (isNonEqualMul(V1, V2, DemandedElts, Q, Depth) ||
      isNonEqualMul(V2, V1, DemandedElts, Q, Depth))
    return true;

  if (isNonEqualShl(V1, V2, DemandedElts, Q, Depth) ||
      isNonEqualShl(V2, V1, DemandedElts, Q, Depth))
    return true;

  if (V1->getType()->isIntOrIntVectorTy()) {
    // Are any known bits in V1 contradictory to known bits in V2? If V1
    // has a known zero where V2 has a known one, they must not be equal.
    KnownBits Known1 = computeKnownBits(V1, DemandedElts, Q, Depth);
    if (!Known1.isUnknown()) {
      KnownBits Known2 = computeKnownBits(V2, DemandedElts, Q, Depth);
      if (Known1.Zero.intersects(Known2.One) ||
          Known2.Zero.intersects(Known1.One))
        return true;
    }
  }

  if (isNonEqualSelect(V1, V2, DemandedElts, Q, Depth) ||
      isNonEqualSelect(V2, V1, DemandedElts, Q, Depth))
    return true;

  if (isNonEqualPointersWithRecursiveGEP(V1, V2, Q) ||
      isNonEqualPointersWithRecursiveGEP(V2, V1, Q))
    return true;

  Value *A, *B;
  // PtrToInts are NonEqual if their Ptrs are NonEqual.
  // Check PtrToInt type matches the pointer size.
  if (match(V1, m_PtrToIntSameSize(Q.DL, m_Value(A))) &&
      match(V2, m_PtrToIntSameSize(Q.DL, m_Value(B))))
    return isKnownNonEqual(A, B, DemandedElts, Q, Depth + 1);

  if (isKnownNonEqualFromContext(V1, V2, Q, Depth))
    return true;

  return false;
}

/// For vector constants, loop over the elements and find the constant with the
/// minimum number of sign bits. Return 0 if the value is not a vector constant
/// or if any element was not analyzed; otherwise, return the count for the
/// element with the minimum number of sign bits.
static unsigned computeNumSignBitsVectorConstant(const Value *V,
                                                 const APInt &DemandedElts,
                                                 unsigned TyBits) {
  const auto *CV = dyn_cast<Constant>(V);
  if (!CV || !isa<FixedVectorType>(CV->getType()))
    return 0;

  unsigned MinSignBits = TyBits;
  unsigned NumElts = cast<FixedVectorType>(CV->getType())->getNumElements();
  for (unsigned i = 0; i != NumElts; ++i) {
    if (!DemandedElts[i])
      continue;
    // If we find a non-ConstantInt, bail out.
    auto *Elt = dyn_cast_or_null<ConstantInt>(CV->getAggregateElement(i));
    if (!Elt)
      return 0;

    MinSignBits = std::min(MinSignBits, Elt->getValue().getNumSignBits());
  }

  return MinSignBits;
}

static unsigned ComputeNumSignBitsImpl(const Value *V,
                                       const APInt &DemandedElts,
                                       const SimplifyQuery &Q, unsigned Depth);

static unsigned ComputeNumSignBits(const Value *V, const APInt &DemandedElts,
                                   const SimplifyQuery &Q, unsigned Depth) {
  unsigned Result = ComputeNumSignBitsImpl(V, DemandedElts, Q, Depth);
  assert(Result > 0 && "At least one sign bit needs to be present!");
  return Result;
}

/// Return the number of times the sign bit of the register is replicated into
/// the other bits. We know that at least 1 bit is always equal to the sign bit
/// (itself), but other cases can give us information. For example, immediately
/// after an "ashr X, 2", we know that the top 3 bits are all equal to each
/// other, so we return 3. For vectors, return the number of sign bits for the
/// vector element with the minimum number of known sign bits of the demanded
/// elements in the vector specified by DemandedElts.
static unsigned ComputeNumSignBitsImpl(const Value *V,
                                       const APInt &DemandedElts,
                                       const SimplifyQuery &Q, unsigned Depth) {
  Type *Ty = V->getType();
#ifndef NDEBUG
  assert(Depth <= MaxAnalysisRecursionDepth && "Limit Search Depth");

  if (auto *FVTy = dyn_cast<FixedVectorType>(Ty)) {
    assert(
        FVTy->getNumElements() == DemandedElts.getBitWidth() &&
        "DemandedElt width should equal the fixed vector number of elements");
  } else {
    assert(DemandedElts == APInt(1, 1) &&
           "DemandedElt width should be 1 for scalars");
  }
#endif

  // We return the minimum number of sign bits that are guaranteed to be present
  // in V, so for undef we have to conservatively return 1.  We don't have the
  // same behavior for poison though -- that's a FIXME today.

  Type *ScalarTy = Ty->getScalarType();
  unsigned TyBits = ScalarTy->isPointerTy() ?
    Q.DL.getPointerTypeSizeInBits(ScalarTy) :
    Q.DL.getTypeSizeInBits(ScalarTy);

  unsigned Tmp, Tmp2;
  unsigned FirstAnswer = 1;

  // Note that ConstantInt is handled by the general computeKnownBits case
  // below.

  if (Depth == MaxAnalysisRecursionDepth)
    return 1;

  if (auto *U = dyn_cast<Operator>(V)) {
    switch (Operator::getOpcode(V)) {
    default: break;
    case Instruction::BitCast: {
      Value *Src = U->getOperand(0);
      Type *SrcTy = Src->getType();

      // Skip if the source type is not an integer or integer vector type
      // This ensures we only process integer-like types
      if (!SrcTy->isIntOrIntVectorTy())
        break;

      unsigned SrcBits = SrcTy->getScalarSizeInBits();

      // Bitcast 'large element' scalar/vector to 'small element' vector.
      if ((SrcBits % TyBits) != 0)
        break;

      // Only proceed if the destination type is a fixed-size vector
      if (isa<FixedVectorType>(Ty)) {
        // Fast case - sign splat can be simply split across the small elements.
        // This works for both vector and scalar sources
        Tmp = ComputeNumSignBits(Src, Q, Depth + 1);
        if (Tmp == SrcBits)
          return TyBits;
      }
      break;
    }
    case Instruction::SExt:
      Tmp = TyBits - U->getOperand(0)->getType()->getScalarSizeInBits();
      return ComputeNumSignBits(U->getOperand(0), DemandedElts, Q, Depth + 1) +
             Tmp;

    case Instruction::SDiv: {
      const APInt *Denominator;
      // sdiv X, C -> adds log(C) sign bits.
      if (match(U->getOperand(1), m_APInt(Denominator))) {

        // Ignore non-positive denominator.
        if (!Denominator->isStrictlyPositive())
          break;

        // Calculate the incoming numerator bits.
        unsigned NumBits =
            ComputeNumSignBits(U->getOperand(0), DemandedElts, Q, Depth + 1);

        // Add floor(log(C)) bits to the numerator bits.
        return std::min(TyBits, NumBits + Denominator->logBase2());
      }
      break;
    }

    case Instruction::SRem: {
      Tmp = ComputeNumSignBits(U->getOperand(0), DemandedElts, Q, Depth + 1);

      const APInt *Denominator;
      // srem X, C -> we know that the result is within [-C+1,C) when C is a
      // positive constant.  This let us put a lower bound on the number of sign
      // bits.
      if (match(U->getOperand(1), m_APInt(Denominator))) {

        // Ignore non-positive denominator.
        if (Denominator->isStrictlyPositive()) {
          // Calculate the leading sign bit constraints by examining the
          // denominator.  Given that the denominator is positive, there are two
          // cases:
          //
          //  1. The numerator is positive. The result range is [0,C) and
          //     [0,C) u< (1 << ceilLogBase2(C)).
          //
          //  2. The numerator is negative. Then the result range is (-C,0] and
          //     integers in (-C,0] are either 0 or >u (-1 << ceilLogBase2(C)).
          //
          // Thus a lower bound on the number of sign bits is `TyBits -
          // ceilLogBase2(C)`.

          unsigned ResBits = TyBits - Denominator->ceilLogBase2();
          Tmp = std::max(Tmp, ResBits);
        }
      }
      return Tmp;
    }

    case Instruction::AShr: {
      Tmp = ComputeNumSignBits(U->getOperand(0), DemandedElts, Q, Depth + 1);
      // ashr X, C   -> adds C sign bits.  Vectors too.
      const APInt *ShAmt;
      if (match(U->getOperand(1), m_APInt(ShAmt))) {
        if (ShAmt->uge(TyBits))
          break; // Bad shift.
        unsigned ShAmtLimited = ShAmt->getZExtValue();
        Tmp += ShAmtLimited;
        if (Tmp > TyBits) Tmp = TyBits;
      }
      return Tmp;
    }
    case Instruction::Shl: {
      const APInt *ShAmt;
      Value *X = nullptr;
      if (match(U->getOperand(1), m_APInt(ShAmt))) {
        // shl destroys sign bits.
        if (ShAmt->uge(TyBits))
          break; // Bad shift.
        // We can look through a zext (more or less treating it as a sext) if
        // all extended bits are shifted out.
        if (match(U->getOperand(0), m_ZExt(m_Value(X))) &&
            ShAmt->uge(TyBits - X->getType()->getScalarSizeInBits())) {
          Tmp = ComputeNumSignBits(X, DemandedElts, Q, Depth + 1);
          Tmp += TyBits - X->getType()->getScalarSizeInBits();
        } else
          Tmp =
              ComputeNumSignBits(U->getOperand(0), DemandedElts, Q, Depth + 1);
        if (ShAmt->uge(Tmp))
          break; // Shifted all sign bits out.
        Tmp2 = ShAmt->getZExtValue();
        return Tmp - Tmp2;
      }
      break;
    }
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor: // NOT is handled here.
      // Logical binary ops preserve the number of sign bits at the worst.
      Tmp = ComputeNumSignBits(U->getOperand(0), DemandedElts, Q, Depth + 1);
      if (Tmp != 1) {
        Tmp2 = ComputeNumSignBits(U->getOperand(1), DemandedElts, Q, Depth + 1);
        FirstAnswer = std::min(Tmp, Tmp2);
        // We computed what we know about the sign bits as our first
        // answer. Now proceed to the generic code that uses
        // computeKnownBits, and pick whichever answer is better.
      }
      break;

    case Instruction::Select: {
      // If we have a clamp pattern, we know that the number of sign bits will
      // be the minimum of the clamp min/max range.
      const Value *X;
      const APInt *CLow, *CHigh;
      if (isSignedMinMaxClamp(U, X, CLow, CHigh))
        return std::min(CLow->getNumSignBits(), CHigh->getNumSignBits());

      Tmp = ComputeNumSignBits(U->getOperand(1), DemandedElts, Q, Depth + 1);
      if (Tmp == 1)
        break;
      Tmp2 = ComputeNumSignBits(U->getOperand(2), DemandedElts, Q, Depth + 1);
      return std::min(Tmp, Tmp2);
    }

    case Instruction::Add:
      // Add can have at most one carry bit.  Thus we know that the output
      // is, at worst, one more bit than the inputs.
      Tmp = ComputeNumSignBits(U->getOperand(0), Q, Depth + 1);
      if (Tmp == 1) break;

      // Special case decrementing a value (ADD X, -1):
      if (const auto *CRHS = dyn_cast<Constant>(U->getOperand(1)))
        if (CRHS->isAllOnesValue()) {
          KnownBits Known(TyBits);
          computeKnownBits(U->getOperand(0), DemandedElts, Known, Q, Depth + 1);

          // If the input is known to be 0 or 1, the output is 0/-1, which is
          // all sign bits set.
          if ((Known.Zero | 1).isAllOnes())
            return TyBits;

          // If we are subtracting one from a positive number, there is no carry
          // out of the result.
          if (Known.isNonNegative())
            return Tmp;
        }

      Tmp2 = ComputeNumSignBits(U->getOperand(1), DemandedElts, Q, Depth + 1);
      if (Tmp2 == 1)
        break;
      return std::min(Tmp, Tmp2) - 1;

    case Instruction::Sub:
      Tmp2 = ComputeNumSignBits(U->getOperand(1), DemandedElts, Q, Depth + 1);
      if (Tmp2 == 1)
        break;

      // Handle NEG.
      if (const auto *CLHS = dyn_cast<Constant>(U->getOperand(0)))
        if (CLHS->isNullValue()) {
          KnownBits Known(TyBits);
          computeKnownBits(U->getOperand(1), DemandedElts, Known, Q, Depth + 1);
          // If the input is known to be 0 or 1, the output is 0/-1, which is
          // all sign bits set.
          if ((Known.Zero | 1).isAllOnes())
            return TyBits;

          // If the input is known to be positive (the sign bit is known clear),
          // the output of the NEG has the same number of sign bits as the
          // input.
          if (Known.isNonNegative())
            return Tmp2;

          // Otherwise, we treat this like a SUB.
        }

      // Sub can have at most one carry bit.  Thus we know that the output
      // is, at worst, one more bit than the inputs.
      Tmp = ComputeNumSignBits(U->getOperand(0), DemandedElts, Q, Depth + 1);
      if (Tmp == 1)
        break;
      return std::min(Tmp, Tmp2) - 1;

    case Instruction::Mul: {
      // The output of the Mul can be at most twice the valid bits in the
      // inputs.
      unsigned SignBitsOp0 =
          ComputeNumSignBits(U->getOperand(0), DemandedElts, Q, Depth + 1);
      if (SignBitsOp0 == 1)
        break;
      unsigned SignBitsOp1 =
          ComputeNumSignBits(U->getOperand(1), DemandedElts, Q, Depth + 1);
      if (SignBitsOp1 == 1)
        break;
      unsigned OutValidBits =
          (TyBits - SignBitsOp0 + 1) + (TyBits - SignBitsOp1 + 1);
      return OutValidBits > TyBits ? 1 : TyBits - OutValidBits + 1;
    }

    case Instruction::PHI: {
      const PHINode *PN = cast<PHINode>(U);
      unsigned NumIncomingValues = PN->getNumIncomingValues();
      // Don't analyze large in-degree PHIs.
      if (NumIncomingValues > 4) break;
      // Unreachable blocks may have zero-operand PHI nodes.
      if (NumIncomingValues == 0) break;

      // Take the minimum of all incoming values.  This can't infinitely loop
      // because of our depth threshold.
      SimplifyQuery RecQ = Q.getWithoutCondContext();
      Tmp = TyBits;
      for (unsigned i = 0, e = NumIncomingValues; i != e; ++i) {
        if (Tmp == 1) return Tmp;
        RecQ.CxtI = PN->getIncomingBlock(i)->getTerminator();
        Tmp = std::min(Tmp, ComputeNumSignBits(PN->getIncomingValue(i),
                                               DemandedElts, RecQ, Depth + 1));
      }
      return Tmp;
    }

    case Instruction::Trunc: {
      // If the input contained enough sign bits that some remain after the
      // truncation, then we can make use of that. Otherwise we don't know
      // anything.
      Tmp = ComputeNumSignBits(U->getOperand(0), Q, Depth + 1);
      unsigned OperandTyBits = U->getOperand(0)->getType()->getScalarSizeInBits();
      if (Tmp > (OperandTyBits - TyBits))
        return Tmp - (OperandTyBits - TyBits);

      return 1;
    }

    case Instruction::ExtractElement:
      // Look through extract element. At the moment we keep this simple and
      // skip tracking the specific element. But at least we might find
      // information valid for all elements of the vector (for example if vector
      // is sign extended, shifted, etc).
      return ComputeNumSignBits(U->getOperand(0), Q, Depth + 1);

    case Instruction::ShuffleVector: {
      // Collect the minimum number of sign bits that are shared by every vector
      // element referenced by the shuffle.
      auto *Shuf = dyn_cast<ShuffleVectorInst>(U);
      if (!Shuf) {
        // FIXME: Add support for shufflevector constant expressions.
        return 1;
      }
      APInt DemandedLHS, DemandedRHS;
      // For undef elements, we don't know anything about the common state of
      // the shuffle result.
      if (!getShuffleDemandedElts(Shuf, DemandedElts, DemandedLHS, DemandedRHS))
        return 1;
      Tmp = std::numeric_limits<unsigned>::max();
      if (!!DemandedLHS) {
        const Value *LHS = Shuf->getOperand(0);
        Tmp = ComputeNumSignBits(LHS, DemandedLHS, Q, Depth + 1);
      }
      // If we don't know anything, early out and try computeKnownBits
      // fall-back.
      if (Tmp == 1)
        break;
      if (!!DemandedRHS) {
        const Value *RHS = Shuf->getOperand(1);
        Tmp2 = ComputeNumSignBits(RHS, DemandedRHS, Q, Depth + 1);
        Tmp = std::min(Tmp, Tmp2);
      }
      // If we don't know anything, early out and try computeKnownBits
      // fall-back.
      if (Tmp == 1)
        break;
      assert(Tmp <= TyBits && "Failed to determine minimum sign bits");
      return Tmp;
    }
    case Instruction::Call: {
      if (const auto *II = dyn_cast<IntrinsicInst>(U)) {
        switch (II->getIntrinsicID()) {
        default:
          break;
        case Intrinsic::abs:
          Tmp =
              ComputeNumSignBits(U->getOperand(0), DemandedElts, Q, Depth + 1);
          if (Tmp == 1)
            break;

          // Absolute value reduces number of sign bits by at most 1.
          return Tmp - 1;
        case Intrinsic::smin:
        case Intrinsic::smax: {
          const APInt *CLow, *CHigh;
          if (isSignedMinMaxIntrinsicClamp(II, CLow, CHigh))
            return std::min(CLow->getNumSignBits(), CHigh->getNumSignBits());
        }
        }
      }
    }
    }
  }

  // Finally, if we can prove that the top bits of the result are 0's or 1's,
  // use this information.

  // If we can examine all elements of a vector constant successfully, we're
  // done (we can't do any better than that). If not, keep trying.
  if (unsigned VecSignBits =
          computeNumSignBitsVectorConstant(V, DemandedElts, TyBits))
    return VecSignBits;

  KnownBits Known(TyBits);
  computeKnownBits(V, DemandedElts, Known, Q, Depth);

  // If we know that the sign bit is either zero or one, determine the number of
  // identical bits in the top of the input value.
  return std::max(FirstAnswer, Known.countMinSignBits());
}

static void computeKnownFPClassFromCond(const Value *V, Value *Cond,
                                        bool CondIsTrue,
                                        const Instruction *CxtI,
                                        KnownFPClass &KnownFromContext,
                                        unsigned Depth = 0) {
  Value *A, *B;
  if (Depth < MaxAnalysisRecursionDepth &&
      (CondIsTrue ? match(Cond, m_LogicalAnd(m_Value(A), m_Value(B)))
                  : match(Cond, m_LogicalOr(m_Value(A), m_Value(B))))) {
    computeKnownFPClassFromCond(V, A, CondIsTrue, CxtI, KnownFromContext,
                                Depth + 1);
    computeKnownFPClassFromCond(V, B, CondIsTrue, CxtI, KnownFromContext,
                                Depth + 1);
    return;
  }
  if (Depth < MaxAnalysisRecursionDepth && match(Cond, m_Not(m_Value(A)))) {
    computeKnownFPClassFromCond(V, A, !CondIsTrue, CxtI, KnownFromContext,
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
    computeKnownFPClassFromCond(V, Q.CC->Cond, !Q.CC->Invert, Q.CxtI,
                                KnownFromContext);

  if (!Q.CxtI)
    return KnownFromContext;

  if (Q.DC && Q.DT) {
    // Handle dominating conditions.
    for (BranchInst *BI : Q.DC->conditionsFor(V)) {
      Value *Cond = BI->getCondition();

      BasicBlockEdge Edge0(BI->getParent(), BI->getSuccessor(0));
      if (Q.DT->dominates(Edge0, Q.CxtI->getParent()))
        computeKnownFPClassFromCond(V, Cond, /*CondIsTrue=*/true, Q.CxtI,
                                    KnownFromContext);

      BasicBlockEdge Edge1(BI->getParent(), BI->getSuccessor(1));
      if (Q.DT->dominates(Edge1, Q.CxtI->getParent()))
        computeKnownFPClassFromCond(V, Cond, /*CondIsTrue=*/false, Q.CxtI,
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

    assert(I->getFunction() == Q.CxtI->getParent()->getParent() &&
           "Got assumption for the wrong function!");
    assert(I->getIntrinsicID() == Intrinsic::assume &&
           "must be an assume intrinsic");

    if (!isValidAssumeForContext(I, Q.CxtI, Q.DT))
      continue;

    computeKnownFPClassFromCond(V, I->getArgOperand(0),
                                /*CondIsTrue=*/true, Q.CxtI, KnownFromContext);
  }

  return KnownFromContext;
}

void llvm::adjustKnownFPClassForSelectArm(KnownFPClass &Known, Value *Cond,
                                          Value *Arm, bool Invert,
                                          const SimplifyQuery &SQ,
                                          unsigned Depth) {

  KnownFPClass KnownSrc;
  computeKnownFPClassFromCond(Arm, Cond,
                              /*CondIsTrue=*/!Invert, SQ.CxtI, KnownSrc,
                              Depth + 1);
  KnownSrc = KnownSrc.unionWith(Known);
  if (KnownSrc.isUnknown())
    return;

  if (isGuaranteedNotToBeUndef(Arm, SQ.AC, SQ.CxtI, SQ.DT, Depth + 1))
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

static void computeKnownFPClassForFPTrunc(const Operator *Op,
                                          const APInt &DemandedElts,
                                          FPClassTest InterestedClasses,
                                          KnownFPClass &Known,
                                          const SimplifyQuery &Q,
                                          unsigned Depth) {
  if ((InterestedClasses &
       (KnownFPClass::OrderedLessThanZeroMask | fcNan)) == fcNone)
    return;

  KnownFPClass KnownSrc;
  computeKnownFPClass(Op->getOperand(0), DemandedElts, InterestedClasses,
                      KnownSrc, Q, Depth + 1);
  Known = KnownFPClass::fptrunc(KnownSrc);
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
    Known.KnownFPClasses = fcPosZero;
    Known.SignBit = false;
    return;
  }

  if (isa<PoisonValue>(V)) {
    Known.KnownFPClasses = fcNone;
    Known.SignBit = false;
    return;
  }

  // Try to handle fixed width vector constants
  auto *VFVTy = dyn_cast<FixedVectorType>(V->getType());
  const Constant *CV = dyn_cast<Constant>(V);
  if (VFVTy && CV) {
    Known.KnownFPClasses = fcNone;
    bool SignBitAllZero = true;
    bool SignBitAllOne = true;

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

      const APFloat &C = CElt->getValueAPF();
      Known.KnownFPClasses |= C.classify();
      if (C.isNegative())
        SignBitAllZero = false;
      else
        SignBitAllOne = false;
    }
    if (SignBitAllOne != SignBitAllZero)
      Known.SignBit = SignBitAllOne;
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
  KnownNotFromFlags |= ~AssumedClasses.KnownFPClasses;

  // We no longer need to find out about these bits from inputs if we can
  // assume this from flags/attributes.
  InterestedClasses &= ~KnownNotFromFlags;

  llvm::scope_exit ClearClassesFromFlags([=, &Known] {
    Known.knownNot(KnownNotFromFlags);
    if (!Known.SignBit && AssumedClasses.SignBit) {
      if (*AssumedClasses.SignBit)
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
      if ((InterestedClasses & fcNegative) == fcNone)
        break;

      if (II->getArgOperand(0) == II->getArgOperand(1) &&
          isGuaranteedNotToBeUndef(II->getArgOperand(0), Q.AC, Q.CxtI, Q.DT,
                                   Depth + 1)) {
        KnownFPClass KnownSrc, KnownAddend;
        computeKnownFPClass(II->getArgOperand(2), DemandedElts,
                            InterestedClasses, KnownAddend, Q, Depth + 1);
        computeKnownFPClass(II->getArgOperand(0), DemandedElts,
                            InterestedClasses, KnownSrc, Q, Depth + 1);

        const Function *F = II->getFunction();
        const fltSemantics &FltSem =
            II->getType()->getScalarType()->getFltSemantics();
        DenormalMode Mode =
            F ? F->getDenormalMode(FltSem) : DenormalMode::getDynamic();

        if (KnownNotFromFlags & fcNan) {
          KnownSrc.knownNot(fcNan);
          KnownAddend.knownNot(fcNan);
        }

        if (KnownNotFromFlags & fcInf) {
          KnownSrc.knownNot(fcInf);
          KnownAddend.knownNot(fcInf);
        }

        Known = KnownFPClass::fma_square(KnownSrc, KnownAddend, Mode);
        break;
      }

      KnownFPClass KnownSrc[3];
      for (int I = 0; I != 3; ++I) {
        computeKnownFPClass(II->getArgOperand(I), DemandedElts,
                            InterestedClasses, KnownSrc[I], Q, Depth + 1);
        if (KnownSrc[I].isUnknown())
          return;

        if (KnownNotFromFlags & fcNan)
          KnownSrc[I].knownNot(fcNan);
        if (KnownNotFromFlags & fcInf)
          KnownSrc[I].knownNot(fcInf);
      }

      const Function *F = II->getFunction();
      const fltSemantics &FltSem =
          II->getType()->getScalarType()->getFltSemantics();
      DenormalMode Mode =
          F ? F->getDenormalMode(FltSem) : DenormalMode::getDynamic();
      Known = KnownFPClass::fma(KnownSrc[0], KnownSrc[1], KnownSrc[2], Mode);
      break;
    }
    case Intrinsic::sqrt:
    case Intrinsic::experimental_constrained_sqrt: {
      KnownFPClass KnownSrc;
      FPClassTest InterestedSrcs = InterestedClasses;
      if (InterestedClasses & fcNan)
        InterestedSrcs |= KnownFPClass::OrderedLessThanZeroMask;

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
    case Intrinsic::sin:
    case Intrinsic::cos: {
      // Return NaN on infinite inputs.
      KnownFPClass KnownSrc;
      computeKnownFPClass(II->getArgOperand(0), DemandedElts, InterestedClasses,
                          KnownSrc, Q, Depth + 1);
      Known.knownNot(fcInf);
      if (KnownSrc.isKnownNeverNaN() && KnownSrc.isKnownNeverInfinity())
        Known.knownNot(fcNan);
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
      DenormalMode DenormMode =
          F ? F->getDenormalMode(
                  II->getType()->getScalarType()->getFltSemantics())
            : DenormalMode::getDynamic();
      Known = KnownFPClass::canonicalize(KnownSrc, DenormMode);
      break;
    }
    case Intrinsic::vector_reduce_fmax:
    case Intrinsic::vector_reduce_fmin:
    case Intrinsic::vector_reduce_fmaximum:
    case Intrinsic::vector_reduce_fminimum: {
      // reduce min/max will choose an element from one of the vector elements,
      // so we can infer and class information that is common to all elements.
      Known = computeKnownFPClass(II->getArgOperand(0), II->getFastMathFlags(),
                                  InterestedClasses, Q, Depth + 1);
      // Can only propagate sign if output is never NaN.
      if (!Known.isKnownNeverNaN())
        Known.SignBit.reset();
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
      if (InterestedSrcs & fcPosFinite)
        InterestedSrcs |= fcPosFinite;
      if (InterestedSrcs & fcNegFinite)
        InterestedSrcs |= fcNegFinite;
      computeKnownFPClass(II->getArgOperand(0), DemandedElts, InterestedSrcs,
                          KnownSrc, Q, Depth + 1);

      Known = KnownFPClass::roundToIntegral(
          KnownSrc, IID == Intrinsic::trunc,
          V->getType()->getScalarType()->isMultiUnitFPType());
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
      computeKnownFPClassForFPTrunc(Op, DemandedElts, InterestedClasses, Known,
                                    Q, Depth);
      break;
    }
    case Intrinsic::log:
    case Intrinsic::log10:
    case Intrinsic::log2:
    case Intrinsic::experimental_constrained_log:
    case Intrinsic::experimental_constrained_log10:
    case Intrinsic::experimental_constrained_log2:
    case Intrinsic::amdgcn_log: {
      Type *EltTy = II->getType()->getScalarType();

      // log(+inf) -> +inf
      // log([+-]0.0) -> -inf
      // log(-inf) -> nan
      // log(-x) -> nan
      if ((InterestedClasses & (fcNan | fcInf)) != fcNone) {
        FPClassTest InterestedSrcs = InterestedClasses;
        if ((InterestedClasses & fcNegInf) != fcNone)
          InterestedSrcs |= fcZero | fcSubnormal;
        if ((InterestedClasses & fcNan) != fcNone)
          InterestedSrcs |= fcNan | fcNegative;

        KnownFPClass KnownSrc;
        computeKnownFPClass(II->getArgOperand(0), DemandedElts, InterestedSrcs,
                            KnownSrc, Q, Depth + 1);

        const Function *F = II->getFunction();
        DenormalMode Mode = F ? F->getDenormalMode(EltTy->getFltSemantics())
                              : DenormalMode::getDynamic();
        Known = KnownFPClass::log(KnownSrc, Mode);
      }

      if (IID == Intrinsic::amdgcn_log && EltTy->isFloatTy())
        Known.knownNot(fcSubnormal);
      break;
    }
    case Intrinsic::powi: {
      if ((InterestedClasses & fcNegative) == fcNone)
        break;

      const Value *Exp = II->getArgOperand(1);
      Type *ExpTy = Exp->getType();
      unsigned BitWidth = ExpTy->getScalarType()->getIntegerBitWidth();
      KnownBits ExponentKnownBits(BitWidth);
      computeKnownBits(Exp, isa<VectorType>(ExpTy) ? DemandedElts : APInt(1, 1),
                       ExponentKnownBits, Q, Depth + 1);

      if (ExponentKnownBits.Zero[0]) { // Is even
        Known.knownNot(fcNegative);
        break;
      }

      // Given that exp is an integer, here are the
      // ways that pow can return a negative value:
      //
      //   pow(-x, exp)   --> negative if exp is odd and x is negative.
      //   pow(-0, exp)   --> -inf if exp is negative odd.
      //   pow(-0, exp)   --> -0 if exp is positive odd.
      //   pow(-inf, exp) --> -0 if exp is negative odd.
      //   pow(-inf, exp) --> -inf if exp is positive odd.
      KnownFPClass KnownSrc;
      computeKnownFPClass(II->getArgOperand(0), DemandedElts, fcNegative,
                          KnownSrc, Q, Depth + 1);
      if (KnownSrc.isKnownNever(fcNegative))
        Known.knownNot(fcNegative);
      break;
    }
    case Intrinsic::ldexp: {
      KnownFPClass KnownSrc;
      computeKnownFPClass(II->getArgOperand(0), DemandedElts, InterestedClasses,
                          KnownSrc, Q, Depth + 1);
      Known.propagateNaN(KnownSrc, /*PropagateSign=*/true);

      // Sign is preserved, but underflows may produce zeroes.
      if (KnownSrc.isKnownNever(fcNegative))
        Known.knownNot(fcNegative);
      else if (KnownSrc.cannotBeOrderedLessThanZero())
        Known.knownNot(KnownFPClass::OrderedLessThanZeroMask);

      if (KnownSrc.isKnownNever(fcPositive))
        Known.knownNot(fcPositive);
      else if (KnownSrc.cannotBeOrderedGreaterThanZero())
        Known.knownNot(KnownFPClass::OrderedGreaterThanZeroMask);

      // Can refine inf/zero handling based on the exponent operand.
      const FPClassTest ExpInfoMask = fcZero | fcSubnormal | fcInf;
      if ((InterestedClasses & ExpInfoMask) == fcNone)
        break;
      if ((KnownSrc.KnownFPClasses & ExpInfoMask) == fcNone)
        break;

      const fltSemantics &Flt =
          II->getType()->getScalarType()->getFltSemantics();
      unsigned Precision = APFloat::semanticsPrecision(Flt);
      const Value *ExpArg = II->getArgOperand(1);
      ConstantRange ExpRange = computeConstantRange(
          ExpArg, true, Q.IIQ.UseInstrInfo, Q.AC, Q.CxtI, Q.DT, Depth + 1);

      const int MantissaBits = Precision - 1;
      if (ExpRange.getSignedMin().sge(static_cast<int64_t>(MantissaBits)))
        Known.knownNot(fcSubnormal);

      const Function *F = II->getFunction();
      const APInt *ConstVal = ExpRange.getSingleElement();
      const fltSemantics &FltSem =
          II->getType()->getScalarType()->getFltSemantics();
      if (ConstVal && ConstVal->isZero()) {
        // ldexp(x, 0) -> x, so propagate everything.
        Known.propagateCanonicalizingSrc(KnownSrc, F->getDenormalMode(FltSem));
      } else if (ExpRange.isAllNegative()) {
        // If we know the power is <= 0, can't introduce inf
        if (KnownSrc.isKnownNeverPosInfinity())
          Known.knownNot(fcPosInf);
        if (KnownSrc.isKnownNeverNegInfinity())
          Known.knownNot(fcNegInf);
      } else if (ExpRange.isAllNonNegative()) {
        // If we know the power is >= 0, can't introduce subnormal or zero
        if (KnownSrc.isKnownNeverPosSubnormal())
          Known.knownNot(fcPosSubnormal);
        if (KnownSrc.isKnownNeverNegSubnormal())
          Known.knownNot(fcNegSubnormal);
        if (F &&
            KnownSrc.isKnownNeverLogicalPosZero(F->getDenormalMode(FltSem)))
          Known.knownNot(fcPosZero);
        if (F &&
            KnownSrc.isKnownNeverLogicalNegZero(F->getDenormalMode(FltSem)))
          Known.knownNot(fcNegZero);
      }

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
    case Intrinsic::amdgcn_rcp: {
      KnownFPClass KnownSrc;
      computeKnownFPClass(II->getArgOperand(0), DemandedElts, InterestedClasses,
                          KnownSrc, Q, Depth + 1);

      Known.propagateNaN(KnownSrc);

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
    default:
      break;
    }

    break;
  }
  case Instruction::FAdd:
  case Instruction::FSub: {
    KnownFPClass KnownLHS, KnownRHS;
    bool WantNegative =
        Op->getOpcode() == Instruction::FAdd &&
        (InterestedClasses & KnownFPClass::OrderedLessThanZeroMask) != fcNone;
    bool WantNaN = (InterestedClasses & fcNan) != fcNone;
    bool WantNegZero = (InterestedClasses & fcNegZero) != fcNone;

    if (!WantNaN && !WantNegative && !WantNegZero)
      break;

    FPClassTest InterestedSrcs = InterestedClasses;
    if (WantNegative)
      InterestedSrcs |= KnownFPClass::OrderedLessThanZeroMask;
    if (InterestedClasses & fcNan)
      InterestedSrcs |= fcInf;
    computeKnownFPClass(Op->getOperand(1), DemandedElts, InterestedSrcs,
                        KnownRHS, Q, Depth + 1);

    // Special case fadd x, x, which is the canonical form of fmul x, 2.
    bool Self = Op->getOperand(0) == Op->getOperand(1) &&
                isGuaranteedNotToBeUndef(Op->getOperand(0), Q.AC, Q.CxtI, Q.DT,
                                         Depth + 1);
    if (Self)
      KnownLHS = KnownRHS;

    if ((WantNaN && KnownRHS.isKnownNeverNaN()) ||
        (WantNegative && KnownRHS.cannotBeOrderedLessThanZero()) ||
        WantNegZero || Opc == Instruction::FSub) {

      // FIXME: Context function should always be passed in separately
      const Function *F = cast<Instruction>(Op)->getFunction();
      const fltSemantics &FltSem =
          Op->getType()->getScalarType()->getFltSemantics();
      DenormalMode Mode =
          F ? F->getDenormalMode(FltSem) : DenormalMode::getDynamic();

      if (Self && Opc == Instruction::FAdd) {
        Known = KnownFPClass::fadd_self(KnownLHS, Mode);
      } else {
        // RHS is canonically cheaper to compute. Skip inspecting the LHS if
        // there's no point.

        if (!Self) {
          computeKnownFPClass(Op->getOperand(0), DemandedElts, InterestedSrcs,
                              KnownLHS, Q, Depth + 1);
        }

        Known = Opc == Instruction::FAdd
                    ? KnownFPClass::fadd(KnownLHS, KnownRHS, Mode)
                    : KnownFPClass::fsub(KnownLHS, KnownRHS, Mode);
      }
    }

    break;
  }
  case Instruction::FMul: {
    const Function *F = cast<Instruction>(Op)->getFunction();
    DenormalMode Mode =
        F ? F->getDenormalMode(
                Op->getType()->getScalarType()->getFltSemantics())
          : DenormalMode::getDynamic();

    // X * X is always non-negative or a NaN.
    if (Op->getOperand(0) == Op->getOperand(1) &&
        isGuaranteedNotToBeUndef(Op->getOperand(0), Q.AC, Q.CxtI, Q.DT)) {
      KnownFPClass KnownSrc;
      computeKnownFPClass(Op->getOperand(0), DemandedElts, fcAllFlags, KnownSrc,
                          Q, Depth + 1);
      Known = KnownFPClass::square(KnownSrc, Mode);
      break;
    }

    KnownFPClass KnownLHS, KnownRHS;

    bool CannotBeSubnormal = false;
    const APFloat *CRHS;
    if (match(Op->getOperand(1), m_APFloat(CRHS))) {
      // Match denormal scaling pattern, similar to the case in ldexp. If the
      // constant's exponent is sufficiently large, the result cannot be
      // subnormal.

      // TODO: Should do general ConstantFPRange analysis.
      const fltSemantics &Flt =
          Op->getType()->getScalarType()->getFltSemantics();
      unsigned Precision = APFloat::semanticsPrecision(Flt);
      const int MantissaBits = Precision - 1;

      int MinKnownExponent = ilogb(*CRHS);
      if (MinKnownExponent >= MantissaBits)
        CannotBeSubnormal = true;

      KnownRHS = KnownFPClass(*CRHS);
    } else {
      computeKnownFPClass(Op->getOperand(1), DemandedElts, fcAllFlags, KnownRHS,
                          Q, Depth + 1);
    }

    // TODO: Improve accuracy in unfused FMA pattern. We can prove an additional
    // not-nan if the addend is known-not negative infinity if the multiply is
    // known-not infinity.

    computeKnownFPClass(Op->getOperand(0), DemandedElts, fcAllFlags, KnownLHS,
                        Q, Depth + 1);

    Known = KnownFPClass::fmul(KnownLHS, KnownRHS, Mode);
    if (CannotBeSubnormal)
      Known.knownNot(fcSubnormal);
    break;
  }
  case Instruction::FDiv:
  case Instruction::FRem: {
    const bool WantNan = (InterestedClasses & fcNan) != fcNone;

    if (Op->getOperand(0) == Op->getOperand(1) &&
        isGuaranteedNotToBeUndef(Op->getOperand(0), Q.AC, Q.CxtI, Q.DT)) {
      if (Op->getOpcode() == Instruction::FDiv) {
        // X / X is always exactly 1.0 or a NaN.
        Known.KnownFPClasses = fcNan | fcPosNormal;
      } else {
        // X % X is always exactly [+-]0.0 or a NaN.
        Known.KnownFPClasses = fcNan | fcZero;
      }

      if (!WantNan)
        break;

      KnownFPClass KnownSrc;
      computeKnownFPClass(Op->getOperand(0), DemandedElts,
                          fcNan | fcInf | fcZero | fcSubnormal, KnownSrc, Q,
                          Depth + 1);
      const Function *F = cast<Instruction>(Op)->getFunction();
      const fltSemantics &FltSem =
          Op->getType()->getScalarType()->getFltSemantics();

      DenormalMode Mode =
          F ? F->getDenormalMode(FltSem) : DenormalMode::getDynamic();

      Known = Op->getOpcode() == Instruction::FDiv
                  ? KnownFPClass::fdiv_self(KnownSrc, Mode)
                  : KnownFPClass::frem_self(KnownSrc, Mode);
      break;
    }

    const bool WantNegative = (InterestedClasses & fcNegative) != fcNone;
    const bool WantPositive =
        Opc == Instruction::FRem && (InterestedClasses & fcPositive) != fcNone;
    if (!WantNan && !WantNegative && !WantPositive)
      break;

    KnownFPClass KnownLHS, KnownRHS;

    computeKnownFPClass(Op->getOperand(1), DemandedElts,
                        fcNan | fcInf | fcZero | fcNegative, KnownRHS, Q,
                        Depth + 1);

    bool KnowSomethingUseful = KnownRHS.isKnownNeverNaN() ||
                               KnownRHS.isKnownNever(fcNegative) ||
                               KnownRHS.isKnownNever(fcPositive);

    if (KnowSomethingUseful || WantPositive) {
      computeKnownFPClass(Op->getOperand(0), DemandedElts, fcAllFlags, KnownLHS,
                          Q, Depth + 1);
    }

    const Function *F = cast<Instruction>(Op)->getFunction();
    const fltSemantics &FltSem =
        Op->getType()->getScalarType()->getFltSemantics();

    if (Op->getOpcode() == Instruction::FDiv) {
      DenormalMode Mode =
          F ? F->getDenormalMode(FltSem) : DenormalMode::getDynamic();
      Known = KnownFPClass::fdiv(KnownLHS, KnownRHS, Mode);
    } else {
      // Inf REM x and x REM 0 produce NaN.
      if (KnownLHS.isKnownNeverNaN() && KnownRHS.isKnownNeverNaN() &&
          KnownLHS.isKnownNeverInfinity() && F &&
          KnownRHS.isKnownNeverLogicalZero(F->getDenormalMode(FltSem))) {
        Known.knownNot(fcNan);
      }

      // The sign for frem is the same as the first operand.
      if (KnownLHS.cannotBeOrderedLessThanZero())
        Known.knownNot(KnownFPClass::OrderedLessThanZeroMask);
      if (KnownLHS.cannotBeOrderedGreaterThanZero())
        Known.knownNot(KnownFPClass::OrderedGreaterThanZeroMask);

      // See if we can be more aggressive about the sign of 0.
      if (KnownLHS.isKnownNever(fcNegative))
        Known.knownNot(fcNegative);
      if (KnownLHS.isKnownNever(fcPositive))
        Known.knownNot(fcPositive);
    }

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

    Known = KnownFPClass::fpext(KnownSrc, DstTy, SrcTy);
    break;
  }
  case Instruction::FPTrunc: {
    computeKnownFPClassForFPTrunc(Op, DemandedElts, InterestedClasses, Known, Q,
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
    if (Op->getOpcode() == Instruction::UIToFP)
      Known.signBitMustBeZero();

    if (InterestedClasses & fcInf) {
      // Get width of largest magnitude integer (remove a bit if signed).
      // This still works for a signed minimum value because the largest FP
      // value is scaled by some fraction close to 2.0 (1.0 + 0.xxxx).
      int IntSize = Op->getOperand(0)->getType()->getScalarSizeInBits();
      if (Op->getOpcode() == Instruction::SIToFP)
        --IntSize;

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
      Known.KnownFPClasses = fcNone;
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
    if (!Shuf || !getShuffleDemandedElts(Shuf, DemandedElts, DemandedLHS, DemandedRHS))
      return;

    if (!!DemandedLHS) {
      const Value *LHS = Shuf->getOperand(0);
      computeKnownFPClass(LHS, DemandedLHS, InterestedClasses, Known, Q,
                          Depth + 1);

      // If we don't know any bits, early out.
      if (Known.isUnknown())
        break;
    } else {
      Known.KnownFPClasses = fcNone;
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
    if (isa<StructType>(Src->getType()) && Indices.size() == 1 &&
        Indices[0] == 0) {
      if (const auto *II = dyn_cast<IntrinsicInst>(Src)) {
        switch (II->getIntrinsicID()) {
        case Intrinsic::frexp: {
          Known.knownNot(fcSubnormal);

          KnownFPClass KnownSrc;
          computeKnownFPClass(II->getArgOperand(0), DemandedElts,
                              InterestedClasses, KnownSrc, Q, Depth + 1);

          const Function *F = cast<Instruction>(Op)->getFunction();
          const fltSemantics &FltSem =
              Op->getType()->getScalarType()->getFltSemantics();

          DenormalMode Mode =
              F ? F->getDenormalMode(FltSem) : DenormalMode::getDynamic();
          Known = KnownFPClass::frexp_mant(KnownSrc, Mode);
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
        Instruction *CxtI;
        breakSelfRecursivePHI(&U, P, IncValue, CxtI);
        // Skip direct self references.
        if (IncValue == P)
          continue;

        KnownFPClass KnownSrc;
        // Recurse, but cap the recursion to two levels, because we don't want
        // to waste time spinning around in loops. We need at least depth 2 to
        // detect known sign bits.
        computeKnownFPClass(IncValue, DemandedElts, InterestedClasses, KnownSrc,
                            Q.getWithoutCondContext().getWithInstruction(CxtI),
                            PhiRecursionLimit);

        if (First) {
          Known = KnownSrc;
          First = false;
        } else {
          Known |= KnownSrc;
        }

        if (Known.KnownFPClasses == fcAllFlags)
          break;
      }
    }

    break;
  }
  case Instruction::BitCast: {
    const Value *Src;
    if (!match(Op, m_ElementWiseBitCast(m_Value(Src))) ||
        !Src->getType()->isIntOrIntVectorTy())
      break;

    const Type *Ty = Op->getType()->getScalarType();
    KnownBits Bits(Ty->getScalarSizeInBits());
    computeKnownBits(Src, DemandedElts, Bits, Q, Depth + 1);

    // Transfer information from the sign bit.
    if (Bits.isNonNegative())
      Known.signBitMustBeZero();
    else if (Bits.isNegative())
      Known.signBitMustBeOne();

    if (Ty->isIEEELikeFPTy()) {
      // IEEE floats are NaN when all bits of the exponent plus at least one of
      // the fraction bits are 1. This means:
      //   - If we assume unknown bits are 0 and the value is NaN, it will
      //     always be NaN
      //   - If we assume unknown bits are 1 and the value is not NaN, it can
      //     never be NaN
      // Note: They do not hold for x86_fp80 format.
      if (APFloat(Ty->getFltSemantics(), Bits.One).isNaN())
        Known.KnownFPClasses = fcNan;
      else if (!APFloat(Ty->getFltSemantics(), ~Bits.Zero).isNaN())
        Known.knownNot(fcNan);

      // Build KnownBits representing Inf and check if it must be equal or
      // unequal to this value.
      auto InfKB = KnownBits::makeConstant(
          APFloat::getInf(Ty->getFltSemantics()).bitcastToAPInt());
      InfKB.Zero.clearSignBit();
      if (const auto InfResult = KnownBits::eq(Bits, InfKB)) {
        assert(!InfResult.value());
        Known.knownNot(fcInf);
      } else if (Bits == InfKB) {
        Known.KnownFPClasses = fcInf;
      }

      // Build KnownBits representing Zero and check if it must be equal or
      // unequal to this value.
      auto ZeroKB = KnownBits::makeConstant(
          APFloat::getZero(Ty->getFltSemantics()).bitcastToAPInt());
      ZeroKB.Zero.clearSignBit();
      if (const auto ZeroResult = KnownBits::eq(Bits, ZeroKB)) {
        assert(!ZeroResult.value());
        Known.knownNot(fcZero);
      } else if (Bits == ZeroKB) {
        Known.KnownFPClasses = fcZero;
      }
    }

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
    const TargetLibraryInfo *TLI, AssumptionCache *AC, const Instruction *CxtI,
    const DominatorTree *DT, bool UseInstrInfo, unsigned Depth) {
  return computeKnownFPClass(V, InterestedClasses,
                             SimplifyQuery(DL, TLI, DT, AC, CxtI, UseInstrInfo),
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
    Result.KnownFPClasses &= ~fcNan;
  if (FMF.noInfs())
    Result.KnownFPClasses &= ~fcInf;
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
  return Known.isKnownNeverNegZero();
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
  return Known.SignBit;
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
  case Instruction::FCmp:
    // fcmp treats both positive and negative zero as equal.
    return true;
  case Instruction::Call:
    if (auto *II = dyn_cast<IntrinsicInst>(User)) {
      switch (II->getIntrinsicID()) {
      case Intrinsic::fabs:
        return true;
      case Intrinsic::copysign:
        return U.getOperandNo() == 0;
      case Intrinsic::is_fpclass:
      case Intrinsic::vp_is_fpclass: {
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

/// Convert ConstantRange OverflowResult into ValueTracking OverflowResult.
static OverflowResult mapOverflowResult(ConstantRange::OverflowResult OR) {
  switch (OR) {
  case ConstantRange::OverflowResult::MayOverflow:
    return OverflowResult::MayOverflow;
  case ConstantRange::OverflowResult::AlwaysOverflowsLow:
    return OverflowResult::AlwaysOverflowsLow;
  case ConstantRange::OverflowResult::AlwaysOverflowsHigh:
    return OverflowResult::AlwaysOverflowsHigh;
  case ConstantRange::OverflowResult::NeverOverflows:
    return OverflowResult::NeverOverflows;
  }
  llvm_unreachable("Unknown OverflowResult");
}

/// Combine constant ranges from computeConstantRange() and computeKnownBits().
ConstantRange
llvm::computeConstantRangeIncludingKnownBits(const WithCache<const Value *> &V,
                                             bool ForSigned,
                                             const SimplifyQuery &SQ) {
  ConstantRange CR1 =
      ConstantRange::fromKnownBits(V.getKnownBits(SQ), ForSigned);
  ConstantRange CR2 = computeConstantRange(V, ForSigned, SQ.IIQ.UseInstrInfo);
  ConstantRange::PreferredRangeType RangeType =
      ForSigned ? ConstantRange::Signed : ConstantRange::Unsigned;
  return CR1.intersectWith(CR2, RangeType);
}

OverflowResult llvm::computeOverflowForUnsignedMul(const Value *LHS,
                                                   const Value *RHS,
                                                   const SimplifyQuery &SQ,
                                                   bool IsNSW) {
  ConstantRange LHSRange =
      computeConstantRangeIncludingKnownBits(LHS, /*ForSigned=*/false, SQ);
  ConstantRange RHSRange =
      computeConstantRangeIncludingKnownBits(RHS, /*ForSigned=*/false, SQ);

  // mul nsw of two non-negative numbers is also nuw.
  if (IsNSW && LHSRange.isAllNonNegative() && RHSRange.isAllNonNegative())
    return OverflowResult::NeverOverflows;

  return mapOverflowResult(LHSRange.unsignedMulMayOverflow(RHSRange));
}

OverflowResult llvm::computeOverflowForSignedMul(const Value *LHS,
                                                 const Value *RHS,
                                                 const SimplifyQuery &SQ) {
  // Multiplying n * m significant bits yields a result of n + m significant
  // bits. If the total number of significant bits does not exceed the
  // result bit width (minus 1), there is no overflow.
  // This means if we have enough leading sign bits in the operands
  // we can guarantee that the result does not overflow.
  // Ref: "Hacker's Delight" by Henry Warren
  unsigned BitWidth = LHS->getType()->getScalarSizeInBits();

  // Note that underestimating the number of sign bits gives a more
  // conservative answer.
  unsigned SignBits =
      ::ComputeNumSignBits(LHS, SQ) + ::ComputeNumSignBits(RHS, SQ);

  // First handle the easy case: if we have enough sign bits there's
  // definitely no overflow.
  if (SignBits > BitWidth + 1)
    return OverflowResult::NeverOverflows;

  // There are two ambiguous cases where there can be no overflow:
  //   SignBits == BitWidth + 1    and
  //   SignBits == BitWidth
  // The second case is difficult to check, therefore we only handle the
  // first case.
  if (SignBits == BitWidth + 1) {
    // It overflows only when both arguments are negative and the true
    // product is exactly the minimum negative number.
    // E.g. mul i16 with 17 sign bits: 0xff00 * 0xff80 = 0x8000
    // For simplicity we just check if at least one side is not negative.
    KnownBits LHSKnown = computeKnownBits(LHS, SQ);
    KnownBits RHSKnown = computeKnownBits(RHS, SQ);
    if (LHSKnown.isNonNegative() || RHSKnown.isNonNegative())
      return OverflowResult::NeverOverflows;
  }
  return OverflowResult::MayOverflow;
}

OverflowResult
llvm::computeOverflowForUnsignedAdd(const WithCache<const Value *> &LHS,
                                    const WithCache<const Value *> &RHS,
                                    const SimplifyQuery &SQ) {
  ConstantRange LHSRange =
      computeConstantRangeIncludingKnownBits(LHS, /*ForSigned=*/false, SQ);
  ConstantRange RHSRange =
      computeConstantRangeIncludingKnownBits(RHS, /*ForSigned=*/false, SQ);
  return mapOverflowResult(LHSRange.unsignedAddMayOverflow(RHSRange));
}

static OverflowResult
computeOverflowForSignedAdd(const WithCache<const Value *> &LHS,
                            const WithCache<const Value *> &RHS,
                            const AddOperator *Add, const SimplifyQuery &SQ) {
  if (Add && Add->hasNoSignedWrap()) {
    return OverflowResult::NeverOverflows;
  }

  // If LHS and RHS each have at least two sign bits, the addition will look
  // like
  //
  // XX..... +
  // YY.....
  //
  // If the carry into the most significant position is 0, X and Y can't both
  // be 1 and therefore the carry out of the addition is also 0.
  //
  // If the carry into the most significant position is 1, X and Y can't both
  // be 0 and therefore the carry out of the addition is also 1.
  //
  // Since the carry into the most significant position is always equal to
  // the carry out of the addition, there is no signed overflow.
  if (::ComputeNumSignBits(LHS, SQ) > 1 && ::ComputeNumSignBits(RHS, SQ) > 1)
    return OverflowResult::NeverOverflows;

  ConstantRange LHSRange =
      computeConstantRangeIncludingKnownBits(LHS, /*ForSigned=*/true, SQ);
  ConstantRange RHSRange =
      computeConstantRangeIncludingKnownBits(RHS, /*ForSigned=*/true, SQ);
  OverflowResult OR =
      mapOverflowResult(LHSRange.signedAddMayOverflow(RHSRange));
  if (OR != OverflowResult::MayOverflow)
    return OR;

  // The remaining code needs Add to be available. Early returns if not so.
  if (!Add)
    return OverflowResult::MayOverflow;

  // If the sign of Add is the same as at least one of the operands, this add
  // CANNOT overflow. If this can be determined from the known bits of the
  // operands the above signedAddMayOverflow() check will have already done so.
  // The only other way to improve on the known bits is from an assumption, so
  // call computeKnownBitsFromContext() directly.
  bool LHSOrRHSKnownNonNegative =
      (LHSRange.isAllNonNegative() || RHSRange.isAllNonNegative());
  bool LHSOrRHSKnownNegative =
      (LHSRange.isAllNegative() || RHSRange.isAllNegative());
  if (LHSOrRHSKnownNonNegative || LHSOrRHSKnownNegative) {
    KnownBits AddKnown(LHSRange.getBitWidth());
    computeKnownBitsFromContext(Add, AddKnown, SQ);
    if ((AddKnown.isNonNegative() && LHSOrRHSKnownNonNegative) ||
        (AddKnown.isNegative() && LHSOrRHSKnownNegative))
      return OverflowResult::NeverOverflows;
  }

  return OverflowResult::MayOverflow;
}

OverflowResult llvm::computeOverflowForUnsignedSub(const Value *LHS,
                                                   const Value *RHS,
                                                   const SimplifyQuery &SQ) {
  // X - (X % ?)
  // The remainder of a value can't have greater magnitude than itself,
  // so the subtraction can't overflow.

  // X - (X -nuw ?)
  // In the minimal case, this would simplify to "?", so there's no subtract
  // at all. But if this analysis is used to peek through casts, for example,
  // then determining no-overflow may allow other transforms.

  // TODO: There are other patterns like this.
  //       See simplifyICmpWithBinOpOnLHS() for candidates.
  if (match(RHS, m_URem(m_Specific(LHS), m_Value())) ||
      match(RHS, m_NUWSub(m_Specific(LHS), m_Value())))
    if (isGuaranteedNotToBeUndef(LHS, SQ.AC, SQ.CxtI, SQ.DT))
      return OverflowResult::NeverOverflows;

  if (auto C = isImpliedByDomCondition(CmpInst::ICMP_UGE, LHS, RHS, SQ.CxtI,
                                       SQ.DL)) {
    if (*C)
      return OverflowResult::NeverOverflows;
    return OverflowResult::AlwaysOverflowsLow;
  }

  ConstantRange LHSRange =
      computeConstantRangeIncludingKnownBits(LHS, /*ForSigned=*/false, SQ);
  ConstantRange RHSRange =
      computeConstantRangeIncludingKnownBits(RHS, /*ForSigned=*/false, SQ);
  return mapOverflowResult(LHSRange.unsignedSubMayOverflow(RHSRange));
}

OverflowResult llvm::computeOverflowForSignedSub(const Value *LHS,
                                                 const Value *RHS,
                                                 const SimplifyQuery &SQ) {
  // X - (X % ?)
  // The remainder of a value can't have greater magnitude than itself,
  // so the subtraction can't overflow.

  // X - (X -nsw ?)
  // In the minimal case, this would simplify to "?", so there's no subtract
  // at all. But if this analysis is used to peek through casts, for example,
  // then determining no-overflow may allow other transforms.
  if (match(RHS, m_SRem(m_Specific(LHS), m_Value())) ||
      match(RHS, m_NSWSub(m_Specific(LHS), m_Value())))
    if (isGuaranteedNotToBeUndef(LHS, SQ.AC, SQ.CxtI, SQ.DT))
      return OverflowResult::NeverOverflows;

  // If LHS and RHS each have at least two sign bits, the subtraction
  // cannot overflow.
  if (::ComputeNumSignBits(LHS, SQ) > 1 && ::ComputeNumSignBits(RHS, SQ) > 1)
    return OverflowResult::NeverOverflows;

  ConstantRange LHSRange =
      computeConstantRangeIncludingKnownBits(LHS, /*ForSigned=*/true, SQ);
  ConstantRange RHSRange =
      computeConstantRangeIncludingKnownBits(RHS, /*ForSigned=*/true, SQ);
  return mapOverflowResult(LHSRange.signedSubMayOverflow(RHSRange));
}

bool llvm::canCreateUndefOrPoison(const Operator *Op,
                                  bool ConsiderFlagsAndMetadata) {
  return ::canCreateUndefOrPoison(Op, UndefPoisonKind::UndefOrPoison,
                                  ConsiderFlagsAndMetadata);
}

bool llvm::canCreatePoison(const Operator *Op, bool ConsiderFlagsAndMetadata) {
  return ::canCreateUndefOrPoison(Op, UndefPoisonKind::PoisonOnly,
                                  ConsiderFlagsAndMetadata);
}

static bool directlyImpliesPoison(const Value *ValAssumedPoison, const Value *V,
                                  unsigned Depth) {
  if (ValAssumedPoison == V)
    return true;

  const unsigned MaxDepth = 2;
  if (Depth >= MaxDepth)
    return false;

  if (const auto *I = dyn_cast<Instruction>(V)) {
    if (any_of(I->operands(), [=](const Use &Op) {
          return propagatesPoison(Op) &&
                 directlyImpliesPoison(ValAssumedPoison, Op, Depth + 1);
        }))
      return true;

    // V  = extractvalue V0, idx
    // V2 = extractvalue V0, idx2
    // V0's elements are all poison or not. (e.g., add_with_overflow)
    const WithOverflowInst *II;
    if (match(I, m_ExtractValue(m_WithOverflowInst(II))) &&
        (match(ValAssumedPoison, m_ExtractValue(m_Specific(II))) ||
         llvm::is_contained(II->args(), ValAssumedPoison)))
      return true;
  }
  return false;
}

static bool impliesPoison(const Value *ValAssumedPoison, const Value *V,
                          unsigned Depth) {
  if (isGuaranteedNotToBePoison(ValAssumedPoison))
    return true;

  if (directlyImpliesPoison(ValAssumedPoison, V, /* Depth */ 0))
    return true;

  const unsigned MaxDepth = 2;
  if (Depth >= MaxDepth)
    return false;

  const auto *I = dyn_cast<Instruction>(ValAssumedPoison);
  if (I && !canCreatePoison(cast<Operator>(I))) {
    return all_of(I->operands(), [=](const Value *Op) {
      return impliesPoison(Op, V, Depth + 1);
    });
  }
  return false;
}

bool llvm::impliesPoison(const Value *ValAssumedPoison, const Value *V) {
  return ::impliesPoison(ValAssumedPoison, V, /* Depth */ 0);
}

static bool isGuaranteedNotToBeUndefOrPoison(
    const Value *V, AssumptionCache *AC, const Instruction *CtxI,
    const DominatorTree *DT, unsigned Depth, UndefPoisonKind Kind) {
  if (Depth >= MaxAnalysisRecursionDepth)
    return false;

  if (isa<MetadataAsValue>(V))
    return false;

  if (const auto *A = dyn_cast<Argument>(V)) {
    if (A->hasAttribute(Attribute::NoUndef) ||
        A->hasAttribute(Attribute::Dereferenceable) ||
        A->hasAttribute(Attribute::DereferenceableOrNull))
      return true;
  }

  if (auto *C = dyn_cast<Constant>(V)) {
    if (isa<PoisonValue>(C))
      return !includesPoison(Kind);

    if (isa<UndefValue>(C))
      return !includesUndef(Kind);

    if (isa<ConstantInt>(C) || isa<GlobalVariable>(C) || isa<ConstantFP>(C) ||
        isa<ConstantPointerNull>(C) || isa<Function>(C))
      return true;

    if (C->getType()->isVectorTy()) {
      if (isa<ConstantExpr>(C)) {
        // Scalable vectors can use a ConstantExpr to build a splat.
        if (Constant *SplatC = C->getSplatValue())
          if (isa<ConstantInt>(SplatC) || isa<ConstantFP>(SplatC))
            return true;
      } else {
        if (includesUndef(Kind) && C->containsUndefElement())
          return false;
        if (includesPoison(Kind) && C->containsPoisonElement())
          return false;
        return !C->containsConstantExpression();
      }
    }
  }

  // Strip cast operations from a pointer value.
  // Note that stripPointerCastsSameRepresentation can strip off getelementptr
  // inbounds with zero offset. To guarantee that the result isn't poison, the
  // stripped pointer is checked as it has to be pointing into an allocated
  // object or be null `null` to ensure `inbounds` getelement pointers with a
  // zero offset could not produce poison.
  // It can strip off addrspacecast that do not change bit representation as
  // well. We believe that such addrspacecast is equivalent to no-op.
  auto *StrippedV = V->stripPointerCastsSameRepresentation();
  if (isa<AllocaInst>(StrippedV) || isa<GlobalVariable>(StrippedV) ||
      isa<Function>(StrippedV) || isa<ConstantPointerNull>(StrippedV))
    return true;

  auto OpCheck = [&](const Value *V) {
    return isGuaranteedNotToBeUndefOrPoison(V, AC, CtxI, DT, Depth + 1, Kind);
  };

  if (auto *Opr = dyn_cast<Operator>(V)) {
    // If the value is a freeze instruction, then it can never
    // be undef or poison.
    if (isa<FreezeInst>(V))
      return true;

    if (const auto *CB = dyn_cast<CallBase>(V)) {
      if (CB->hasRetAttr(Attribute::NoUndef) ||
          CB->hasRetAttr(Attribute::Dereferenceable) ||
          CB->hasRetAttr(Attribute::DereferenceableOrNull))
        return true;
    }

    if (!::canCreateUndefOrPoison(Opr, Kind,
                                  /*ConsiderFlagsAndMetadata=*/true)) {
      if (const auto *PN = dyn_cast<PHINode>(V)) {
        unsigned Num = PN->getNumIncomingValues();
        bool IsWellDefined = true;
        for (unsigned i = 0; i < Num; ++i) {
          if (PN == PN->getIncomingValue(i))
            continue;
          auto *TI = PN->getIncomingBlock(i)->getTerminator();
          if (!isGuaranteedNotToBeUndefOrPoison(PN->getIncomingValue(i), AC, TI,
                                                DT, Depth + 1, Kind)) {
            IsWellDefined = false;
            break;
          }
        }
        if (IsWellDefined)
          return true;
      } else if (auto *Splat = isa<ShuffleVectorInst>(Opr) ? getSplatValue(Opr)
                                                           : nullptr) {
        // For splats we only need to check the value being splatted.
        if (OpCheck(Splat))
          return true;
      } else if (all_of(Opr->operands(), OpCheck))
        return true;
    }
  }

  if (auto *I = dyn_cast<LoadInst>(V))
    if (I->hasMetadata(LLVMContext::MD_noundef) ||
        I->hasMetadata(LLVMContext::MD_dereferenceable) ||
        I->hasMetadata(LLVMContext::MD_dereferenceable_or_null))
      return true;

  if (programUndefinedIfUndefOrPoison(V, !includesUndef(Kind)))
    return true;

  // CxtI may be null or a cloned instruction.
  if (!CtxI || !CtxI->getParent() || !DT)
    return false;

  auto *DNode = DT->getNode(CtxI->getParent());
  if (!DNode)
    // Unreachable block
    return false;

  // If V is used as a branch condition before reaching CtxI, V cannot be
  // undef or poison.
  //   br V, BB1, BB2
  // BB1:
  //   CtxI ; V cannot be undef or poison here
  auto *Dominator = DNode->getIDom();
  // This check is purely for compile time reasons: we can skip the IDom walk
  // if what we are checking for includes undef and the value is not an integer.
  if (!includesUndef(Kind) || V->getType()->isIntegerTy())
    while (Dominator) {
      auto *TI = Dominator->getBlock()->getTerminator();

      Value *Cond = nullptr;
      if (auto BI = dyn_cast_or_null<BranchInst>(TI)) {
        if (BI->isConditional())
          Cond = BI->getCondition();
      } else if (auto SI = dyn_cast_or_null<SwitchInst>(TI)) {
        Cond = SI->getCondition();
      }

      if (Cond) {
        if (Cond == V)
          return true;
        else if (!includesUndef(Kind) && isa<Operator>(Cond)) {
          // For poison, we can analyze further
          auto *Opr = cast<Operator>(Cond);
          if (any_of(Opr->operands(), [V](const Use &U) {
                return V == U && propagatesPoison(U);
              }))
            return true;
        }
      }

      Dominator = Dominator->getIDom();
    }

  if (AC && getKnowledgeValidInContext(V, {Attribute::NoUndef}, *AC, CtxI, DT))
    return true;

  return false;
}

bool llvm::isGuaranteedNotToBeUndefOrPoison(const Value *V, AssumptionCache *AC,
                                            const Instruction *CtxI,
                                            const DominatorTree *DT,
                                            unsigned Depth) {
  return ::isGuaranteedNotToBeUndefOrPoison(V, AC, CtxI, DT, Depth,
                                            UndefPoisonKind::UndefOrPoison);
}

bool llvm::isGuaranteedNotToBePoison(const Value *V, AssumptionCache *AC,
                                     const Instruction *CtxI,
                                     const DominatorTree *DT, unsigned Depth) {
  return ::isGuaranteedNotToBeUndefOrPoison(V, AC, CtxI, DT, Depth,
                                            UndefPoisonKind::PoisonOnly);
}

bool llvm::isGuaranteedNotToBeUndef(const Value *V, AssumptionCache *AC,
                                    const Instruction *CtxI,
                                    const DominatorTree *DT, unsigned Depth) {
  return ::isGuaranteedNotToBeUndefOrPoison(V, AC, CtxI, DT, Depth,
                                            UndefPoisonKind::UndefOnly);
}

OverflowResult llvm::computeOverflowForSignedAdd(const AddOperator *Add,
                                                 const SimplifyQuery &SQ) {
  return ::computeOverflowForSignedAdd(Add->getOperand(0), Add->getOperand(1),
                                       Add, SQ);
}

OverflowResult
llvm::computeOverflowForSignedAdd(const WithCache<const Value *> &LHS,
                                  const WithCache<const Value *> &RHS,
                                  const SimplifyQuery &SQ) {
  return ::computeOverflowForSignedAdd(LHS, RHS, nullptr, SQ);
}

bool llvm::programUndefinedIfUndefOrPoison(const Instruction *Inst) {
  return ::programUndefinedIfUndefOrPoison(Inst, false);
}

bool llvm::programUndefinedIfPoison(const Instruction *Inst) {
  return ::programUndefinedIfUndefOrPoison(Inst, true);
}

/// Match clamp pattern for float types without care about NaNs or signed zeros.
/// Given non-min/max outer cmp/select from the clamp pattern this
/// function recognizes if it can be substitued by a "canonical" min/max
/// pattern.
static SelectPatternResult matchFastFloatClamp(CmpInst::Predicate Pred,
                                               Value *CmpLHS, Value *CmpRHS,
                                               Value *TrueVal, Value *FalseVal,
                                               Value *&LHS, Value *&RHS) {
  // Try to match
  //   X < C1 ? C1 : Min(X, C2) --> Max(C1, Min(X, C2))
  //   X > C1 ? C1 : Max(X, C2) --> Min(C1, Max(X, C2))
  // and return description of the outer Max/Min.

  // First, check if select has inverse order:
  if (CmpRHS == FalseVal) {
    std::swap(TrueVal, FalseVal);
    Pred = CmpInst::getInversePredicate(Pred);
  }

  // Assume success now. If there's no match, callers should not use these anyway.
  LHS = TrueVal;
  RHS = FalseVal;

  const APFloat *FC1;
  if (CmpRHS != TrueVal || !match(CmpRHS, m_APFloat(FC1)) || !FC1->isFinite())
    return {SPF_UNKNOWN, SPNB_NA, false};

  const APFloat *FC2;
  switch (Pred) {
  case CmpInst::FCMP_OLT:
  case CmpInst::FCMP_OLE:
  case CmpInst::FCMP_ULT:
  case CmpInst::FCMP_ULE:
    if (match(FalseVal, m_OrdOrUnordFMin(m_Specific(CmpLHS), m_APFloat(FC2))) &&
        *FC1 < *FC2)
      return {SPF_FMAXNUM, SPNB_RETURNS_ANY, false};
    break;
  case CmpInst::FCMP_OGT:
  case CmpInst::FCMP_OGE:
  case CmpInst::FCMP_UGT:
  case CmpInst::FCMP_UGE:
    if (match(FalseVal, m_OrdOrUnordFMax(m_Specific(CmpLHS), m_APFloat(FC2))) &&
        *FC1 > *FC2)
      return {SPF_FMINNUM, SPNB_RETURNS_ANY, false};
    break;
  default:
    break;
  }

  return {SPF_UNKNOWN, SPNB_NA, false};
}

/// Recognize variations of:
///   CLAMP(v,l,h) ==> ((v) < (l) ? (l) : ((v) > (h) ? (h) : (v)))
static SelectPatternResult matchClamp(CmpInst::Predicate Pred,
                                      Value *CmpLHS, Value *CmpRHS,
                                      Value *TrueVal, Value *FalseVal) {
  // Swap the select operands and predicate to match the patterns below.
  if (CmpRHS != TrueVal) {
    Pred = ICmpInst::getSwappedPredicate(Pred);
    std::swap(TrueVal, FalseVal);
  }
  const APInt *C1;
  if (CmpRHS == TrueVal && match(CmpRHS, m_APInt(C1))) {
    const APInt *C2;
    // (X <s C1) ? C1 : SMIN(X, C2) ==> SMAX(SMIN(X, C2), C1)
    if (match(FalseVal, m_SMin(m_Specific(CmpLHS), m_APInt(C2))) &&
        C1->slt(*C2) && Pred == CmpInst::ICMP_SLT)
      return {SPF_SMAX, SPNB_NA, false};

    // (X >s C1) ? C1 : SMAX(X, C2) ==> SMIN(SMAX(X, C2), C1)
    if (match(FalseVal, m_SMax(m_Specific(CmpLHS), m_APInt(C2))) &&
        C1->sgt(*C2) && Pred == CmpInst::ICMP_SGT)
      return {SPF_SMIN, SPNB_NA, false};

    // (X <u C1) ? C1 : UMIN(X, C2) ==> UMAX(UMIN(X, C2), C1)
    if (match(FalseVal, m_UMin(m_Specific(CmpLHS), m_APInt(C2))) &&
        C1->ult(*C2) && Pred == CmpInst::ICMP_ULT)
      return {SPF_UMAX, SPNB_NA, false};

    // (X >u C1) ? C1 : UMAX(X, C2) ==> UMIN(UMAX(X, C2), C1)
    if (match(FalseVal, m_UMax(m_Specific(CmpLHS), m_APInt(C2))) &&
        C1->ugt(*C2) && Pred == CmpInst::ICMP_UGT)
      return {SPF_UMIN, SPNB_NA, false};
  }
  return {SPF_UNKNOWN, SPNB_NA, false};
}

/// Recognize variations of:
///   a < c ? min(a,b) : min(b,c) ==> min(min(a,b),min(b,c))
static SelectPatternResult matchMinMaxOfMinMax(CmpInst::Predicate Pred,
                                               Value *CmpLHS, Value *CmpRHS,
                                               Value *TVal, Value *FVal,
                                               unsigned Depth) {
  // TODO: Allow FP min/max with nnan/nsz.
  assert(CmpInst::isIntPredicate(Pred) && "Expected integer comparison");

  Value *A = nullptr, *B = nullptr;
  SelectPatternResult L = matchSelectPattern(TVal, A, B, nullptr, Depth + 1);
  if (!SelectPatternResult::isMinOrMax(L.Flavor))
    return {SPF_UNKNOWN, SPNB_NA, false};

  Value *C = nullptr, *D = nullptr;
  SelectPatternResult R = matchSelectPattern(FVal, C, D, nullptr, Depth + 1);
  if (L.Flavor != R.Flavor)
    return {SPF_UNKNOWN, SPNB_NA, false};

  // We have something like: x Pred y ? min(a, b) : min(c, d).
  // Try to match the compare to the min/max operations of the select operands.
  // First, make sure we have the right compare predicate.
  switch (L.Flavor) {
  case SPF_SMIN:
    if (Pred == ICmpInst::ICMP_SGT || Pred == ICmpInst::ICMP_SGE) {
      Pred = ICmpInst::getSwappedPredicate(Pred);
      std::swap(CmpLHS, CmpRHS);
    }
    if (Pred == ICmpInst::ICMP_SLT || Pred == ICmpInst::ICMP_SLE)
      break;
    return {SPF_UNKNOWN, SPNB_NA, false};
  case SPF_SMAX:
    if (Pred == ICmpInst::ICMP_SLT || Pred == ICmpInst::ICMP_SLE) {
      Pred = ICmpInst::getSwappedPredicate(Pred);
      std::swap(CmpLHS, CmpRHS);
    }
    if (Pred == ICmpInst::ICMP_SGT || Pred == ICmpInst::ICMP_SGE)
      break;
    return {SPF_UNKNOWN, SPNB_NA, false};
  case SPF_UMIN:
    if (Pred == ICmpInst::ICMP_UGT || Pred == ICmpInst::ICMP_UGE) {
      Pred = ICmpInst::getSwappedPredicate(Pred);
      std::swap(CmpLHS, CmpRHS);
    }
    if (Pred == ICmpInst::ICMP_ULT || Pred == ICmpInst::ICMP_ULE)
      break;
    return {SPF_UNKNOWN, SPNB_NA, false};
  case SPF_UMAX:
    if (Pred == ICmpInst::ICMP_ULT || Pred == ICmpInst::ICMP_ULE) {
      Pred = ICmpInst::getSwappedPredicate(Pred);
      std::swap(CmpLHS, CmpRHS);
    }
    if (Pred == ICmpInst::ICMP_UGT || Pred == ICmpInst::ICMP_UGE)
      break;
    return {SPF_UNKNOWN, SPNB_NA, false};
  default:
    return {SPF_UNKNOWN, SPNB_NA, false};
  }

  // If there is a common operand in the already matched min/max and the other
  // min/max operands match the compare operands (either directly or inverted),
  // then this is min/max of the same flavor.

  // a pred c ? m(a, b) : m(c, b) --> m(m(a, b), m(c, b))
  // ~c pred ~a ? m(a, b) : m(c, b) --> m(m(a, b), m(c, b))
  if (D == B) {
    if ((CmpLHS == A && CmpRHS == C) || (match(C, m_Not(m_Specific(CmpLHS))) &&
                                         match(A, m_Not(m_Specific(CmpRHS)))))
      return {L.Flavor, SPNB_NA, false};
  }
  // a pred d ? m(a, b) : m(b, d) --> m(m(a, b), m(b, d))
  // ~d pred ~a ? m(a, b) : m(b, d) --> m(m(a, b), m(b, d))
  if (C == B) {
    if ((CmpLHS == A && CmpRHS == D) || (match(D, m_Not(m_Specific(CmpLHS))) &&
                                         match(A, m_Not(m_Specific(CmpRHS)))))
      return {L.Flavor, SPNB_NA, false};
  }
  // b pred c ? m(a, b) : m(c, a) --> m(m(a, b), m(c, a))
  // ~c pred ~b ? m(a, b) : m(c, a) --> m(m(a, b), m(c, a))
  if (D == A) {
    if ((CmpLHS == B && CmpRHS == C) || (match(C, m_Not(m_Specific(CmpLHS))) &&
                                         match(B, m_Not(m_Specific(CmpRHS)))))
      return {L.Flavor, SPNB_NA, false};
  }
  // b pred d ? m(a, b) : m(a, d) --> m(m(a, b), m(a, d))
  // ~d pred ~b ? m(a, b) : m(a, d) --> m(m(a, b), m(a, d))
  if (C == A) {
    if ((CmpLHS == B && CmpRHS == D) || (match(D, m_Not(m_Specific(CmpLHS))) &&
                                         match(B, m_Not(m_Specific(CmpRHS)))))
      return {L.Flavor, SPNB_NA, false};
  }

  return {SPF_UNKNOWN, SPNB_NA, false};
}

/// If the input value is the result of a 'not' op, constant integer, or vector
/// splat of a constant integer, return the bitwise-not source value.
/// TODO: This could be extended to handle non-splat vector integer constants.
static Value *getNotValue(Value *V) {
  Value *NotV;
  if (match(V, m_Not(m_Value(NotV))))
    return NotV;

  const APInt *C;
  if (match(V, m_APInt(C)))
    return ConstantInt::get(V->getType(), ~(*C));

  return nullptr;
}

/// Match non-obvious integer minimum and maximum sequences.
static SelectPatternResult matchMinMax(CmpInst::Predicate Pred,
                                       Value *CmpLHS, Value *CmpRHS,
                                       Value *TrueVal, Value *FalseVal,
                                       Value *&LHS, Value *&RHS,
                                       unsigned Depth) {
  // Assume success. If there's no match, callers should not use these anyway.
  LHS = TrueVal;
  RHS = FalseVal;

  SelectPatternResult SPR = matchClamp(Pred, CmpLHS, CmpRHS, TrueVal, FalseVal);
  if (SPR.Flavor != SelectPatternFlavor::SPF_UNKNOWN)
    return SPR;

  SPR = matchMinMaxOfMinMax(Pred, CmpLHS, CmpRHS, TrueVal, FalseVal, Depth);
  if (SPR.Flavor != SelectPatternFlavor::SPF_UNKNOWN)
    return SPR;

  // Look through 'not' ops to find disguised min/max.
  // (X > Y) ? ~X : ~Y ==> (~X < ~Y) ? ~X : ~Y ==> MIN(~X, ~Y)
  // (X < Y) ? ~X : ~Y ==> (~X > ~Y) ? ~X : ~Y ==> MAX(~X, ~Y)
  if (CmpLHS == getNotValue(TrueVal) && CmpRHS == getNotValue(FalseVal)) {
    switch (Pred) {
    case CmpInst::ICMP_SGT: return {SPF_SMIN, SPNB_NA, false};
    case CmpInst::ICMP_SLT: return {SPF_SMAX, SPNB_NA, false};
    case CmpInst::ICMP_UGT: return {SPF_UMIN, SPNB_NA, false};
    case CmpInst::ICMP_ULT: return {SPF_UMAX, SPNB_NA, false};
    default: break;
    }
  }

  // (X > Y) ? ~Y : ~X ==> (~X < ~Y) ? ~Y : ~X ==> MAX(~Y, ~X)
  // (X < Y) ? ~Y : ~X ==> (~X > ~Y) ? ~Y : ~X ==> MIN(~Y, ~X)
  if (CmpLHS == getNotValue(FalseVal) && CmpRHS == getNotValue(TrueVal)) {
    switch (Pred) {
    case CmpInst::ICMP_SGT: return {SPF_SMAX, SPNB_NA, false};
    case CmpInst::ICMP_SLT: return {SPF_SMIN, SPNB_NA, false};
    case CmpInst::ICMP_UGT: return {SPF_UMAX, SPNB_NA, false};
    case CmpInst::ICMP_ULT: return {SPF_UMIN, SPNB_NA, false};
    default: break;
    }
  }

  if (Pred != CmpInst::ICMP_SGT && Pred != CmpInst::ICMP_SLT)
    return {SPF_UNKNOWN, SPNB_NA, false};

  const APInt *C1;
  if (!match(CmpRHS, m_APInt(C1)))
    return {SPF_UNKNOWN, SPNB_NA, false};

  // An unsigned min/max can be written with a signed compare.
  const APInt *C2;
  if ((CmpLHS == TrueVal && match(FalseVal, m_APInt(C2))) ||
      (CmpLHS == FalseVal && match(TrueVal, m_APInt(C2)))) {
    // Is the sign bit set?
    // (X <s 0) ? X : MAXVAL ==> (X >u MAXVAL) ? X : MAXVAL ==> UMAX
    // (X <s 0) ? MAXVAL : X ==> (X >u MAXVAL) ? MAXVAL : X ==> UMIN
    if (Pred == CmpInst::ICMP_SLT && C1->isZero() && C2->isMaxSignedValue())
      return {CmpLHS == TrueVal ? SPF_UMAX : SPF_UMIN, SPNB_NA, false};

    // Is the sign bit clear?
    // (X >s -1) ? MINVAL : X ==> (X <u MINVAL) ? MINVAL : X ==> UMAX
    // (X >s -1) ? X : MINVAL ==> (X <u MINVAL) ? X : MINVAL ==> UMIN
    if (Pred == CmpInst::ICMP_SGT && C1->isAllOnes() && C2->isMinSignedValue())
      return {CmpLHS == FalseVal ? SPF_UMAX : SPF_UMIN, SPNB_NA, false};
  }

  return {SPF_UNKNOWN, SPNB_NA, false};
}

static SelectPatternResult matchSelectPattern(CmpInst::Predicate Pred,
                                              FastMathFlags FMF,
                                              Value *CmpLHS, Value *CmpRHS,
                                              Value *TrueVal, Value *FalseVal,
                                              Value *&LHS, Value *&RHS,
                                              unsigned Depth) {
  bool HasMismatchedZeros = false;
  if (CmpInst::isFPPredicate(Pred)) {
    // IEEE-754 ignores the sign of 0.0 in comparisons. So if the select has one
    // 0.0 operand, set the compare's 0.0 operands to that same value for the
    // purpose of identifying min/max. Disregard vector constants with undefined
    // elements because those can not be back-propagated for analysis.
    Value *OutputZeroVal = nullptr;
    if (match(TrueVal, m_AnyZeroFP()) && !match(FalseVal, m_AnyZeroFP()) &&
        !cast<Constant>(TrueVal)->containsUndefOrPoisonElement())
      OutputZeroVal = TrueVal;
    else if (match(FalseVal, m_AnyZeroFP()) && !match(TrueVal, m_AnyZeroFP()) &&
             !cast<Constant>(FalseVal)->containsUndefOrPoisonElement())
      OutputZeroVal = FalseVal;

    if (OutputZeroVal) {
      if (match(CmpLHS, m_AnyZeroFP()) && CmpLHS != OutputZeroVal) {
        HasMismatchedZeros = true;
        CmpLHS = OutputZeroVal;
      }
      if (match(CmpRHS, m_AnyZeroFP()) && CmpRHS != OutputZeroVal) {
        HasMismatchedZeros = true;
        CmpRHS = OutputZeroVal;
      }
    }
  }

  LHS = CmpLHS;
  RHS = CmpRHS;

  // Signed zero may return inconsistent results between implementations.
  //  (0.0 <= -0.0) ? 0.0 : -0.0 // Returns 0.0
  //  minNum(0.0, -0.0)          // May return -0.0 or 0.0 (IEEE 754-2008 5.3.1)
  // Therefore, we behave conservatively and only proceed if at least one of the
  // operands is known to not be zero or if we don't care about signed zero.
  switch (Pred) {
  default: break;
  case CmpInst::FCMP_OGT: case CmpInst::FCMP_OLT:
  case CmpInst::FCMP_UGT: case CmpInst::FCMP_ULT:
    if (!HasMismatchedZeros)
      break;
    [[fallthrough]];
  case CmpInst::FCMP_OGE: case CmpInst::FCMP_OLE:
  case CmpInst::FCMP_UGE: case CmpInst::FCMP_ULE:
    if (!FMF.noSignedZeros() && !isKnownNonZero(CmpLHS) &&
        !isKnownNonZero(CmpRHS))
      return {SPF_UNKNOWN, SPNB_NA, false};
  }

  SelectPatternNaNBehavior NaNBehavior = SPNB_NA;
  bool Ordered = false;

  // When given one NaN and one non-NaN input:
  //   - maxnum/minnum (C99 fmaxf()/fminf()) return the non-NaN input.
  //   - A simple C99 (a < b ? a : b) construction will return 'b' (as the
  //     ordered comparison fails), which could be NaN or non-NaN.
  // so here we discover exactly what NaN behavior is required/accepted.
  if (CmpInst::isFPPredicate(Pred)) {
    bool LHSSafe = isKnownNonNaN(CmpLHS, FMF);
    bool RHSSafe = isKnownNonNaN(CmpRHS, FMF);

    if (LHSSafe && RHSSafe) {
      // Both operands are known non-NaN.
      NaNBehavior = SPNB_RETURNS_ANY;
      Ordered = CmpInst::isOrdered(Pred);
    } else if (CmpInst::isOrdered(Pred)) {
      // An ordered comparison will return false when given a NaN, so it
      // returns the RHS.
      Ordered = true;
      if (LHSSafe)
        // LHS is non-NaN, so if RHS is NaN then NaN will be returned.
        NaNBehavior = SPNB_RETURNS_NAN;
      else if (RHSSafe)
        NaNBehavior = SPNB_RETURNS_OTHER;
      else
        // Completely unsafe.
        return {SPF_UNKNOWN, SPNB_NA, false};
    } else {
      Ordered = false;
      // An unordered comparison will return true when given a NaN, so it
      // returns the LHS.
      if (LHSSafe)
        // LHS is non-NaN, so if RHS is NaN then non-NaN will be returned.
        NaNBehavior = SPNB_RETURNS_OTHER;
      else if (RHSSafe)
        NaNBehavior = SPNB_RETURNS_NAN;
      else
        // Completely unsafe.
        return {SPF_UNKNOWN, SPNB_NA, false};
    }
  }

  if (TrueVal == CmpRHS && FalseVal == CmpLHS) {
    std::swap(CmpLHS, CmpRHS);
    Pred = CmpInst::getSwappedPredicate(Pred);
    if (NaNBehavior == SPNB_RETURNS_NAN)
      NaNBehavior = SPNB_RETURNS_OTHER;
    else if (NaNBehavior == SPNB_RETURNS_OTHER)
      NaNBehavior = SPNB_RETURNS_NAN;
    Ordered = !Ordered;
  }

  // ([if]cmp X, Y) ? X : Y
  if (TrueVal == CmpLHS && FalseVal == CmpRHS)
    return getSelectPattern(Pred, NaNBehavior, Ordered);

  if (isKnownNegation(TrueVal, FalseVal)) {
    // Sign-extending LHS does not change its sign, so TrueVal/FalseVal can
    // match against either LHS or sext(LHS).
    auto MaybeSExtCmpLHS =
        m_CombineOr(m_Specific(CmpLHS), m_SExt(m_Specific(CmpLHS)));
    auto ZeroOrAllOnes = m_CombineOr(m_ZeroInt(), m_AllOnes());
    auto ZeroOrOne = m_CombineOr(m_ZeroInt(), m_One());
    if (match(TrueVal, MaybeSExtCmpLHS)) {
      // Set the return values. If the compare uses the negated value (-X >s 0),
      // swap the return values because the negated value is always 'RHS'.
      LHS = TrueVal;
      RHS = FalseVal;
      if (match(CmpLHS, m_Neg(m_Specific(FalseVal))))
        std::swap(LHS, RHS);

      // (X >s 0) ? X : -X or (X >s -1) ? X : -X --> ABS(X)
      // (-X >s 0) ? -X : X or (-X >s -1) ? -X : X --> ABS(X)
      if (Pred == ICmpInst::ICMP_SGT && match(CmpRHS, ZeroOrAllOnes))
        return {SPF_ABS, SPNB_NA, false};

      // (X >=s 0) ? X : -X or (X >=s 1) ? X : -X --> ABS(X)
      if (Pred == ICmpInst::ICMP_SGE && match(CmpRHS, ZeroOrOne))
        return {SPF_ABS, SPNB_NA, false};

      // (X <s 0) ? X : -X or (X <s 1) ? X : -X --> NABS(X)
      // (-X <s 0) ? -X : X or (-X <s 1) ? -X : X --> NABS(X)
      if (Pred == ICmpInst::ICMP_SLT && match(CmpRHS, ZeroOrOne))
        return {SPF_NABS, SPNB_NA, false};
    }
    else if (match(FalseVal, MaybeSExtCmpLHS)) {
      // Set the return values. If the compare uses the negated value (-X >s 0),
      // swap the return values because the negated value is always 'RHS'.
      LHS = FalseVal;
      RHS = TrueVal;
      if (match(CmpLHS, m_Neg(m_Specific(TrueVal))))
        std::swap(LHS, RHS);

      // (X >s 0) ? -X : X or (X >s -1) ? -X : X --> NABS(X)
      // (-X >s 0) ? X : -X or (-X >s -1) ? X : -X --> NABS(X)
      if (Pred == ICmpInst::ICMP_SGT && match(CmpRHS, ZeroOrAllOnes))
        return {SPF_NABS, SPNB_NA, false};

      // (X <s 0) ? -X : X or (X <s 1) ? -X : X --> ABS(X)
      // (-X <s 0) ? X : -X or (-X <s 1) ? X : -X --> ABS(X)
      if (Pred == ICmpInst::ICMP_SLT && match(CmpRHS, ZeroOrOne))
        return {SPF_ABS, SPNB_NA, false};
    }
  }

  if (CmpInst::isIntPredicate(Pred))
    return matchMinMax(Pred, CmpLHS, CmpRHS, TrueVal, FalseVal, LHS, RHS,
                       Depth);

  // According to (IEEE 754-2008 5.3.1), minNum(0.0, -0.0) and similar
  // may return either -0.0 or 0.0, so fcmp/select pair has stricter
  // semantics than minNum. Be conservative in such case.
  if (NaNBehavior != SPNB_RETURNS_ANY ||
      (!FMF.noSignedZeros() && !isKnownNonZero(CmpLHS) &&
       !isKnownNonZero(CmpRHS)))
    return {SPF_UNKNOWN, SPNB_NA, false};

  return matchFastFloatClamp(Pred, CmpLHS, CmpRHS, TrueVal, FalseVal, LHS, RHS);
}

SelectPatternResult llvm::matchSelectPattern(Value *V, Value *&LHS, Value *&RHS,
                                             Instruction::CastOps *CastOp,
                                             unsigned Depth) {
  if (Depth >= MaxAnalysisRecursionDepth)
    return {SPF_UNKNOWN, SPNB_NA, false};

  SelectInst *SI = dyn_cast<SelectInst>(V);
  if (!SI)
    return {SPF_UNKNOWN, SPNB_NA, false};

  CmpInst *CmpI = dyn_cast<CmpInst>(SI->getCondition());
  if (!CmpI)
    return {SPF_UNKNOWN, SPNB_NA, false};

  Value *TrueVal = SI->getTrueValue();
  Value *FalseVal = SI->getFalseValue();

  return llvm::matchDecomposedSelectPattern(
      CmpI, TrueVal, FalseVal, LHS, RHS,
      isa<FPMathOperator>(SI) ? SI->getFastMathFlags() : FastMathFlags(),
      CastOp, Depth);
}

SelectPatternResult llvm::matchDecomposedSelectPattern(
    CmpInst *CmpI, Value *TrueVal, Value *FalseVal, Value *&LHS, Value *&RHS,
    FastMathFlags FMF, Instruction::CastOps *CastOp, unsigned Depth) {
  CmpInst::Predicate Pred = CmpI->getPredicate();
  Value *CmpLHS = CmpI->getOperand(0);
  Value *CmpRHS = CmpI->getOperand(1);
  if (isa<FPMathOperator>(CmpI) && CmpI->hasNoNaNs())
    FMF.setNoNaNs();

  // Bail out early.
  if (CmpI->isEquality())
    return {SPF_UNKNOWN, SPNB_NA, false};

  // Deal with type mismatches.
  if (CastOp && CmpLHS->getType() != TrueVal->getType()) {
    if (Value *C = lookThroughCast(CmpI, TrueVal, FalseVal, CastOp)) {
      // If this is a potential fmin/fmax with a cast to integer, then ignore
      // -0.0 because there is no corresponding integer value.
      if (*CastOp == Instruction::FPToSI || *CastOp == Instruction::FPToUI)
        FMF.setNoSignedZeros();
      return ::matchSelectPattern(Pred, FMF, CmpLHS, CmpRHS,
                                  cast<CastInst>(TrueVal)->getOperand(0), C,
                                  LHS, RHS, Depth);
    }
    if (Value *C = lookThroughCast(CmpI, FalseVal, TrueVal, CastOp)) {
      // If this is a potential fmin/fmax with a cast to integer, then ignore
      // -0.0 because there is no corresponding integer value.
      if (*CastOp == Instruction::FPToSI || *CastOp == Instruction::FPToUI)
        FMF.setNoSignedZeros();
      return ::matchSelectPattern(Pred, FMF, CmpLHS, CmpRHS, C,
                                  cast<CastInst>(FalseVal)->getOperand(0), LHS,
                                  RHS, Depth);
    }
  }
  return ::matchSelectPattern(Pred, FMF, CmpLHS, CmpRHS, TrueVal, FalseVal, LHS,
                              RHS, Depth);
}
/// Return true if LHS implies RHS (expanded to its components as "R0 RPred R1")
/// is true.  Return false if LHS implies RHS is false. Otherwise, return
/// std::nullopt if we can't infer anything.
static std::optional<bool>
isImpliedCondICmps(CmpPredicate LPred, const Value *L0, const Value *L1,
                   CmpPredicate RPred, const Value *R0, const Value *R1,
                   const DataLayout &DL, bool LHSIsTrue) {
  // The rest of the logic assumes the LHS condition is true.  If that's not the
  // case, invert the predicate to make it so.
  if (!LHSIsTrue)
    LPred = ICmpInst::getInverseCmpPredicate(LPred);

  // We can have non-canonical operands, so try to normalize any common operand
  // to L0/R0.
  if (L0 == R1) {
    std::swap(R0, R1);
    RPred = ICmpInst::getSwappedCmpPredicate(RPred);
  }
  if (R0 == L1) {
    std::swap(L0, L1);
    LPred = ICmpInst::getSwappedCmpPredicate(LPred);
  }
  if (L1 == R1) {
    // If we have L0 == R0 and L1 == R1, then make L1/R1 the constants.
    if (L0 != R0 || match(L0, m_ImmConstant())) {
      std::swap(L0, L1);
      LPred = ICmpInst::getSwappedCmpPredicate(LPred);
      std::swap(R0, R1);
      RPred = ICmpInst::getSwappedCmpPredicate(RPred);
    }
  }

  // See if we can infer anything if operand-0 matches and we have at least one
  // constant.
  const APInt *Unused;
  if (L0 == R0 && (match(L1, m_APInt(Unused)) || match(R1, m_APInt(Unused)))) {
    // Potential TODO: We could also further use the constant range of L0/R0 to
    // further constraint the constant ranges. At the moment this leads to
    // several regressions related to not transforming `multi_use(A + C0) eq/ne
    // C1` (see discussion: D58633).
    ConstantRange LCR = computeConstantRange(
        L1, ICmpInst::isSigned(LPred), /* UseInstrInfo=*/true, /*AC=*/nullptr,
        /*CxtI=*/nullptr, /*DT=*/nullptr, MaxAnalysisRecursionDepth - 1);
    ConstantRange RCR = computeConstantRange(
        R1, ICmpInst::isSigned(RPred), /* UseInstrInfo=*/true, /*AC=*/nullptr,
        /*CxtI=*/nullptr, /*DT=*/nullptr, MaxAnalysisRecursionDepth - 1);
    // Even if L1/R1 are not both constant, we can still sometimes deduce
    // relationship from a single constant. For example X u> Y implies X != 0.
    if (auto R = isImpliedCondCommonOperandWithCR(LPred, LCR, RPred, RCR))
      return R;
    // If both L1/R1 were exact constant ranges and we didn't get anything
    // here, we won't be able to deduce this.
    if (match(L1, m_APInt(Unused)) && match(R1, m_APInt(Unused)))
      return std::nullopt;
  }

  // Can we infer anything when the two compares have matching operands?
  if (L0 == R0 && L1 == R1)
    return ICmpInst::isImpliedByMatchingCmp(LPred, RPred);

  // It only really makes sense in the context of signed comparison for "X - Y
  // must be positive if X >= Y and no overflow".
  // Take SGT as an example:  L0:x > L1:y and C >= 0
  //                      ==> R0:(x -nsw y) < R1:(-C) is false
  CmpInst::Predicate SignedLPred = LPred.getPreferredSignedPredicate();
  if ((SignedLPred == ICmpInst::ICMP_SGT ||
       SignedLPred == ICmpInst::ICMP_SGE) &&
      match(R0, m_NSWSub(m_Specific(L0), m_Specific(L1)))) {
    if (match(R1, m_NonPositive()) &&
        ICmpInst::isImpliedByMatchingCmp(SignedLPred, RPred) == false)
      return false;
  }

  // Take SLT as an example:  L0:x < L1:y and C <= 0
  //                      ==> R0:(x -nsw y) < R1:(-C) is true
  if ((SignedLPred == ICmpInst::ICMP_SLT ||
       SignedLPred == ICmpInst::ICMP_SLE) &&
      match(R0, m_NSWSub(m_Specific(L0), m_Specific(L1)))) {
    if (match(R1, m_NonNegative()) &&
        ICmpInst::isImpliedByMatchingCmp(SignedLPred, RPred) == true)
      return true;
  }

  // a - b == NonZero -> a != b
  // ptrtoint(a) - ptrtoint(b) == NonZero -> a != b
  const APInt *L1C;
  Value *A, *B;
  if (LPred == ICmpInst::ICMP_EQ && ICmpInst::isEquality(RPred) &&
      match(L1, m_APInt(L1C)) && !L1C->isZero() &&
      match(L0, m_Sub(m_Value(A), m_Value(B))) &&
      ((A == R0 && B == R1) || (A == R1 && B == R0) ||
       (match(A, m_PtrToIntOrAddr(m_Specific(R0))) &&
        match(B, m_PtrToIntOrAddr(m_Specific(R1)))) ||
       (match(A, m_PtrToIntOrAddr(m_Specific(R1))) &&
        match(B, m_PtrToIntOrAddr(m_Specific(R0)))))) {
    return RPred.dropSameSign() == ICmpInst::ICMP_NE;
  }

  // L0 = R0 = L1 + R1, L0 >=u L1 implies R0 >=u R1, L0 <u L1 implies R0 <u R1
  if (L0 == R0 &&
      (LPred == ICmpInst::ICMP_ULT || LPred == ICmpInst::ICMP_UGE) &&
      (RPred == ICmpInst::ICMP_ULT || RPred == ICmpInst::ICMP_UGE) &&
      match(L0, m_c_Add(m_Specific(L1), m_Specific(R1))))
    return CmpPredicate::getMatching(LPred, RPred).has_value();

  if (auto P = CmpPredicate::getMatching(LPred, RPred))
    return isImpliedCondOperands(*P, L0, L1, R0, R1);

  return std::nullopt;
}

/// Return true if LHS implies RHS (expanded to its components as "R0 RPred R1")
/// is true.  Return false if LHS implies RHS is false. Otherwise, return
/// std::nullopt if we can't infer anything.
static std::optional<bool>
isImpliedCondFCmps(FCmpInst::Predicate LPred, const Value *L0, const Value *L1,
                   FCmpInst::Predicate RPred, const Value *R0, const Value *R1,
                   const DataLayout &DL, bool LHSIsTrue) {
  // The rest of the logic assumes the LHS condition is true.  If that's not the
  // case, invert the predicate to make it so.
  if (!LHSIsTrue)
    LPred = FCmpInst::getInversePredicate(LPred);

  // We can have non-canonical operands, so try to normalize any common operand
  // to L0/R0.
  if (L0 == R1) {
    std::swap(R0, R1);
    RPred = FCmpInst::getSwappedPredicate(RPred);
  }
  if (R0 == L1) {
    std::swap(L0, L1);
    LPred = FCmpInst::getSwappedPredicate(LPred);
  }
  if (L1 == R1) {
    // If we have L0 == R0 and L1 == R1, then make L1/R1 the constants.
    if (L0 != R0 || match(L0, m_ImmConstant())) {
      std::swap(L0, L1);
      LPred = ICmpInst::getSwappedCmpPredicate(LPred);
      std::swap(R0, R1);
      RPred = ICmpInst::getSwappedCmpPredicate(RPred);
    }
  }

  // Can we infer anything when the two compares have matching operands?
  if (L0 == R0 && L1 == R1) {
    if ((LPred & RPred) == LPred)
      return true;
    if ((LPred & ~RPred) == LPred)
      return false;
  }

  // See if we can infer anything if operand-0 matches and we have at least one
  // constant.
  const APFloat *L1C, *R1C;
  if (L0 == R0 && match(L1, m_APFloat(L1C)) && match(R1, m_APFloat(R1C))) {
    if (std::optional<ConstantFPRange> DomCR =
            ConstantFPRange::makeExactFCmpRegion(LPred, *L1C)) {
      if (std::optional<ConstantFPRange> ImpliedCR =
              ConstantFPRange::makeExactFCmpRegion(RPred, *R1C)) {
        if (ImpliedCR->contains(*DomCR))
          return true;
      }
      if (std::optional<ConstantFPRange> ImpliedCR =
              ConstantFPRange::makeExactFCmpRegion(
                  FCmpInst::getInversePredicate(RPred), *R1C)) {
        if (ImpliedCR->contains(*DomCR))
          return false;
      }
    }
  }

  return std::nullopt;
}

/// Return true if LHS implies RHS is true.  Return false if LHS implies RHS is
/// false.  Otherwise, return std::nullopt if we can't infer anything.  We
/// expect the RHS to be an icmp and the LHS to be an 'and', 'or', or a 'select'
/// instruction.
static std::optional<bool>
isImpliedCondAndOr(const Instruction *LHS, CmpPredicate RHSPred,
                   const Value *RHSOp0, const Value *RHSOp1,
                   const DataLayout &DL, bool LHSIsTrue, unsigned Depth) {
  // The LHS must be an 'or', 'and', or a 'select' instruction.
  assert((LHS->getOpcode() == Instruction::And ||
          LHS->getOpcode() == Instruction::Or ||
          LHS->getOpcode() == Instruction::Select) &&
         "Expected LHS to be 'and', 'or', or 'select'.");

  assert(Depth <= MaxAnalysisRecursionDepth && "Hit recursion limit");

  // If the result of an 'or' is false, then we know both legs of the 'or' are
  // false.  Similarly, if the result of an 'and' is true, then we know both
  // legs of the 'and' are true.
  const Value *ALHS, *ARHS;
  if ((!LHSIsTrue && match(LHS, m_LogicalOr(m_Value(ALHS), m_Value(ARHS)))) ||
      (LHSIsTrue && match(LHS, m_LogicalAnd(m_Value(ALHS), m_Value(ARHS))))) {
    // FIXME: Make this non-recursion.
    if (std::optional<bool> Implication = isImpliedCondition(
            ALHS, RHSPred, RHSOp0, RHSOp1, DL, LHSIsTrue, Depth + 1))
      return Implication;
    if (std::optional<bool> Implication = isImpliedCondition(
            ARHS, RHSPred, RHSOp0, RHSOp1, DL, LHSIsTrue, Depth + 1))
      return Implication;
    return std::nullopt;
  }
  return std::nullopt;
}

std::optional<bool>
llvm::isImpliedCondition(const Value *LHS, CmpPredicate RHSPred,
                         const Value *RHSOp0, const Value *RHSOp1,
                         const DataLayout &DL, bool LHSIsTrue, unsigned Depth) {
  // Bail out when we hit the limit.
  if (Depth == MaxAnalysisRecursionDepth)
    return std::nullopt;

  // A mismatch occurs when we compare a scalar cmp to a vector cmp, for
  // example.
  if (RHSOp0->getType()->isVectorTy() != LHS->getType()->isVectorTy())
    return std::nullopt;

  assert(LHS->getType()->isIntOrIntVectorTy(1) &&
         "Expected integer type only!");

  // Match not
  if (match(LHS, m_Not(m_Value(LHS))))
    LHSIsTrue = !LHSIsTrue;

  // Both LHS and RHS are icmps.
  if (RHSOp0->getType()->getScalarType()->isIntOrPtrTy()) {
    if (const auto *LHSCmp = dyn_cast<ICmpInst>(LHS))
      return isImpliedCondICmps(LHSCmp->getCmpPredicate(),
                                LHSCmp->getOperand(0), LHSCmp->getOperand(1),
                                RHSPred, RHSOp0, RHSOp1, DL, LHSIsTrue);
    const Value *V;
    if (match(LHS, m_NUWTrunc(m_Value(V))))
      return isImpliedCondICmps(CmpInst::ICMP_NE, V,
                                ConstantInt::get(V->getType(), 0), RHSPred,
                                RHSOp0, RHSOp1, DL, LHSIsTrue);
  } else {
    assert(RHSOp0->getType()->isFPOrFPVectorTy() &&
           "Expected floating point type only!");
    if (const auto *LHSCmp = dyn_cast<FCmpInst>(LHS))
      return isImpliedCondFCmps(LHSCmp->getPredicate(), LHSCmp->getOperand(0),
                                LHSCmp->getOperand(1), RHSPred, RHSOp0, RHSOp1,
                                DL, LHSIsTrue);
  }

  /// The LHS should be an 'or', 'and', or a 'select' instruction.  We expect
  /// the RHS to be an icmp.
  /// FIXME: Add support for and/or/select on the RHS.
  if (const Instruction *LHSI = dyn_cast<Instruction>(LHS)) {
    if ((LHSI->getOpcode() == Instruction::And ||
         LHSI->getOpcode() == Instruction::Or ||
         LHSI->getOpcode() == Instruction::Select))
      return isImpliedCondAndOr(LHSI, RHSPred, RHSOp0, RHSOp1, DL, LHSIsTrue,
                                Depth);
  }
  return std::nullopt;
}

std::optional<bool> llvm::isImpliedCondition(const Value *LHS, const Value *RHS,
                                             const DataLayout &DL,
                                             bool LHSIsTrue, unsigned Depth) {
  // LHS ==> RHS by definition
  if (LHS == RHS)
    return LHSIsTrue;

  // Match not
  bool InvertRHS = false;
  if (match(RHS, m_Not(m_Value(RHS)))) {
    if (LHS == RHS)
      return !LHSIsTrue;
    InvertRHS = true;
  }

  if (const ICmpInst *RHSCmp = dyn_cast<ICmpInst>(RHS)) {
    if (auto Implied = isImpliedCondition(
            LHS, RHSCmp->getCmpPredicate(), RHSCmp->getOperand(0),
            RHSCmp->getOperand(1), DL, LHSIsTrue, Depth))
      return InvertRHS ? !*Implied : *Implied;
    return std::nullopt;
  }
  if (const FCmpInst *RHSCmp = dyn_cast<FCmpInst>(RHS)) {
    if (auto Implied = isImpliedCondition(
            LHS, RHSCmp->getPredicate(), RHSCmp->getOperand(0),
            RHSCmp->getOperand(1), DL, LHSIsTrue, Depth))
      return InvertRHS ? !*Implied : *Implied;
    return std::nullopt;
  }

  const Value *V;
  if (match(RHS, m_NUWTrunc(m_Value(V)))) {
    if (auto Implied = isImpliedCondition(LHS, CmpInst::ICMP_NE, V,
                                          ConstantInt::get(V->getType(), 0), DL,
                                          LHSIsTrue, Depth))
      return InvertRHS ? !*Implied : *Implied;
    return std::nullopt;
  }

  if (Depth == MaxAnalysisRecursionDepth)
    return std::nullopt;

  // LHS ==> (RHS1 || RHS2) if LHS ==> RHS1 or LHS ==> RHS2
  // LHS ==> !(RHS1 && RHS2) if LHS ==> !RHS1 or LHS ==> !RHS2
  const Value *RHS1, *RHS2;
  if (match(RHS, m_LogicalOr(m_Value(RHS1), m_Value(RHS2)))) {
    if (std::optional<bool> Imp =
            isImpliedCondition(LHS, RHS1, DL, LHSIsTrue, Depth + 1))
      if (*Imp == true)
        return !InvertRHS;
    if (std::optional<bool> Imp =
            isImpliedCondition(LHS, RHS2, DL, LHSIsTrue, Depth + 1))
      if (*Imp == true)
        return !InvertRHS;
  }
  if (match(RHS, m_LogicalAnd(m_Value(RHS1), m_Value(RHS2)))) {
    if (std::optional<bool> Imp =
            isImpliedCondition(LHS, RHS1, DL, LHSIsTrue, Depth + 1))
      if (*Imp == false)
        return InvertRHS;
    if (std::optional<bool> Imp =
            isImpliedCondition(LHS, RHS2, DL, LHSIsTrue, Depth + 1))
      if (*Imp == false)
        return InvertRHS;
  }

  return std::nullopt;
}

// Returns a pair (Condition, ConditionIsTrue), where Condition is a branch
// condition dominating ContextI or nullptr, if no condition is found.
static std::pair<Value *, bool>
getDomPredecessorCondition(const Instruction *ContextI) {
  if (!ContextI || !ContextI->getParent())
    return {nullptr, false};

  // TODO: This is a poor/cheap way to determine dominance. Should we use a
  // dominator tree (eg, from a SimplifyQuery) instead?
  const BasicBlock *ContextBB = ContextI->getParent();
  const BasicBlock *PredBB = ContextBB->getSinglePredecessor();
  if (!PredBB)
    return {nullptr, false};

  // We need a conditional branch in the predecessor.
  Value *PredCond;
  BasicBlock *TrueBB, *FalseBB;
  if (!match(PredBB->getTerminator(), m_Br(m_Value(PredCond), TrueBB, FalseBB)))
    return {nullptr, false};

  // The branch should get simplified. Don't bother simplifying this condition.
  if (TrueBB == FalseBB)
    return {nullptr, false};

  assert((TrueBB == ContextBB || FalseBB == ContextBB) &&
         "Predecessor block does not point to successor?");

  // Is this condition implied by the predecessor condition?
  return {PredCond, TrueBB == ContextBB};
}

std::optional<bool> llvm::isImpliedByDomCondition(const Value *Cond,
                                                  const Instruction *ContextI,
                                                  const DataLayout &DL) {
  assert(Cond->getType()->isIntOrIntVectorTy(1) && "Condition must be bool");
  auto PredCond = getDomPredecessorCondition(ContextI);
  if (PredCond.first)
    return isImpliedCondition(PredCond.first, Cond, DL, PredCond.second);
  return std::nullopt;
}

std::optional<bool> llvm::isImpliedByDomCondition(CmpPredicate Pred,
                                                  const Value *LHS,
                                                  const Value *RHS,
                                                  const Instruction *ContextI,
                                                  const DataLayout &DL) {
  auto PredCond = getDomPredecessorCondition(ContextI);
  if (PredCond.first)
    return isImpliedCondition(PredCond.first, Pred, LHS, RHS, DL,
                              PredCond.second);
  return std::nullopt;
}

ConstantRange llvm::computeConstantRange(const Value *V, bool ForSigned,
                                         bool UseInstrInfo, AssumptionCache *AC,
                                         const Instruction *CtxI,
                                         const DominatorTree *DT,
                                         unsigned Depth) {
  assert(V->getType()->isIntOrIntVectorTy() && "Expected integer instruction");

  if (Depth == MaxAnalysisRecursionDepth)
    return ConstantRange::getFull(V->getType()->getScalarSizeInBits());

  if (auto *C = dyn_cast<Constant>(V))
    return C->toConstantRange();

  unsigned BitWidth = V->getType()->getScalarSizeInBits();
  InstrInfoQuery IIQ(UseInstrInfo);
  ConstantRange CR = ConstantRange::getFull(BitWidth);
  if (auto *BO = dyn_cast<BinaryOperator>(V)) {
    APInt Lower = APInt(BitWidth, 0);
    APInt Upper = APInt(BitWidth, 0);
    // TODO: Return ConstantRange.
    setLimitsForBinOp(*BO, Lower, Upper, IIQ, ForSigned);
    CR = ConstantRange::getNonEmpty(Lower, Upper);
  } else if (auto *II = dyn_cast<IntrinsicInst>(V))
    CR = getRangeForIntrinsic(*II, UseInstrInfo);
  else if (auto *SI = dyn_cast<SelectInst>(V)) {
    ConstantRange CRTrue = computeConstantRange(
        SI->getTrueValue(), ForSigned, UseInstrInfo, AC, CtxI, DT, Depth + 1);
    ConstantRange CRFalse = computeConstantRange(
        SI->getFalseValue(), ForSigned, UseInstrInfo, AC, CtxI, DT, Depth + 1);
    CR = CRTrue.unionWith(CRFalse);
    CR = CR.intersectWith(getRangeForSelectPattern(*SI, IIQ));
  } else if (isa<FPToUIInst>(V) || isa<FPToSIInst>(V)) {
    APInt Lower = APInt(BitWidth, 0);
    APInt Upper = APInt(BitWidth, 0);
    // TODO: Return ConstantRange.
    setLimitForFPToI(cast<Instruction>(V), Lower, Upper);
    CR = ConstantRange::getNonEmpty(Lower, Upper);
  } else if (const auto *A = dyn_cast<Argument>(V))
    if (std::optional<ConstantRange> Range = A->getRange())
      CR = *Range;

  if (auto *I = dyn_cast<Instruction>(V)) {
    if (auto *Range = IIQ.getMetadata(I, LLVMContext::MD_range))
      CR = CR.intersectWith(getConstantRangeFromMetadata(*Range));

    if (const auto *CB = dyn_cast<CallBase>(V))
      if (std::optional<ConstantRange> Range = CB->getRange())
        CR = CR.intersectWith(*Range);
  }

  if (CtxI && AC) {
    // Try to restrict the range based on information from assumptions.
    for (auto &AssumeVH : AC->assumptionsFor(V)) {
      if (!AssumeVH)
        continue;
      CallInst *I = cast<CallInst>(AssumeVH);
      assert(I->getParent()->getParent() == CtxI->getParent()->getParent() &&
             "Got assumption for the wrong function!");
      assert(I->getIntrinsicID() == Intrinsic::assume &&
             "must be an assume intrinsic");

      if (!isValidAssumeForContext(I, CtxI, DT))
        continue;
      Value *Arg = I->getArgOperand(0);
      ICmpInst *Cmp = dyn_cast<ICmpInst>(Arg);
      // Currently we just use information from comparisons.
      if (!Cmp || Cmp->getOperand(0) != V)
        continue;
      // TODO: Set "ForSigned" parameter via Cmp->isSigned()?
      ConstantRange RHS =
          computeConstantRange(Cmp->getOperand(1), /* ForSigned */ false,
                               UseInstrInfo, AC, I, DT, Depth + 1);
      CR = CR.intersectWith(
          ConstantRange::makeAllowedICmpRegion(Cmp->getPredicate(), RHS));
    }
  }

  return CR;
}
