//===- ConstantRange.cpp - ConstantRange implementation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Represent a range of possible values that may occur when the program is run
// for an integral value.  This keeps track of a lower and upper bound for the
// constant, which MAY wrap around the end of the numeric range.  To do this, it
// keeps track of a [lower, upper) bound, which specifies an interval just like
// STL iterators.  When used with boolean values, the following are important
// ranges (other integral ranges use min/max values for special range values):
//
//  [F, F) = {}     = Empty set
//  [T, F) = {T}
//  [F, T) = {F}
//  [T, T) = {F, T} = Full set
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/APInt.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstdint>

using namespace llvm;

ConstantRange::ConstantRange(uint32_t BitWidth, bool Full)
    : Lower(Full ? APInt::getMaxValue(BitWidth) : APInt::getMinValue(BitWidth)),
      Upper(Lower) {}

ConstantRange::ConstantRange(APInt V)
    : Lower(std::move(V)), Upper(Lower + 1) {}

ConstantRange::ConstantRange(APInt L, APInt U)
    : Lower(std::move(L)), Upper(std::move(U)) {
  assert(Lower.getBitWidth() == Upper.getBitWidth() &&
         "ConstantRange with unequal bit widths");
  assert((Lower != Upper || (Lower.isMaxValue() || Lower.isMinValue())) &&
         "Lower == Upper, but they aren't min or max value!");
}

ConstantRange ConstantRange::fromKnownBits(const KnownBits &Known,
                                           bool IsSigned) {
  assert(!Known.hasConflict() && "Expected valid KnownBits");

  if (Known.isUnknown())
    return getFull(Known.getBitWidth());

  // For unsigned ranges, or signed ranges with known sign bit, create a simple
  // range between the smallest and largest possible value.
  if (!IsSigned || Known.isNegative() || Known.isNonNegative())
    return ConstantRange(Known.One, ~Known.Zero + 1);

  // If we don't know the sign bit, pick the lower bound as a negative number
  // and the upper bound as a non-negative one.
  APInt Lower = Known.One, Upper = ~Known.Zero;
  Lower.setSignBit();
  Upper.clearSignBit();
  return ConstantRange(Lower, Upper + 1);
}

ConstantRange ConstantRange::makeAllowedICmpRegion(CmpInst::Predicate Pred,
                                                   const ConstantRange &CR) {
  if (CR.isEmptySet())
    return CR;

  uint32_t W = CR.getBitWidth();
  switch (Pred) {
  default:
    llvm_unreachable("Invalid ICmp predicate to makeAllowedICmpRegion()");
  case CmpInst::ICMP_EQ:
    return CR;
  case CmpInst::ICMP_NE:
    if (CR.isSingleElement())
      return ConstantRange(CR.getUpper(), CR.getLower());
    return getFull(W);
  case CmpInst::ICMP_ULT: {
    APInt UMax(CR.getUnsignedMax());
    if (UMax.isMinValue())
      return getEmpty(W);
    return ConstantRange(APInt::getMinValue(W), std::move(UMax));
  }
  case CmpInst::ICMP_SLT: {
    APInt SMax(CR.getSignedMax());
    if (SMax.isMinSignedValue())
      return getEmpty(W);
    return ConstantRange(APInt::getSignedMinValue(W), std::move(SMax));
  }
  case CmpInst::ICMP_ULE:
    return getNonEmpty(APInt::getMinValue(W), CR.getUnsignedMax() + 1);
  case CmpInst::ICMP_SLE:
    return getNonEmpty(APInt::getSignedMinValue(W), CR.getSignedMax() + 1);
  case CmpInst::ICMP_UGT: {
    APInt UMin(CR.getUnsignedMin());
    if (UMin.isMaxValue())
      return getEmpty(W);
    return ConstantRange(std::move(UMin) + 1, APInt::getNullValue(W));
  }
  case CmpInst::ICMP_SGT: {
    APInt SMin(CR.getSignedMin());
    if (SMin.isMaxSignedValue())
      return getEmpty(W);
    return ConstantRange(std::move(SMin) + 1, APInt::getSignedMinValue(W));
  }
  case CmpInst::ICMP_UGE:
    return getNonEmpty(CR.getUnsignedMin(), APInt::getNullValue(W));
  case CmpInst::ICMP_SGE:
    return getNonEmpty(CR.getSignedMin(), APInt::getSignedMinValue(W));
  }
}

ConstantRange ConstantRange::makeSatisfyingICmpRegion(CmpInst::Predicate Pred,
                                                      const ConstantRange &CR) {
  // Follows from De-Morgan's laws:
  //
  // ~(~A union ~B) == A intersect B.
  //
  return makeAllowedICmpRegion(CmpInst::getInversePredicate(Pred), CR)
      .inverse();
}

ConstantRange ConstantRange::makeExactICmpRegion(CmpInst::Predicate Pred,
                                                 const APInt &C) {
  // Computes the exact range that is equal to both the constant ranges returned
  // by makeAllowedICmpRegion and makeSatisfyingICmpRegion. This is always true
  // when RHS is a singleton such as an APInt and so the assert is valid.
  // However for non-singleton RHS, for example ult [2,5) makeAllowedICmpRegion
  // returns [0,4) but makeSatisfyICmpRegion returns [0,2).
  //
  assert(makeAllowedICmpRegion(Pred, C) == makeSatisfyingICmpRegion(Pred, C));
  return makeAllowedICmpRegion(Pred, C);
}

bool ConstantRange::getEquivalentICmp(CmpInst::Predicate &Pred,
                                      APInt &RHS) const {
  bool Success = false;

  if (isFullSet() || isEmptySet()) {
    Pred = isEmptySet() ? CmpInst::ICMP_ULT : CmpInst::ICMP_UGE;
    RHS = APInt(getBitWidth(), 0);
    Success = true;
  } else if (auto *OnlyElt = getSingleElement()) {
    Pred = CmpInst::ICMP_EQ;
    RHS = *OnlyElt;
    Success = true;
  } else if (auto *OnlyMissingElt = getSingleMissingElement()) {
    Pred = CmpInst::ICMP_NE;
    RHS = *OnlyMissingElt;
    Success = true;
  } else if (getLower().isMinSignedValue() || getLower().isMinValue()) {
    Pred =
        getLower().isMinSignedValue() ? CmpInst::ICMP_SLT : CmpInst::ICMP_ULT;
    RHS = getUpper();
    Success = true;
  } else if (getUpper().isMinSignedValue() || getUpper().isMinValue()) {
    Pred =
        getUpper().isMinSignedValue() ? CmpInst::ICMP_SGE : CmpInst::ICMP_UGE;
    RHS = getLower();
    Success = true;
  }

  assert((!Success || ConstantRange::makeExactICmpRegion(Pred, RHS) == *this) &&
         "Bad result!");

  return Success;
}

ConstantRange
ConstantRange::makeGuaranteedNoWrapRegion(Instruction::BinaryOps BinOp,
                                          const ConstantRange &Other,
                                          unsigned NoWrapKind) {
  using OBO = OverflowingBinaryOperator;

  // Computes the intersection of CR0 and CR1.  It is different from
  // intersectWith in that the ConstantRange returned will only contain elements
  // in both CR0 and CR1 (i.e. SubsetIntersect(X, Y) is a *subset*, proper or
  // not, of both X and Y).
  auto SubsetIntersect =
      [](const ConstantRange &CR0, const ConstantRange &CR1) {
    return CR0.inverse().unionWith(CR1.inverse()).inverse();
  };

  assert(Instruction::isBinaryOp(BinOp) && "Binary operators only!");

  assert((NoWrapKind == OBO::NoSignedWrap ||
          NoWrapKind == OBO::NoUnsignedWrap) &&
         "NoWrapKind invalid!");

  unsigned BitWidth = Other.getBitWidth();
  ConstantRange Result(BitWidth, /* full */ true);

  switch (BinOp) {
  default:
    // Conservative answer: empty set
    return getEmpty(BitWidth);

  case Instruction::Add:
    if (auto *C = Other.getSingleElement())
      if (C->isNullValue())
        // Full set: nothing signed / unsigned wraps when added to 0.
        return getFull(BitWidth);

    if (NoWrapKind == OBO::NoUnsignedWrap)
      return ConstantRange(APInt::getNullValue(BitWidth),
                           -Other.getUnsignedMax());

    if (NoWrapKind == OBO::NoSignedWrap) {
      const APInt &SignedMin = Other.getSignedMin();
      const APInt &SignedMax = Other.getSignedMax();
      if (SignedMax.isStrictlyPositive())
        Result = SubsetIntersect(
            Result,
            ConstantRange(APInt::getSignedMinValue(BitWidth),
                          APInt::getSignedMinValue(BitWidth) - SignedMax));
      if (SignedMin.isNegative())
        Result = SubsetIntersect(
            Result,
            ConstantRange(APInt::getSignedMinValue(BitWidth) - SignedMin,
                          APInt::getSignedMinValue(BitWidth)));
    }
    return Result;

  case Instruction::Sub:
    if (auto *C = Other.getSingleElement())
      if (C->isNullValue())
        // Full set: nothing signed / unsigned wraps when subtracting 0.
        return getFull(BitWidth);

    if (NoWrapKind == OBO::NoUnsignedWrap)
      return ConstantRange(Other.getUnsignedMax(),
                           APInt::getMinValue(BitWidth));

    if (NoWrapKind == OBO::NoSignedWrap) {
      const APInt &SignedMin = Other.getSignedMin();
      const APInt &SignedMax = Other.getSignedMax();
      if (SignedMax.isStrictlyPositive())
        Result = SubsetIntersect(
            Result,
            ConstantRange(APInt::getSignedMinValue(BitWidth) + SignedMax,
                          APInt::getSignedMinValue(BitWidth)));
      if (SignedMin.isNegative())
        Result = SubsetIntersect(
            Result,
            ConstantRange(APInt::getSignedMinValue(BitWidth),
                          APInt::getSignedMinValue(BitWidth) + SignedMin));
    }
    return Result;

  case Instruction::Mul: {
    // Equivalent to calling makeGuaranteedNoWrapRegion() on [V, V+1).
    const bool Unsigned = NoWrapKind == OBO::NoUnsignedWrap;
    const auto makeSingleValueRegion = [Unsigned,
                                        BitWidth](APInt V) -> ConstantRange {
      // Handle special case for 0, -1 and 1. See the last for reason why we
      // specialize -1 and 1.
      if (V == 0 || V.isOneValue())
        return getFull(BitWidth);

      APInt MinValue, MaxValue;
      if (Unsigned) {
        MinValue = APInt::getMinValue(BitWidth);
        MaxValue = APInt::getMaxValue(BitWidth);
      } else {
        MinValue = APInt::getSignedMinValue(BitWidth);
        MaxValue = APInt::getSignedMaxValue(BitWidth);
      }
      // e.g. Returning [-127, 127], represented as [-127, -128).
      if (!Unsigned && V.isAllOnesValue())
        return ConstantRange(-MaxValue, MinValue);

      APInt Lower, Upper;
      if (!Unsigned && V.isNegative()) {
        Lower = APIntOps::RoundingSDiv(MaxValue, V, APInt::Rounding::UP);
        Upper = APIntOps::RoundingSDiv(MinValue, V, APInt::Rounding::DOWN);
      } else if (Unsigned) {
        Lower = APIntOps::RoundingUDiv(MinValue, V, APInt::Rounding::UP);
        Upper = APIntOps::RoundingUDiv(MaxValue, V, APInt::Rounding::DOWN);
      } else {
        Lower = APIntOps::RoundingSDiv(MinValue, V, APInt::Rounding::UP);
        Upper = APIntOps::RoundingSDiv(MaxValue, V, APInt::Rounding::DOWN);
      }
      // ConstantRange ctor take a half inclusive interval [Lower, Upper + 1).
      // Upper + 1 is guanranteed not to overflow, because |divisor| > 1. 0, -1,
      // and 1 are already handled as special cases.
      return ConstantRange(Lower, Upper + 1);
    };

    if (Unsigned)
      return makeSingleValueRegion(Other.getUnsignedMax());

    return SubsetIntersect(makeSingleValueRegion(Other.getSignedMin()),
                           makeSingleValueRegion(Other.getSignedMax()));
  }
  }
}

bool ConstantRange::isFullSet() const {
  return Lower == Upper && Lower.isMaxValue();
}

bool ConstantRange::isEmptySet() const {
  return Lower == Upper && Lower.isMinValue();
}

bool ConstantRange::isWrappedSet() const {
  return Lower.ugt(Upper) && !Upper.isNullValue();
}

bool ConstantRange::isUpperWrapped() const {
  return Lower.ugt(Upper);
}

bool ConstantRange::isSignWrappedSet() const {
  return Lower.sgt(Upper) && !Upper.isMinSignedValue();
}

bool ConstantRange::isUpperSignWrapped() const {
  return Lower.sgt(Upper);
}

bool
ConstantRange::isSizeStrictlySmallerThan(const ConstantRange &Other) const {
  assert(getBitWidth() == Other.getBitWidth());
  if (isFullSet())
    return false;
  if (Other.isFullSet())
    return true;
  return (Upper - Lower).ult(Other.Upper - Other.Lower);
}

bool
ConstantRange::isSizeLargerThan(uint64_t MaxSize) const {
  assert(MaxSize && "MaxSize can't be 0.");
  // If this a full set, we need special handling to avoid needing an extra bit
  // to represent the size.
  if (isFullSet())
    return APInt::getMaxValue(getBitWidth()).ugt(MaxSize - 1);

  return (Upper - Lower).ugt(MaxSize);
}

bool ConstantRange::isAllNegative() const {
  // Empty set is all negative, full set is not.
  if (isEmptySet())
    return true;
  if (isFullSet())
    return false;

  return !isUpperSignWrapped() && !Upper.isStrictlyPositive();
}

bool ConstantRange::isAllNonNegative() const {
  // Empty and full set are automatically treated correctly.
  return !isSignWrappedSet() && Lower.isNonNegative();
}

APInt ConstantRange::getUnsignedMax() const {
  if (isFullSet() || isUpperWrapped())
    return APInt::getMaxValue(getBitWidth());
  return getUpper() - 1;
}

APInt ConstantRange::getUnsignedMin() const {
  if (isFullSet() || isWrappedSet())
    return APInt::getMinValue(getBitWidth());
  return getLower();
}

APInt ConstantRange::getSignedMax() const {
  if (isFullSet() || isUpperSignWrapped())
    return APInt::getSignedMaxValue(getBitWidth());
  return getUpper() - 1;
}

APInt ConstantRange::getSignedMin() const {
  if (isFullSet() || isSignWrappedSet())
    return APInt::getSignedMinValue(getBitWidth());
  return getLower();
}

bool ConstantRange::contains(const APInt &V) const {
  if (Lower == Upper)
    return isFullSet();

  if (!isUpperWrapped())
    return Lower.ule(V) && V.ult(Upper);
  return Lower.ule(V) || V.ult(Upper);
}

bool ConstantRange::contains(const ConstantRange &Other) const {
  if (isFullSet() || Other.isEmptySet()) return true;
  if (isEmptySet() || Other.isFullSet()) return false;

  if (!isUpperWrapped()) {
    if (Other.isUpperWrapped())
      return false;

    return Lower.ule(Other.getLower()) && Other.getUpper().ule(Upper);
  }

  if (!Other.isUpperWrapped())
    return Other.getUpper().ule(Upper) ||
           Lower.ule(Other.getLower());

  return Other.getUpper().ule(Upper) && Lower.ule(Other.getLower());
}

ConstantRange ConstantRange::subtract(const APInt &Val) const {
  assert(Val.getBitWidth() == getBitWidth() && "Wrong bit width");
  // If the set is empty or full, don't modify the endpoints.
  if (Lower == Upper)
    return *this;
  return ConstantRange(Lower - Val, Upper - Val);
}

ConstantRange ConstantRange::difference(const ConstantRange &CR) const {
  return intersectWith(CR.inverse());
}

static ConstantRange getPreferredRange(
    const ConstantRange &CR1, const ConstantRange &CR2,
    ConstantRange::PreferredRangeType Type) {
  if (Type == ConstantRange::Unsigned) {
    if (!CR1.isWrappedSet() && CR2.isWrappedSet())
      return CR1;
    if (CR1.isWrappedSet() && !CR2.isWrappedSet())
      return CR2;
  } else if (Type == ConstantRange::Signed) {
    if (!CR1.isSignWrappedSet() && CR2.isSignWrappedSet())
      return CR1;
    if (CR1.isSignWrappedSet() && !CR2.isSignWrappedSet())
      return CR2;
  }

  if (CR1.isSizeStrictlySmallerThan(CR2))
    return CR1;
  return CR2;
}

ConstantRange ConstantRange::intersectWith(const ConstantRange &CR,
                                           PreferredRangeType Type) const {
  assert(getBitWidth() == CR.getBitWidth() &&
         "ConstantRange types don't agree!");

  // Handle common cases.
  if (   isEmptySet() || CR.isFullSet()) return *this;
  if (CR.isEmptySet() ||    isFullSet()) return CR;

  if (!isUpperWrapped() && CR.isUpperWrapped())
    return CR.intersectWith(*this, Type);

  if (!isUpperWrapped() && !CR.isUpperWrapped()) {
    if (Lower.ult(CR.Lower)) {
      // L---U       : this
      //       L---U : CR
      if (Upper.ule(CR.Lower))
        return getEmpty();

      // L---U       : this
      //   L---U     : CR
      if (Upper.ult(CR.Upper))
        return ConstantRange(CR.Lower, Upper);

      // L-------U   : this
      //   L---U     : CR
      return CR;
    }
    //   L---U     : this
    // L-------U   : CR
    if (Upper.ult(CR.Upper))
      return *this;

    //   L-----U   : this
    // L-----U     : CR
    if (Lower.ult(CR.Upper))
      return ConstantRange(Lower, CR.Upper);

    //       L---U : this
    // L---U       : CR
    return getEmpty();
  }

  if (isUpperWrapped() && !CR.isUpperWrapped()) {
    if (CR.Lower.ult(Upper)) {
      // ------U   L--- : this
      //  L--U          : CR
      if (CR.Upper.ult(Upper))
        return CR;

      // ------U   L--- : this
      //  L------U      : CR
      if (CR.Upper.ule(Lower))
        return ConstantRange(CR.Lower, Upper);

      // ------U   L--- : this
      //  L----------U  : CR
      return getPreferredRange(*this, CR, Type);
    }
    if (CR.Lower.ult(Lower)) {
      // --U      L---- : this
      //     L--U       : CR
      if (CR.Upper.ule(Lower))
        return getEmpty();

      // --U      L---- : this
      //     L------U   : CR
      return ConstantRange(Lower, CR.Upper);
    }

    // --U  L------ : this
    //        L--U  : CR
    return CR;
  }

  if (CR.Upper.ult(Upper)) {
    // ------U L-- : this
    // --U L------ : CR
    if (CR.Lower.ult(Upper))
      return getPreferredRange(*this, CR, Type);

    // ----U   L-- : this
    // --U   L---- : CR
    if (CR.Lower.ult(Lower))
      return ConstantRange(Lower, CR.Upper);

    // ----U L---- : this
    // --U     L-- : CR
    return CR;
  }
  if (CR.Upper.ule(Lower)) {
    // --U     L-- : this
    // ----U L---- : CR
    if (CR.Lower.ult(Lower))
      return *this;

    // --U   L---- : this
    // ----U   L-- : CR
    return ConstantRange(CR.Lower, Upper);
  }

  // --U L------ : this
  // ------U L-- : CR
  return getPreferredRange(*this, CR, Type);
}

ConstantRange ConstantRange::unionWith(const ConstantRange &CR,
                                       PreferredRangeType Type) const {
  assert(getBitWidth() == CR.getBitWidth() &&
         "ConstantRange types don't agree!");

  if (   isFullSet() || CR.isEmptySet()) return *this;
  if (CR.isFullSet() ||    isEmptySet()) return CR;

  if (!isUpperWrapped() && CR.isUpperWrapped())
    return CR.unionWith(*this, Type);

  if (!isUpperWrapped() && !CR.isUpperWrapped()) {
    //        L---U  and  L---U        : this
    //  L---U                   L---U  : CR
    // result in one of
    //  L---------U
    // -----U L-----
    if (CR.Upper.ult(Lower) || Upper.ult(CR.Lower))
      return getPreferredRange(
          ConstantRange(Lower, CR.Upper), ConstantRange(CR.Lower, Upper), Type);

    APInt L = CR.Lower.ult(Lower) ? CR.Lower : Lower;
    APInt U = (CR.Upper - 1).ugt(Upper - 1) ? CR.Upper : Upper;

    if (L.isNullValue() && U.isNullValue())
      return getFull();

    return ConstantRange(std::move(L), std::move(U));
  }

  if (!CR.isUpperWrapped()) {
    // ------U   L-----  and  ------U   L----- : this
    //   L--U                            L--U  : CR
    if (CR.Upper.ule(Upper) || CR.Lower.uge(Lower))
      return *this;

    // ------U   L----- : this
    //    L---------U   : CR
    if (CR.Lower.ule(Upper) && Lower.ule(CR.Upper))
      return getFull();

    // ----U       L---- : this
    //       L---U       : CR
    // results in one of
    // ----------U L----
    // ----U L----------
    if (Upper.ult(CR.Lower) && CR.Upper.ult(Lower))
      return getPreferredRange(
          ConstantRange(Lower, CR.Upper), ConstantRange(CR.Lower, Upper), Type);

    // ----U     L----- : this
    //        L----U    : CR
    if (Upper.ult(CR.Lower) && Lower.ule(CR.Upper))
      return ConstantRange(CR.Lower, Upper);

    // ------U    L---- : this
    //    L-----U       : CR
    assert(CR.Lower.ule(Upper) && CR.Upper.ult(Lower) &&
           "ConstantRange::unionWith missed a case with one range wrapped");
    return ConstantRange(Lower, CR.Upper);
  }

  // ------U    L----  and  ------U    L---- : this
  // -U  L-----------  and  ------------U  L : CR
  if (CR.Lower.ule(Upper) || Lower.ule(CR.Upper))
    return getFull();

  APInt L = CR.Lower.ult(Lower) ? CR.Lower : Lower;
  APInt U = CR.Upper.ugt(Upper) ? CR.Upper : Upper;

  return ConstantRange(std::move(L), std::move(U));
}

ConstantRange ConstantRange::castOp(Instruction::CastOps CastOp,
                                    uint32_t ResultBitWidth) const {
  switch (CastOp) {
  default:
    llvm_unreachable("unsupported cast type");
  case Instruction::Trunc:
    return truncate(ResultBitWidth);
  case Instruction::SExt:
    return signExtend(ResultBitWidth);
  case Instruction::ZExt:
    return zeroExtend(ResultBitWidth);
  case Instruction::BitCast:
    return *this;
  case Instruction::FPToUI:
  case Instruction::FPToSI:
    if (getBitWidth() == ResultBitWidth)
      return *this;
    else
      return getFull();
  case Instruction::UIToFP: {
    // TODO: use input range if available
    auto BW = getBitWidth();
    APInt Min = APInt::getMinValue(BW).zextOrSelf(ResultBitWidth);
    APInt Max = APInt::getMaxValue(BW).zextOrSelf(ResultBitWidth);
    return ConstantRange(std::move(Min), std::move(Max));
  }
  case Instruction::SIToFP: {
    // TODO: use input range if available
    auto BW = getBitWidth();
    APInt SMin = APInt::getSignedMinValue(BW).sextOrSelf(ResultBitWidth);
    APInt SMax = APInt::getSignedMaxValue(BW).sextOrSelf(ResultBitWidth);
    return ConstantRange(std::move(SMin), std::move(SMax));
  }
  case Instruction::FPTrunc:
  case Instruction::FPExt:
  case Instruction::IntToPtr:
  case Instruction::PtrToInt:
  case Instruction::AddrSpaceCast:
    // Conservatively return getFull set.
    return getFull();
  };
}

ConstantRange ConstantRange::zeroExtend(uint32_t DstTySize) const {
  if (isEmptySet()) return getEmpty(DstTySize);

  unsigned SrcTySize = getBitWidth();
  assert(SrcTySize < DstTySize && "Not a value extension");
  if (isFullSet() || isUpperWrapped()) {
    // Change into [0, 1 << src bit width)
    APInt LowerExt(DstTySize, 0);
    if (!Upper) // special case: [X, 0) -- not really wrapping around
      LowerExt = Lower.zext(DstTySize);
    return ConstantRange(std::move(LowerExt),
                         APInt::getOneBitSet(DstTySize, SrcTySize));
  }

  return ConstantRange(Lower.zext(DstTySize), Upper.zext(DstTySize));
}

ConstantRange ConstantRange::signExtend(uint32_t DstTySize) const {
  if (isEmptySet()) return getEmpty(DstTySize);

  unsigned SrcTySize = getBitWidth();
  assert(SrcTySize < DstTySize && "Not a value extension");

  // special case: [X, INT_MIN) -- not really wrapping around
  if (Upper.isMinSignedValue())
    return ConstantRange(Lower.sext(DstTySize), Upper.zext(DstTySize));

  if (isFullSet() || isSignWrappedSet()) {
    return ConstantRange(APInt::getHighBitsSet(DstTySize,DstTySize-SrcTySize+1),
                         APInt::getLowBitsSet(DstTySize, SrcTySize-1) + 1);
  }

  return ConstantRange(Lower.sext(DstTySize), Upper.sext(DstTySize));
}

ConstantRange ConstantRange::truncate(uint32_t DstTySize) const {
  assert(getBitWidth() > DstTySize && "Not a value truncation");
  if (isEmptySet())
    return getEmpty(DstTySize);
  if (isFullSet())
    return getFull(DstTySize);

  APInt LowerDiv(Lower), UpperDiv(Upper);
  ConstantRange Union(DstTySize, /*isFullSet=*/false);

  // Analyze wrapped sets in their two parts: [0, Upper) \/ [Lower, MaxValue]
  // We use the non-wrapped set code to analyze the [Lower, MaxValue) part, and
  // then we do the union with [MaxValue, Upper)
  if (isUpperWrapped()) {
    // If Upper is greater than or equal to MaxValue(DstTy), it covers the whole
    // truncated range.
    if (Upper.getActiveBits() > DstTySize ||
        Upper.countTrailingOnes() == DstTySize)
      return getFull(DstTySize);

    Union = ConstantRange(APInt::getMaxValue(DstTySize),Upper.trunc(DstTySize));
    UpperDiv.setAllBits();

    // Union covers the MaxValue case, so return if the remaining range is just
    // MaxValue(DstTy).
    if (LowerDiv == UpperDiv)
      return Union;
  }

  // Chop off the most significant bits that are past the destination bitwidth.
  if (LowerDiv.getActiveBits() > DstTySize) {
    // Mask to just the signficant bits and subtract from LowerDiv/UpperDiv.
    APInt Adjust = LowerDiv & APInt::getBitsSetFrom(getBitWidth(), DstTySize);
    LowerDiv -= Adjust;
    UpperDiv -= Adjust;
  }

  unsigned UpperDivWidth = UpperDiv.getActiveBits();
  if (UpperDivWidth <= DstTySize)
    return ConstantRange(LowerDiv.trunc(DstTySize),
                         UpperDiv.trunc(DstTySize)).unionWith(Union);

  // The truncated value wraps around. Check if we can do better than fullset.
  if (UpperDivWidth == DstTySize + 1) {
    // Clear the MSB so that UpperDiv wraps around.
    UpperDiv.clearBit(DstTySize);
    if (UpperDiv.ult(LowerDiv))
      return ConstantRange(LowerDiv.trunc(DstTySize),
                           UpperDiv.trunc(DstTySize)).unionWith(Union);
  }

  return getFull(DstTySize);
}

ConstantRange ConstantRange::zextOrTrunc(uint32_t DstTySize) const {
  unsigned SrcTySize = getBitWidth();
  if (SrcTySize > DstTySize)
    return truncate(DstTySize);
  if (SrcTySize < DstTySize)
    return zeroExtend(DstTySize);
  return *this;
}

ConstantRange ConstantRange::sextOrTrunc(uint32_t DstTySize) const {
  unsigned SrcTySize = getBitWidth();
  if (SrcTySize > DstTySize)
    return truncate(DstTySize);
  if (SrcTySize < DstTySize)
    return signExtend(DstTySize);
  return *this;
}

ConstantRange ConstantRange::binaryOp(Instruction::BinaryOps BinOp,
                                      const ConstantRange &Other) const {
  assert(Instruction::isBinaryOp(BinOp) && "Binary operators only!");

  switch (BinOp) {
  case Instruction::Add:
    return add(Other);
  case Instruction::Sub:
    return sub(Other);
  case Instruction::Mul:
    return multiply(Other);
  case Instruction::UDiv:
    return udiv(Other);
  case Instruction::URem:
    return urem(Other);
  case Instruction::Shl:
    return shl(Other);
  case Instruction::LShr:
    return lshr(Other);
  case Instruction::AShr:
    return ashr(Other);
  case Instruction::And:
    return binaryAnd(Other);
  case Instruction::Or:
    return binaryOr(Other);
  // Note: floating point operations applied to abstract ranges are just
  // ideal integer operations with a lossy representation
  case Instruction::FAdd:
    return add(Other);
  case Instruction::FSub:
    return sub(Other);
  case Instruction::FMul:
    return multiply(Other);
  default:
    // Conservatively return getFull set.
    return getFull();
  }
}

ConstantRange
ConstantRange::add(const ConstantRange &Other) const {
  if (isEmptySet() || Other.isEmptySet())
    return getEmpty();
  if (isFullSet() || Other.isFullSet())
    return getFull();

  APInt NewLower = getLower() + Other.getLower();
  APInt NewUpper = getUpper() + Other.getUpper() - 1;
  if (NewLower == NewUpper)
    return getFull();

  ConstantRange X = ConstantRange(std::move(NewLower), std::move(NewUpper));
  if (X.isSizeStrictlySmallerThan(*this) ||
      X.isSizeStrictlySmallerThan(Other))
    // We've wrapped, therefore, full set.
    return getFull();
  return X;
}

ConstantRange ConstantRange::addWithNoSignedWrap(const APInt &Other) const {
  // Calculate the subset of this range such that "X + Other" is
  // guaranteed not to wrap (overflow) for all X in this subset.
  // makeGuaranteedNoWrapRegion will produce an exact NSW range.
  auto NSWRange = ConstantRange::makeGuaranteedNoWrapRegion(BinaryOperator::Add,
                                      ConstantRange(Other),
                                      OverflowingBinaryOperator::NoSignedWrap);
  auto NSWConstrainedRange = intersectWith(NSWRange);

  return NSWConstrainedRange.add(ConstantRange(Other));
}

ConstantRange
ConstantRange::sub(const ConstantRange &Other) const {
  if (isEmptySet() || Other.isEmptySet())
    return getEmpty();
  if (isFullSet() || Other.isFullSet())
    return getFull();

  APInt NewLower = getLower() - Other.getUpper() + 1;
  APInt NewUpper = getUpper() - Other.getLower();
  if (NewLower == NewUpper)
    return getFull();

  ConstantRange X = ConstantRange(std::move(NewLower), std::move(NewUpper));
  if (X.isSizeStrictlySmallerThan(*this) ||
      X.isSizeStrictlySmallerThan(Other))
    // We've wrapped, therefore, full set.
    return getFull();
  return X;
}

ConstantRange
ConstantRange::multiply(const ConstantRange &Other) const {
  // TODO: If either operand is a single element and the multiply is known to
  // be non-wrapping, round the result min and max value to the appropriate
  // multiple of that element. If wrapping is possible, at least adjust the
  // range according to the greatest power-of-two factor of the single element.

  if (isEmptySet() || Other.isEmptySet())
    return getEmpty();

  // Multiplication is signedness-independent. However different ranges can be
  // obtained depending on how the input ranges are treated. These different
  // ranges are all conservatively correct, but one might be better than the
  // other. We calculate two ranges; one treating the inputs as unsigned
  // and the other signed, then return the smallest of these ranges.

  // Unsigned range first.
  APInt this_min = getUnsignedMin().zext(getBitWidth() * 2);
  APInt this_max = getUnsignedMax().zext(getBitWidth() * 2);
  APInt Other_min = Other.getUnsignedMin().zext(getBitWidth() * 2);
  APInt Other_max = Other.getUnsignedMax().zext(getBitWidth() * 2);

  ConstantRange Result_zext = ConstantRange(this_min * Other_min,
                                            this_max * Other_max + 1);
  ConstantRange UR = Result_zext.truncate(getBitWidth());

  // If the unsigned range doesn't wrap, and isn't negative then it's a range
  // from one positive number to another which is as good as we can generate.
  // In this case, skip the extra work of generating signed ranges which aren't
  // going to be better than this range.
  if (!UR.isUpperWrapped() &&
      (UR.getUpper().isNonNegative() || UR.getUpper().isMinSignedValue()))
    return UR;

  // Now the signed range. Because we could be dealing with negative numbers
  // here, the lower bound is the smallest of the cartesian product of the
  // lower and upper ranges; for example:
  //   [-1,4) * [-2,3) = min(-1*-2, -1*2, 3*-2, 3*2) = -6.
  // Similarly for the upper bound, swapping min for max.

  this_min = getSignedMin().sext(getBitWidth() * 2);
  this_max = getSignedMax().sext(getBitWidth() * 2);
  Other_min = Other.getSignedMin().sext(getBitWidth() * 2);
  Other_max = Other.getSignedMax().sext(getBitWidth() * 2);

  auto L = {this_min * Other_min, this_min * Other_max,
            this_max * Other_min, this_max * Other_max};
  auto Compare = [](const APInt &A, const APInt &B) { return A.slt(B); };
  ConstantRange Result_sext(std::min(L, Compare), std::max(L, Compare) + 1);
  ConstantRange SR = Result_sext.truncate(getBitWidth());

  return UR.isSizeStrictlySmallerThan(SR) ? UR : SR;
}

ConstantRange
ConstantRange::smax(const ConstantRange &Other) const {
  // X smax Y is: range(smax(X_smin, Y_smin),
  //                    smax(X_smax, Y_smax))
  if (isEmptySet() || Other.isEmptySet())
    return getEmpty();
  APInt NewL = APIntOps::smax(getSignedMin(), Other.getSignedMin());
  APInt NewU = APIntOps::smax(getSignedMax(), Other.getSignedMax()) + 1;
  return getNonEmpty(std::move(NewL), std::move(NewU));
}

ConstantRange
ConstantRange::umax(const ConstantRange &Other) const {
  // X umax Y is: range(umax(X_umin, Y_umin),
  //                    umax(X_umax, Y_umax))
  if (isEmptySet() || Other.isEmptySet())
    return getEmpty();
  APInt NewL = APIntOps::umax(getUnsignedMin(), Other.getUnsignedMin());
  APInt NewU = APIntOps::umax(getUnsignedMax(), Other.getUnsignedMax()) + 1;
  return getNonEmpty(std::move(NewL), std::move(NewU));
}

ConstantRange
ConstantRange::smin(const ConstantRange &Other) const {
  // X smin Y is: range(smin(X_smin, Y_smin),
  //                    smin(X_smax, Y_smax))
  if (isEmptySet() || Other.isEmptySet())
    return getEmpty();
  APInt NewL = APIntOps::smin(getSignedMin(), Other.getSignedMin());
  APInt NewU = APIntOps::smin(getSignedMax(), Other.getSignedMax()) + 1;
  return getNonEmpty(std::move(NewL), std::move(NewU));
}

ConstantRange
ConstantRange::umin(const ConstantRange &Other) const {
  // X umin Y is: range(umin(X_umin, Y_umin),
  //                    umin(X_umax, Y_umax))
  if (isEmptySet() || Other.isEmptySet())
    return getEmpty();
  APInt NewL = APIntOps::umin(getUnsignedMin(), Other.getUnsignedMin());
  APInt NewU = APIntOps::umin(getUnsignedMax(), Other.getUnsignedMax()) + 1;
  return getNonEmpty(std::move(NewL), std::move(NewU));
}

ConstantRange
ConstantRange::udiv(const ConstantRange &RHS) const {
  if (isEmptySet() || RHS.isEmptySet() || RHS.getUnsignedMax().isNullValue())
    return getEmpty();

  APInt Lower = getUnsignedMin().udiv(RHS.getUnsignedMax());

  APInt RHS_umin = RHS.getUnsignedMin();
  if (RHS_umin.isNullValue()) {
    // We want the lowest value in RHS excluding zero. Usually that would be 1
    // except for a range in the form of [X, 1) in which case it would be X.
    if (RHS.getUpper() == 1)
      RHS_umin = RHS.getLower();
    else
      RHS_umin = 1;
  }

  APInt Upper = getUnsignedMax().udiv(RHS_umin) + 1;
  return getNonEmpty(std::move(Lower), std::move(Upper));
}

ConstantRange ConstantRange::urem(const ConstantRange &RHS) const {
  if (isEmptySet() || RHS.isEmptySet() || RHS.getUnsignedMax().isNullValue())
    return getEmpty();

  // L % R for L < R is L.
  if (getUnsignedMax().ult(RHS.getUnsignedMin()))
    return *this;

  // L % R is <= L and < R.
  APInt Upper = APIntOps::umin(getUnsignedMax(), RHS.getUnsignedMax() - 1) + 1;
  return getNonEmpty(APInt::getNullValue(getBitWidth()), std::move(Upper));
}

ConstantRange
ConstantRange::binaryAnd(const ConstantRange &Other) const {
  if (isEmptySet() || Other.isEmptySet())
    return getEmpty();

  // TODO: replace this with something less conservative

  APInt umin = APIntOps::umin(Other.getUnsignedMax(), getUnsignedMax());
  return getNonEmpty(APInt::getNullValue(getBitWidth()), std::move(umin) + 1);
}

ConstantRange
ConstantRange::binaryOr(const ConstantRange &Other) const {
  if (isEmptySet() || Other.isEmptySet())
    return getEmpty();

  // TODO: replace this with something less conservative

  APInt umax = APIntOps::umax(getUnsignedMin(), Other.getUnsignedMin());
  return getNonEmpty(std::move(umax), APInt::getNullValue(getBitWidth()));
}

ConstantRange
ConstantRange::shl(const ConstantRange &Other) const {
  if (isEmptySet() || Other.isEmptySet())
    return getEmpty();

  APInt max = getUnsignedMax();
  APInt Other_umax = Other.getUnsignedMax();

  // If we are shifting by maximum amount of
  // zero return return the original range.
  if (Other_umax.isNullValue())
    return *this;
  // there's overflow!
  if (Other_umax.ugt(max.countLeadingZeros()))
    return getFull();

  // FIXME: implement the other tricky cases

  APInt min = getUnsignedMin();
  min <<= Other.getUnsignedMin();
  max <<= Other_umax;

  return ConstantRange(std::move(min), std::move(max) + 1);
}

ConstantRange
ConstantRange::lshr(const ConstantRange &Other) const {
  if (isEmptySet() || Other.isEmptySet())
    return getEmpty();

  APInt max = getUnsignedMax().lshr(Other.getUnsignedMin()) + 1;
  APInt min = getUnsignedMin().lshr(Other.getUnsignedMax());
  return getNonEmpty(std::move(min), std::move(max));
}

ConstantRange
ConstantRange::ashr(const ConstantRange &Other) const {
  if (isEmptySet() || Other.isEmptySet())
    return getEmpty();

  // May straddle zero, so handle both positive and negative cases.
  // 'PosMax' is the upper bound of the result of the ashr
  // operation, when Upper of the LHS of ashr is a non-negative.
  // number. Since ashr of a non-negative number will result in a
  // smaller number, the Upper value of LHS is shifted right with
  // the minimum value of 'Other' instead of the maximum value.
  APInt PosMax = getSignedMax().ashr(Other.getUnsignedMin()) + 1;

  // 'PosMin' is the lower bound of the result of the ashr
  // operation, when Lower of the LHS is a non-negative number.
  // Since ashr of a non-negative number will result in a smaller
  // number, the Lower value of LHS is shifted right with the
  // maximum value of 'Other'.
  APInt PosMin = getSignedMin().ashr(Other.getUnsignedMax());

  // 'NegMax' is the upper bound of the result of the ashr
  // operation, when Upper of the LHS of ashr is a negative number.
  // Since 'ashr' of a negative number will result in a bigger
  // number, the Upper value of LHS is shifted right with the
  // maximum value of 'Other'.
  APInt NegMax = getSignedMax().ashr(Other.getUnsignedMax()) + 1;

  // 'NegMin' is the lower bound of the result of the ashr
  // operation, when Lower of the LHS of ashr is a negative number.
  // Since 'ashr' of a negative number will result in a bigger
  // number, the Lower value of LHS is shifted right with the
  // minimum value of 'Other'.
  APInt NegMin = getSignedMin().ashr(Other.getUnsignedMin());

  APInt max, min;
  if (getSignedMin().isNonNegative()) {
    // Upper and Lower of LHS are non-negative.
    min = PosMin;
    max = PosMax;
  } else if (getSignedMax().isNegative()) {
    // Upper and Lower of LHS are negative.
    min = NegMin;
    max = NegMax;
  } else {
    // Upper is non-negative and Lower is negative.
    min = NegMin;
    max = PosMax;
  }
  return getNonEmpty(std::move(min), std::move(max));
}

ConstantRange ConstantRange::uadd_sat(const ConstantRange &Other) const {
  if (isEmptySet() || Other.isEmptySet())
    return getEmpty();

  APInt NewL = getUnsignedMin().uadd_sat(Other.getUnsignedMin());
  APInt NewU = getUnsignedMax().uadd_sat(Other.getUnsignedMax()) + 1;
  return getNonEmpty(std::move(NewL), std::move(NewU));
}

ConstantRange ConstantRange::sadd_sat(const ConstantRange &Other) const {
  if (isEmptySet() || Other.isEmptySet())
    return getEmpty();

  APInt NewL = getSignedMin().sadd_sat(Other.getSignedMin());
  APInt NewU = getSignedMax().sadd_sat(Other.getSignedMax()) + 1;
  return getNonEmpty(std::move(NewL), std::move(NewU));
}

ConstantRange ConstantRange::usub_sat(const ConstantRange &Other) const {
  if (isEmptySet() || Other.isEmptySet())
    return getEmpty();

  APInt NewL = getUnsignedMin().usub_sat(Other.getUnsignedMax());
  APInt NewU = getUnsignedMax().usub_sat(Other.getUnsignedMin()) + 1;
  return getNonEmpty(std::move(NewL), std::move(NewU));
}

ConstantRange ConstantRange::ssub_sat(const ConstantRange &Other) const {
  if (isEmptySet() || Other.isEmptySet())
    return getEmpty();

  APInt NewL = getSignedMin().ssub_sat(Other.getSignedMax());
  APInt NewU = getSignedMax().ssub_sat(Other.getSignedMin()) + 1;
  return getNonEmpty(std::move(NewL), std::move(NewU));
}

ConstantRange ConstantRange::inverse() const {
  if (isFullSet())
    return getEmpty();
  if (isEmptySet())
    return getFull();
  return ConstantRange(Upper, Lower);
}

ConstantRange::OverflowResult ConstantRange::unsignedAddMayOverflow(
    const ConstantRange &Other) const {
  if (isEmptySet() || Other.isEmptySet())
    return OverflowResult::MayOverflow;

  APInt Min = getUnsignedMin(), Max = getUnsignedMax();
  APInt OtherMin = Other.getUnsignedMin(), OtherMax = Other.getUnsignedMax();

  // a u+ b overflows iff a u> ~b.
  if (Min.ugt(~OtherMin))
    return OverflowResult::AlwaysOverflows;
  if (Max.ugt(~OtherMax))
    return OverflowResult::MayOverflow;
  return OverflowResult::NeverOverflows;
}

ConstantRange::OverflowResult ConstantRange::signedAddMayOverflow(
    const ConstantRange &Other) const {
  if (isEmptySet() || Other.isEmptySet())
    return OverflowResult::MayOverflow;

  APInt Min = getSignedMin(), Max = getSignedMax();
  APInt OtherMin = Other.getSignedMin(), OtherMax = Other.getSignedMax();

  APInt SignedMin = APInt::getSignedMinValue(getBitWidth());
  APInt SignedMax = APInt::getSignedMaxValue(getBitWidth());

  // a s+ b overflows high iff a s>=0 && b s>= 0 && a s> smax - b.
  // a s+ b overflows low iff a s< 0 && b s< 0 && a s< smin - b.
  if (Min.isNonNegative() && OtherMin.isNonNegative() &&
      Min.sgt(SignedMax - OtherMin))
    return OverflowResult::AlwaysOverflows;
  if (Max.isNegative() && OtherMax.isNegative() &&
      Max.slt(SignedMin - OtherMax))
    return OverflowResult::AlwaysOverflows;

  if (Max.isNonNegative() && OtherMax.isNonNegative() &&
      Max.sgt(SignedMax - OtherMax))
    return OverflowResult::MayOverflow;
  if (Min.isNegative() && OtherMin.isNegative() &&
      Min.slt(SignedMin - OtherMin))
    return OverflowResult::MayOverflow;

  return OverflowResult::NeverOverflows;
}

ConstantRange::OverflowResult ConstantRange::unsignedSubMayOverflow(
    const ConstantRange &Other) const {
  if (isEmptySet() || Other.isEmptySet())
    return OverflowResult::MayOverflow;

  APInt Min = getUnsignedMin(), Max = getUnsignedMax();
  APInt OtherMin = Other.getUnsignedMin(), OtherMax = Other.getUnsignedMax();

  // a u- b overflows iff a u< b.
  if (Max.ult(OtherMin))
    return OverflowResult::AlwaysOverflows;
  if (Min.ult(OtherMax))
    return OverflowResult::MayOverflow;
  return OverflowResult::NeverOverflows;
}

ConstantRange::OverflowResult ConstantRange::signedSubMayOverflow(
    const ConstantRange &Other) const {
  if (isEmptySet() || Other.isEmptySet())
    return OverflowResult::MayOverflow;

  APInt Min = getSignedMin(), Max = getSignedMax();
  APInt OtherMin = Other.getSignedMin(), OtherMax = Other.getSignedMax();

  APInt SignedMin = APInt::getSignedMinValue(getBitWidth());
  APInt SignedMax = APInt::getSignedMaxValue(getBitWidth());

  // a s- b overflows high iff a s>=0 && b s< 0 && a s> smax + b.
  // a s- b overflows low iff a s< 0 && b s>= 0 && a s< smin + b.
  if (Min.isNonNegative() && OtherMax.isNegative() &&
      Min.sgt(SignedMax + OtherMax))
    return OverflowResult::AlwaysOverflows;
  if (Max.isNegative() && OtherMin.isNonNegative() &&
      Max.slt(SignedMin + OtherMin))
    return OverflowResult::AlwaysOverflows;

  if (Max.isNonNegative() && OtherMin.isNegative() &&
      Max.sgt(SignedMax + OtherMin))
    return OverflowResult::MayOverflow;
  if (Min.isNegative() && OtherMax.isNonNegative() &&
      Min.slt(SignedMin + OtherMax))
    return OverflowResult::MayOverflow;

  return OverflowResult::NeverOverflows;
}

ConstantRange::OverflowResult ConstantRange::unsignedMulMayOverflow(
    const ConstantRange &Other) const {
  if (isEmptySet() || Other.isEmptySet())
    return OverflowResult::MayOverflow;

  APInt Min = getUnsignedMin(), Max = getUnsignedMax();
  APInt OtherMin = Other.getUnsignedMin(), OtherMax = Other.getUnsignedMax();
  bool Overflow;

  (void) Min.umul_ov(OtherMin, Overflow);
  if (Overflow)
    return OverflowResult::AlwaysOverflows;

  (void) Max.umul_ov(OtherMax, Overflow);
  if (Overflow)
    return OverflowResult::MayOverflow;

  return OverflowResult::NeverOverflows;
}

void ConstantRange::print(raw_ostream &OS) const {
  if (isFullSet())
    OS << "full-set";
  else if (isEmptySet())
    OS << "empty-set";
  else
    OS << "[" << Lower << "," << Upper << ")";
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void ConstantRange::dump() const {
  print(dbgs());
}
#endif

ConstantRange llvm::getConstantRangeFromMetadata(const MDNode &Ranges) {
  const unsigned NumRanges = Ranges.getNumOperands() / 2;
  assert(NumRanges >= 1 && "Must have at least one range!");
  assert(Ranges.getNumOperands() % 2 == 0 && "Must be a sequence of pairs");

  auto *FirstLow = mdconst::extract<ConstantInt>(Ranges.getOperand(0));
  auto *FirstHigh = mdconst::extract<ConstantInt>(Ranges.getOperand(1));

  ConstantRange CR(FirstLow->getValue(), FirstHigh->getValue());

  for (unsigned i = 1; i < NumRanges; ++i) {
    auto *Low = mdconst::extract<ConstantInt>(Ranges.getOperand(2 * i + 0));
    auto *High = mdconst::extract<ConstantInt>(Ranges.getOperand(2 * i + 1));

    // Note: unionWith will potentially create a range that contains values not
    // contained in any of the original N ranges.
    CR = CR.unionWith(ConstantRange(Low->getValue(), High->getValue()));
  }

  return CR;
}
