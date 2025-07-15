//===- ConstantFPRange.cpp - ConstantFPRange implementation ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/ConstantFPRange.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

using namespace llvm;

void ConstantFPRange::makeEmpty() {
  auto &Sem = Lower.getSemantics();
  Lower = APFloat::getInf(Sem, /*Negative=*/false);
  Upper = APFloat::getInf(Sem, /*Negative=*/true);
  MayBeQNaN = false;
  MayBeSNaN = false;
}

void ConstantFPRange::makeFull() {
  auto &Sem = Lower.getSemantics();
  Lower = APFloat::getInf(Sem, /*Negative=*/true);
  Upper = APFloat::getInf(Sem, /*Negative=*/false);
  MayBeQNaN = true;
  MayBeSNaN = true;
}

bool ConstantFPRange::isNaNOnly() const {
  return Lower.isPosInfinity() && Upper.isNegInfinity();
}

ConstantFPRange::ConstantFPRange(const fltSemantics &Sem, bool IsFullSet)
    : Lower(Sem, APFloat::uninitialized), Upper(Sem, APFloat::uninitialized) {
  Lower = APFloat::getInf(Sem, /*Negative=*/IsFullSet);
  Upper = APFloat::getInf(Sem, /*Negative=*/!IsFullSet);
  MayBeQNaN = IsFullSet;
  MayBeSNaN = IsFullSet;
}

ConstantFPRange::ConstantFPRange(const APFloat &Value)
    : Lower(Value.getSemantics(), APFloat::uninitialized),
      Upper(Value.getSemantics(), APFloat::uninitialized) {
  if (Value.isNaN()) {
    makeEmpty();
    bool IsSNaN = Value.isSignaling();
    MayBeQNaN = !IsSNaN;
    MayBeSNaN = IsSNaN;
  } else {
    Lower = Upper = Value;
    MayBeQNaN = MayBeSNaN = false;
  }
}

// We treat that -0 is less than 0 here.
static APFloat::cmpResult strictCompare(const APFloat &LHS,
                                        const APFloat &RHS) {
  assert(!LHS.isNaN() && !RHS.isNaN() && "Unordered compare");
  if (LHS.isZero() && RHS.isZero()) {
    if (LHS.isNegative() == RHS.isNegative())
      return APFloat::cmpEqual;
    return LHS.isNegative() ? APFloat::cmpLessThan : APFloat::cmpGreaterThan;
  }
  return LHS.compare(RHS);
}

static bool isNonCanonicalEmptySet(const APFloat &Lower, const APFloat &Upper) {
  return strictCompare(Lower, Upper) == APFloat::cmpGreaterThan &&
         !(Lower.isInfinity() && Upper.isInfinity());
}

static void canonicalizeRange(APFloat &Lower, APFloat &Upper) {
  if (isNonCanonicalEmptySet(Lower, Upper)) {
    Lower = APFloat::getInf(Lower.getSemantics(), /*Negative=*/false);
    Upper = APFloat::getInf(Upper.getSemantics(), /*Negative=*/true);
  }
}

ConstantFPRange::ConstantFPRange(APFloat LowerVal, APFloat UpperVal,
                                 bool MayBeQNaNVal, bool MayBeSNaNVal)
    : Lower(std::move(LowerVal)), Upper(std::move(UpperVal)),
      MayBeQNaN(MayBeQNaNVal), MayBeSNaN(MayBeSNaNVal) {
  assert(&Lower.getSemantics() == &Upper.getSemantics() &&
         "Should only use the same semantics");
  assert(!isNonCanonicalEmptySet(Lower, Upper) && "Non-canonical form");
}

ConstantFPRange ConstantFPRange::getFinite(const fltSemantics &Sem) {
  return ConstantFPRange(APFloat::getLargest(Sem, /*Negative=*/true),
                         APFloat::getLargest(Sem, /*Negative=*/false),
                         /*MayBeQNaN=*/false, /*MayBeSNaN=*/false);
}

ConstantFPRange ConstantFPRange::getNaNOnly(const fltSemantics &Sem,
                                            bool MayBeQNaN, bool MayBeSNaN) {
  return ConstantFPRange(APFloat::getInf(Sem, /*Negative=*/false),
                         APFloat::getInf(Sem, /*Negative=*/true), MayBeQNaN,
                         MayBeSNaN);
}

ConstantFPRange ConstantFPRange::getNonNaN(const fltSemantics &Sem) {
  return ConstantFPRange(APFloat::getInf(Sem, /*Negative=*/true),
                         APFloat::getInf(Sem, /*Negative=*/false),
                         /*MayBeQNaN=*/false, /*MayBeSNaN=*/false);
}

/// Return true for ULT/UGT/OLT/OGT
static bool fcmpPredExcludesEqual(FCmpInst::Predicate Pred) {
  return !(Pred & FCmpInst::FCMP_OEQ);
}

/// Return [-inf, V) or [-inf, V]
static ConstantFPRange makeLessThan(APFloat V, FCmpInst::Predicate Pred) {
  const fltSemantics &Sem = V.getSemantics();
  if (fcmpPredExcludesEqual(Pred)) {
    if (V.isNegInfinity())
      return ConstantFPRange::getEmpty(Sem);
    V.next(/*nextDown=*/true);
  }
  return ConstantFPRange::getNonNaN(APFloat::getInf(Sem, /*Negative=*/true),
                                    std::move(V));
}

/// Return (V, +inf] or [V, +inf]
static ConstantFPRange makeGreaterThan(APFloat V, FCmpInst::Predicate Pred) {
  const fltSemantics &Sem = V.getSemantics();
  if (fcmpPredExcludesEqual(Pred)) {
    if (V.isPosInfinity())
      return ConstantFPRange::getEmpty(Sem);
    V.next(/*nextDown=*/false);
  }
  return ConstantFPRange::getNonNaN(std::move(V),
                                    APFloat::getInf(Sem, /*Negative=*/false));
}

/// Make sure that +0/-0 are both included in the range.
static ConstantFPRange extendZeroIfEqual(const ConstantFPRange &CR,
                                         FCmpInst::Predicate Pred) {
  if (fcmpPredExcludesEqual(Pred))
    return CR;

  APFloat Lower = CR.getLower();
  APFloat Upper = CR.getUpper();
  if (Lower.isPosZero())
    Lower = APFloat::getZero(Lower.getSemantics(), /*Negative=*/true);
  if (Upper.isNegZero())
    Upper = APFloat::getZero(Upper.getSemantics(), /*Negative=*/false);
  return ConstantFPRange(std::move(Lower), std::move(Upper), CR.containsQNaN(),
                         CR.containsSNaN());
}

static ConstantFPRange setNaNField(const ConstantFPRange &CR,
                                   FCmpInst::Predicate Pred) {
  bool ContainsNaN = FCmpInst::isUnordered(Pred);
  return ConstantFPRange(CR.getLower(), CR.getUpper(),
                         /*MayBeQNaN=*/ContainsNaN, /*MayBeSNaN=*/ContainsNaN);
}

ConstantFPRange
ConstantFPRange::makeAllowedFCmpRegion(FCmpInst::Predicate Pred,
                                       const ConstantFPRange &Other) {
  if (Other.isEmptySet())
    return Other;
  if (Other.containsNaN() && FCmpInst::isUnordered(Pred))
    return getFull(Other.getSemantics());
  if (Other.isNaNOnly() && FCmpInst::isOrdered(Pred))
    return getEmpty(Other.getSemantics());

  switch (Pred) {
  case FCmpInst::FCMP_TRUE:
    return getFull(Other.getSemantics());
  case FCmpInst::FCMP_FALSE:
    return getEmpty(Other.getSemantics());
  case FCmpInst::FCMP_ORD:
    return getNonNaN(Other.getSemantics());
  case FCmpInst::FCMP_UNO:
    return getNaNOnly(Other.getSemantics(), /*MayBeQNaN=*/true,
                      /*MayBeSNaN=*/true);
  case FCmpInst::FCMP_OEQ:
  case FCmpInst::FCMP_UEQ:
    return setNaNField(extendZeroIfEqual(Other, Pred), Pred);
  case FCmpInst::FCMP_ONE:
  case FCmpInst::FCMP_UNE:
    if (const APFloat *SingleElement =
            Other.getSingleElement(/*ExcludesNaN=*/true)) {
      const fltSemantics &Sem = SingleElement->getSemantics();
      if (SingleElement->isPosInfinity())
        return setNaNField(
            getNonNaN(APFloat::getInf(Sem, /*Negative=*/true),
                      APFloat::getLargest(Sem, /*Negative=*/false)),
            Pred);
      if (SingleElement->isNegInfinity())
        return setNaNField(
            getNonNaN(APFloat::getLargest(Sem, /*Negative=*/true),
                      APFloat::getInf(Sem, /*Negative=*/false)),
            Pred);
    }
    return Pred == FCmpInst::FCMP_ONE ? getNonNaN(Other.getSemantics())
                                      : getFull(Other.getSemantics());
  case FCmpInst::FCMP_OLT:
  case FCmpInst::FCMP_OLE:
  case FCmpInst::FCMP_ULT:
  case FCmpInst::FCMP_ULE:
    return setNaNField(
        extendZeroIfEqual(makeLessThan(Other.getUpper(), Pred), Pred), Pred);
  case FCmpInst::FCMP_OGT:
  case FCmpInst::FCMP_OGE:
  case FCmpInst::FCMP_UGT:
  case FCmpInst::FCMP_UGE:
    return setNaNField(
        extendZeroIfEqual(makeGreaterThan(Other.getLower(), Pred), Pred), Pred);
  default:
    llvm_unreachable("Unexpected predicate");
  }
}

ConstantFPRange
ConstantFPRange::makeSatisfyingFCmpRegion(FCmpInst::Predicate Pred,
                                          const ConstantFPRange &Other) {
  if (Other.isEmptySet())
    return getFull(Other.getSemantics());
  if (Other.containsNaN() && FCmpInst::isOrdered(Pred))
    return getEmpty(Other.getSemantics());
  if (Other.isNaNOnly() && FCmpInst::isUnordered(Pred))
    return getFull(Other.getSemantics());

  switch (Pred) {
  case FCmpInst::FCMP_TRUE:
    return getFull(Other.getSemantics());
  case FCmpInst::FCMP_FALSE:
    return getEmpty(Other.getSemantics());
  case FCmpInst::FCMP_ORD:
    return getNonNaN(Other.getSemantics());
  case FCmpInst::FCMP_UNO:
    return getNaNOnly(Other.getSemantics(), /*MayBeQNaN=*/true,
                      /*MayBeSNaN=*/true);
  case FCmpInst::FCMP_OEQ:
  case FCmpInst::FCMP_UEQ:
    return setNaNField(Other.isSingleElement(/*ExcludesNaN=*/true) ||
                               ((Other.classify() & ~fcNan) == fcZero)
                           ? extendZeroIfEqual(Other, Pred)
                           : getEmpty(Other.getSemantics()),
                       Pred);
  case FCmpInst::FCMP_ONE:
  case FCmpInst::FCMP_UNE:
    return getEmpty(Other.getSemantics());
  case FCmpInst::FCMP_OLT:
  case FCmpInst::FCMP_OLE:
  case FCmpInst::FCMP_ULT:
  case FCmpInst::FCMP_ULE:
    return setNaNField(
        extendZeroIfEqual(makeLessThan(Other.getLower(), Pred), Pred), Pred);
  case FCmpInst::FCMP_OGT:
  case FCmpInst::FCMP_OGE:
  case FCmpInst::FCMP_UGT:
  case FCmpInst::FCMP_UGE:
    return setNaNField(
        extendZeroIfEqual(makeGreaterThan(Other.getUpper(), Pred), Pred), Pred);
  default:
    llvm_unreachable("Unexpected predicate");
  }
}

std::optional<ConstantFPRange>
ConstantFPRange::makeExactFCmpRegion(FCmpInst::Predicate Pred,
                                     const APFloat &Other) {
  if ((Pred == FCmpInst::FCMP_UNE || Pred == FCmpInst::FCMP_ONE) &&
      !Other.isNaN())
    return std::nullopt;
  return makeSatisfyingFCmpRegion(Pred, ConstantFPRange(Other));
}

bool ConstantFPRange::fcmp(FCmpInst::Predicate Pred,
                           const ConstantFPRange &Other) const {
  return makeSatisfyingFCmpRegion(Pred, Other).contains(*this);
}

bool ConstantFPRange::isFullSet() const {
  return Lower.isNegInfinity() && Upper.isPosInfinity() && MayBeQNaN &&
         MayBeSNaN;
}

bool ConstantFPRange::isEmptySet() const {
  return Lower.isPosInfinity() && Upper.isNegInfinity() && !MayBeQNaN &&
         !MayBeSNaN;
}

bool ConstantFPRange::contains(const APFloat &Val) const {
  assert(&getSemantics() == &Val.getSemantics() &&
         "Should only use the same semantics");

  if (Val.isNaN())
    return Val.isSignaling() ? MayBeSNaN : MayBeQNaN;
  return strictCompare(Lower, Val) != APFloat::cmpGreaterThan &&
         strictCompare(Val, Upper) != APFloat::cmpGreaterThan;
}

bool ConstantFPRange::contains(const ConstantFPRange &CR) const {
  assert(&getSemantics() == &CR.getSemantics() &&
         "Should only use the same semantics");

  if (CR.MayBeQNaN && !MayBeQNaN)
    return false;

  if (CR.MayBeSNaN && !MayBeSNaN)
    return false;

  return strictCompare(Lower, CR.Lower) != APFloat::cmpGreaterThan &&
         strictCompare(CR.Upper, Upper) != APFloat::cmpGreaterThan;
}

const APFloat *ConstantFPRange::getSingleElement(bool ExcludesNaN) const {
  if (!ExcludesNaN && (MayBeSNaN || MayBeQNaN))
    return nullptr;
  return Lower.bitwiseIsEqual(Upper) ? &Lower : nullptr;
}

std::optional<bool> ConstantFPRange::getSignBit() const {
  if (!MayBeSNaN && !MayBeQNaN && Lower.isNegative() == Upper.isNegative())
    return Lower.isNegative();
  return std::nullopt;
}

bool ConstantFPRange::operator==(const ConstantFPRange &CR) const {
  if (MayBeSNaN != CR.MayBeSNaN || MayBeQNaN != CR.MayBeQNaN)
    return false;
  return Lower.bitwiseIsEqual(CR.Lower) && Upper.bitwiseIsEqual(CR.Upper);
}

FPClassTest ConstantFPRange::classify() const {
  uint32_t Mask = fcNone;
  if (MayBeSNaN)
    Mask |= fcSNan;
  if (MayBeQNaN)
    Mask |= fcQNan;
  if (!isNaNOnly()) {
    FPClassTest LowerMask = Lower.classify();
    FPClassTest UpperMask = Upper.classify();
    assert(LowerMask <= UpperMask && "Range is nan-only.");
    // Set all bits from log2(LowerMask) to log2(UpperMask).
    Mask |= (UpperMask << 1) - LowerMask;
  }
  return static_cast<FPClassTest>(Mask);
}

void ConstantFPRange::print(raw_ostream &OS) const {
  if (isFullSet())
    OS << "full-set";
  else if (isEmptySet())
    OS << "empty-set";
  else {
    bool NaNOnly = isNaNOnly();
    if (!NaNOnly)
      OS << '[' << Lower << ", " << Upper << ']';

    if (MayBeSNaN || MayBeQNaN) {
      if (!NaNOnly)
        OS << " with ";
      if (MayBeSNaN && MayBeQNaN)
        OS << "NaN";
      else if (MayBeSNaN)
        OS << "SNaN";
      else if (MayBeQNaN)
        OS << "QNaN";
    }
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void ConstantFPRange::dump() const { print(dbgs()); }
#endif

ConstantFPRange
ConstantFPRange::intersectWith(const ConstantFPRange &CR) const {
  assert(&getSemantics() == &CR.getSemantics() &&
         "Should only use the same semantics");
  APFloat NewLower = maxnum(Lower, CR.Lower);
  APFloat NewUpper = minnum(Upper, CR.Upper);
  canonicalizeRange(NewLower, NewUpper);
  return ConstantFPRange(std::move(NewLower), std::move(NewUpper),
                         MayBeQNaN & CR.MayBeQNaN, MayBeSNaN & CR.MayBeSNaN);
}

ConstantFPRange ConstantFPRange::unionWith(const ConstantFPRange &CR) const {
  assert(&getSemantics() == &CR.getSemantics() &&
         "Should only use the same semantics");
  return ConstantFPRange(minnum(Lower, CR.Lower), maxnum(Upper, CR.Upper),
                         MayBeQNaN | CR.MayBeQNaN, MayBeSNaN | CR.MayBeSNaN);
}
