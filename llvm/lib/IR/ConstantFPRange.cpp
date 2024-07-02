//===- ConstantFPRange.cpp - ConstantFPRange implementation ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/ConstantFPRange.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

using namespace llvm;

// A floating point format must supports NaN, Inf and -0.
bool ConstantFPRange::isSupportedSemantics(const fltSemantics &Sem) {
  switch (APFloat::SemanticsToEnum(Sem)) {
  default:
    return false;
  case APFloat::S_IEEEhalf:
  case APFloat::S_BFloat:
  case APFloat::S_IEEEsingle:
  case APFloat::S_IEEEdouble:
  case APFloat::S_IEEEquad:
    return true;
  }
}

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
  assert(isSupportedSemantics(Sem) && "Unsupported fp format");
  Lower = APFloat::getInf(Sem, /*Negative=*/IsFullSet);
  Upper = APFloat::getInf(Sem, /*Negative=*/!IsFullSet);
  MayBeQNaN = IsFullSet;
  MayBeSNaN = IsFullSet;
}

ConstantFPRange::ConstantFPRange(const APFloat &Value)
    : Lower(Value.getSemantics(), APFloat::uninitialized),
      Upper(Value.getSemantics(), APFloat::uninitialized) {
  assert(isSupportedSemantics(getSemantics()) && "Unsupported fp format");

  if (Value.isNaN()) {
    makeEmpty();
    bool isSNaN = Value.isSignaling();
    MayBeQNaN = !isSNaN;
    MayBeSNaN = isSNaN;
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

ConstantFPRange::ConstantFPRange(APFloat LowerVal, APFloat UpperVal,
                                 bool MayBeQNaN, bool MaybeSNaN)
    : Lower(std::move(LowerVal)), Upper(std::move(UpperVal)) {
  assert(isSupportedSemantics(getSemantics()) && "Unsupported fp format");

  // Canonicalize empty set into [Inf, -Inf].
  if (strictCompare(Lower, Upper) == APFloat::cmpGreaterThan &&
      !(Lower.isInfinity() && Upper.isInfinity()))
    makeEmpty();
  this->MayBeQNaN = MayBeQNaN;
  this->MayBeSNaN = MayBeSNaN;
}

ConstantFPRange
ConstantFPRange::makeAllowedFCmpRegion(FCmpInst::Predicate Pred,
                                       const ConstantFPRange &Other) {
  // TODO
  return getFull(Other.getSemantics());
}

ConstantFPRange
ConstantFPRange::makeSatisfyingFCmpRegion(FCmpInst::Predicate Pred,
                                          const ConstantFPRange &Other) {
  // TODO
  return getEmpty(Other.getSemantics());
}

ConstantFPRange ConstantFPRange::makeExactFCmpRegion(FCmpInst::Predicate Pred,
                                                     const APFloat &Other) {
  return makeAllowedFCmpRegion(Pred, Other);
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

const APFloat *ConstantFPRange::getSingleElement() const {
  if (MayBeSNaN || MayBeQNaN)
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
    for (uint32_t I = LowerMask; I <= UpperMask; I <<= 1)
      Mask |= I;
  }
  return static_cast<FPClassTest>(Mask);
}

KnownFPClass ConstantFPRange::toKnownFPClass() const {
  KnownFPClass Result;
  Result.KnownFPClasses = classify();
  Result.SignBit = getSignBit();
  return Result;
}

void ConstantFPRange::print(raw_ostream &OS) const {
  if (isFullSet())
    OS << "full-set";
  else if (isEmptySet())
    OS << "empty-set";
  else {
    bool NaNOnly = isNaNOnly();
    if (!NaNOnly) {
      OS << '[';
      Lower.print(OS);
      OS << ", ";
      Upper.print(OS);
      OS << ']';
    }

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
  return ConstantFPRange(maxnum(Lower, CR.Lower), minnum(Upper, CR.Upper),
                         MayBeQNaN & CR.MayBeQNaN, MayBeSNaN & CR.MayBeSNaN);
}

ConstantFPRange ConstantFPRange::unionWith(const ConstantFPRange &CR) const {
  return ConstantFPRange(minnum(Lower, CR.Lower), maxnum(Upper, CR.Upper),
                         MayBeQNaN | CR.MayBeQNaN, MayBeSNaN | CR.MayBeSNaN);
}
