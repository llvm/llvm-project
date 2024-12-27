//===- ValueLattice.cpp - Value constraint analysis -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ValueLattice.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/IR/Instructions.h"

namespace llvm {
Constant *
ValueLatticeElement::getCompare(CmpInst::Predicate Pred, Type *Ty,
                                const ValueLatticeElement &Other,
                                const DataLayout &DL) const {
  // Not yet resolved.
  if (isUnknown() || Other.isUnknown())
    return nullptr;

  // TODO: Can be made more precise, but always returning undef would be
  // incorrect.
  if (isUndef() || Other.isUndef())
    return nullptr;

  if (isConstant() && Other.isConstant())
    return ConstantFoldCompareInstOperands(Pred, getConstant(),
                                           Other.getConstant(), DL);

  if (ICmpInst::isEquality(Pred)) {
    // not(C) != C => true, not(C) == C => false.
    if ((isNotConstant() && Other.isConstant() &&
         getNotConstant() == Other.getConstant()) ||
        (isConstant() && Other.isNotConstant() &&
         getConstant() == Other.getNotConstant()))
      return Pred == ICmpInst::ICMP_NE ? ConstantInt::getTrue(Ty)
                                       : ConstantInt::getFalse(Ty);
  }

  // Integer constants are represented as ConstantRanges with single
  // elements.
  if (!isConstantRange() || !Other.isConstantRange())
    return nullptr;

  const auto &CR = getConstantRange();
  const auto &OtherCR = Other.getConstantRange();
  if (CR.icmp(Pred, OtherCR))
    return ConstantInt::getTrue(Ty);
  if (CR.icmp(CmpInst::getInversePredicate(Pred), OtherCR))
    return ConstantInt::getFalse(Ty);

  return nullptr;
}

static bool hasSingleValue(const ValueLatticeElement &Val) {
  if (Val.isConstantRange() && Val.getConstantRange().isSingleElement())
    // Integer constants are single element ranges
    return true;
  return Val.isConstant();
}

/// Combine two sets of facts about the same value into a single set of
/// facts.  Note that this method is not suitable for merging facts along
/// different paths in a CFG; that's what the mergeIn function is for.  This
/// is for merging facts gathered about the same value at the same location
/// through two independent means.
/// Notes:
/// * This method does not promise to return the most precise possible lattice
///   value implied by A and B.  It is allowed to return any lattice element
///   which is at least as strong as *either* A or B (unless our facts
///   conflict, see below).
/// * Due to unreachable code, the intersection of two lattice values could be
///   contradictory.  If this happens, we return some valid lattice value so as
///   not confuse the rest of LVI.  Ideally, we'd always return Undefined, but
///   we do not make this guarantee.  TODO: This would be a useful enhancement.
ValueLatticeElement
ValueLatticeElement::intersect(const ValueLatticeElement &Other) const {
  if (isUnknown())
    return *this;
  if (Other.isUnknown())
    return Other;

  // If we gave up for one, but got a useable fact from the other, use it.
  if (isOverdefined())
    return Other;
  if (Other.isOverdefined())
    return *this;

  // Can't get any more precise than constants.
  if (hasSingleValue(*this))
    return *this;
  if (hasSingleValue(Other))
    return Other;

  // Could be either constant range or not constant here.
  if (!isConstantRange() || !Other.isConstantRange()) {
    // TODO: Arbitrary choice, could be improved
    return *this;
  }

  // Intersect two constant ranges
  ConstantRange Range =
      getConstantRange().intersectWith(Other.getConstantRange());
  // Note: An empty range is implicitly converted to unknown or undef depending
  // on MayIncludeUndef internally.
  return ValueLatticeElement::getRange(
      std::move(Range), /*MayIncludeUndef=*/isConstantRangeIncludingUndef() ||
                            Other.isConstantRangeIncludingUndef());
}

raw_ostream &operator<<(raw_ostream &OS, const ValueLatticeElement &Val) {
  if (Val.isUnknown())
    return OS << "unknown";
  if (Val.isUndef())
    return OS << "undef";
  if (Val.isOverdefined())
    return OS << "overdefined";

  if (Val.isNotConstant())
    return OS << "notconstant<" << *Val.getNotConstant() << ">";

  if (Val.isConstantRangeIncludingUndef())
    return OS << "constantrange incl. undef <"
              << Val.getConstantRange(true).getLower() << ", "
              << Val.getConstantRange(true).getUpper() << ">";

  if (Val.isConstantRange())
    return OS << "constantrange<" << Val.getConstantRange().getLower() << ", "
              << Val.getConstantRange().getUpper() << ">";
  return OS << "constant<" << *Val.getConstant() << ">";
}
} // end namespace llvm
