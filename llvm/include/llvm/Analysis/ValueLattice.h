//===- ValueLattice.h - Value constraint analysis ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_VALUELATTICE_H
#define LLVM_ANALYSIS_VALUELATTICE_H

#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Constants.h"
//
//===----------------------------------------------------------------------===//
//                               ValueLatticeElement
//===----------------------------------------------------------------------===//

/// This class represents lattice values for constants.
///
/// FIXME: This is basically just for bringup, this can be made a lot more rich
/// in the future.
///

namespace llvm {
class ValueLatticeElement {
  enum ValueLatticeElementTy {
    /// This Value has no known value yet.  As a result, this implies the
    /// producing instruction is dead.  Caution: We use this as the starting
    /// state in our local meet rules.  In this usage, it's taken to mean
    /// "nothing known yet".
    unknown,

    /// This Value is an UndefValue constant or produces undef. Undefined values
    /// can be merged with constants (or single element constant ranges),
    /// assuming all uses of the result will be replaced.
    undef,

    /// This Value has a specific constant value.  (For constant integers,
    /// constantrange is used instead.  Integer typed constantexprs can appear
    /// as constant.)
    constant,

    /// This Value is known to not have the specified value.  (For constant
    /// integers, constantrange is used instead.  As above, integer typed
    /// constantexprs can appear here.)
    notconstant,

    /// The Value falls within this range. (Used only for integer typed values.)
    constantrange,

    /// We can not precisely model the dynamic values this value might take.
    overdefined
  };

  ValueLatticeElementTy Tag;

  /// The union either stores a pointer to a constant or a constant range,
  /// associated to the lattice element. We have to ensure that Range is
  /// initialized or destroyed when changing state to or from constantrange.
  union {
    Constant *ConstVal;
    ConstantRange Range;
  };

public:
  // Const and Range are initialized on-demand.
  ValueLatticeElement() : Tag(unknown) {}

  /// Custom destructor to ensure Range is properly destroyed, when the object
  /// is deallocated.
  ~ValueLatticeElement() {
    switch (Tag) {
    case overdefined:
    case unknown:
    case undef:
    case constant:
    case notconstant:
      break;
    case constantrange:
      Range.~ConstantRange();
      break;
    };
  }

  /// Custom copy constructor, to ensure Range gets initialized when
  /// copying a constant range lattice element.
  ValueLatticeElement(const ValueLatticeElement &Other) : Tag(unknown) {
    *this = Other;
  }

  /// Custom assignment operator, to ensure Range gets initialized when
  /// assigning a constant range lattice element.
  ValueLatticeElement &operator=(const ValueLatticeElement &Other) {
    // If we change the state of this from constant range to non constant range,
    // destroy Range.
    if (isConstantRange() && !Other.isConstantRange())
      Range.~ConstantRange();

    // If we change the state of this from a valid ConstVal to another a state
    // without a valid ConstVal, zero the pointer.
    if ((isConstant() || isNotConstant()) && !Other.isConstant() &&
        !Other.isNotConstant())
      ConstVal = nullptr;

    switch (Other.Tag) {
    case constantrange:
      if (!isConstantRange())
        new (&Range) ConstantRange(Other.Range);
      else
        Range = Other.Range;
      break;
    case constant:
    case notconstant:
      ConstVal = Other.ConstVal;
      break;
    case overdefined:
    case unknown:
    case undef:
      break;
    }
    Tag = Other.Tag;
    return *this;
  }

  static ValueLatticeElement get(Constant *C) {
    ValueLatticeElement Res;
    if (isa<UndefValue>(C))
      Res.markUndef();
    else
      Res.markConstant(C);
    return Res;
  }
  static ValueLatticeElement getNot(Constant *C) {
    ValueLatticeElement Res;
    assert(!isa<UndefValue>(C) && "!= undef is not supported");
    Res.markNotConstant(C);
    return Res;
  }
  static ValueLatticeElement getRange(ConstantRange CR) {
    ValueLatticeElement Res;
    Res.markConstantRange(std::move(CR));
    return Res;
  }
  static ValueLatticeElement getOverdefined() {
    ValueLatticeElement Res;
    Res.markOverdefined();
    return Res;
  }

  bool isUndef() const { return Tag == undef; }
  bool isUnknown() const { return Tag == unknown; }
  bool isUnknownOrUndef() const { return Tag == unknown || Tag == undef; }
  bool isConstant() const { return Tag == constant; }
  bool isNotConstant() const { return Tag == notconstant; }
  bool isConstantRange() const { return Tag == constantrange; }
  bool isOverdefined() const { return Tag == overdefined; }

  Constant *getConstant() const {
    assert(isConstant() && "Cannot get the constant of a non-constant!");
    return ConstVal;
  }

  Constant *getNotConstant() const {
    assert(isNotConstant() && "Cannot get the constant of a non-notconstant!");
    return ConstVal;
  }

  const ConstantRange &getConstantRange() const {
    assert(isConstantRange() &&
           "Cannot get the constant-range of a non-constant-range!");
    return Range;
  }

  Optional<APInt> asConstantInteger() const {
    if (isConstant() && isa<ConstantInt>(getConstant())) {
      return cast<ConstantInt>(getConstant())->getValue();
    } else if (isConstantRange() && getConstantRange().isSingleElement()) {
      return *getConstantRange().getSingleElement();
    }
    return None;
  }

  bool markOverdefined() {
    if (isOverdefined())
      return false;
    if (isConstant() || isNotConstant())
      ConstVal = nullptr;
    if (isConstantRange())
      Range.~ConstantRange();
    Tag = overdefined;
    return true;
  }

  bool markUndef() {
    if (isUndef())
      return false;

    assert(isUnknown());
    Tag = undef;
    return true;
  }

  bool markConstant(Constant *V) {
    if (isa<UndefValue>(V))
      return markUndef();

    if (isConstant()) {
      assert(getConstant() == V && "Marking constant with different value");
      return false;
    }

    if (ConstantInt *CI = dyn_cast<ConstantInt>(V))
      return markConstantRange(ConstantRange(CI->getValue()));

    assert(isUnknown() || isUndef());
    Tag = constant;
    ConstVal = V;
    return true;
  }

  bool markNotConstant(Constant *V) {
    assert(V && "Marking constant with NULL");
    if (ConstantInt *CI = dyn_cast<ConstantInt>(V))
      return markConstantRange(
          ConstantRange(CI->getValue() + 1, CI->getValue()));

    if (isa<UndefValue>(V))
      return false;

    if (isNotConstant()) {
      assert(getNotConstant() == V && "Marking !constant with different value");
      return false;
    }

    assert(isUnknown());
    Tag = notconstant;
    ConstVal = V;
    return true;
  }

  /// Mark the object as constant range with \p NewR. If the object is already a
  /// constant range, nothing changes if the existing range is equal to \p
  /// NewR. Otherwise \p NewR must be a superset of the existing range or the
  /// object must be undef.
  bool markConstantRange(ConstantRange NewR) {
    if (isConstantRange()) {
      if (getConstantRange() == NewR)
        return false;

      if (NewR.isEmptySet())
        return markOverdefined();

      assert(NewR.contains(getConstantRange()) &&
             "Existing range must be a subset of NewR");
      Range = std::move(NewR);
      return true;
    }

    assert(isUnknown() || isUndef());
    if (NewR.isEmptySet())
      return markOverdefined();

    Tag = constantrange;
    new (&Range) ConstantRange(std::move(NewR));
    return true;
  }

  /// Updates this object to approximate both this object and RHS. Returns
  /// true if this object has been changed.
  bool mergeIn(const ValueLatticeElement &RHS, const DataLayout &DL) {
    if (RHS.isUnknown() || isOverdefined())
      return false;
    if (RHS.isOverdefined()) {
      markOverdefined();
      return true;
    }

    if (isUndef()) {
      assert(!RHS.isUnknown());
      if (RHS.isUndef())
        return false;
      if (RHS.isConstant())
        return markConstant(RHS.getConstant());
      if (RHS.isConstantRange() && RHS.getConstantRange().isSingleElement())
        return markConstantRange(RHS.getConstantRange());
      return markOverdefined();
    }

    if (isUnknown()) {
      assert(!RHS.isUnknown() && "Unknow RHS should be handled earlier");
      *this = RHS;
      return true;
    }

    if (isConstant()) {
      if (RHS.isConstant() && getConstant() == RHS.getConstant())
        return false;
      if (RHS.isUndef())
        return false;
      markOverdefined();
      return true;
    }

    if (isNotConstant()) {
      if (RHS.isNotConstant() && getNotConstant() == RHS.getNotConstant())
        return false;
      markOverdefined();
      return true;
    }

    assert(isConstantRange() && "New ValueLattice type?");
    if (RHS.isUndef() && getConstantRange().isSingleElement())
      return false;

    if (!RHS.isConstantRange()) {
      // We can get here if we've encountered a constantexpr of integer type
      // and merge it with a constantrange.
      markOverdefined();
      return true;
    }
    ConstantRange NewR = getConstantRange().unionWith(RHS.getConstantRange());
    if (NewR.isFullSet())
      return markOverdefined();
    else if (NewR == getConstantRange())
      return false;
    else
      return markConstantRange(std::move(NewR));
  }

  // Compares this symbolic value with Other using Pred and returns either
  /// true, false or undef constants, or nullptr if the comparison cannot be
  /// evaluated.
  Constant *getCompare(CmpInst::Predicate Pred, Type *Ty,
                       const ValueLatticeElement &Other) const {
    if (isUnknownOrUndef() || Other.isUnknownOrUndef())
      return UndefValue::get(Ty);

    if (isConstant() && Other.isConstant())
      return ConstantExpr::getCompare(Pred, getConstant(), Other.getConstant());

    // Integer constants are represented as ConstantRanges with single
    // elements.
    if (!isConstantRange() || !Other.isConstantRange())
      return nullptr;

    const auto &CR = getConstantRange();
    const auto &OtherCR = Other.getConstantRange();
    if (ConstantRange::makeSatisfyingICmpRegion(Pred, OtherCR).contains(CR))
      return ConstantInt::getTrue(Ty);
    if (ConstantRange::makeSatisfyingICmpRegion(
            CmpInst::getInversePredicate(Pred), OtherCR)
            .contains(CR))
      return ConstantInt::getFalse(Ty);

    return nullptr;
  }
};

raw_ostream &operator<<(raw_ostream &OS, const ValueLatticeElement &Val);

} // end namespace llvm
#endif
