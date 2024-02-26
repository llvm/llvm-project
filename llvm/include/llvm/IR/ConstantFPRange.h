//===- ConstantFPRange.h - Represent a range for floating-point -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Represent a range of possible values that may occur when the program is run
// for a floating-point value. This keeps track of a lower and upper bound for
// the constant.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_CONSTANTFPRANGE_H
#define LLVM_IR_CONSTANTFPRANGE_H

#include "llvm/ADT/APFloat.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/Compiler.h"
#include <optional>

namespace llvm {

class raw_ostream;
struct KnownFPClass;

/// This class represents a range of floating-point values.
class [[nodiscard]] ConstantFPRange {
  APFloat Lower, Upper;
  bool MaybeQNaN : 1;
  bool MaybeSNaN : 1;
  bool SignBitMaybeZero : 1;
  bool SignBitMaybeOne : 1;

  /// Create empty constant range with same semantics.
  ConstantFPRange getEmpty() const {
    return ConstantFPRange(getSemantics(), /*IsFullSet=*/false);
  }

  /// Create full constant range with same semantics.
  ConstantFPRange getFull() const {
    return ConstantFPRange(getSemantics(), /*IsFullSet=*/true);
  }

public:
  /// Initialize a full or empty set for the specified semantics.
  explicit ConstantFPRange(const fltSemantics &FloatSema, bool IsFullSet);

  /// Initialize a range to hold the single specified value.
  ConstantFPRange(const APFloat &Value);

  /// Initialize a range of values explicitly.
  ConstantFPRange(APFloat Lower, APFloat Upper, bool MaybeQNaN, bool MaybeSNaN,
                  bool SignBitMaybeZero, bool SignBitMaybeOne);

  /// Create empty constant range with the given semantics.
  static ConstantFPRange getEmpty(const fltSemantics &FloatSema) {
    return ConstantFPRange(FloatSema, /*IsFullSet=*/false);
  }

  /// Create full constant range with the given semantics.
  static ConstantFPRange getFull(const fltSemantics &FloatSema) {
    return ConstantFPRange(FloatSema, /*IsFullSet=*/true);
  }

  /// Initialize a range based on a known floating-point classes constraint.
  static ConstantFPRange fromKnownFPClass(const KnownFPClass &Known);

  /// Produce the exact range such that all values in the returned range satisfy
  /// the given predicate with any value contained within Other. Formally, this
  /// returns the exact answer when the superset of 'union over all y in Other
  /// is exactly same as the subset of intersection over all y in Other.
  /// { x : fcmp op x y is true}'.
  ///
  /// Example: Pred = olt and Other = float 3 returns [-inf, 3)
  static ConstantFPRange makeExactFCmpRegion(FCmpInst::Predicate Pred,
                                             const APFloat &Other);

  /// Does the predicate \p Pred hold between ranges this and \p Other?
  /// NOTE: false does not mean that inverse predicate holds!
  bool fcmp(FCmpInst::Predicate Pred, const ConstantFPRange &Other) const;

  /// Return the lower value for this range.
  const APFloat &getLower() const { return Lower; }

  /// Return the upper value for this range.
  const APFloat &getUpper() const { return Upper; }

  /// Get the semantics of this ConstantFPRange.
  const fltSemantics &getSemantics() const { return Lower.getSemantics(); }

  /// Return true if this set contains all of the elements possible
  /// for this data-type.
  bool isFullSet() const;

  /// Return true if this set contains no members.
  bool isEmptySet() const;

  /// Return true if the specified value is in the set.
  bool contains(const APFloat &Val) const;

  /// Return true if the other range is a subset of this one.
  bool contains(const ConstantFPRange &CR) const;

  /// If this set contains a single element, return it, otherwise return null.
  const APFloat *getSingleElement() const;

  /// Return true if this set contains exactly one member.
  bool isSingleElement() const { return getSingleElement() != nullptr; }

  /// Return true if the sign bit of all values in this range is 1.
  /// Return false if the sign bit of all values in this range is 0.
  /// Otherwise, return std::nullopt.
  std::optional<bool> getSignBit();

  /// Return true if this range is equal to another range.
  bool operator==(const ConstantFPRange &CR) const;
  bool operator!=(const ConstantFPRange &CR) const { return !operator==(CR); }

  /// Return known floating-point classes for values in this range.
  KnownFPClass toKnownFPClass();

  /// Print out the bounds to a stream.
  void print(raw_ostream &OS) const;

  /// Allow printing from a debugger easily.
  void dump() const;
};

inline raw_ostream &operator<<(raw_ostream &OS, const ConstantFPRange &CR) {
  CR.print(OS);
  return OS;
}

} // end namespace llvm

#endif // LLVM_IR_CONSTANTFPRANGE_H
