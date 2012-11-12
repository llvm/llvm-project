//===-- llvm/Support/ConstantRange.h - Represent a range --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Represent a range of possible values that may occur when the program is run
// for an integral value.  This keeps track of a lower and upper bound for the
// constant, which MAY wrap around the end of the numeric range.  To do this, it
// keeps track of a [lower, upper) bound, which specifies an interval just like
// STL iterators.  When used with boolean values, the following are important
// ranges: :
//
//  [F, F) = {}     = Empty set
//  [T, F) = {T}
//  [F, T) = {F}
//  [T, T) = {F, T} = Full set
//
// The other integral ranges use min/max values for special range values. For
// example, for 8-bit types, it uses:
// [0, 0)     = {}       = Empty set
// [255, 255) = {0..255} = Full Set
//
// Note that ConstantRange can be used to represent either signed or
// unsigned ranges.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_CONSTANT_RANGE_H
#define LLVM_SUPPORT_CONSTANT_RANGE_H

#include "llvm/ADT/APInt.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {

/// ConstantRange - This class represents an range of values.
///
class ConstantRange {
  APInt Lower, Upper;

public:
  /// Initialize a full (the default) or empty set for the specified bit width.
  ///
  explicit ConstantRange(uint32_t BitWidth, bool isFullSet = true);

  /// Initialize a range to hold the single specified value.
  ///
  ConstantRange(const APInt &Value);

  /// @brief Initialize a range of values explicitly. This will assert out if
  /// Lower==Upper and Lower != Min or Max value for its type. It will also
  /// assert out if the two APInt's are not the same bit width.
  ConstantRange(const APInt &Lower, const APInt &Upper);

  /// makeICmpRegion - Produce the smallest range that contains all values that
  /// might satisfy the comparison specified by Pred when compared to any value
  /// contained within Other.
  ///
  /// Solves for range X in 'for all x in X, there exists a y in Y such that
  /// icmp op x, y is true'. Every value that might make the comparison true
  /// is included in the resulting range.
  static ConstantRange makeICmpRegion(unsigned Pred,
                                      const ConstantRange &Other);

  /// getLower - Return the lower value for this range...
  ///
  const APInt &getLower() const { return Lower; }

  /// getUpper - Return the upper value for this range...
  ///
  const APInt &getUpper() const { return Upper; }

  /// getBitWidth - get the bit width of this ConstantRange
  ///
  uint32_t getBitWidth() const { return Lower.getBitWidth(); }

  /// isFullSet - Return true if this set contains all of the elements possible
  /// for this data-type
  ///
  bool isFullSet() const;

  /// isEmptySet - Return true if this set contains no members.
  ///
  bool isEmptySet() const;

  /// isWrappedSet - Return true if this set wraps around the top of the range,
  /// for example: [100, 8)
  ///
  bool isWrappedSet() const;

  /// isSignWrappedSet - Return true if this set wraps around the INT_MIN of
  /// its bitwidth, for example: i8 [120, 140).
  ///
  bool isSignWrappedSet() const;

  /// contains - Return true if the specified value is in the set.
  ///
  bool contains(const APInt &Val) const;

  /// contains - Return true if the other range is a subset of this one.
  ///
  bool contains(const ConstantRange &CR) const;

  /// getSingleElement - If this set contains a single element, return it,
  /// otherwise return null.
  ///
  const APInt *getSingleElement() const {
    if (Upper == Lower + 1)
      return &Lower;
    return 0;
  }

  /// isSingleElement - Return true if this set contains exactly one member.
  ///
  bool isSingleElement() const { return getSingleElement() != 0; }

  /// getSetSize - Return the number of elements in this set.
  ///
  APInt getSetSize() const;

  /// getUnsignedMax - Return the largest unsigned value contained in the
  /// ConstantRange.
  ///
  APInt getUnsignedMax() const;

  /// getUnsignedMin - Return the smallest unsigned value contained in the
  /// ConstantRange.
  ///
  APInt getUnsignedMin() const;

  /// getSignedMax - Return the largest signed value contained in the
  /// ConstantRange.
  ///
  APInt getSignedMax() const;

  /// getSignedMin - Return the smallest signed value contained in the
  /// ConstantRange.
  ///
  APInt getSignedMin() const;

  /// operator== - Return true if this range is equal to another range.
  ///
  bool operator==(const ConstantRange &CR) const {
    return Lower == CR.Lower && Upper == CR.Upper;
  }
  bool operator!=(const ConstantRange &CR) const {
    return !operator==(CR);
  }

  /// subtract - Subtract the specified constant from the endpoints of this
  /// constant range.
  ConstantRange subtract(const APInt &CI) const;

  /// \brief Subtract the specified range from this range (aka relative
  /// complement of the sets).
  ConstantRange difference(const ConstantRange &CR) const;

  /// intersectWith - Return the range that results from the intersection of
  /// this range with another range.  The resultant range is guaranteed to
  /// include all elements contained in both input ranges, and to have the
  /// smallest possible set size that does so.  Because there may be two
  /// intersections with the same set size, A.intersectWith(B) might not
  /// be equal to B.intersectWith(A).
  ///
  ConstantRange intersectWith(const ConstantRange &CR) const;

  /// unionWith - Return the range that results from the union of this range
  /// with another range.  The resultant range is guaranteed to include the
  /// elements of both sets, but may contain more.  For example, [3, 9) union
  /// [12,15) is [3, 15), which includes 9, 10, and 11, which were not included
  /// in either set before.
  ///
  ConstantRange unionWith(const ConstantRange &CR) const;

  /// zeroExtend - Return a new range in the specified integer type, which must
  /// be strictly larger than the current type.  The returned range will
  /// correspond to the possible range of values if the source range had been
  /// zero extended to BitWidth.
  ConstantRange zeroExtend(uint32_t BitWidth) const;

  /// signExtend - Return a new range in the specified integer type, which must
  /// be strictly larger than the current type.  The returned range will
  /// correspond to the possible range of values if the source range had been
  /// sign extended to BitWidth.
  ConstantRange signExtend(uint32_t BitWidth) const;

  /// truncate - Return a new range in the specified integer type, which must be
  /// strictly smaller than the current type.  The returned range will
  /// correspond to the possible range of values if the source range had been
  /// truncated to the specified type.
  ConstantRange truncate(uint32_t BitWidth) const;

  /// zextOrTrunc - make this range have the bit width given by \p BitWidth. The
  /// value is zero extended, truncated, or left alone to make it that width.
  ConstantRange zextOrTrunc(uint32_t BitWidth) const;
  
  /// sextOrTrunc - make this range have the bit width given by \p BitWidth. The
  /// value is sign extended, truncated, or left alone to make it that width.
  ConstantRange sextOrTrunc(uint32_t BitWidth) const;

  /// add - Return a new range representing the possible values resulting
  /// from an addition of a value in this range and a value in \p Other.
  ConstantRange add(const ConstantRange &Other) const;

  /// sub - Return a new range representing the possible values resulting
  /// from a subtraction of a value in this range and a value in \p Other.
  ConstantRange sub(const ConstantRange &Other) const;

  /// multiply - Return a new range representing the possible values resulting
  /// from a multiplication of a value in this range and a value in \p Other.
  /// TODO: This isn't fully implemented yet.
  ConstantRange multiply(const ConstantRange &Other) const;

  /// smax - Return a new range representing the possible values resulting
  /// from a signed maximum of a value in this range and a value in \p Other.
  ConstantRange smax(const ConstantRange &Other) const;

  /// umax - Return a new range representing the possible values resulting
  /// from an unsigned maximum of a value in this range and a value in \p Other.
  ConstantRange umax(const ConstantRange &Other) const;

  /// udiv - Return a new range representing the possible values resulting
  /// from an unsigned division of a value in this range and a value in
  /// \p Other.
  ConstantRange udiv(const ConstantRange &Other) const;

  /// binaryAnd - return a new range representing the possible values resulting
  /// from a binary-and of a value in this range by a value in \p Other.
  ConstantRange binaryAnd(const ConstantRange &Other) const;

  /// binaryOr - return a new range representing the possible values resulting
  /// from a binary-or of a value in this range by a value in \p Other.
  ConstantRange binaryOr(const ConstantRange &Other) const;

  /// shl - Return a new range representing the possible values resulting
  /// from a left shift of a value in this range by a value in \p Other.
  /// TODO: This isn't fully implemented yet.
  ConstantRange shl(const ConstantRange &Other) const;

  /// lshr - Return a new range representing the possible values resulting
  /// from a logical right shift of a value in this range and a value in
  /// \p Other.
  ConstantRange lshr(const ConstantRange &Other) const;

  /// inverse - Return a new range that is the logical not of the current set.
  ///
  ConstantRange inverse() const;
  
  /// print - Print out the bounds to a stream...
  ///
  void print(raw_ostream &OS) const;

  /// dump - Allow printing from a debugger easily...
  ///
  void dump() const;
};

inline raw_ostream &operator<<(raw_ostream &OS, const ConstantRange &CR) {
  CR.print(OS);
  return OS;
}

} // End llvm namespace

#endif
