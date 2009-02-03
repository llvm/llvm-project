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
// Note that ConstantRange always keeps unsigned values.
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_CONSTANT_RANGE_H
#define LLVM_SUPPORT_CONSTANT_RANGE_H

#include "llvm/ADT/APInt.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {

class ConstantRange {
  APInt Lower, Upper;
  static ConstantRange intersect1Wrapped(const ConstantRange &LHS,
                                         const ConstantRange &RHS);
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
  ConstantRange(const APInt& Lower, const APInt& Upper);

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

  /// contains - Return true if the specified value is in the set.
  ///
  bool contains(const APInt &Val) const;

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

  /// intersectWith - Return the range that results from the intersection of
  /// this range with another range.  The resultant range is pruned as much as
  /// possible, but there may be cases where elements are included that are in
  /// one of the sets but not the other.  For example: [100, 8) intersect [3,
  /// 120) yields [3, 120)
  ///
  ConstantRange intersectWith(const ConstantRange &CR) const;

  /// maximalIntersectWith - Return the range that results from the intersection
  /// of this range with another range.  The resultant range is guaranteed to
  /// include all elements contained in both input ranges, and to have the
  /// smallest possible set size that does so.  Because there may be two
  /// intersections with the same set size, A.maximalIntersectWith(B) might not
  /// be equal to B.maximalIntersectWith(A).
  ///
  ConstantRange maximalIntersectWith(const ConstantRange &CR) const;

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
