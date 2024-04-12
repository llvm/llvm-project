//===- ConstantRangeList.h - A list of constant range -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Represent a list of signed ConstantRange and do NOT support wrap around the
// end of the numeric range. Ranges in the list are ordered and no overlapping.
// Ranges should have the same bitwidth. Each range's lower should be less than
// its upper. Special lists (take 8-bit as an example):
//
// {[0, 0)}     = Empty set
// {[255, 255)} = Full Set
//
// For EmptySet or FullSet, the list size is 1 but it's not allowed to access
// the range.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_CONSTANTRANGELIST_H
#define LLVM_IR_CONSTANTRANGELIST_H

#include "llvm/ADT/APInt.h"
#include "llvm/IR/ConstantRange.h"
#include <cstddef>
#include <cstdint>

namespace llvm {

class raw_ostream;

/// This class represents a list of constant ranges.
class [[nodiscard]] ConstantRangeList {
  SmallVector<ConstantRange, 2> Ranges;

public:
  /// Initialize a full or empty set for the specified bit width.
  explicit ConstantRangeList(uint32_t BitWidth, bool isFullSet);

  ConstantRangeList(int64_t Lower, int64_t Upper);

  /// It's not allowed to access EmptySet's or FullSet's range.
  SmallVectorImpl<ConstantRange>::iterator begin() {
    assert(!isEmptySet() && !isFullSet());
    return Ranges.begin();
  }
  SmallVectorImpl<ConstantRange>::iterator end() { return Ranges.end(); }
  SmallVectorImpl<ConstantRange>::const_iterator begin() const {
    assert(!isEmptySet() && !isFullSet());
    return Ranges.begin();
  }
  SmallVectorImpl<ConstantRange>::const_iterator end() const {
    return Ranges.end();
  }
  ConstantRange getRange(unsigned i) const {
    assert(!isEmptySet() && !isFullSet() && i < Ranges.size());
    return Ranges[i];
  }

  /// Return true if this set contains no members.
  bool isEmptySet() const {
    return Ranges.size() == 1 && Ranges[0].isEmptySet();
  }

  /// Return true if this set contains all of the elements possible
  /// for this data-type.
  bool isFullSet() const { return Ranges.size() == 1 && Ranges[0].isFullSet(); }

  /// Get the bit width of this ConstantRangeList.
  uint32_t getBitWidth() const { return Ranges[0].getBitWidth(); }

  /// For EmptySet or FullSet, the CRL size is 1 not 0.
  size_t size() const { return Ranges.size(); }

  /// Insert a range to Ranges. Keep the list ordered
  /// and no overlapping (merge ranges if needed).
  void insert(const ConstantRange &Range);
  void insert(int64_t Lower, int64_t Upper) {
    insert(ConstantRange(APInt(64, Lower, /*isSigned=*/true),
                         APInt(64, Upper, /*isSigned=*/true)));
  }

  /// Return true if this range is equal to another range.
  bool operator==(const ConstantRangeList &CRL) const {
    if (size() != CRL.size())
      return false;
    for (size_t i = 0; i < size(); ++i) {
      if (Ranges[i] != CRL.Ranges[i])
        return false;
    }
    return true;
  }
  bool operator!=(const ConstantRangeList &CRL) const {
    return !operator==(CRL);
  }

  /// Print out the ranges to a stream.
  void print(raw_ostream &OS) const;
};

} // end namespace llvm

#endif // LLVM_IR_CONSTANTRANGELIST_H
