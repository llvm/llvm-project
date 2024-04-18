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
// its upper.
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
  SmallVectorImpl<ConstantRange>::iterator begin() { return Ranges.begin(); }
  SmallVectorImpl<ConstantRange>::iterator end() { return Ranges.end(); }
  SmallVectorImpl<ConstantRange>::const_iterator begin() const {
    return Ranges.begin();
  }
  SmallVectorImpl<ConstantRange>::const_iterator end() const {
    return Ranges.end();
  }
  ConstantRange getRange(unsigned i) const {
    assert(i < Ranges.size());
    return Ranges[i];
  }

  /// Return true if this list contains no members.
  bool empty() const { return Ranges.empty(); }

  /// Get the bit width of this ConstantRangeList.
  uint32_t getBitWidth() const { return 64; }

  /// Return the number of ranges in this ConstantRangeList.
  size_t size() const { return Ranges.size(); }

  /// Insert a new range to Ranges and keep the list ordered.
  void insert(const ConstantRange &NewRange);
  void insert(int64_t Lower, int64_t Upper) {
    insert(ConstantRange(APInt(64, Lower, /*isSigned=*/true),
                         APInt(64, Upper, /*isSigned=*/true)));
  }

  // Append a new Range to Ranges. Caller should make sure
  // the list is still ordered after appending.
  void append(const ConstantRange &Range) { Ranges.push_back(Range); }
  void append(int64_t Lower, int64_t Upper) {
    append(ConstantRange(APInt(64, Lower, /*isSigned=*/true),
                         APInt(64, Upper, /*isSigned=*/true)));
  }

  /// Return true if this range list is equal to another range list.
  bool operator==(const ConstantRangeList &CRL) const {
    return Ranges == CRL.Ranges;
  }
  bool operator!=(const ConstantRangeList &CRL) const {
    return !operator==(CRL);
  }

  /// Print out the ranges to a stream.
  void print(raw_ostream &OS) const;
};

} // end namespace llvm

#endif // LLVM_IR_CONSTANTRANGELIST_H
