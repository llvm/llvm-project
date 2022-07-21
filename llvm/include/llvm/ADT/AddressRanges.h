//===- AddressRanges.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_ADDRESSRANGES_H
#define LLVM_ADT_ADDRESSRANGES_H

#include "llvm/ADT/Optional.h"
#include <cassert>
#include <stdint.h>
#include <vector>

namespace llvm {

/// A class that represents an address range. The range is specified using
/// a start and an end address: [Start, End).
class AddressRange {
public:
  AddressRange() {}
  AddressRange(uint64_t S, uint64_t E) : Start(S), End(E) {
    assert(Start <= End);
  }
  uint64_t start() const { return Start; }
  uint64_t end() const { return End; }
  uint64_t size() const { return End - Start; }
  bool contains(uint64_t Addr) const { return Start <= Addr && Addr < End; }
  bool intersects(const AddressRange &R) const {
    return Start < R.End && R.Start < End;
  }
  bool operator==(const AddressRange &R) const {
    return Start == R.Start && End == R.End;
  }
  bool operator!=(const AddressRange &R) const { return !(*this == R); }
  bool operator<(const AddressRange &R) const {
    return std::make_pair(Start, End) < std::make_pair(R.Start, R.End);
  }

private:
  uint64_t Start = 0;
  uint64_t End = 0;
};

/// The AddressRanges class helps normalize address range collections.
/// This class keeps a sorted vector of AddressRange objects and can perform
/// insertions and searches efficiently. The address ranges are always sorted
/// and never contain any invalid or empty address ranges. Intersecting
/// address ranges are combined during insertion.
class AddressRanges {
protected:
  using Collection = std::vector<AddressRange>;
  Collection Ranges;

public:
  void clear() { Ranges.clear(); }
  bool empty() const { return Ranges.empty(); }
  bool contains(uint64_t Addr) const;
  bool contains(AddressRange Range) const;
  Optional<AddressRange> getRangeThatContains(uint64_t Addr) const;
  void insert(AddressRange Range);
  void reserve(size_t Capacity) { Ranges.reserve(Capacity); }
  size_t size() const { return Ranges.size(); }
  bool operator==(const AddressRanges &RHS) const {
    return Ranges == RHS.Ranges;
  }
  const AddressRange &operator[](size_t i) const {
    assert(i < Ranges.size());
    return Ranges[i];
  }
  Collection::const_iterator begin() const { return Ranges.begin(); }
  Collection::const_iterator end() const { return Ranges.end(); }
};

} // namespace llvm

#endif // LLVM_ADT_ADDRESSRANGES_H
