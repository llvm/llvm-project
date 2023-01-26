//===- AddressRanges.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_ADDRESSRANGES_H
#define LLVM_ADT_ADDRESSRANGES_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <cassert>
#include <optional>
#include <stdint.h>

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
/// and never contain any invalid or empty address ranges.
/// Intersecting([100,200), [150,300)) and adjacent([100,200), [200,300))
/// address ranges are combined during insertion.
class AddressRanges {
protected:
  using Collection = SmallVector<AddressRange>;
  Collection Ranges;

public:
  void clear() { Ranges.clear(); }
  bool empty() const { return Ranges.empty(); }
  bool contains(uint64_t Addr) const { return find(Addr) != Ranges.end(); }
  bool contains(AddressRange Range) const {
    return find(Range) != Ranges.end();
  }
  std::optional<AddressRange> getRangeThatContains(uint64_t Addr) const {
    Collection::const_iterator It = find(Addr);
    if (It == Ranges.end())
      return std::nullopt;

    return *It;
  }
  Collection::const_iterator insert(AddressRange Range);
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

protected:
  Collection::const_iterator find(uint64_t Addr) const;
  Collection::const_iterator find(AddressRange Range) const;
};

/// AddressRangesMap class maps values to the address ranges.
/// It keeps address ranges and corresponding values. If ranges
/// are combined during insertion, then combined range keeps
/// newly inserted value.
template <typename T> class AddressRangesMap : protected AddressRanges {
public:
  void clear() {
    Ranges.clear();
    Values.clear();
  }
  bool empty() const { return AddressRanges::empty(); }
  bool contains(uint64_t Addr) const { return AddressRanges::contains(Addr); }
  bool contains(AddressRange Range) const {
    return AddressRanges::contains(Range);
  }
  void insert(AddressRange Range, T Value) {
    size_t InputSize = Ranges.size();
    Collection::const_iterator RangesIt = AddressRanges::insert(Range);
    if (RangesIt == Ranges.end())
      return;

    // make Values match to Ranges.
    size_t Idx = RangesIt - Ranges.begin();
    typename ValuesCollection::iterator ValuesIt = Values.begin() + Idx;
    if (InputSize < Ranges.size())
      Values.insert(ValuesIt, T());
    else if (InputSize > Ranges.size())
      Values.erase(ValuesIt, ValuesIt + InputSize - Ranges.size());
    assert(Ranges.size() == Values.size());

    // set value to the inserted or combined range.
    Values[Idx] = Value;
  }
  size_t size() const {
    assert(Ranges.size() == Values.size());
    return AddressRanges::size();
  }
  std::optional<std::pair<AddressRange, T>>
  getRangeValueThatContains(uint64_t Addr) const {
    Collection::const_iterator It = find(Addr);
    if (It == Ranges.end())
      return std::nullopt;

    return std::make_pair(*It, Values[It - Ranges.begin()]);
  }
  std::pair<AddressRange, T> operator[](size_t Idx) const {
    return std::make_pair(Ranges[Idx], Values[Idx]);
  }

protected:
  using ValuesCollection = SmallVector<T>;
  ValuesCollection Values;
};

} // namespace llvm

#endif // LLVM_ADT_ADDRESSRANGES_H
