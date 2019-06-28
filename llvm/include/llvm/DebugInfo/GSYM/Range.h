//===- AddressRange.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_GSYM_RANGE_H
#define LLVM_DEBUGINFO_GSYM_RANGE_H

#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <stdint.h>
#include <vector>

#define HEX8(v) llvm::format_hex(v, 4)
#define HEX16(v) llvm::format_hex(v, 6)
#define HEX32(v) llvm::format_hex(v, 10)
#define HEX64(v) llvm::format_hex(v, 18)

namespace llvm {
class raw_ostream;

namespace gsym {

/// A class that represents an address range. The range is specified using
/// a start and an end address.
struct AddressRange {
  uint64_t Start;
  uint64_t End;
  AddressRange() : Start(0), End(0) {}
  AddressRange(uint64_t S, uint64_t E) : Start(S), End(E) {}
  uint64_t size() const { return End - Start; }
  bool contains(uint64_t Addr) const { return Start <= Addr && Addr < End; }
  bool intersects(const AddressRange &R) const {
    return Start < R.End && R.Start < End;
  }

  bool operator==(const AddressRange &R) const {
    return Start == R.Start && End == R.End;
  }
  bool operator!=(const AddressRange &R) const {
    return !(*this == R);
  }
  bool operator<(const AddressRange &R) const {
    return std::make_pair(Start, End) < std::make_pair(R.Start, R.End);
  }
};

raw_ostream &operator<<(raw_ostream &OS, const AddressRange &R);

/// The AddressRanges class helps normalize address range collections.
/// This class keeps a sorted vector of AddressRange objects and can perform
/// insertions and searches efficiently. The address ranges are always sorted
/// and never contain any invalid or empty address ranges. This allows us to
/// emit address ranges into the GSYM file efficiently. Intersecting address
/// ranges are combined during insertion so that we can emit the most compact
/// representation for address ranges when writing to disk.
class AddressRanges {
protected:
  using Collection = std::vector<AddressRange>;
  Collection Ranges;
public:
  void clear() { Ranges.clear(); }
  bool empty() const { return Ranges.empty(); }
  bool contains(uint64_t Addr) const;
  void insert(AddressRange Range);
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

raw_ostream &operator<<(raw_ostream &OS, const AddressRanges &AR);

} // namespace gsym
} // namespace llvm

#endif // #ifndef LLVM_DEBUGINFO_GSYM_RANGE_H
