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
class AddressRange {
  uint64_t Start;
  uint64_t End;
public:
  AddressRange(uint64_t S = 0, uint64_t E = 0) : Start(S), End(E) {}
  /// Access to the size must use the size() accessor to ensure the correct
  /// answer. This allows an AddressRange to be constructed with invalid
  /// address ranges where the end address is less that the start address
  /// either because it was not set, or because of incorrect data.
  uint64_t size() const { return Start < End ? End - Start : 0; }
  void setStartAddress(uint64_t Addr) { Start = Addr; }
  void setEndAddress(uint64_t Addr) { End = Addr; }
  void setSize(uint64_t Size) { End = Start + Size; }
  uint64_t startAddress() const { return Start; }
  /// Access to the end address must use the size() accessor to ensure the
  /// correct answer. This allows an AddressRange to be constructed with
  /// invalid address ranges where the end address is less that the start
  /// address either because it was not set, or because of incorrect data.
  uint64_t endAddress() const { return Start + size(); }
  void clear() {
    Start = 0;
    End = 0;
  }
  bool contains(uint64_t Addr) const { return Start <= Addr && Addr < endAddress(); }
  bool isContiguousWith(const AddressRange &R) const {
    return (Start <= R.endAddress()) && (endAddress() >= R.Start);
  }
  bool intersects(const AddressRange &R) const {
    return (Start < R.endAddress()) && (endAddress() > R.Start);
  }
  bool intersect(const AddressRange &R) {
    if (intersects(R)) {
      Start = std::min<uint64_t>(Start, R.Start);
      End = std::max<uint64_t>(endAddress(), R.endAddress());
      return true;
    }
    return false;
  }
};

inline bool operator==(const AddressRange &LHS, const AddressRange &RHS) {
  return LHS.startAddress() == RHS.startAddress() && LHS.endAddress() == RHS.endAddress();
}
inline bool operator!=(const AddressRange &LHS, const AddressRange &RHS) {
  return LHS.startAddress() != RHS.startAddress() || LHS.endAddress() != RHS.endAddress();
}
inline bool operator<(const AddressRange &LHS, const AddressRange &RHS) {
  if (LHS.startAddress() == RHS.startAddress())
    return LHS.endAddress() < RHS.endAddress();
  return LHS.startAddress() < RHS.startAddress();
}

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
  void insert(const AddressRange &R);
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
