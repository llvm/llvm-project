//===- Range.h --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
class DataExtractor;
class raw_ostream;

namespace gsym {

class FileWriter;

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
  /// AddressRange objects are encoded and decoded to be relative to a base
  /// address. This will be the FunctionInfo's start address if the AddressRange
  /// is directly contained in a FunctionInfo, or a base address of the
  /// containing parent AddressRange or AddressRanges. This allows address
  /// ranges to be efficiently encoded using ULEB128 encodings as we encode the
  /// offset and size of each range instead of full addresses. This also makes
  /// encoded addresses easy to relocate as we just need to relocate one base
  /// address.
  /// @{
  void decode(DataExtractor &Data, uint64_t BaseAddr, uint64_t &Offset);
  void encode(FileWriter &O, uint64_t BaseAddr) const;
  /// @}

  /// Skip an address range object in the specified data a the specified
  /// offset.
  ///
  /// \param Data The binary stream to read the data from.
  ///
  /// \param Offset The byte offset within \a Data.
  static void skip(DataExtractor &Data, uint64_t &Offset);
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
  bool contains(AddressRange Range) const;
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

  /// Address ranges are decoded and encoded to be relative to a base address.
  /// See the AddressRange comment for the encode and decode methods for full
  /// details.
  /// @{
  void decode(DataExtractor &Data, uint64_t BaseAddr, uint64_t &Offset);
  void encode(FileWriter &O, uint64_t BaseAddr) const;
  /// @}

  /// Skip an address range object in the specified data a the specified
  /// offset.
  ///
  /// \param Data The binary stream to read the data from.
  ///
  /// \param Offset The byte offset within \a Data.
  ///
  /// \returns The number of address ranges that were skipped.
  static uint64_t skip(DataExtractor &Data, uint64_t &Offset);
};

raw_ostream &operator<<(raw_ostream &OS, const AddressRanges &AR);

} // namespace gsym
} // namespace llvm

#endif // #ifndef LLVM_DEBUGINFO_GSYM_RANGE_H
