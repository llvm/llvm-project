//===-- DWARFDebugRangeList.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARFDEBUGRANGELIST_H
#define LLVM_DEBUGINFO_DWARFDEBUGRANGELIST_H

#include "llvm/Support/DataExtractor.h"
#include <vector>

namespace llvm {

class raw_ostream;

class DWARFDebugRangeList {
public:
  struct RangeListEntry {
    // A beginning address offset. This address offset has the size of an
    // address and is relative to the applicable base address of the
    // compilation unit referencing this range list. It marks the beginning
    // of an address range.
    uint64_t StartAddress;
    // An ending address offset. This address offset again has the size of
    // an address and is relative to the applicable base address of the
    // compilation unit referencing this range list. It marks the first
    // address past the end of the address range. The ending address must
    // be greater than or equal to the beginning address.
    uint64_t EndAddress;
    // The end of any given range list is marked by an end of list entry,
    // which consists of a 0 for the beginning address offset
    // and a 0 for the ending address offset.
    bool isEndOfListEntry() const {
      return (StartAddress == 0) && (EndAddress == 0);
    }
    // A base address selection entry consists of:
    // 1. The value of the largest representable address offset
    // (for example, 0xffffffff when the size of an address is 32 bits).
    // 2. An address, which defines the appropriate base address for
    // use in interpreting the beginning and ending address offsets of
    // subsequent entries of the location list.
    bool isBaseAddressSelectionEntry(uint8_t AddressSize) const {
      assert(AddressSize == 4 || AddressSize == 8);
      if (AddressSize == 4)
        return StartAddress == -1U;
      else
        return StartAddress == -1ULL;
    }
    bool containsAddress(uint64_t BaseAddress, uint64_t Address) const {
      return (BaseAddress + StartAddress <= Address) &&
             (Address < BaseAddress + EndAddress);
    }
  };

private:
  // Offset in .debug_ranges section.
  uint32_t Offset;
  uint8_t AddressSize;
  std::vector<RangeListEntry> Entries;

public:
  DWARFDebugRangeList() { clear(); }
  void clear();
  void dump(raw_ostream &OS) const;
  bool extract(DataExtractor data, uint32_t *offset_ptr);
  /// containsAddress - Returns true if range list contains the given
  /// address. Has to be passed base address of the compile unit that
  /// references this range list.
  bool containsAddress(uint64_t BaseAddress, uint64_t Address) const;
};

}  // namespace llvm

#endif  // LLVM_DEBUGINFO_DWARFDEBUGRANGELIST_H
