//===-- DWARFDebugAranges.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARFDEBUGARANGES_H
#define LLVM_DEBUGINFO_DWARFDEBUGARANGES_H

#include "DWARFDebugArangeSet.h"
#include <list>

namespace llvm {

class DWARFContext;

class DWARFDebugAranges {
public:
  struct Range {
    explicit Range(uint64_t lo = -1ULL, uint64_t hi = -1ULL,
                   uint32_t off = -1U)
      : LoPC(lo), Length(hi-lo), Offset(off) {}

    void clear() {
      LoPC = -1ULL;
      Length = 0;
      Offset = -1U;
    }

    void setHiPC(uint64_t HiPC) {
      if (HiPC == -1ULL || HiPC <= LoPC)
        Length = 0;
      else
        Length = HiPC - LoPC;
    }
    uint64_t HiPC() const {
      if (Length)
        return LoPC + Length;
      return -1ULL;
    }
    bool isValidRange() const { return Length > 0; }

    static bool SortedOverlapCheck(const Range &curr_range,
                                   const Range &next_range, uint32_t n) {
      if (curr_range.Offset != next_range.Offset)
        return false;
      return curr_range.HiPC() + n >= next_range.LoPC;
    }

    bool contains(const Range &range) const {
      return LoPC <= range.LoPC && range.HiPC() <= HiPC();
    }

    void dump(raw_ostream &OS) const;
    uint64_t LoPC; // Start of address range
    uint32_t Length; // End of address range (not including this address)
    uint32_t Offset; // Offset of the compile unit or die
  };

  void clear() { Aranges.clear(); }
  bool allRangesAreContiguous(uint64_t& LoPC, uint64_t& HiPC) const;
  bool getMaxRange(uint64_t& LoPC, uint64_t& HiPC) const;
  bool extract(DataExtractor debug_aranges_data);
  bool generate(DWARFContext *ctx);

  // Use append range multiple times and then call sort
  void appendRange(uint32_t cu_offset, uint64_t low_pc, uint64_t high_pc);
  void sort(bool minimize, uint32_t n);

  const Range *rangeAtIndex(uint32_t idx) const {
    if (idx < Aranges.size())
      return &Aranges[idx];
    return NULL;
  }
  void dump(raw_ostream &OS) const;
  uint32_t findAddress(uint64_t address) const;
  bool isEmpty() const { return Aranges.empty(); }
  uint32_t getNumRanges() const { return Aranges.size(); }

  uint32_t offsetAtIndex(uint32_t idx) const {
    if (idx < Aranges.size())
      return Aranges[idx].Offset;
    return -1U;
  }

  typedef std::vector<Range>              RangeColl;
  typedef RangeColl::const_iterator       RangeCollIterator;

private:
  RangeColl Aranges;
};

}

#endif
