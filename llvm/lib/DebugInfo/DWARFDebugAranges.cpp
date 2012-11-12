//===-- DWARFDebugAranges.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DWARFDebugAranges.h"
#include "DWARFCompileUnit.h"
#include "DWARFContext.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
using namespace llvm;

// Compare function DWARFDebugAranges::Range structures
static bool RangeLessThan(const DWARFDebugAranges::Range &range1,
                          const DWARFDebugAranges::Range &range2) {
  return range1.LoPC < range2.LoPC;
}

namespace {
  class CountArangeDescriptors {
  public:
    CountArangeDescriptors(uint32_t &count_ref) : Count(count_ref) {}
    void operator()(const DWARFDebugArangeSet &set) {
      Count += set.getNumDescriptors();
    }
    uint32_t &Count;
  };

  class AddArangeDescriptors {
  public:
    AddArangeDescriptors(DWARFDebugAranges::RangeColl &ranges)
      : RangeCollection(ranges) {}
    void operator()(const DWARFDebugArangeSet& set) {
      const DWARFDebugArangeSet::Descriptor* arange_desc_ptr;
      DWARFDebugAranges::Range range;
      range.Offset = set.getCompileUnitDIEOffset();

      for (uint32_t i=0; (arange_desc_ptr = set.getDescriptor(i)) != NULL; ++i){
        range.LoPC = arange_desc_ptr->Address;
        range.Length = arange_desc_ptr->Length;

        // Insert each item in increasing address order so binary searching
        // can later be done!
        DWARFDebugAranges::RangeColl::iterator insert_pos =
          std::lower_bound(RangeCollection.begin(), RangeCollection.end(),
                           range, RangeLessThan);
        RangeCollection.insert(insert_pos, range);
      }
    }
    DWARFDebugAranges::RangeColl& RangeCollection;
  };
}

bool DWARFDebugAranges::extract(DataExtractor debug_aranges_data) {
  if (debug_aranges_data.isValidOffset(0)) {
    uint32_t offset = 0;

    typedef std::vector<DWARFDebugArangeSet> SetCollection;
    SetCollection sets;

    DWARFDebugArangeSet set;
    Range range;
    while (set.extract(debug_aranges_data, &offset))
      sets.push_back(set);

    uint32_t count = 0;

    std::for_each(sets.begin(), sets.end(), CountArangeDescriptors(count));

    if (count > 0) {
      Aranges.reserve(count);
      AddArangeDescriptors range_adder(Aranges);
      std::for_each(sets.begin(), sets.end(), range_adder);
    }
  }
  return false;
}

bool DWARFDebugAranges::generate(DWARFContext *ctx) {
  clear();
  if (ctx) {
    const uint32_t num_compile_units = ctx->getNumCompileUnits();
    for (uint32_t cu_idx = 0; cu_idx < num_compile_units; ++cu_idx) {
      DWARFCompileUnit *cu = ctx->getCompileUnitAtIndex(cu_idx);
      if (cu)
        cu->buildAddressRangeTable(this, true);
    }
  }
  sort(true, /* overlap size */ 0);
  return !isEmpty();
}

void DWARFDebugAranges::dump(raw_ostream &OS) const {
  const uint32_t num_ranges = getNumRanges();
  for (uint32_t i = 0; i < num_ranges; ++i) {
    const Range &range = Aranges[i];
    OS << format("0x%8.8x: [0x%8.8" PRIx64 " - 0x%8.8" PRIx64 ")\n",
                 range.Offset, (uint64_t)range.LoPC, (uint64_t)range.HiPC());
  }
}

void DWARFDebugAranges::Range::dump(raw_ostream &OS) const {
  OS << format("{0x%8.8x}: [0x%8.8" PRIx64 " - 0x%8.8" PRIx64 ")\n",
               Offset, LoPC, HiPC());
}

void DWARFDebugAranges::appendRange(uint32_t offset, uint64_t low_pc,
                                    uint64_t high_pc) {
  if (!Aranges.empty()) {
    if (Aranges.back().Offset == offset && Aranges.back().HiPC() == low_pc) {
      Aranges.back().setHiPC(high_pc);
      return;
    }
  }
  Aranges.push_back(Range(low_pc, high_pc, offset));
}

void DWARFDebugAranges::sort(bool minimize, uint32_t n) {
  const size_t orig_arange_size = Aranges.size();
  // Size of one? If so, no sorting is needed
  if (orig_arange_size <= 1)
    return;
  // Sort our address range entries
  std::stable_sort(Aranges.begin(), Aranges.end(), RangeLessThan);

  if (!minimize)
    return;

  // Most address ranges are contiguous from function to function
  // so our new ranges will likely be smaller. We calculate the size
  // of the new ranges since although std::vector objects can be resized,
  // the will never reduce their allocated block size and free any excesss
  // memory, so we might as well start a brand new collection so it is as
  // small as possible.

  // First calculate the size of the new minimal arange vector
  // so we don't have to do a bunch of re-allocations as we
  // copy the new minimal stuff over to the new collection.
  size_t minimal_size = 1;
  for (size_t i = 1; i < orig_arange_size; ++i) {
    if (!Range::SortedOverlapCheck(Aranges[i-1], Aranges[i], n))
      ++minimal_size;
  }

  // If the sizes are the same, then no consecutive aranges can be
  // combined, we are done.
  if (minimal_size == orig_arange_size)
    return;

  // Else, make a new RangeColl that _only_ contains what we need.
  RangeColl minimal_aranges;
  minimal_aranges.resize(minimal_size);
  uint32_t j = 0;
  minimal_aranges[j] = Aranges[0];
  for (size_t i = 1; i < orig_arange_size; ++i) {
    if(Range::SortedOverlapCheck (minimal_aranges[j], Aranges[i], n)) {
      minimal_aranges[j].setHiPC (Aranges[i].HiPC());
    } else {
      // Only increment j if we aren't merging
      minimal_aranges[++j] = Aranges[i];
    }
  }
  assert (j+1 == minimal_size);

  // Now swap our new minimal aranges into place. The local
  // minimal_aranges will then contian the old big collection
  // which will get freed.
  minimal_aranges.swap(Aranges);
}

uint32_t DWARFDebugAranges::findAddress(uint64_t address) const {
  if (!Aranges.empty()) {
    Range range(address);
    RangeCollIterator begin = Aranges.begin();
    RangeCollIterator end = Aranges.end();
    RangeCollIterator pos = lower_bound(begin, end, range, RangeLessThan);

    if (pos != end && pos->LoPC <= address && address < pos->HiPC()) {
      return pos->Offset;
    } else if (pos != begin) {
      --pos;
      if (pos->LoPC <= address && address < pos->HiPC())
        return (*pos).Offset;
    }
  }
  return -1U;
}

bool
DWARFDebugAranges::allRangesAreContiguous(uint64_t &LoPC, uint64_t &HiPC) const{
  if (Aranges.empty())
    return false;

  uint64_t next_addr = 0;
  RangeCollIterator begin = Aranges.begin();
  for (RangeCollIterator pos = begin, end = Aranges.end(); pos != end;
       ++pos) {
    if (pos != begin && pos->LoPC != next_addr)
      return false;
    next_addr = pos->HiPC();
  }
  // We checked for empty at the start of function so front() will be valid.
  LoPC = Aranges.front().LoPC;
  // We checked for empty at the start of function so back() will be valid.
  HiPC = Aranges.back().HiPC();
  return true;
}

bool DWARFDebugAranges::getMaxRange(uint64_t &LoPC, uint64_t &HiPC) const {
  if (Aranges.empty())
    return false;
  // We checked for empty at the start of function so front() will be valid.
  LoPC = Aranges.front().LoPC;
  // We checked for empty at the start of function so back() will be valid.
  HiPC = Aranges.back().HiPC();
  return true;
}
