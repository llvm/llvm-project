//===-- DWARFDebugAranges.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DWARFDebugAranges.h"
#include "DWARFDebugArangeSet.h"
#include "DWARFUnit.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Timer.h"

using namespace lldb;
using namespace lldb_private;

// Constructor
DWARFDebugAranges::DWARFDebugAranges() : m_aranges() {}

// CountArangeDescriptors
class CountArangeDescriptors {
public:
  CountArangeDescriptors(uint32_t &count_ref) : count(count_ref) {
    //      printf("constructor CountArangeDescriptors()\n");
  }
  void operator()(const DWARFDebugArangeSet &set) {
    count += set.NumDescriptors();
  }
  uint32_t &count;
};

// Extract
llvm::Error
DWARFDebugAranges::extract(const DWARFDataExtractor &debug_aranges_data) {
  lldb::offset_t offset = 0;

  DWARFDebugArangeSet set;
  Range range;
  while (debug_aranges_data.ValidOffset(offset)) {
    llvm::Error error = set.extract(debug_aranges_data, &offset);
    if (!error)
      return error;

    const uint32_t num_descriptors = set.NumDescriptors();
    if (num_descriptors > 0) {
      const dw_offset_t cu_offset = set.GetHeader().cu_offset;

      for (uint32_t i = 0; i < num_descriptors; ++i) {
        const DWARFDebugArangeSet::Descriptor &descriptor =
            set.GetDescriptorRef(i);
        m_aranges.Append(RangeToDIE::Entry(descriptor.address,
                                           descriptor.length, cu_offset));
      }
    }
    set.Clear();
  }
  return llvm::ErrorSuccess();
}

void DWARFDebugAranges::Dump(Log *log) const {
  if (log == nullptr)
    return;

  const size_t num_entries = m_aranges.GetSize();
  for (size_t i = 0; i < num_entries; ++i) {
    const RangeToDIE::Entry *entry = m_aranges.GetEntryAtIndex(i);
    if (entry)
      LLDB_LOGF(log, "0x%8.8x: [0x%" PRIx64 " - 0x%" PRIx64 ")", entry->data,
                entry->GetRangeBase(), entry->GetRangeEnd());
  }
}

void DWARFDebugAranges::AppendRange(dw_offset_t offset, dw_addr_t low_pc,
                                    dw_addr_t high_pc) {
  if (high_pc > low_pc)
    m_aranges.Append(RangeToDIE::Entry(low_pc, high_pc - low_pc, offset));
}

void DWARFDebugAranges::Sort(bool minimize) {
  static Timer::Category func_cat(LLVM_PRETTY_FUNCTION);
  Timer scoped_timer(func_cat, "%s this = %p", LLVM_PRETTY_FUNCTION,
                     static_cast<void *>(this));

  m_aranges.Sort();
  m_aranges.CombineConsecutiveEntriesWithEqualData();
}

// FindAddress
dw_offset_t DWARFDebugAranges::FindAddress(dw_addr_t address) const {
  const RangeToDIE::Entry *entry = m_aranges.FindEntryThatContains(address);
  if (entry)
    return entry->data;
  return DW_INVALID_OFFSET;
}
