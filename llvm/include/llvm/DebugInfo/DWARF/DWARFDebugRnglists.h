//===- DWARFDebugRnglists.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARFDEBUGRNGLISTS_H
#define LLVM_DEBUGINFO_DWARFDEBUGRNGLISTS_H

#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/DebugInfo/DWARF/DWARFAddressRange.h"
#include "llvm/DebugInfo/DWARF/DWARFDataExtractor.h"
#include "llvm/DebugInfo/DWARF/DWARFListTable.h"
#include <cstdint>
#include <map>
#include <vector>

namespace llvm {

struct BaseAddress;
class DWARFContext;
class Error;
class raw_ostream;

/// A class representing a single range list entry.
struct RangeListEntry : public DWARFListEntryBase {
  /// The values making up the range list entry. Most represent a range with
  /// a start and end address or a start address and a length. Others are
  /// single value base addresses or end-of-list with no values. The unneeded
  /// values are semantically undefined, but initialized to 0.
  uint64_t Value0;
  uint64_t Value1;

  Error extract(DWARFDataExtractor Data, uint32_t End, uint16_t Version,
                StringRef SectionName, uint32_t *OffsetPtr, bool isDWO = false);
  bool isEndOfList() const { return EntryKind == dwarf::DW_RLE_end_of_list; }
  bool isBaseAddressSelectionEntry() const {
    return EntryKind == dwarf::DW_RLE_base_address;
  }
  uint64_t getStartAddress() const {
    assert((EntryKind == dwarf::DW_RLE_start_end ||
            EntryKind == dwarf::DW_RLE_offset_pair ||
            EntryKind == dwarf::DW_RLE_startx_length) &&
           "Unexpected range list entry kind");
    return Value0;
  }
  uint64_t getEndAddress() const {
    assert((EntryKind == dwarf::DW_RLE_start_end ||
            EntryKind == dwarf::DW_RLE_offset_pair) &&
           "Unexpected range list entry kind");
    return Value1;
  }
  void dump(raw_ostream &OS, DWARFContext *C, uint8_t AddrSize,
            uint64_t &CurrentBase, unsigned Indent, uint16_t Version,
            uint8_t MaxEncodingStringLength = 0,
            DIDumpOptions DumpOpts = {}) const;
};

/// A class representing a single rangelist.
class DWARFDebugRnglist : public DWARFListType<RangeListEntry> {
public:
  /// Build a DWARFAddressRangesVector from a rangelist.
  DWARFAddressRangesVector
  getAbsoluteRanges(llvm::Optional<BaseAddress> BaseAddr) const;
};

class DWARFDebugRnglistTable : public DWARFListTableBase<DWARFDebugRnglist> {
public:
  DWARFDebugRnglistTable(DWARFContext *C, StringRef SectionName,
                         bool isDWO = false)
      : DWARFListTableBase(C, SectionName, isDWO,
                           /* HeaderString   = */ "ranges:",
                           /* ListTypeString = */ "range",
                           dwarf::RangeListEncodingString) {}
};

} // end namespace llvm

#endif // LLVM_DEBUGINFO_DWARFDEBUGRNGLISTS_H
