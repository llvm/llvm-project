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

#include "llvm/DebugInfo/DIContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDataExtractor.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugRangeList.h"
#include <cstdint>
#include <vector>

namespace llvm {

class Error;
class raw_ostream;

class DWARFDebugRnglists {
private:
  struct Header {
    /// The total length of the entries for this table, not including the length
    /// field itself.
    uint32_t Length = 0;
    /// The DWARF version number.
    uint16_t Version;
    /// The size in bytes of an address on the target architecture. For
    /// segmented addressing, this is the size of the offset portion of the
    /// address.
    uint8_t AddrSize;
    /// The size in bytes of a segment selector on the target architecture.
    /// If the target system uses a flat address space, this value is 0.
    uint8_t SegSize;
    /// The number of offsets that follow the header before the range lists.
    uint32_t OffsetEntryCount;
  };

public:
  struct RangeListEntry {
    /// The offset at which the entry is located in the section.
    const uint32_t Offset;
    /// The DWARF encoding (DW_RLE_*).
    const uint8_t EntryKind;
    /// The values making up the range list entry. Most represent a range with
    /// a start and end address or a start address and a length. Others are
    /// single value base addresses or end-of-list with no values. The unneeded
    /// values are semantically undefined, but initialized to 0.
    const uint64_t Value0;
    const uint64_t Value1;
  };

  using DWARFRangeList = std::vector<RangeListEntry>;

private:
  uint32_t HeaderOffset;
  Header HeaderData;
  std::vector<uint32_t> Offsets;
  std::vector<DWARFRangeList> Ranges;
  // The length of the longest encoding string we encountered during parsing.
  uint8_t MaxEncodingStringLength = 0;

public:
  void clear();
  Error extract(DWARFDataExtractor Data, uint32_t *OffsetPtr);
  uint32_t getHeaderOffset() const { return HeaderOffset; }
  void dump(raw_ostream &OS, DIDumpOptions DumpOpts) const;

  /// Returns the length of this table, including the length field, or 0 if the
  /// length has not been determined (e.g. because the table has not yet been
  /// parsed, or there was a problem in parsing).
  uint32_t length() const;
};

} // end namespace llvm

#endif // LLVM_DEBUGINFO_DWARFDEBUGRNGLISTS_H
