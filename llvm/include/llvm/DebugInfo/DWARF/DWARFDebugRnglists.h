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
#include <map>
#include <vector>

namespace llvm {

class Error;
class raw_ostream;

/// A class representing a single rangelist.
class DWARFDebugRnglist {
public:
  struct RangeListEntry {
    /// The offset at which the entry is located in the section.
    uint32_t Offset;
    /// The DWARF encoding (DW_RLE_*).
    uint8_t EntryKind;
    /// The values making up the range list entry. Most represent a range with
    /// a start and end address or a start address and a length. Others are
    /// single value base addresses or end-of-list with no values. The unneeded
    /// values are semantically undefined, but initialized to 0.
    uint64_t Value0;
    uint64_t Value1;

    Error extract(DWARFDataExtractor Data, uint32_t End, uint32_t *OffsetPtr);
  };

  using RngListEntries = std::vector<RangeListEntry>;

private:
  RngListEntries Entries;

public:
  const RngListEntries &getEntries() const { return Entries; }
  bool empty() const { return Entries.empty(); }
  void clear() { Entries.clear(); }
  Error extract(DWARFDataExtractor Data, uint32_t HeaderOffset, uint32_t End,
                uint32_t *OffsetPtr);
};

/// A class representing a table of range lists as specified in DWARF v5.
/// The table consists of a header followed by an array of offsets into the
/// .debug_rnglists section, followed by one or more rangelists. The rangelists
/// are kept in a map where the keys are the lists' section offsets.
class DWARFDebugRnglistTable {
public:
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

private:
  uint32_t HeaderOffset;
  Header HeaderData;
  std::vector<uint32_t> Offsets;
  std::map<uint32_t, DWARFDebugRnglist> Ranges;

public:
  void clear();
  /// Extract the table header and the array of offsets.
  Error extractHeaderAndOffsets(DWARFDataExtractor Data, uint32_t *OffsetPtr);
  /// Extract an entire table, including all rangelists.
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
