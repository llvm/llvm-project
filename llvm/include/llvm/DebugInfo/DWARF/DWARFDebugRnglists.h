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

  Header HeaderData;
  std::vector<uint64_t> Offsets;
  std::vector<DWARFAddressRangesVector> Ranges;

public:
  void clear();
  Error extract(DWARFDataExtractor Data, uint32_t *OffsetPtr);
  void dump(raw_ostream &OS) const;

  /// Returns the length of this table, including the length field, or 0 if the
  /// length has not been determined (e.g. because the table has not yet been
  /// parsed, or there was a problem in parsing).
  uint32_t length() const;
};

} // end namespace llvm

#endif // LLVM_DEBUGINFO_DWARFDEBUGRNGLISTS_H
