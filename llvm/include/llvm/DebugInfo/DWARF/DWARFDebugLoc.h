//===- DWARFDebugLoc.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARF_DWARFDEBUGLOC_H
#define LLVM_DEBUGINFO_DWARF_DWARFDEBUGLOC_H

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDataExtractor.h"
#include "llvm/DebugInfo/DWARF/DWARFRelocMap.h"
#include <cstdint>

namespace llvm {
class DWARFUnit;
class MCRegisterInfo;
class raw_ostream;

class DWARFDebugLoc {
public:
  /// A single location within a location list.
  struct Entry {
    /// The beginning address of the instruction range.
    uint64_t Begin;
    /// The ending address of the instruction range.
    uint64_t End;
    /// The location of the variable within the specified range.
    SmallVector<uint8_t, 4> Loc;
  };

  /// A list of locations that contain one variable.
  struct LocationList {
    /// The beginning offset where this location list is stored in the debug_loc
    /// section.
    uint64_t Offset;
    /// All the locations in which the variable is stored.
    SmallVector<Entry, 2> Entries;
    /// Dump this list on OS.
    void dump(raw_ostream &OS, uint64_t BaseAddress, bool IsLittleEndian,
              unsigned AddressSize, const MCRegisterInfo *MRI, DWARFUnit *U,
              DIDumpOptions DumpOpts,
              unsigned Indent) const;
  };

private:
  using LocationLists = SmallVector<LocationList, 4>;

  /// A list of all the variables in the debug_loc section, each one describing
  /// the locations in which the variable is stored.
  LocationLists Locations;

  unsigned AddressSize;

  bool IsLittleEndian;

public:
  /// Print the location lists found within the debug_loc section.
  void dump(raw_ostream &OS, const MCRegisterInfo *RegInfo, DIDumpOptions DumpOpts,
            Optional<uint64_t> Offset) const;

  /// Parse the debug_loc section accessible via the 'data' parameter using the
  /// address size also given in 'data' to interpret the address ranges.
  void parse(const DWARFDataExtractor &data);

  /// Return the location list at the given offset or nullptr.
  LocationList const *getLocationListAtOffset(uint64_t Offset) const;

  Expected<LocationList>
  parseOneLocationList(const DWARFDataExtractor &Data, uint64_t *Offset);
};

class DWARFDebugLoclists {
public:
  // Unconstructible.
  DWARFDebugLoclists() = delete;

  struct Entry {
    uint8_t Kind;
    uint64_t Offset;
    uint64_t Value0;
    uint64_t Value1;
    SmallVector<uint8_t, 4> Loc;
    void dump(raw_ostream &OS, uint64_t &BaseAddr, bool IsLittleEndian,
              unsigned AddressSize, const MCRegisterInfo *MRI, DWARFUnit *U,
              DIDumpOptions DumpOpts, unsigned Indent, size_t MaxEncodingStringLength) const;
  };

  /// Call the user-provided callback for each entry (including the end-of-list
  /// entry) in the location list starting at \p Offset. The callback can return
  /// false to terminate the iteration early. Returns an error if it was unable
  /// to parse the entire location list correctly. Upon successful termination
  /// \p Offset will be updated point past the end of the list.
  static Error visitLocationList(const DWARFDataExtractor &Data,
                                 uint64_t *Offset, uint16_t Version,
                                 llvm::function_ref<bool(const Entry &)> F);

  /// Dump the location list at the given \p Offset. The function returns true
  /// iff it has successfully reched the end of the list. This means that one
  /// can attempt to parse another list after the current one (\p Offset will be
  /// updated to point past the end of the current list).
  static bool dumpLocationList(const DWARFDataExtractor &Data, uint64_t *Offset,
                               uint16_t Version, raw_ostream &OS,
                               uint64_t BaseAddr, const MCRegisterInfo *MRI,
                               DWARFUnit *U, DIDumpOptions DumpOpts,
                               unsigned Indent);

  /// Dump all location lists within the given range.
  static void dumpRange(const DWARFDataExtractor &Data, uint64_t StartOffset,
                        uint64_t Size, uint16_t Version, raw_ostream &OS,
                        uint64_t BaseAddr, const MCRegisterInfo *MRI,
                        DIDumpOptions DumpOpts);
};

} // end namespace llvm

#endif // LLVM_DEBUGINFO_DWARF_DWARFDEBUGLOC_H
