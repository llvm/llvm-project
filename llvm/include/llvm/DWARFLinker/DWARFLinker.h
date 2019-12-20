//===- DWARFLinker.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DWARFLINKER_DWARFLINKER_H
#define LLVM_DWARFLINKER_DWARFLINKER_H

#include "llvm/CodeGen/NonRelocatableStringpool.h"
#include "llvm/DWARFLinker/DWARFLinkerDeclContext.h"
#include "llvm/DebugInfo/DWARF/DWARFCompileUnit.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include <map>

namespace llvm {

enum class DwarfLinkerClient { Dsymutil, LLD, General };

/// Partial address range. Besides an offset, only the
/// HighPC is stored. The structure is stored in a map where the LowPC is the
/// key.
struct ObjFileAddressRange {
  /// Function HighPC.
  uint64_t HighPC;
  /// Offset to apply to the linked address.
  /// should be 0 for not-linked object file.
  int64_t Offset;

  ObjFileAddressRange(uint64_t EndPC, int64_t Offset)
      : HighPC(EndPC), Offset(Offset) {}

  ObjFileAddressRange() : HighPC(0), Offset(0) {}
};

/// Map LowPC to ObjFileAddressRange.
using RangesTy = std::map<uint64_t, ObjFileAddressRange>;

/// AddressesMap represents information about valid addresses used
/// by debug information. Valid addresses are those which points to
/// live code sections. i.e. relocations for these addresses point
/// into sections which would be/are placed into resulting binary.
class AddressesMap {
public:
  virtual ~AddressesMap();

  /// Returns true if represented addresses are from linked file.
  /// Returns false if represented addresses are from not-linked
  /// object file.
  virtual bool areRelocationsResolved() const = 0;

  /// Checks that there are valid relocations against a .debug_info
  /// section. Reset current relocation pointer if neccessary.
  virtual bool hasValidRelocs(bool ResetRelocsPtr = true) = 0;

  /// Checks that there is a relocation against .debug_info
  /// table between \p StartOffset and \p NextOffset.
  ///
  /// This function must be called with offsets in strictly ascending
  /// order because it never looks back at relocations it already 'went past'.
  /// \returns true and sets Info.InDebugMap if it is the case.
  virtual bool hasValidRelocationAt(uint64_t StartOffset, uint64_t EndOffset,
                                    CompileUnit::DIEInfo &Info) = 0;

  /// Apply the valid relocations to the buffer \p Data, taking into
  /// account that Data is at \p BaseOffset in the debug_info section.
  ///
  /// This function must be called with monotonic \p BaseOffset values.
  ///
  /// \returns true whether any reloc has been applied.
  virtual bool applyValidRelocs(MutableArrayRef<char> Data, uint64_t BaseOffset,
                                bool IsLittleEndian) = 0;

  /// Returns all valid functions address ranges(i.e., those ranges
  /// which points to sections with code).
  virtual RangesTy &getValidAddressRanges() = 0;

  /// Erases all data.
  virtual void clear() = 0;
};

} // end namespace llvm

#endif // LLVM_DWARFLINKER_DWARFLINKER_H
