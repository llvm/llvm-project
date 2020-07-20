//===- DWARFLinker.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DWARFLINKER_DWARFLINKER_H
#define LLVM_DWARFLINKER_DWARFLINKER_H

#include "llvm/CodeGen/AccelTable.h"
#include "llvm/CodeGen/NonRelocatableStringpool.h"
#include "llvm/DWARFLinker/DWARFLinkerDeclContext.h"
#include "llvm/DebugInfo/DWARF/DWARFCompileUnit.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/MC/MCDwarf.h"
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

/// DwarfEmitter presents interface to generate all debug info tables.
class DwarfEmitter {
public:
  virtual ~DwarfEmitter();

  /// Emit DIE containing warnings.
  virtual void emitPaperTrailWarningsDie(const Triple &Triple, DIE &Die) = 0;

  /// Emit section named SecName with content equals to
  /// corresponding section in Obj.
  virtual void emitSectionContents(const object::ObjectFile &Obj,
                                   StringRef SecName) = 0;

  /// Emit the abbreviation table \p Abbrevs to the debug_abbrev section.
  virtual void
  emitAbbrevs(const std::vector<std::unique_ptr<DIEAbbrev>> &Abbrevs,
              unsigned DwarfVersion) = 0;

  /// Emit the string table described by \p Pool.
  virtual void emitStrings(const NonRelocatableStringpool &Pool) = 0;

  /// Emit DWARF debug names.
  virtual void
  emitDebugNames(AccelTable<DWARF5AccelTableStaticData> &Table) = 0;

  /// Emit Apple namespaces accelerator table.
  virtual void
  emitAppleNamespaces(AccelTable<AppleAccelTableStaticOffsetData> &Table) = 0;

  /// Emit Apple names accelerator table.
  virtual void
  emitAppleNames(AccelTable<AppleAccelTableStaticOffsetData> &Table) = 0;

  /// Emit Apple Objective-C accelerator table.
  virtual void
  emitAppleObjc(AccelTable<AppleAccelTableStaticOffsetData> &Table) = 0;

  /// Emit Apple type accelerator table.
  virtual void
  emitAppleTypes(AccelTable<AppleAccelTableStaticTypeData> &Table) = 0;

  /// Emit debug_ranges for \p FuncRange by translating the
  /// original \p Entries.
  virtual void emitRangesEntries(
      int64_t UnitPcOffset, uint64_t OrigLowPc,
      const FunctionIntervals::const_iterator &FuncRange,
      const std::vector<DWARFDebugRangeList::RangeListEntry> &Entries,
      unsigned AddressSize) = 0;

  /// Emit debug_aranges entries for \p Unit and if \p DoRangesSection is true,
  /// also emit the debug_ranges entries for the DW_TAG_compile_unit's
  /// DW_AT_ranges attribute.
  virtual void emitUnitRangesEntries(CompileUnit &Unit,
                                     bool DoRangesSection) = 0;

  /// Copy the debug_line over to the updated binary while unobfuscating the
  /// file names and directories.
  virtual void translateLineTable(DataExtractor LineData, uint64_t Offset) = 0;

  /// Emit the line table described in \p Rows into the debug_line section.
  virtual void emitLineTableForUnit(MCDwarfLineTableParams Params,
                                    StringRef PrologueBytes,
                                    unsigned MinInstLength,
                                    std::vector<DWARFDebugLine::Row> &Rows,
                                    unsigned AdddressSize) = 0;

  /// Emit the .debug_pubnames contribution for \p Unit.
  virtual void emitPubNamesForUnit(const CompileUnit &Unit) = 0;

  /// Emit the .debug_pubtypes contribution for \p Unit.
  virtual void emitPubTypesForUnit(const CompileUnit &Unit) = 0;

  /// Emit a CIE.
  virtual void emitCIE(StringRef CIEBytes) = 0;

  /// Emit an FDE with data \p Bytes.
  virtual void emitFDE(uint32_t CIEOffset, uint32_t AddreSize, uint32_t Address,
                       StringRef Bytes) = 0;

  /// Emit the debug_loc contribution for \p Unit by copying the entries from
  /// \p Dwarf and offsetting them. Update the location attributes to point to
  /// the new entries.
  virtual void emitLocationsForUnit(
      const CompileUnit &Unit, DWARFContext &Dwarf,
      std::function<void(StringRef, SmallVectorImpl<uint8_t> &)>
          ProcessExpr) = 0;

  /// Emit the compilation unit header for \p Unit in the
  /// debug_info section.
  ///
  /// As a side effect, this also switches the current Dwarf version
  /// of the MC layer to the one of U.getOrigUnit().
  virtual void emitCompileUnitHeader(CompileUnit &Unit) = 0;

  /// Recursively emit the DIE tree rooted at \p Die.
  virtual void emitDIE(DIE &Die) = 0;

  /// Returns size of generated .debug_line section.
  virtual uint64_t getLineSectionSize() const = 0;

  /// Returns size of generated .debug_frame section.
  virtual uint64_t getFrameSectionSize() const = 0;

  /// Returns size of generated .debug_ranges section.
  virtual uint64_t getRangesSectionSize() const = 0;

  /// Returns size of generated .debug_info section.
  virtual uint64_t getDebugInfoSectionSize() const = 0;
};

} // end namespace llvm

#endif // LLVM_DWARFLINKER_DWARFLINKER_H
