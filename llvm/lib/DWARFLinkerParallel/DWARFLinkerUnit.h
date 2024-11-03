//===- DWARFLinkerUnit.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_DWARFLINKERPARALLEL_DWARFLINKERUNIT_H
#define LLVM_LIB_DWARFLINKERPARALLEL_DWARFLINKERUNIT_H

#include "DWARFLinkerGlobalData.h"
#include "OutputSections.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/CodeGen/DIE.h"
#include "llvm/DWARFLinkerParallel/DWARFLinker.h"
#include "llvm/DWARFLinkerParallel/StringPool.h"
#include "llvm/DebugInfo/DWARF/DWARFUnit.h"
#include "llvm/Support/LEB128.h"

namespace llvm {
namespace dwarflinker_parallel {

class DwarfUnit;
using MacroOffset2UnitMapTy = DenseMap<uint64_t, DwarfUnit *>;

/// Base class for all Dwarf units(Compile unit/Type table unit).
class DwarfUnit : public OutputSections {
public:
  virtual ~DwarfUnit() {}
  DwarfUnit(LinkingGlobalData &GlobalData, unsigned ID,
            StringRef ClangModuleName)
      : OutputSections(GlobalData), ID(ID), ClangModuleName(ClangModuleName),
        OutUnitDIE(nullptr) {}

  /// Unique id of the unit.
  unsigned getUniqueID() const { return ID; }

  /// Return language of this unit.
  uint16_t getLanguage() const { return Language; }

  /// Set size of this(newly generated) compile unit.
  void setUnitSize(uint64_t UnitSize) { this->UnitSize = UnitSize; }

  /// Returns size of this(newly generated) compile unit.
  uint64_t getUnitSize() const { return UnitSize; }

  /// Returns this unit name.
  StringRef getUnitName() const { return UnitName; }

  /// Return the DW_AT_LLVM_sysroot of the compile unit or an empty StringRef.
  StringRef getSysRoot() { return SysRoot; }

  /// Return true if this compile unit is from Clang module.
  bool isClangModule() const { return !ClangModuleName.empty(); }

  /// Return Clang module name;
  const std::string &getClangModuleName() const { return ClangModuleName; }

  /// Return global data.
  LinkingGlobalData &getGlobalData() { return GlobalData; }

  /// Returns true if unit is inter-connected(it references/referenced by other
  /// unit).
  bool isInterconnectedCU() const { return IsInterconnectedCU; }

  /// Mark this unit as inter-connected(it references/referenced by other unit).
  void setInterconnectedCU() { IsInterconnectedCU = true; }

  /// Adds \p Abbrev into unit`s abbreviation table.
  void assignAbbrev(DIEAbbrev &Abbrev);

  /// Returns abbreviations for this compile unit.
  const std::vector<std::unique_ptr<DIEAbbrev>> &getAbbreviations() const {
    return Abbreviations;
  }

  /// Returns output unit DIE.
  DIE *getOutUnitDIE() { return OutUnitDIE; }

  /// Set output unit DIE.
  void setOutUnitDIE(DIE *UnitDie) { OutUnitDIE = UnitDie; }

  /// \defgroup Methods used to emit unit's debug info:
  ///
  /// @{
  /// Emit unit's abbreviations.
  Error emitAbbreviations();

  /// Emit .debug_info section for unit DIEs.
  Error emitDebugInfo(Triple &TargetTriple);

  /// Emit .debug_line section.
  Error emitDebugLine(Triple &TargetTriple,
                      const DWARFDebugLine::LineTable &OutLineTable);
  /// @}

  /// This structure keeps fields which would be used for creating accelerator
  /// table.
  struct AccelInfo {
    AccelInfo(StringEntry *Name, const DIE *Die, bool SkipPubSection = false);
    AccelInfo(StringEntry *Name, const DIE *Die, uint32_t QualifiedNameHash,
              bool ObjCClassIsImplementation);

    /// Name of the entry.
    StringEntry *Name = nullptr;

    /// Tag of the DIE this entry describes.
    dwarf::Tag Tag = dwarf::DW_TAG_null;

    /// Output offset of the DIE this entry describes.
    uint64_t OutOffset = 0;

    /// Hash of the fully qualified name.
    uint32_t QualifiedNameHash = 0;

    /// Emit this entry only in the apple_* sections.
    bool SkipPubSection = false;

    /// Is this an ObjC class implementation?
    bool ObjcClassImplementation = false;

    /// Cloned Die containing acceleration info.
    const DIE *Die = nullptr;
  };

  /// \defgroup Methods used for reporting warnings and errors:
  ///
  /// @{
  void warn(const Twine &Warning) { GlobalData.warn(Warning, getUnitName()); }

  void error(const Twine &Err) { GlobalData.warn(Err, getUnitName()); }
  /// @}

protected:
  /// Emit single abbreviation entry.
  void emitDwarfAbbrevEntry(const DIEAbbrev &Abbrev,
                            SectionDescriptor &AbbrevSection);

  /// Unique ID for the unit.
  unsigned ID = 0;

  /// The DW_AT_language of this unit.
  uint16_t Language = 0;

  /// The name of this unit.
  std::string UnitName;

  /// The DW_AT_LLVM_sysroot of this unit.
  std::string SysRoot;

  /// If this is a Clang module, this holds the module's name.
  std::string ClangModuleName;

  uint64_t UnitSize = 0;

  /// true if current unit references_to/is_referenced by other unit.
  std::atomic<bool> IsInterconnectedCU = {false};

  /// FoldingSet that uniques the abbreviations.
  FoldingSet<DIEAbbrev> AbbreviationsSet;

  /// Storage for the unique Abbreviations.
  std::vector<std::unique_ptr<DIEAbbrev>> Abbreviations;

  /// Output unit DIE.
  DIE *OutUnitDIE = nullptr;
};

} // end of namespace dwarflinker_parallel
} // end namespace llvm

#endif // LLVM_LIB_DWARFLINKERPARALLEL_DWARFLINKERUNIT_H
