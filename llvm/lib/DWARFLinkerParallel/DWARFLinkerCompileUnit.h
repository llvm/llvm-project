//===- DWARFLinkerCompileUnit.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_DWARFLINKERPARALLEL_DWARFLINKERCOMPILEUNIT_H
#define LLVM_LIB_DWARFLINKERPARALLEL_DWARFLINKERCOMPILEUNIT_H

#include "DWARFLinkerUnit.h"
#include "llvm/DWARFLinkerParallel/DWARFFile.h"
#include "llvm/DWARFLinkerParallel/DWARFLinker.h"
#include <optional>

namespace llvm {
namespace dwarflinker_parallel {

struct LinkContext;
class DWARFFile;

/// Stores all information related to a compile unit, be it in its original
/// instance of the object file or its brand new cloned and generated DIE tree.
class CompileUnit : public DwarfUnit {
public:
  CompileUnit(LinkContext &, unsigned ID, StringRef ClangModuleName,
              DWARFFile &File,
              DWARFLinker::SwiftInterfacesMapTy *,
              UnitMessageHandlerTy WarningHandler)
      : DwarfUnit(ID, ClangModuleName, WarningHandler), ContaingFile(File) {
    FormParams.Version = 4;
    FormParams.Format = dwarf::DWARF32;
    FormParams.AddrSize = 4;
    UnitName = ContaingFile.FileName;
  }

  CompileUnit(LinkContext &, DWARFUnit &OrigUnit, unsigned ID,
              StringRef ClangModuleName, DWARFFile &File,
              UnitMessageHandlerTy WarningHandler)
      : DwarfUnit(ID, ClangModuleName, WarningHandler),
        ContaingFile(File), OrigUnit(&OrigUnit) {
    DWARFDie CUDie = OrigUnit.getUnitDIE();
    if (!CUDie)
      return;

    if (File.Dwarf)
      Endianess = File.Dwarf->isLittleEndian() ? support::endianness::little
                                               : support::endianness::big;

    FormParams.Version = OrigUnit.getVersion();
    FormParams.Format = dwarf::DWARF32;
    FormParams.AddrSize = OrigUnit.getAddressByteSize();

    Language = dwarf::toUnsigned(CUDie.find(dwarf::DW_AT_language), 0);

    UnitName = ContaingFile.FileName;
    SysRoot = dwarf::toStringRef(CUDie.find(dwarf::DW_AT_LLVM_sysroot)).str();
  }

  /// \defgroup Helper methods to access OrigUnit.
  ///
  /// @{

  /// Returns paired compile unit from input DWARF.
  DWARFUnit &getOrigUnit() const {
    assert(OrigUnit != nullptr);
    return *OrigUnit;
  }

  const DWARFDebugInfoEntry *
  getFirstChildEntry(const DWARFDebugInfoEntry *Die) const {
    assert(OrigUnit != nullptr);
    return OrigUnit->getFirstChildEntry(Die);
  }

  const DWARFDebugInfoEntry *
  getSiblingEntry(const DWARFDebugInfoEntry *Die) const {
    assert(OrigUnit != nullptr);
    return OrigUnit->getSiblingEntry(Die);
  }

  DWARFDie getParent(const DWARFDebugInfoEntry *Die) {
    assert(OrigUnit != nullptr);
    return OrigUnit->getParent(Die);
  }

  DWARFDie getDIEAtIndex(unsigned Index) {
    assert(OrigUnit != nullptr);
    return OrigUnit->getDIEAtIndex(Index);
  }

  const DWARFDebugInfoEntry *getDebugInfoEntry(unsigned Index) const {
    assert(OrigUnit != nullptr);
    return OrigUnit->getDebugInfoEntry(Index);
  }

  DWARFDie getUnitDIE(bool ExtractUnitDIEOnly = true) {
    assert(OrigUnit != nullptr);
    return OrigUnit->getUnitDIE(ExtractUnitDIEOnly);
  }

  DWARFDie getDIE(const DWARFDebugInfoEntry *Die) {
    assert(OrigUnit != nullptr);
    return DWARFDie(OrigUnit, Die);
  }

  uint32_t getDIEIndex(const DWARFDebugInfoEntry *Die) const {
    assert(OrigUnit != nullptr);
    return OrigUnit->getDIEIndex(Die);
  }

  uint32_t getDIEIndex(const DWARFDie &Die) const {
    assert(OrigUnit != nullptr);
    return OrigUnit->getDIEIndex(Die);
  }

  std::optional<DWARFFormValue> find(uint32_t DieIdx,
                                     ArrayRef<dwarf::Attribute> Attrs) const {
    assert(OrigUnit != nullptr);
    return find(OrigUnit->getDebugInfoEntry(DieIdx), Attrs);
  }

  std::optional<DWARFFormValue> find(const DWARFDebugInfoEntry *Die,
                                     ArrayRef<dwarf::Attribute> Attrs) const {
    if (!Die)
      return std::nullopt;
    auto AbbrevDecl = Die->getAbbreviationDeclarationPtr();
    if (AbbrevDecl) {
      for (auto Attr : Attrs) {
        if (auto Value = AbbrevDecl->getAttributeValue(Die->getOffset(), Attr,
                                                       *OrigUnit))
          return Value;
      }
    }
    return std::nullopt;
  }

  std::optional<uint32_t> getDIEIndexForOffset(uint64_t Offset) {
    return OrigUnit->getDIEIndexForOffset(Offset);
  }

  /// @}

private:
  /// DWARFFile containing this compile unit.
  DWARFFile &ContaingFile;

  /// Pointer to the paired compile unit from the input DWARF.
  DWARFUnit *OrigUnit = nullptr;
};

} // end of namespace dwarflinker_parallel
} // end namespace llvm

#endif // LLVM_LIB_DWARFLINKERPARALLEL_DWARFLINKERCOMPILEUNIT_H
