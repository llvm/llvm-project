//===- DWARFLinkerUnit.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_DWARFLINKERPARALLEL_DWARFLINKERUNIT_H
#define LLVM_LIB_DWARFLINKERPARALLEL_DWARFLINKERUNIT_H

#include "OutputSections.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/CodeGen/DIE.h"
#include "llvm/DWARFLinkerParallel/StringPool.h"
#include "llvm/DebugInfo/DWARF/DWARFUnit.h"
#include "llvm/Support/LEB128.h"

namespace llvm {
namespace dwarflinker_parallel {

using UnitMessageHandlerTy = function_ref<void(
    const Twine &Error, StringRef Context, const DWARFDie *DIE)>;

/// Each unit keeps output data as a file with debug tables
/// corresponding to the concrete unit.
using OutTablesFileTy = SmallString<0>;

/// Base class for all Dwarf units(Compile unit/Type table unit).
class DwarfUnit : public OutputSections {
public:
  virtual ~DwarfUnit() {}
  DwarfUnit(unsigned ID, StringRef ClangModuleName,
            UnitMessageHandlerTy WarningHandler)
      : ID(ID), ClangModuleName(ClangModuleName),
        WarningHandler(WarningHandler) {
    FormParams.Version = 4;
    FormParams.Format = dwarf::DWARF32;
    FormParams.AddrSize = 4;
  }

  /// Endiannes for the compile unit.
  support::endianness getEndianness() const { return Endianess; }

  /// Return DWARF version.
  uint16_t getVersion() const { return FormParams.Version; }

  /// Return size of header of debug_info table.
  uint16_t getHeaderSize() const { return FormParams.Version >= 5 ? 12 : 11; }

  /// Return size of address.
  uint8_t getAddressByteSize() const { return FormParams.AddrSize; }

  /// Return size of reference.
  uint8_t getRefAddrByteSize() const { return FormParams.getRefAddrByteSize(); }

  /// Return format of the Dwarf(DWARF32 or DWARF64).
  /// TODO: DWARF64 is not currently supported.
  dwarf::DwarfFormat getDwarfFormat() const { return FormParams.Format; }

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

  /// Create a Die for this unit.
  void setOutputDIE(DIE *UnitDie) { NewUnit = UnitDie; }

  /// Return Die for this compile unit.
  DIE *getOutputUnitDIE() const { return NewUnit; }

  /// Return true if this compile unit is from Clang module.
  bool isClangModule() const { return !ClangModuleName.empty(); }

  /// Return Clang module name;
  const std::string &getClangModuleName() const { return ClangModuleName; }

  /// Returns generated file keeping debug tables for this compile unit.
  OutTablesFileTy &getOutDwarfBits() { return OutDebugInfoBits; }

  /// Erases generated file keeping debug tables for this compile unit.
  void eraseDwarfBits() { OutDebugInfoBits = OutTablesFileTy(); }

  MCSymbol *getLabelBegin() { return LabelBegin; }
  void setLabelBegin(MCSymbol *S) { LabelBegin = S; }

  /// Error reporting methods.
  /// @{

  void reportWarning(const Twine &Warning,
                     const DWARFDie *Die = nullptr) const {
    if (WarningHandler)
      WarningHandler(Warning, getUnitName(), Die);
  }
  void reportWarning(Error Warning) const {
    handleAllErrors(std::move(Warning), [&](ErrorInfoBase &Info) {
      if (WarningHandler)
        WarningHandler(Info.message(), getUnitName(), nullptr);
    });
  }
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

protected:
  /// Unique ID for the unit.
  unsigned ID = 0;

  /// Properties of the unit.
  dwarf::FormParams FormParams;

  /// DIE for newly generated compile unit.
  DIE *NewUnit = nullptr;

  /// The DW_AT_language of this unit.
  uint16_t Language = 0;

  /// The name of this unit.
  std::string UnitName;

  /// The DW_AT_LLVM_sysroot of this unit.
  std::string SysRoot;

  /// If this is a Clang module, this holds the module's name.
  std::string ClangModuleName;

  uint64_t UnitSize = 0;

  /// Elf file containg generated debug tables for this compile unit.
  OutTablesFileTy OutDebugInfoBits;

  /// Endiannes for this compile unit.
  support::endianness Endianess = support::endianness::little;

  MCSymbol *LabelBegin = nullptr;

  /// true if current unit references_to/is_referenced by other unit.
  std::atomic<bool> IsInterconnectedCU = {false};

  UnitMessageHandlerTy WarningHandler;
};

} // end of namespace dwarflinker_parallel
} // end namespace llvm

#endif // LLVM_LIB_DWARFLINKERPARALLEL_DWARFLINKERUNIT_H
