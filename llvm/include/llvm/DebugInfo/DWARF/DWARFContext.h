//===- DWARFContext.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===/

#ifndef LLVM_DEBUGINFO_DWARF_DWARFCONTEXT_H
#define LLVM_DEBUGINFO_DWARF_DWARFCONTEXT_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugLine.h"
#include "llvm/DebugInfo/DWARF/DWARFDie.h"
#include "llvm/DebugInfo/DWARF/DWARFObject.h"
#include "llvm/DebugInfo/DWARF/DWARFUnit.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Host.h"
#include <cstdint>
#include <memory>

namespace llvm {

class MCRegisterInfo;
class MemoryBuffer;
class AppleAcceleratorTable;
class DWARFCompileUnit;
class DWARFDebugAbbrev;
class DWARFDebugAranges;
class DWARFDebugFrame;
class DWARFDebugLoc;
class DWARFDebugMacro;
class DWARFDebugNames;
class DWARFGdbIndex;
class DWARFTypeUnit;
class DWARFUnitIndex;

/// DWARFContext
/// This data structure is the top level entity that deals with dwarf debug
/// information parsing. The actual data is supplied through DWARFObj.
class DWARFContext : public DIContext {
  DWARFUnitVector NormalUnits;
  Optional<DenseMap<uint64_t, DWARFTypeUnit*>> NormalTypeUnits;
  std::unique_ptr<DWARFUnitIndex> CUIndex;
  std::unique_ptr<DWARFGdbIndex> GdbIndex;
  std::unique_ptr<DWARFUnitIndex> TUIndex;
  std::unique_ptr<DWARFDebugAbbrev> Abbrev;
  std::unique_ptr<DWARFDebugLoc> Loc;
  std::unique_ptr<DWARFDebugAranges> Aranges;
  std::unique_ptr<DWARFDebugLine> Line;
  std::unique_ptr<DWARFDebugFrame> DebugFrame;
  std::unique_ptr<DWARFDebugFrame> EHFrame;
  std::unique_ptr<DWARFDebugMacro> Macro;
  std::unique_ptr<DWARFDebugMacro> Macinfo;
  std::unique_ptr<DWARFDebugNames> Names;
  std::unique_ptr<AppleAcceleratorTable> AppleNames;
  std::unique_ptr<AppleAcceleratorTable> AppleTypes;
  std::unique_ptr<AppleAcceleratorTable> AppleNamespaces;
  std::unique_ptr<AppleAcceleratorTable> AppleObjC;

  DWARFUnitVector DWOUnits;
  Optional<DenseMap<uint64_t, DWARFTypeUnit*>> DWOTypeUnits;
  std::unique_ptr<DWARFDebugAbbrev> AbbrevDWO;
  std::unique_ptr<DWARFDebugMacro> MacinfoDWO;
  std::unique_ptr<DWARFDebugMacro> MacroDWO;

  /// The maximum DWARF version of all units.
  unsigned MaxVersion = 0;

  struct DWOFile {
    object::OwningBinary<object::ObjectFile> File;
    std::unique_ptr<DWARFContext> Context;
  };
  StringMap<std::weak_ptr<DWOFile>> DWOFiles;
  std::weak_ptr<DWOFile> DWP;
  bool CheckedForDWP = false;
  std::string DWPName;

  std::unique_ptr<MCRegisterInfo> RegInfo;

  std::function<void(Error)> RecoverableErrorHandler =
      WithColor::defaultErrorHandler;
  std::function<void(Error)> WarningHandler = WithColor::defaultWarningHandler;

  /// Read compile units from the debug_info section (if necessary)
  /// and type units from the debug_types sections (if necessary)
  /// and store them in NormalUnits.
  void parseNormalUnits();

  /// Read compile units from the debug_info.dwo section (if necessary)
  /// and type units from the debug_types.dwo section (if necessary)
  /// and store them in DWOUnits.
  /// If \p Lazy is true, set up to parse but don't actually parse them.
  enum { EagerParse = false, LazyParse = true };
  void parseDWOUnits(bool Lazy = false);

  std::unique_ptr<const DWARFObject> DObj;

  /// Helper enum to distinguish between macro[.dwo] and macinfo[.dwo]
  /// section.
  enum MacroSecType {
    MacinfoSection,
    MacinfoDwoSection,
    MacroSection,
    MacroDwoSection
  };

public:
  DWARFContext(std::unique_ptr<const DWARFObject> DObj,
               std::string DWPName = "",
               std::function<void(Error)> RecoverableErrorHandler =
                   WithColor::defaultErrorHandler,
               std::function<void(Error)> WarningHandler =
                   WithColor::defaultWarningHandler);
  ~DWARFContext();

  DWARFContext(DWARFContext &) = delete;
  DWARFContext &operator=(DWARFContext &) = delete;

  const DWARFObject &getDWARFObj() const { return *DObj; }

  static bool classof(const DIContext *DICtx) {
    return DICtx->getKind() == CK_DWARF;
  }

  /// Dump a textual representation to \p OS. If any \p DumpOffsets are present,
  /// dump only the record at the specified offset.
  void dump(raw_ostream &OS, DIDumpOptions DumpOpts,
            std::array<Optional<uint64_t>, DIDT_ID_Count> DumpOffsets);

  void dump(raw_ostream &OS, DIDumpOptions DumpOpts) override {
    std::array<Optional<uint64_t>, DIDT_ID_Count> DumpOffsets;
    dump(OS, DumpOpts, DumpOffsets);
  }

  bool verify(raw_ostream &OS, DIDumpOptions DumpOpts = {}) override;

  using unit_iterator_range = DWARFUnitVector::iterator_range;
  using compile_unit_range = DWARFUnitVector::compile_unit_range;

  /// Get units from .debug_info in this context.
  unit_iterator_range info_section_units() {
    parseNormalUnits();
    return unit_iterator_range(NormalUnits.begin(),
                               NormalUnits.begin() +
                                   NormalUnits.getNumInfoUnits());
  }

  const DWARFUnitVector &getNormalUnitsVector() {
    parseNormalUnits();
    return NormalUnits;
  }

  /// Get units from .debug_types in this context.
  unit_iterator_range types_section_units() {
    parseNormalUnits();
    return unit_iterator_range(
        NormalUnits.begin() + NormalUnits.getNumInfoUnits(), NormalUnits.end());
  }

  /// Get compile units in this context.
  compile_unit_range compile_units() {
    return make_filter_range(info_section_units(), isCompileUnit);
  }

  // If you want type_units(), it'll need to be a concat iterator of a filter of
  // TUs in info_section + all the (all type) units in types_section

  /// Get all normal compile/type units in this context.
  unit_iterator_range normal_units() {
    parseNormalUnits();
    return unit_iterator_range(NormalUnits.begin(), NormalUnits.end());
  }

  /// Get units from .debug_info..dwo in the DWO context.
  unit_iterator_range dwo_info_section_units() {
    parseDWOUnits();
    return unit_iterator_range(DWOUnits.begin(),
                               DWOUnits.begin() + DWOUnits.getNumInfoUnits());
  }

  const DWARFUnitVector &getDWOUnitsVector() {
    parseDWOUnits();
    return DWOUnits;
  }

  /// Get units from .debug_types.dwo in the DWO context.
  unit_iterator_range dwo_types_section_units() {
    parseDWOUnits();
    return unit_iterator_range(DWOUnits.begin() + DWOUnits.getNumInfoUnits(),
                               DWOUnits.end());
  }

  /// Get compile units in the DWO context.
  compile_unit_range dwo_compile_units() {
    return make_filter_range(dwo_info_section_units(), isCompileUnit);
  }

  // If you want dwo_type_units(), it'll need to be a concat iterator of a
  // filter of TUs in dwo_info_section + all the (all type) units in
  // dwo_types_section.

  /// Get all units in the DWO context.
  unit_iterator_range dwo_units() {
    parseDWOUnits();
    return unit_iterator_range(DWOUnits.begin(), DWOUnits.end());
  }

  /// Get the number of compile units in this context.
  unsigned getNumCompileUnits() {
    parseNormalUnits();
    return NormalUnits.getNumInfoUnits();
  }

  /// Get the number of type units in this context.
  unsigned getNumTypeUnits() {
    parseNormalUnits();
    return NormalUnits.getNumTypesUnits();
  }

  /// Get the number of compile units in the DWO context.
  unsigned getNumDWOCompileUnits() {
    parseDWOUnits();
    return DWOUnits.getNumInfoUnits();
  }

  /// Get the number of type units in the DWO context.
  unsigned getNumDWOTypeUnits() {
    parseDWOUnits();
    return DWOUnits.getNumTypesUnits();
  }

  /// Get the unit at the specified index.
  DWARFUnit *getUnitAtIndex(unsigned index) {
    parseNormalUnits();
    return NormalUnits[index].get();
  }

  /// Get the unit at the specified index for the DWO units.
  DWARFUnit *getDWOUnitAtIndex(unsigned index) {
    parseDWOUnits();
    return DWOUnits[index].get();
  }

  DWARFCompileUnit *getDWOCompileUnitForHash(uint64_t Hash);
  DWARFTypeUnit *getTypeUnitForHash(uint16_t Version, uint64_t Hash, bool IsDWO);

  /// Return the compile unit that includes an offset (relative to .debug_info).
  DWARFCompileUnit *getCompileUnitForOffset(uint64_t Offset);

  /// Get a DIE given an exact offset.
  DWARFDie getDIEForOffset(uint64_t Offset);

  unsigned getMaxVersion() {
    // Ensure info units have been parsed to discover MaxVersion
    info_section_units();
    return MaxVersion;
  }

  unsigned getMaxDWOVersion() {
    // Ensure DWO info units have been parsed to discover MaxVersion
    dwo_info_section_units();
    return MaxVersion;
  }

  void setMaxVersionIfGreater(unsigned Version) {
    if (Version > MaxVersion)
      MaxVersion = Version;
  }

  const DWARFUnitIndex &getCUIndex();
  DWARFGdbIndex &getGdbIndex();
  const DWARFUnitIndex &getTUIndex();

  /// Get a pointer to the parsed DebugAbbrev object.
  const DWARFDebugAbbrev *getDebugAbbrev();

  /// Get a pointer to the parsed DebugLoc object.
  const DWARFDebugLoc *getDebugLoc();

  /// Get a pointer to the parsed dwo abbreviations object.
  const DWARFDebugAbbrev *getDebugAbbrevDWO();

  /// Get a pointer to the parsed DebugAranges object.
  const DWARFDebugAranges *getDebugAranges();

  /// Get a pointer to the parsed frame information object.
  Expected<const DWARFDebugFrame *> getDebugFrame();

  /// Get a pointer to the parsed eh frame information object.
  Expected<const DWARFDebugFrame *> getEHFrame();

  /// Get a pointer to the parsed DebugMacinfo information object.
  const DWARFDebugMacro *getDebugMacinfo();

  /// Get a pointer to the parsed DebugMacinfoDWO information object.
  const DWARFDebugMacro *getDebugMacinfoDWO();

  /// Get a pointer to the parsed DebugMacro information object.
  const DWARFDebugMacro *getDebugMacro();

  /// Get a pointer to the parsed DebugMacroDWO information object.
  const DWARFDebugMacro *getDebugMacroDWO();

  /// Get a reference to the parsed accelerator table object.
  const DWARFDebugNames &getDebugNames();

  /// Get a reference to the parsed accelerator table object.
  const AppleAcceleratorTable &getAppleNames();

  /// Get a reference to the parsed accelerator table object.
  const AppleAcceleratorTable &getAppleTypes();

  /// Get a reference to the parsed accelerator table object.
  const AppleAcceleratorTable &getAppleNamespaces();

  /// Get a reference to the parsed accelerator table object.
  const AppleAcceleratorTable &getAppleObjC();

  /// Get a pointer to a parsed line table corresponding to a compile unit.
  /// Report any parsing issues as warnings on stderr.
  const DWARFDebugLine::LineTable *getLineTableForUnit(DWARFUnit *U);

  /// Get a pointer to a parsed line table corresponding to a compile unit.
  /// Report any recoverable parsing problems using the handler.
  Expected<const DWARFDebugLine::LineTable *>
  getLineTableForUnit(DWARFUnit *U,
                      function_ref<void(Error)> RecoverableErrorHandler);

  DataExtractor getStringExtractor() const {
    return DataExtractor(DObj->getStrSection(), false, 0);
  }
  DataExtractor getStringDWOExtractor() const {
    return DataExtractor(DObj->getStrDWOSection(), false, 0);
  }
  DataExtractor getLineStringExtractor() const {
    return DataExtractor(DObj->getLineStrSection(), false, 0);
  }

  /// Wraps the returned DIEs for a given address.
  struct DIEsForAddress {
    DWARFCompileUnit *CompileUnit = nullptr;
    DWARFDie FunctionDIE;
    DWARFDie BlockDIE;
    explicit operator bool() const { return CompileUnit != nullptr; }
  };

  /// Get the compilation unit, the function DIE and lexical block DIE for the
  /// given address where applicable.
  /// TODO: change input parameter from "uint64_t Address"
  ///       into "SectionedAddress Address"
  DIEsForAddress getDIEsForAddress(uint64_t Address);

  DILineInfo getLineInfoForAddress(
      object::SectionedAddress Address,
      DILineInfoSpecifier Specifier = DILineInfoSpecifier()) override;
  DILineInfoTable getLineInfoForAddressRange(
      object::SectionedAddress Address, uint64_t Size,
      DILineInfoSpecifier Specifier = DILineInfoSpecifier()) override;
  DIInliningInfo getInliningInfoForAddress(
      object::SectionedAddress Address,
      DILineInfoSpecifier Specifier = DILineInfoSpecifier()) override;

  std::vector<DILocal>
  getLocalsForAddress(object::SectionedAddress Address) override;

  bool isLittleEndian() const { return DObj->isLittleEndian(); }
  static unsigned getMaxSupportedVersion() { return 5; }
  static bool isSupportedVersion(unsigned version) {
    return version >= 2 && version <= getMaxSupportedVersion();
  }

  static SmallVector<uint8_t, 3> getSupportedAddressSizes() {
    return {2, 4, 8};
  }
  static bool isAddressSizeSupported(unsigned AddressSize) {
    return llvm::is_contained(getSupportedAddressSizes(), AddressSize);
  }
  template <typename... Ts>
  static Error checkAddressSizeSupported(unsigned AddressSize,
                                         std::error_code EC, char const *Fmt,
                                         const Ts &...Vals) {
    if (isAddressSizeSupported(AddressSize))
      return Error::success();
    std::string Buffer;
    raw_string_ostream Stream(Buffer);
    Stream << format(Fmt, Vals...)
           << " has unsupported address size: " << AddressSize
           << " (supported are ";
    ListSeparator LS;
    for (unsigned Size : DWARFContext::getSupportedAddressSizes())
      Stream << LS << Size;
    Stream << ')';
    return make_error<StringError>(Stream.str(), EC);
  }

  std::shared_ptr<DWARFContext> getDWOContext(StringRef AbsolutePath);

  const MCRegisterInfo *getRegisterInfo() const { return RegInfo.get(); }

  function_ref<void(Error)> getRecoverableErrorHandler() {
    return RecoverableErrorHandler;
  }

  function_ref<void(Error)> getWarningHandler() { return WarningHandler; }

  enum class ProcessDebugRelocations { Process, Ignore };

  static std::unique_ptr<DWARFContext>
  create(const object::ObjectFile &Obj,
         ProcessDebugRelocations RelocAction = ProcessDebugRelocations::Process,
         const LoadedObjectInfo *L = nullptr, std::string DWPName = "",
         std::function<void(Error)> RecoverableErrorHandler =
             WithColor::defaultErrorHandler,
         std::function<void(Error)> WarningHandler =
             WithColor::defaultWarningHandler);

  static std::unique_ptr<DWARFContext>
  create(const StringMap<std::unique_ptr<MemoryBuffer>> &Sections,
         uint8_t AddrSize, bool isLittleEndian = sys::IsLittleEndianHost,
         std::function<void(Error)> RecoverableErrorHandler =
             WithColor::defaultErrorHandler,
         std::function<void(Error)> WarningHandler =
             WithColor::defaultWarningHandler);

  /// Loads register info for the architecture of the provided object file.
  /// Improves readability of dumped DWARF expressions. Requires the caller to
  /// have initialized the relevant target descriptions.
  Error loadRegisterInfo(const object::ObjectFile &Obj);

  /// Get address size from CUs.
  /// TODO: refactor compile_units() to make this const.
  uint8_t getCUAddrSize();

  Triple::ArchType getArch() const {
    return getDWARFObj().getFile()->getArch();
  }

  /// Return the compile unit which contains instruction with provided
  /// address.
  /// TODO: change input parameter from "uint64_t Address"
  ///       into "SectionedAddress Address"
  DWARFCompileUnit *getCompileUnitForAddress(uint64_t Address);

private:
  /// Parse a macro[.dwo] or macinfo[.dwo] section.
  std::unique_ptr<DWARFDebugMacro>
  parseMacroOrMacinfo(MacroSecType SectionType);

  void addLocalsForDie(DWARFCompileUnit *CU, DWARFDie Subprogram, DWARFDie Die,
                       std::vector<DILocal> &Result);
};

} // end namespace llvm

#endif // LLVM_DEBUGINFO_DWARF_DWARFCONTEXT_H
