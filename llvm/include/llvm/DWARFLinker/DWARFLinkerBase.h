//===- DWARFLinkerBase.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DWARFLINKER_DWARFLINKERBASE_H
#define LLVM_DWARFLINKER_DWARFLINKERBASE_H

#include "AddressesMap.h"
#include "DWARFFile.h"
#include "IndexedValuesMap.h"
#include "llvm/ADT/AddressRanges.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugLine.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugRangeList.h"
#include "llvm/DebugInfo/DWARF/DWARFDie.h"
#include "llvm/DebugInfo/DWARF/DWARFExpression.h"
#include <map>

namespace llvm {
class DWARFUnit;

namespace dwarflinker {

/// List of tracked debug tables.
enum class DebugSectionKind : uint8_t {
  DebugInfo = 0,
  DebugLine,
  DebugFrame,
  DebugRange,
  DebugRngLists,
  DebugLoc,
  DebugLocLists,
  DebugARanges,
  DebugAbbrev,
  DebugMacinfo,
  DebugMacro,
  DebugAddr,
  DebugStr,
  DebugLineStr,
  DebugStrOffsets,
  DebugPubNames,
  DebugPubTypes,
  DebugNames,
  AppleNames,
  AppleNamespaces,
  AppleObjC,
  AppleTypes,
  NumberOfEnumEntries // must be last
};

static constexpr size_t SectionKindsNum =
    static_cast<size_t>(DebugSectionKind::NumberOfEnumEntries);

static constexpr StringLiteral SectionNames[SectionKindsNum] = {
    "debug_info",     "debug_line",     "debug_frame",       "debug_ranges",
    "debug_rnglists", "debug_loc",      "debug_loclists",    "debug_aranges",
    "debug_abbrev",   "debug_macinfo",  "debug_macro",       "debug_addr",
    "debug_str",      "debug_line_str", "debug_str_offsets", "debug_pubnames",
    "debug_pubtypes", "debug_names",    "apple_names",       "apple_namespac",
    "apple_objc",     "apple_types"};

/// Return the name of the section.
static constexpr const StringLiteral &
getSectionName(DebugSectionKind SectionKind) {
  return SectionNames[static_cast<uint8_t>(SectionKind)];
}

/// Recognise the table name and match it with the DebugSectionKind.
std::optional<DebugSectionKind> parseDebugTableName(StringRef Name);

/// The base interface for DWARFLinker implementations.
class DWARFLinkerBase {
public:
  virtual ~DWARFLinkerBase() = default;

  using MessageHandlerTy = std::function<void(
      const Twine &Warning, StringRef Context, const DWARFDie *DIE)>;
  using ObjFileLoaderTy = std::function<ErrorOr<DWARFFile &>(
      StringRef ContainerName, StringRef Path)>;
  using InputVerificationHandlerTy =
      std::function<void(const DWARFFile &File, llvm::StringRef Output)>;
  using ObjectPrefixMapTy = std::map<std::string, std::string>;
  using CompileUnitHandlerTy = function_ref<void(const DWARFUnit &Unit)>;
  using TranslatorFuncTy = std::function<StringRef(StringRef)>;
  using SwiftInterfacesMapTy = std::map<std::string, std::string>;

  /// Type of output file.
  enum class OutputFileType : uint8_t {
    Object,
    Assembly,
  };

  /// The kind of accelerator tables we should emit.
  enum class AccelTableKind : uint8_t {
    Apple,     ///< .apple_names, .apple_namespaces, .apple_types, .apple_objc.
    Pub,       ///< .debug_pubnames, .debug_pubtypes
    DebugNames ///< .debug_names.
  };

  /// Add object file to be linked. Pre-load compile unit die. Call
  /// \p OnCUDieLoaded for each compile unit die. If specified \p File
  /// has reference to the Clang module then such module would be
  /// pre-loaded by \p Loader for !Update case.
  ///
  /// \pre NoODR, Update options should be set before call to addObjectFile.
  virtual void addObjectFile(
      DWARFFile &File, ObjFileLoaderTy Loader = nullptr,
      CompileUnitHandlerTy OnCUDieLoaded = [](const DWARFUnit &) {}) = 0;

  /// Link debug info for added objFiles. Object files are linked all together.
  virtual Error link() = 0;

  /// A number of methods setting various linking options:

  /// Allows to generate log of linking process to the standard output.
  virtual void setVerbosity(bool Verbose) = 0;

  /// Print statistics to standard output.
  virtual void setStatistics(bool Statistics) = 0;

  /// Verify the input DWARF.
  virtual void setVerifyInputDWARF(bool Verify) = 0;

  /// Do not unique types according to ODR.
  virtual void setNoODR(bool NoODR) = 0;

  /// Update index tables only(do not modify rest of DWARF).
  virtual void setUpdateIndexTablesOnly(bool Update) = 0;

  /// Allow generating valid, but non-deterministic output.
  virtual void setAllowNonDeterministicOutput(bool) = 0;

  /// Set whether to keep the enclosing function for a static variable.
  virtual void setKeepFunctionForStatic(bool KeepFunctionForStatic) = 0;

  /// Use specified number of threads for parallel files linking.
  virtual void setNumThreads(unsigned NumThreads) = 0;

  /// Add kind of accelerator tables to be generated.
  virtual void addAccelTableKind(AccelTableKind Kind) = 0;

  /// Set prepend path for clang modules.
  virtual void setPrependPath(const std::string &Ppath) = 0;

  /// Set estimated objects files amount, for preliminary data allocation.
  virtual void setEstimatedObjfilesAmount(unsigned ObjFilesNum) = 0;

  /// Set verification handler which would be used to report verification
  /// errors.
  virtual void
  setInputVerificationHandler(InputVerificationHandlerTy Handler) = 0;

  /// Set map for Swift interfaces.
  virtual void setSwiftInterfacesMap(SwiftInterfacesMapTy *Map) = 0;

  /// Set prefix map for objects.
  virtual void setObjectPrefixMap(ObjectPrefixMapTy *Map) = 0;

  /// Set target DWARF version.
  virtual Error setTargetDWARFVersion(uint16_t TargetDWARFVersion) = 0;
};

} // end namespace dwarflinker
} // end namespace llvm

#endif // LLVM_DWARFLINKER_DWARFLINKERBASE_H
