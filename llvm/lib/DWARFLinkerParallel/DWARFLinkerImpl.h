//===- DWARFLinkerImpl.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_DWARFLINKERPARALLEL_DWARFLINKERIMPL_H
#define LLVM_LIB_DWARFLINKERPARALLEL_DWARFLINKERIMPL_H

#include "DWARFEmitterImpl.h"
#include "DWARFLinkerCompileUnit.h"
#include "llvm/ADT/AddressRanges.h"
#include "llvm/CodeGen/AccelTable.h"
#include "llvm/DWARFLinkerParallel/DWARFLinker.h"
#include "llvm/DWARFLinkerParallel/StringPool.h"
#include "llvm/DWARFLinkerParallel/StringTable.h"

namespace llvm {
namespace dwarflinker_parallel {

using Offset2UnitMapTy = DenseMap<uint64_t, CompileUnit *>;

struct RangeAttrPatch;
struct LocAttrPatch;

class DWARFLinkerImpl : public DWARFLinker {
public:
  DWARFLinkerImpl(MessageHandlerTy ErrorHandler,
                  MessageHandlerTy WarningHandler,
                  TranslatorFuncTy StringsTranslator)
      : UniqueUnitID(0), ErrorHandler(ErrorHandler),
        WarningHandler(WarningHandler),
        OutputStrings(Strings, StringsTranslator) {}

  Error createEmitter(const Triple &TheTriple, OutputFileType FileType,
                      raw_pwrite_stream &OutFile) override;

  ExtraDwarfEmitter *getEmitter() override;

  /// Add object file to be linked. Pre-load compile unit die. Call
  /// \p OnCUDieLoaded for each compile unit die. If specified \p File
  /// has reference to the Clang module then such module would be
  /// pre-loaded by \p Loader for !Update case.
  ///
  /// \pre NoODR, Update options should be set before call to addObjectFile.
  void addObjectFile(
      DWARFFile &File, ObjFileLoaderTy Loader = nullptr,
      CompileUnitHandlerTy OnCUDieLoaded = [](const DWARFUnit &) {}) override {}

  /// Link debug info for added files.
  Error link() override {
    reportWarning("LLVM parallel dwarflinker is not implemented yet.", "");
    return Error::success();
  }

  /// \defgroup Methods setting various linking options:
  ///
  /// @{
  ///

  /// Allows to generate log of linking process to the standard output.
  void setVerbosity(bool Verbose) override { Options.Verbose = Verbose; }

  /// Print statistics to standard output.
  void setStatistics(bool Statistics) override {
    Options.Statistics = Statistics;
  }

  /// Verify the input DWARF.
  void setVerifyInputDWARF(bool Verify) override {
    Options.VerifyInputDWARF = Verify;
  }

  /// Do not unique types according to ODR.
  void setNoODR(bool NoODR) override { Options.NoODR = NoODR; }

  /// Update index tables only(do not modify rest of DWARF).
  void setUpdateIndexTablesOnly(bool UpdateIndexTablesOnly) override {
    Options.UpdateIndexTablesOnly = UpdateIndexTablesOnly;
  }

  /// Allow generating valid, but non-deterministic output.
  void
  setAllowNonDeterministicOutput(bool AllowNonDeterministicOutput) override {
    Options.AllowNonDeterministicOutput = AllowNonDeterministicOutput;
  }

  /// Set to keep the enclosing function for a static variable.
  void setKeepFunctionForStatic(bool KeepFunctionForStatic) override {
    Options.KeepFunctionForStatic = KeepFunctionForStatic;
  }

  /// Use specified number of threads for parallel files linking.
  void setNumThreads(unsigned NumThreads) override {
    Options.Threads = NumThreads;
  }

  /// Add kind of accelerator tables to be generated.
  void addAccelTableKind(AccelTableKind Kind) override {
    assert(!llvm::is_contained(Options.AccelTables, Kind));
    Options.AccelTables.emplace_back(Kind);
  }

  /// Set prepend path for clang modules.
  void setPrependPath(const std::string &Ppath) override {
    Options.PrependPath = Ppath;
  }

  /// Set estimated objects files amount, for preliminary data allocation.
  void setEstimatedObjfilesAmount(unsigned ObjFilesNum) override {
    ObjectContexts.reserve(ObjFilesNum);
  }

  /// Set verification handler which would be used to report verification
  /// errors.
  void
  setInputVerificationHandler(InputVerificationHandlerTy Handler) override {
    Options.InputVerificationHandler = Handler;
  }

  /// Set map for Swift interfaces.
  void setSwiftInterfacesMap(SwiftInterfacesMapTy *Map) override {
    Options.ParseableSwiftInterfaces = Map;
  }

  /// Set prefix map for objects.
  void setObjectPrefixMap(ObjectPrefixMapTy *Map) override {
    Options.ObjectPrefixMap = Map;
  }

  /// Set target DWARF version.
  Error setTargetDWARFVersion(uint16_t TargetDWARFVersion) override {
    if ((TargetDWARFVersion < 1) || (TargetDWARFVersion > 5))
      return createStringError(std::errc::invalid_argument,
                               "unsupported DWARF version: %d",
                               TargetDWARFVersion);

    Options.TargetDWARFVersion = TargetDWARFVersion;
    return Error::success();
  }
  /// @}

protected:
  /// Reports Warning.
  void reportWarning(const Twine &Warning, const DWARFFile &File,
                     const DWARFDie *DIE = nullptr) const {
    if (WarningHandler != nullptr)
      WarningHandler(Warning, File.FileName, DIE);
  }

  /// Reports Warning.
  void reportWarning(const Twine &Warning, StringRef FileName,
                     const DWARFDie *DIE = nullptr) const {
    if (WarningHandler != nullptr)
      WarningHandler(Warning, FileName, DIE);
  }

  /// Reports Error.
  void reportError(const Twine &Warning, StringRef FileName,
                   const DWARFDie *DIE = nullptr) const {
    if (ErrorHandler != nullptr)
      ErrorHandler(Warning, FileName, DIE);
  }

  /// Returns next available unique Compile Unit ID.
  unsigned getNextUniqueUnitID() { return UniqueUnitID.fetch_add(1); }

  /// Keeps track of data associated with one object during linking.
  /// i.e. source file descriptor, compilation units, output data
  /// for compilation units common tables.
  struct LinkContext : public OutputSections {
    using UnitListTy = SmallVector<std::unique_ptr<CompileUnit>>;

    /// Keep information for referenced clang module: already loaded DWARF info
    /// of the clang module and a CompileUnit of the module.
    struct RefModuleUnit {
      RefModuleUnit(DWARFFile &File, std::unique_ptr<CompileUnit> Unit)
          : File(File), Unit(std::move(Unit)) {}
      RefModuleUnit(RefModuleUnit &&Other)
          : File(Other.File), Unit(std::move(Other.Unit)) {}
      RefModuleUnit(const RefModuleUnit &) = delete;

      DWARFFile &File;
      std::unique_ptr<CompileUnit> Unit;
    };
    using ModuleUnitListTy = SmallVector<RefModuleUnit>;

    /// Object file descriptor.
    DWARFFile &File;

    /// Set of Compilation Units(may be accessed asynchroniously for reading).
    UnitListTy CompileUnits;

    /// Set of Compile Units for modules.
    ModuleUnitListTy ModulesCompileUnits;

    /// Size of Debug info before optimizing.
    uint64_t OriginalDebugInfoSize = 0;

    /// Output sections, common for all compilation units.
    OutTablesFileTy OutDebugInfoBytes;

    /// Endianness for the final file.
    support::endianness Endianess = support::endianness::little;

    LinkContext(DWARFFile &File) : File(File) {
      if (File.Dwarf) {
        if (!File.Dwarf->compile_units().empty())
          CompileUnits.reserve(File.Dwarf->getNumCompileUnits());

        Endianess = File.Dwarf->isLittleEndian() ? support::endianness::little
                                                 : support::endianness::big;
      }
    }

    /// Add Compile Unit corresponding to the module.
    void addModulesCompileUnit(RefModuleUnit &&Unit) {
      ModulesCompileUnits.emplace_back(std::move(Unit));
    }

    /// Return Endiannes of the source DWARF information.
    support::endianness getEndianness() { return Endianess; }

    /// \returns pointer to compilation unit which corresponds \p Offset.
    CompileUnit *getUnitForOffset(CompileUnit &CU, uint64_t Offset) const;
  };

  /// linking options
  struct DWARFLinkerOptions {
    /// DWARF version for the output.
    uint16_t TargetDWARFVersion = 0;

    /// Generate processing log to the standard output.
    bool Verbose = false;

    /// Print statistics.
    bool Statistics = false;

    /// Verify the input DWARF.
    bool VerifyInputDWARF = false;

    /// Do not unique types according to ODR
    bool NoODR = false;

    /// Update index tables.
    bool UpdateIndexTablesOnly = false;

    /// Whether we want a static variable to force us to keep its enclosing
    /// function.
    bool KeepFunctionForStatic = false;

    /// Allow to generate valid, but non deterministic output.
    bool AllowNonDeterministicOutput = false;

    /// Number of threads.
    unsigned Threads = 1;

    /// The accelerator table kinds
    SmallVector<AccelTableKind, 1> AccelTables;

    /// Prepend path for the clang modules.
    std::string PrependPath;

    /// input verification handler(it might be called asynchronously).
    InputVerificationHandlerTy InputVerificationHandler = nullptr;

    /// A list of all .swiftinterface files referenced by the debug
    /// info, mapping Module name to path on disk. The entries need to
    /// be uniqued and sorted and there are only few entries expected
    /// per compile unit, which is why this is a std::map.
    /// this is dsymutil specific fag.
    ///
    /// (it might be called asynchronously).
    SwiftInterfacesMapTy *ParseableSwiftInterfaces = nullptr;

    /// A list of remappings to apply to file paths.
    ///
    /// (it might be called asynchronously).
    ObjectPrefixMapTy *ObjectPrefixMap = nullptr;
  } Options;

  /// \defgroup Data members accessed asinchroniously.
  ///
  /// @{

  /// Unique ID for compile unit.
  std::atomic<unsigned> UniqueUnitID;

  /// Strings pool. Keeps all strings.
  StringPool Strings;

  /// error handler(it might be called asynchronously).
  MessageHandlerTy ErrorHandler = nullptr;

  /// warning handler(it might be called asynchronously).
  MessageHandlerTy WarningHandler = nullptr;
  /// @}

  /// \defgroup Data members accessed sequentially.
  ///
  /// @{

  /// Set of strings which should be emitted.
  StringTable OutputStrings;

  /// Keeps all linking contexts.
  SmallVector<std::unique_ptr<LinkContext>> ObjectContexts;

  /// The emitter of final dwarf file.
  std::unique_ptr<DwarfEmitterImpl> TheDwarfEmitter;
  /// @}
};

} // end namespace dwarflinker_parallel
} // end namespace llvm

#endif // LLVM_LIB_DWARFLINKERPARALLEL_DWARFLINKERIMPL_H
