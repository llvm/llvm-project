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
#include "StringEntryToDwarfStringPoolEntryMap.h"
#include "llvm/ADT/AddressRanges.h"
#include "llvm/CodeGen/AccelTable.h"
#include "llvm/DWARFLinkerParallel/DWARFLinker.h"
#include "llvm/DWARFLinkerParallel/StringPool.h"

namespace llvm {
namespace dwarflinker_parallel {

/// This class links debug info.
class DWARFLinkerImpl : public DWARFLinker {
public:
  DWARFLinkerImpl(MessageHandlerTy ErrorHandler,
                  MessageHandlerTy WarningHandler,
                  TranslatorFuncTy StringsTranslator)
      : UniqueUnitID(0), DebugStrStrings(GlobalData),
        DebugLineStrStrings(GlobalData), CommonSections(GlobalData) {
    GlobalData.setTranslator(StringsTranslator);
    GlobalData.setErrorHandler(ErrorHandler);
    GlobalData.setWarningHandler(WarningHandler);
  }

  /// Create debug info emitter.
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

      CompileUnitHandlerTy OnCUDieLoaded = [](const DWARFUnit &) {}) override;

  /// Link debug info for added files.
  Error link() override;

  /// \defgroup Methods setting various linking options:
  ///
  /// @{
  ///

  /// Allows to generate log of linking process to the standard output.
  void setVerbosity(bool Verbose) override {
    GlobalData.Options.Verbose = Verbose;
  }

  /// Print statistics to standard output.
  void setStatistics(bool Statistics) override {
    GlobalData.Options.Statistics = Statistics;
  }

  /// Verify the input DWARF.
  void setVerifyInputDWARF(bool Verify) override {
    GlobalData.Options.VerifyInputDWARF = Verify;
  }

  /// Do not unique types according to ODR.
  void setNoODR(bool) override {
    // FIXME: set option when ODR mode will be supported.
    // getOptions().NoODR = NoODR;
    GlobalData.Options.NoODR = true;
  }

  /// Update index tables only(do not modify rest of DWARF).
  void setUpdateIndexTablesOnly(bool UpdateIndexTablesOnly) override {
    GlobalData.Options.UpdateIndexTablesOnly = UpdateIndexTablesOnly;
  }

  /// Allow generating valid, but non-deterministic output.
  void
  setAllowNonDeterministicOutput(bool AllowNonDeterministicOutput) override {
    GlobalData.Options.AllowNonDeterministicOutput =
        AllowNonDeterministicOutput;
  }

  /// Set to keep the enclosing function for a static variable.
  void setKeepFunctionForStatic(bool KeepFunctionForStatic) override {
    GlobalData.Options.KeepFunctionForStatic = KeepFunctionForStatic;
  }

  /// Use specified number of threads for parallel files linking.
  void setNumThreads(unsigned NumThreads) override {
    GlobalData.Options.Threads = NumThreads;
  }

  /// Add kind of accelerator tables to be generated.
  void addAccelTableKind(AccelTableKind Kind) override {
    assert(!llvm::is_contained(GlobalData.getOptions().AccelTables, Kind));
    GlobalData.Options.AccelTables.emplace_back(Kind);
  }

  /// Set prepend path for clang modules.
  void setPrependPath(const std::string &Ppath) override {
    GlobalData.Options.PrependPath = Ppath;
  }

  /// Set estimated objects files amount, for preliminary data allocation.
  void setEstimatedObjfilesAmount(unsigned ObjFilesNum) override {
    ObjectContexts.reserve(ObjFilesNum);
  }

  /// Set verification handler which would be used to report verification
  /// errors.
  void
  setInputVerificationHandler(InputVerificationHandlerTy Handler) override {
    GlobalData.Options.InputVerificationHandler = Handler;
  }

  /// Set map for Swift interfaces.
  void setSwiftInterfacesMap(SwiftInterfacesMapTy *Map) override {
    GlobalData.Options.ParseableSwiftInterfaces = Map;
  }

  /// Set prefix map for objects.
  void setObjectPrefixMap(ObjectPrefixMapTy *Map) override {
    GlobalData.Options.ObjectPrefixMap = Map;
  }

  /// Set target DWARF version.
  Error setTargetDWARFVersion(uint16_t TargetDWARFVersion) override {
    if ((TargetDWARFVersion < 1) || (TargetDWARFVersion > 5))
      return createStringError(std::errc::invalid_argument,
                               "unsupported DWARF version: %d",
                               TargetDWARFVersion);

    GlobalData.Options.TargetDWARFVersion = TargetDWARFVersion;
    return Error::success();
  }
  /// @}

protected:
  /// Verify input DWARF file.
  void verifyInput(const DWARFFile &File);

  /// Validate specified options.
  Error validateAndUpdateOptions();

  /// Take already linked compile units and glue them into single file.
  void glueCompileUnitsAndWriteToTheOutput();

  /// Hold the input and output of the debug info size in bytes.
  struct DebugInfoSize {
    uint64_t Input;
    uint64_t Output;
  };

  friend class DependencyTracker;
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
    DWARFFile &InputDWARFFile;

    /// Set of Compilation Units(may be accessed asynchroniously for reading).
    UnitListTy CompileUnits;

    /// Set of Compile Units for modules.
    ModuleUnitListTy ModulesCompileUnits;

    /// Size of Debug info before optimizing.
    uint64_t OriginalDebugInfoSize = 0;

    /// Flag indicating that all inter-connected units are loaded
    /// and the dwarf linking process for these units is started.
    bool InterCUProcessingStarted = false;

    StringMap<uint64_t> &ClangModules;

    std::optional<Triple> TargetTriple;

    /// Flag indicating that new inter-connected compilation units were
    /// discovered. It is used for restarting units processing
    /// if new inter-connected units were found.
    std::atomic<bool> HasNewInterconnectedCUs = {false};

    /// Counter for compile units ID.
    std::atomic<size_t> &UniqueUnitID;

    LinkContext(LinkingGlobalData &GlobalData, DWARFFile &File,
                StringMap<uint64_t> &ClangModules,
                std::atomic<size_t> &UniqueUnitID,
                std::optional<Triple> TargetTriple)
        : OutputSections(GlobalData), InputDWARFFile(File),
          ClangModules(ClangModules), TargetTriple(TargetTriple),
          UniqueUnitID(UniqueUnitID) {

      if (File.Dwarf) {
        if (!File.Dwarf->compile_units().empty())
          CompileUnits.reserve(File.Dwarf->getNumCompileUnits());

        // Set context format&endianness based on the input file.
        Format.Version = File.Dwarf->getMaxVersion();
        Format.AddrSize = File.Dwarf->getCUAddrSize();
        Endianness = File.Dwarf->isLittleEndian() ? llvm::endianness::little
                                                  : llvm::endianness::big;
      }
    }

    /// Check whether specified \p CUDie is a Clang module reference.
    /// if \p Quiet is false then display error messages.
    /// \return first == true if CUDie is a Clang module reference.
    ///         second == true if module is already loaded.
    std::pair<bool, bool> isClangModuleRef(const DWARFDie &CUDie,
                                           std::string &PCMFile,
                                           unsigned Indent, bool Quiet);

    /// If this compile unit is really a skeleton CU that points to a
    /// clang module, register it in ClangModules and return true.
    ///
    /// A skeleton CU is a CU without children, a DW_AT_gnu_dwo_name
    /// pointing to the module, and a DW_AT_gnu_dwo_id with the module
    /// hash.
    bool registerModuleReference(const DWARFDie &CUDie, ObjFileLoaderTy Loader,
                                 CompileUnitHandlerTy OnCUDieLoaded,
                                 unsigned Indent = 0);

    /// Recursively add the debug info in this clang module .pcm
    /// file (and all the modules imported by it in a bottom-up fashion)
    /// to ModuleUnits.
    Error loadClangModule(ObjFileLoaderTy Loader, const DWARFDie &CUDie,
                          const std::string &PCMFile,
                          CompileUnitHandlerTy OnCUDieLoaded,
                          unsigned Indent = 0);

    /// Add Compile Unit corresponding to the module.
    void addModulesCompileUnit(RefModuleUnit &&Unit) {
      ModulesCompileUnits.emplace_back(std::move(Unit));
    }

    /// Computes the total size of the debug info.
    uint64_t getInputDebugInfoSize() const {
      uint64_t Size = 0;

      if (InputDWARFFile.Dwarf == nullptr)
        return Size;

      for (auto &Unit : InputDWARFFile.Dwarf->compile_units())
        Size += Unit->getLength();

      return Size;
    }

    /// Link compile units for this context.
    Error link();

    /// Link specified compile unit until specified stage.
    void linkSingleCompileUnit(
        CompileUnit &CU,
        enum CompileUnit::Stage DoUntilStage = CompileUnit::Stage::Cleaned);

    /// Emit invariant sections.
    Error emitInvariantSections();

    /// Clone and emit .debug_frame.
    Error cloneAndEmitDebugFrame();

    /// Emit FDE record.
    void emitFDE(uint32_t CIEOffset, uint32_t AddrSize, uint64_t Address,
                 StringRef FDEBytes, SectionDescriptor &Section);

    std::function<CompileUnit *(uint64_t)> getUnitForOffset =
        [&](uint64_t Offset) -> CompileUnit * {
      auto CU = llvm::upper_bound(
          CompileUnits, Offset,
          [](uint64_t LHS, const std::unique_ptr<CompileUnit> &RHS) {
            return LHS < RHS->getOrigUnit().getNextUnitOffset();
          });

      return CU != CompileUnits.end() ? CU->get() : nullptr;
    };
  };

  /// Enumerate all compile units and assign offsets to their sections and
  /// strings.
  void assignOffsets();

  /// Enumerate all compile units and assign offsets to their sections.
  void assignOffsetsToSections();

  /// Enumerate all compile units and assign offsets to their strings.
  void assignOffsetsToStrings();

  /// Print statistic for processed Debug Info.
  void printStatistic();

  enum StringDestinationKind : uint8_t { DebugStr, DebugLineStr };

  /// Enumerates all strings.
  void forEachOutputString(
      function_ref<void(StringDestinationKind, const StringEntry *)>
          StringHandler);

  /// Enumerates sections for modules, invariant for object files, compile
  /// units.
  void forEachObjectSectionsSet(
      function_ref<void(OutputSections &SectionsSet)> SectionsSetHandler);

  /// Enumerates all comple units.
  void forEachCompileUnit(function_ref<void(CompileUnit *CU)> UnitHandler);

  /// Enumerates all patches and update them with the correct values.
  void patchOffsetsAndSizes();

  /// Emit debug sections common for all input files.
  void emitCommonSectionsAndWriteCompileUnitsToTheOutput();

  /// Emit apple accelerator sections.
  void emitAppleAcceleratorSections(const Triple &TargetTriple);

  /// Emit .debug_names section.
  void emitDWARFv5DebugNamesSection(const Triple &TargetTriple);

  /// Emit string sections.
  void emitStringSections();

  /// Cleanup data(string pools) after output sections are generated.
  void cleanupDataAfterDWARFOutputIsWritten();

  /// Enumerate all compile units and put their data into the output stream.
  void writeCompileUnitsToTheOutput();

  /// Enumerate common sections and put their data into the output stream.
  void writeCommonSectionsToTheOutput();

  /// \defgroup Data members accessed asinchroniously.
  ///
  /// @{

  /// Unique ID for compile unit.
  std::atomic<size_t> UniqueUnitID;

  /// Mapping the PCM filename to the DwoId.
  StringMap<uint64_t> ClangModules;
  std::mutex ClangModulesMutex;
  /// @}

  /// \defgroup Data members accessed sequentially.
  ///
  /// @{
  /// DwarfStringPoolEntries for .debug_str section.
  StringEntryToDwarfStringPoolEntryMap DebugStrStrings;

  /// DwarfStringPoolEntries for .debug_line_str section.
  StringEntryToDwarfStringPoolEntryMap DebugLineStrStrings;

  /// Keeps all linking contexts.
  SmallVector<std::unique_ptr<LinkContext>> ObjectContexts;

  /// Common sections.
  OutputSections CommonSections;

  /// The emitter of final dwarf file.
  std::unique_ptr<DwarfEmitterImpl> TheDwarfEmitter;

  /// Overall compile units number.
  uint64_t OverallNumberOfCU = 0;

  /// Data global for the whole linking process.
  LinkingGlobalData GlobalData;
  /// @}
};

} // end namespace dwarflinker_parallel
} // end namespace llvm

#endif // LLVM_LIB_DWARFLINKERPARALLEL_DWARFLINKERIMPL_H
