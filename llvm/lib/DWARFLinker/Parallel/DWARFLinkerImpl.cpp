//=== DWARFLinkerImpl.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DWARFLinkerImpl.h"
#include "DependencyTracker.h"
#include "llvm/DWARFLinker/Utils.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugAbbrev.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/ThreadPool.h"

using namespace llvm;
using namespace dwarf_linker;
using namespace dwarf_linker::parallel;

DWARFLinkerImpl::DWARFLinkerImpl(MessageHandlerTy ErrorHandler,
                                 MessageHandlerTy WarningHandler)
    : UniqueUnitID(0), DebugStrStrings(GlobalData),
      DebugLineStrStrings(GlobalData), CommonSections(GlobalData) {
  GlobalData.setErrorHandler(ErrorHandler);
  GlobalData.setWarningHandler(WarningHandler);
}

DWARFLinkerImpl::LinkContext::LinkContext(LinkingGlobalData &GlobalData,
                                          DWARFFile &File, uint64_t ObjFileIdx,
                                          StringMap<uint64_t> &ClangModules,
                                          uint64_t &ModuleUnitIdx,
                                          std::atomic<size_t> &UniqueUnitID)
    : OutputSections(GlobalData), InputDWARFFile(File),
      ObjectFileIdx(ObjFileIdx), ClangModules(ClangModules),
      ModuleUnitIdx(ModuleUnitIdx), UniqueUnitID(UniqueUnitID) {

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

void DWARFLinkerImpl::addObjectFile(DWARFFile &File, ObjFileLoaderTy Loader,
                                    CompileUnitHandlerTy OnCUDieLoaded) {
  ObjectContexts.emplace_back(std::make_unique<LinkContext>(
      GlobalData, File, FirstObjFileIdx + ObjectContexts.size(), ClangModules,
      ModuleUnitIdx, UniqueUnitID));

  if (ObjectContexts.back()->InputDWARFFile.Dwarf) {
    for (const std::unique_ptr<DWARFUnit> &CU :
         ObjectContexts.back()->InputDWARFFile.Dwarf->compile_units()) {
      DWARFDie CUDie = CU->getUnitDIE();

      if (!CUDie)
        continue;

      OnCUDieLoaded(*CU);

      // Register mofule reference.
      if (!GlobalData.getOptions().UpdateIndexTablesOnly)
        ObjectContexts.back()->registerModuleReference(CUDie, Loader,
                                                       OnCUDieLoaded);
    }
  }
}

void DWARFLinkerImpl::setEstimatedObjfilesAmount(unsigned ObjFilesNum) {
  ObjectContexts.reserve(ObjFilesNum);
}

Error DWARFLinkerImpl::link() {
  // UniqueUnitID is initialized by the constructor and must not be reset
  // here. addObjectFile() may have already handed out IDs to clang module
  // CUs loaded from .pcm files, and the IDs handed out below must stay
  // disjoint from those.

  if (Error Err = validateAndUpdateOptions())
    return Err;

  dwarf::FormParams GlobalFormat = {GlobalData.getOptions().TargetDWARFVersion,
                                    0, dwarf::DwarfFormat::DWARF32};
  llvm::endianness GlobalEndianness = llvm::endianness::native;

  if (std::optional<std::reference_wrapper<const Triple>> CurTriple =
          GlobalData.getTargetTriple()) {
    GlobalEndianness = (*CurTriple).get().isLittleEndian()
                           ? llvm::endianness::little
                           : llvm::endianness::big;
  }
  std::optional<uint16_t> Language;

  for (std::unique_ptr<LinkContext> &Context : ObjectContexts) {
    if (Context->InputDWARFFile.Dwarf == nullptr) {
      Context->setOutputFormat(Context->getFormParams(), GlobalEndianness);
      continue;
    }

    if (GlobalData.getOptions().Verbose) {
      outs() << "DEBUG MAP OBJECT: " << Context->InputDWARFFile.FileName
             << "\n";

      for (const std::unique_ptr<DWARFUnit> &OrigCU :
           Context->InputDWARFFile.Dwarf->compile_units()) {
        outs() << "Input compilation unit:";
        DIDumpOptions DumpOpts;
        DumpOpts.ChildRecurseDepth = 0;
        DumpOpts.Verbose = GlobalData.getOptions().Verbose;
        OrigCU->getUnitDIE().dump(outs(), 0, DumpOpts);
      }
    }

    // Verify input DWARF if requested.
    if (GlobalData.getOptions().VerifyInputDWARF)
      verifyInput(Context->InputDWARFFile);

    if (!GlobalData.getTargetTriple())
      GlobalEndianness = Context->getEndianness();
    GlobalFormat.AddrSize =
        std::max(GlobalFormat.AddrSize, Context->getFormParams().AddrSize);

    Context->setOutputFormat(Context->getFormParams(), GlobalEndianness);

    // FIXME: move creation of CompileUnits into the addObjectFile.
    // This would allow to not scan for context Language and Modules state
    // twice. And then following handling might be removed.
    for (const std::unique_ptr<DWARFUnit> &OrigCU :
         Context->InputDWARFFile.Dwarf->compile_units()) {
      DWARFDie UnitDie = OrigCU->getUnitDIE();

      if (!Language) {
        if (std::optional<uint64_t> LangVal = UnitDie.getLanguage())
          if (isODRLanguage(*LangVal))
            Language = static_cast<uint16_t>(*LangVal);
      }
    }

    // Clang module units decide their ODR availability from their own
    // language, so they have to be part of this scan as well. A module unit
    // can be the only ODR unit of a link, and any unit which deduplicates
    // types requires the artificial type unit to exist.
    for (const std::unique_ptr<CompileUnit> &Module :
         Context->ModulesCompileUnits) {
      if (!Language) {
        if (std::optional<uint16_t> LangVal = Module->getLanguage())
          if (isODRLanguage(*LangVal))
            Language = *LangVal;
      }
    }
  }

  if (GlobalFormat.AddrSize == 0) {
    if (std::optional<std::reference_wrapper<const Triple>> TargetTriple =
            GlobalData.getTargetTriple())
      GlobalFormat.AddrSize = (*TargetTriple).get().isArch32Bit() ? 4 : 8;
    else
      GlobalFormat.AddrSize = 8;
  }

  CommonSections.setOutputFormat(GlobalFormat, GlobalEndianness);

  if (!GlobalData.Options.NoODR && Language.has_value()) {
    llvm::parallel::TaskGroup TGroup;
    TGroup.spawn([&]() {
      ArtificialTypeUnit = std::make_unique<TypeUnit>(
          GlobalData, UniqueUnitID++, Language, GlobalFormat, GlobalEndianness);
    });
  }

  // Set this process-global once. link() runs per architecture and dsymutil
  // may run those links concurrently, so assigning it from each would be a
  // data race; the thread count is the same for every architecture, so the
  // first assignment suffices. Size the executor from that thread count rather
  // than the per-architecture CU count, which is moot once it is shared.
  static llvm::once_flag ParallelStrategyFlag;
  llvm::call_once(ParallelStrategyFlag, [&] {
    llvm::parallel::strategy =
        hardware_concurrency(GlobalData.getOptions().Threads);
  });

  // Link object files.
  if (GlobalData.getOptions().Threads == 1) {
    for (std::unique_ptr<LinkContext> &Context : ObjectContexts) {
      // Link object file.
      if (Error Err = Context->link(ArtificialTypeUnit.get()))
        GlobalData.error(std::move(Err), Context->InputDWARFFile.FileName);
      if (Error Err = Context->unloadInput())
        GlobalData.error(std::move(Err), Context->InputDWARFFile.FileName);
    }
  } else {
    assert(ThreadPool && "setThreadPool() must be called before link()");
    ThreadPoolTaskGroup Group(*ThreadPool);
    for (std::unique_ptr<LinkContext> &Context : ObjectContexts)
      Group.async([&]() {
        // Link object file.
        if (Error Err = Context->link(ArtificialTypeUnit.get()))
          GlobalData.error(std::move(Err), Context->InputDWARFFile.FileName);
        if (Error Err = Context->unloadInput())
          GlobalData.error(std::move(Err), Context->InputDWARFFile.FileName);
      });
  }

  // Merge staged parseable Swift interface entries into the shared map. Done
  // serially so that the final map contents and any conflict warnings are
  // deterministic.
  if (DWARFLinkerBase::SwiftInterfacesMapTy *SwiftInterfaces =
          GlobalData.Options.ParseableSwiftInterfaces) {
    for (std::unique_ptr<LinkContext> &Context : ObjectContexts) {
      for (std::unique_ptr<CompileUnit> &ModuleUnit :
           Context->ModulesCompileUnits)
        ModuleUnit->mergeSwiftInterfaces(*SwiftInterfaces);
      for (std::unique_ptr<CompileUnit> &CU : Context->CompileUnits)
        CU->mergeSwiftInterfaces(*SwiftInterfaces);
    }
  }

  // Build the linker-wide CIE registry, then emit each context's
  // .debug_frame in parallel. See CIERegistry for the ownership rules.
  if (!GlobalData.getOptions().UpdateIndexTablesOnly) {
    LinkContext::CIERegistry CIEs;
    for (std::unique_ptr<LinkContext> &Context : ObjectContexts)
      if (Context->FrameScan)
        Context->registerCIEs(CIEs);

    llvm::parallel::TaskGroup TGroup;
    for (std::unique_ptr<LinkContext> &Context : ObjectContexts) {
      if (!Context->FrameScan)
        continue;
      TGroup.spawn([&]() {
        if (Error Err = Context->emitDebugFrame(CIEs))
          GlobalData.error(std::move(Err), Context->InputDWARFFile.FileName);
      });
    }
  }

  if (ArtificialTypeUnit != nullptr && !ArtificialTypeUnit->getTypePool()
                                            .getRoot()
                                            ->getValue()
                                            .load()
                                            ->Children.empty()) {
    if (GlobalData.getTargetTriple().has_value())
      if (Error Err = ArtificialTypeUnit->finishCloningAndEmit(
              (*GlobalData.getTargetTriple()).get()))
        return Err;
  }

  // At this stage each compile units are cloned to their own set of debug
  // sections. Now, update patches, assign offsets and assemble final file
  // glueing debug tables from each compile unit.
  glueCompileUnitsAndWriteToTheOutput();

  return Error::success();
}

void DWARFLinkerImpl::verifyInput(const DWARFFile &File) {
  assert(File.Dwarf);

  std::string Buffer;
  raw_string_ostream OS(Buffer);
  DIDumpOptions DumpOpts;
  if (!File.Dwarf->verify(OS, DumpOpts.noImplicitRecursion())) {
    if (GlobalData.getOptions().InputVerificationHandler)
      GlobalData.getOptions().InputVerificationHandler(File, OS.str());
  }
}

Error DWARFLinkerImpl::validateAndUpdateOptions() {
  if (GlobalData.getOptions().TargetDWARFVersion == 0)
    return createStringError(std::errc::invalid_argument,
                             "target DWARF version is not set");

  if (GlobalData.getOptions().Verbose && GlobalData.getOptions().Threads != 1) {
    GlobalData.Options.Threads = 1;
    GlobalData.warn(
        "set number of threads to 1 to make --verbose to work properly.", "");
  }

  // Do not do types deduplication in case --update.
  if (GlobalData.getOptions().UpdateIndexTablesOnly &&
      !GlobalData.Options.NoODR)
    GlobalData.Options.NoODR = true;

  return Error::success();
}

/// Resolve the relative path to a build artifact referenced by DWARF of \p File
/// by applying DW_AT_comp_dir.
static void resolveRelativeObjectPath(SmallVectorImpl<char> &Buf, DWARFDie CU,
                                      const DWARFFile &File) {
  sys::path::append(
      Buf, File.remapPath(dwarf::toStringRef(CU.find(dwarf::DW_AT_comp_dir))));
}

static uint64_t getDwoId(const DWARFDie &CUDie) {
  auto DwoId = dwarf::toUnsigned(
      CUDie.find({dwarf::DW_AT_dwo_id, dwarf::DW_AT_GNU_dwo_id}));
  if (DwoId)
    return *DwoId;
  return 0;
}

static std::string
remapPath(StringRef Path,
          const DWARFLinker::ObjectPrefixMapTy &ObjectPrefixMap) {
  if (ObjectPrefixMap.empty())
    return Path.str();

  SmallString<256> p = Path;
  for (const auto &Entry : ObjectPrefixMap)
    if (llvm::sys::path::replace_path_prefix(p, Entry.first, Entry.second))
      break;
  return p.str().str();
}

static std::string getPCMFile(const DWARFDie &CUDie,
                              DWARFLinker::ObjectPrefixMapTy *ObjectPrefixMap,
                              const DWARFFile &File) {
  std::string PCMFile = dwarf::toString(
      CUDie.find({dwarf::DW_AT_dwo_name, dwarf::DW_AT_GNU_dwo_name}), "");

  if (PCMFile.empty())
    return PCMFile;

  if (ObjectPrefixMap)
    PCMFile = remapPath(PCMFile, *ObjectPrefixMap);

  return File.remapPath(PCMFile);
}

std::pair<bool, bool> DWARFLinkerImpl::LinkContext::isClangModuleRef(
    const DWARFDie &CUDie, std::string &PCMFile, unsigned Indent, bool Quiet) {
  if (PCMFile.empty())
    return std::make_pair(false, false);

  // Clang module DWARF skeleton CUs abuse this for the path to the module.
  uint64_t DwoId = getDwoId(CUDie);

  std::string Name = dwarf::toString(CUDie.find(dwarf::DW_AT_name), "");
  if (Name.empty()) {
    if (!Quiet)
      GlobalData.warn("anonymous module skeleton CU for " + PCMFile + ".",
                      InputDWARFFile.FileName);
    return std::make_pair(true, true);
  }

  if (!Quiet && GlobalData.getOptions().Verbose) {
    outs().indent(Indent);
    outs() << "Found clang module reference " << PCMFile;
  }

  auto Cached = ClangModules.find(PCMFile);
  if (Cached != ClangModules.end()) {
    // FIXME: Until PR27449 (https://llvm.org/bugs/show_bug.cgi?id=27449) is
    // fixed in clang, only warn about DWO_id mismatches in verbose mode.
    // ASTFileSignatures will change randomly when a module is rebuilt.
    if (!Quiet && GlobalData.getOptions().Verbose && (Cached->second != DwoId))
      GlobalData.warn(
          Twine("hash mismatch: this object file was built against a "
                "different version of the module ") +
              PCMFile + ".",
          InputDWARFFile.FileName);
    if (!Quiet && GlobalData.getOptions().Verbose)
      outs() << " [cached].\n";
    return std::make_pair(true, true);
  }

  return std::make_pair(true, false);
}

/// If this compile unit is really a skeleton CU that points to a
/// clang module, register it in ClangModules and return true.
///
/// A skeleton CU is a CU without children, a DW_AT_gnu_dwo_name
/// pointing to the module, and a DW_AT_gnu_dwo_id with the module
/// hash.
bool DWARFLinkerImpl::LinkContext::registerModuleReference(
    const DWARFDie &CUDie, ObjFileLoaderTy Loader,
    CompileUnitHandlerTy OnCUDieLoaded, unsigned Indent) {
  std::string PCMFile = getPCMFile(
      CUDie, GlobalData.getOptions().ObjectPrefixMap, InputDWARFFile);
  std::pair<bool, bool> IsClangModuleRef =
      isClangModuleRef(CUDie, PCMFile, Indent, false);

  if (!IsClangModuleRef.first)
    return false;

  if (IsClangModuleRef.second)
    return true;

  if (GlobalData.getOptions().Verbose)
    outs() << " ...\n";

  // Cyclic dependencies are disallowed by Clang, but we still
  // shouldn't run into an infinite loop, so mark it as processed now.
  ClangModules.insert({PCMFile, getDwoId(CUDie)});

  if (Error E =
          loadClangModule(Loader, CUDie, PCMFile, OnCUDieLoaded, Indent + 2)) {
    consumeError(std::move(E));
    return false;
  }
  return true;
}

Error DWARFLinkerImpl::LinkContext::loadClangModule(
    ObjFileLoaderTy Loader, const DWARFDie &CUDie, const std::string &PCMFile,
    CompileUnitHandlerTy OnCUDieLoaded, unsigned Indent) {

  uint64_t DwoId = getDwoId(CUDie);
  std::string ModuleName = dwarf::toString(CUDie.find(dwarf::DW_AT_name), "");

  /// Using a SmallString<0> because loadClangModule() is recursive.
  SmallString<0> Path(GlobalData.getOptions().PrependPath);
  if (sys::path::is_relative(PCMFile))
    resolveRelativeObjectPath(Path, CUDie, InputDWARFFile);
  sys::path::append(Path, PCMFile);
  // Don't use the cached binary holder because we have no thread-safety
  // guarantee and the lifetime is limited.

  if (Loader == nullptr) {
    GlobalData.error("cann't load clang module: loader is not specified.",
                     InputDWARFFile.FileName);
    return Error::success();
  }

  auto ErrOrObj = Loader(InputDWARFFile.FileName, Path);
  if (!ErrOrObj)
    return Error::success();

  // The module was built by the same build as the object file referencing it,
  // so the same prefix map applies.
  if (!ErrOrObj->PrefixMap)
    ErrOrObj->PrefixMap = InputDWARFFile.PrefixMap;

  std::unique_ptr<CompileUnit> Unit;
  for (const auto &CU : ErrOrObj->Dwarf->compile_units()) {
    OnCUDieLoaded(*CU);
    // Recursively get all modules imported by this one.
    auto ChildCUDie = CU->getUnitDIE();
    if (!ChildCUDie)
      continue;
    if (!registerModuleReference(ChildCUDie, Loader, OnCUDieLoaded, Indent)) {
      if (Unit) {
        std::string Err =
            (PCMFile +
             ": Clang modules are expected to have exactly 1 compile unit.\n");
        GlobalData.error(Err, InputDWARFFile.FileName);
        return make_error<StringError>(Err, inconvertibleErrorCode());
      }
      // FIXME: Until PR27449 (https://llvm.org/bugs/show_bug.cgi?id=27449) is
      // fixed in clang, only warn about DWO_id mismatches in verbose mode.
      // ASTFileSignatures will change randomly when a module is rebuilt.
      uint64_t PCMDwoId = getDwoId(ChildCUDie);
      if (PCMDwoId != DwoId) {
        if (GlobalData.getOptions().Verbose)
          GlobalData.warn(
              Twine("hash mismatch: this object file was built against a "
                    "different version of the module ") +
                  PCMFile + ".",
              InputDWARFFile.FileName);
        // Update the cache entry with the DwoId of the module loaded from disk.
        ClangModules[PCMFile] = PCMDwoId;
      }

      // Empty modules units should not be cloned.
      if (!ChildCUDie.hasChildren())
        continue;

      // Add this module.
      Unit = std::make_unique<CompileUnit>(
          GlobalData, *CU, UniqueUnitID.fetch_add(1), ModuleName, *ErrOrObj,
          getUnitForOffset, CU->getFormParams(), getEndianness());
    }
  }

  if (Unit) {
    // Incrementing the shared counter needs no synchronization: reaching this
    // point requires a loader, and only the serial pass over the object files
    // supplies one.
    if (Error E = Unit->setPriority(ModuleUnitObjFileIdx, ModuleUnitIdx++))
      return E;

    ModulesCompileUnits.emplace_back(std::move(Unit));
    // Preload line table, as it can't be loaded asynchronously.
    ModulesCompileUnits.back()->loadLineTable();
  }

  return Error::success();
}

Error DWARFLinkerImpl::LinkContext::link(TypeUnit *ArtificialTypeUnit) {
  InterCUProcessingStarted = false;
  if (!InputDWARFFile.Dwarf)
    return Error::success();

  // Preload macro tables, as they can't be loaded asynchronously.
  InputDWARFFile.Dwarf->getDebugMacinfo();
  InputDWARFFile.Dwarf->getDebugMacro();

  // Link modules compile units first.
  parallelForEach(ModulesCompileUnits, [&](std::unique_ptr<CompileUnit> &Mod) {
    // A module unit describes DIEs which no address reaches, so nothing marks
    // it inter-connected and the inter-connected loops below, which iterate
    // CompileUnits alone, would never advance it.
    assert(!Mod->isInterconnectedCU() && "module unit is inter-connected");
    linkSingleCompileUnit(*Mod, ArtificialTypeUnit);
  });

  // Check for live relocations. If there is no any live relocation then we
  // can skip entire object file.
  if (!GlobalData.getOptions().UpdateIndexTablesOnly &&
      !InputDWARFFile.Addresses->hasValidRelocs()) {
    if (GlobalData.getOptions().Verbose)
      outs() << "No valid relocations found. Skipping.\n";
    return Error::success();
  }

  OriginalDebugInfoSize = getInputDebugInfoSize();

  // Create CompileUnit structures to keep information about source
  // DWARFUnit`s, load line tables.
  uint64_t LocalCUIdx = 0;
  for (const auto &OrigCU : InputDWARFFile.Dwarf->compile_units()) {
    // Load only unit DIE at this stage.
    auto CUDie = OrigCU->getUnitDIE();
    std::string PCMFile = getPCMFile(
        CUDie, GlobalData.getOptions().ObjectPrefixMap, InputDWARFFile);

    // The !isClangModuleRef condition effectively skips over fully resolved
    // skeleton units.
    if (!CUDie || GlobalData.getOptions().UpdateIndexTablesOnly ||
        !isClangModuleRef(CUDie, PCMFile, 0, true).first) {
      CompileUnits.emplace_back(std::make_unique<CompileUnit>(
          GlobalData, *OrigCU, UniqueUnitID.fetch_add(1), "", InputDWARFFile,
          getUnitForOffset, OrigCU->getFormParams(), getEndianness()));
      if (llvm::Error E =
              CompileUnits.back()->setPriority(ObjectFileIdx, LocalCUIdx++))
        return E;

      // Preload line table, as it can't be loaded asynchronously.
      CompileUnits.back()->loadLineTable();
    }
  };

  HasNewInterconnectedCUs = false;

  // Link self-sufficient compile units and discover inter-connected compile
  // units.
  parallelForEach(CompileUnits, [&](std::unique_ptr<CompileUnit> &CU) {
    linkSingleCompileUnit(*CU, ArtificialTypeUnit);
  });

  // Link all inter-connected units.
  if (HasNewInterconnectedCUs) {
    InterCUProcessingStarted = true;

    if (Error Err = finiteLoop([&]() -> Expected<bool> {
          HasNewInterconnectedCUs = false;

          // Load inter-connected units and find their live roots.
          parallelForEach(CompileUnits, [&](std::unique_ptr<CompileUnit> &CU) {
            if (CU->isInterconnectedCU()) {
              CU->maybeResetToLoadedStage();
              linkSingleCompileUnit(*CU, ArtificialTypeUnit,
                                    CompileUnit::Stage::LiveRootsCollected);
            }
          });

          // Do liveness analysis for inter-connected units.
          parallelForEach(CompileUnits, [&](std::unique_ptr<CompileUnit> &CU) {
            linkSingleCompileUnit(*CU, ArtificialTypeUnit,
                                  CompileUnit::Stage::LivenessAnalysisDone);
          });

          return HasNewInterconnectedCUs.load();
        }))
      return Err;

    // Update dependencies.
    if (Error Err = finiteLoop([&]() -> Expected<bool> {
          HasNewGlobalDependency = false;
          parallelForEach(CompileUnits, [&](std::unique_ptr<CompileUnit> &CU) {
            linkSingleCompileUnit(
                *CU, ArtificialTypeUnit,
                CompileUnit::Stage::UpdateDependenciesCompleteness);
          });
          return HasNewGlobalDependency.load();
        }))
      return Err;
    parallelForEach(CompileUnits, [&](std::unique_ptr<CompileUnit> &CU) {
      if (CU->isInterconnectedCU() &&
          CU->getStage() == CompileUnit::Stage::LivenessAnalysisDone)
        CU->setStage(CompileUnit::Stage::UpdateDependenciesCompleteness);
    });

    // Assign type names.
    parallelForEach(CompileUnits, [&](std::unique_ptr<CompileUnit> &CU) {
      linkSingleCompileUnit(*CU, ArtificialTypeUnit,
                            CompileUnit::Stage::TypeNamesAssigned);
    });

    // Clone inter-connected units.
    parallelForEach(CompileUnits, [&](std::unique_ptr<CompileUnit> &CU) {
      linkSingleCompileUnit(*CU, ArtificialTypeUnit,
                            CompileUnit::Stage::Cloned);
    });

    // Update patches for inter-connected units.
    parallelForEach(CompileUnits, [&](std::unique_ptr<CompileUnit> &CU) {
      linkSingleCompileUnit(*CU, ArtificialTypeUnit,
                            CompileUnit::Stage::PatchesUpdated);
    });

    // Release data.
    parallelForEach(CompileUnits, [&](std::unique_ptr<CompileUnit> &CU) {
      linkSingleCompileUnit(*CU, ArtificialTypeUnit,
                            CompileUnit::Stage::Cleaned);
    });
  }

  // The input is unloaded next, while the units are only placed once all the
  // inputs are linked.
  for (std::unique_ptr<CompileUnit> &CU : CompileUnits)
    if (CU->getStage() != CompileUnit::Stage::Skipped)
      CU->saveRelocations();

  if (GlobalData.getOptions().UpdateIndexTablesOnly) {
    // Emit Invariant sections.

    if (Error Err = emitInvariantSections())
      return Err;
  }

  return Error::success();
}

void DWARFLinkerImpl::LinkContext::linkSingleCompileUnit(
    CompileUnit &CU, TypeUnit *ArtificialTypeUnit,
    enum CompileUnit::Stage DoUntilStage) {
  if (InterCUProcessingStarted != CU.isInterconnectedCU())
    return;

  if (Error Err = finiteLoop([&]() -> Expected<bool> {
        if (CU.getStage() >= DoUntilStage)
          return false;

        switch (CU.getStage()) {
        case CompileUnit::Stage::CreatedNotLoaded: {
          // Load input compilation unit DIEs.
          // Analyze properties of DIEs.
          if (!CU.loadInputDIEs()) {
            // We do not need to do liveness analysis for invalid compilation
            // unit.
            CU.setStage(CompileUnit::Stage::Skipped);
          } else {
            CU.analyzeDWARFStructure();

            // The registerModuleReference() condition effectively skips
            // over fully resolved skeleton units. This second pass of
            // registerModuleReferences doesn't do any new work, but it
            // will collect top-level errors, which are suppressed. Module
            // warnings were already displayed in the first iteration.
            //
            // Runs concurrently, so it must stay a no-op: it may only be
            // entered when the serial pass in addObjectFile() has already
            // populated the module map.
            if (!GlobalData.getOptions().UpdateIndexTablesOnly &&
                registerModuleReference(
                    CU.getOrigUnit().getUnitDIE(), nullptr,
                    [](const DWARFUnit &) {}, 0))
              CU.setStage(CompileUnit::Stage::PatchesUpdated);
            else
              CU.setStage(CompileUnit::Stage::Loaded);
          }
        } break;

        case CompileUnit::Stage::Loaded:
          CU.collectLiveRoots();
          CU.setStage(CompileUnit::Stage::LiveRootsCollected);
          break;

        case CompileUnit::Stage::LiveRootsCollected: {
          // Mark all the DIEs that need to be present in the generated output.
          // If ODR requested, build type names.
          if (!CU.resolveDependenciesAndMarkLiveness(InterCUProcessingStarted,
                                                     HasNewInterconnectedCUs)) {
            assert(HasNewInterconnectedCUs &&
                   "Flag indicating new inter-connections is not set");
            return false;
          }

          CU.setStage(CompileUnit::Stage::LivenessAnalysisDone);
        } break;

        case CompileUnit::Stage::LivenessAnalysisDone: {
          if (InterCUProcessingStarted) {
            if (CU.updateDependenciesCompleteness())
              HasNewGlobalDependency = true;
            return false;
          } else {
            if (Error Err = finiteLoop([&]() -> Expected<bool> {
                  return CU.updateDependenciesCompleteness();
                }))
              return std::move(Err);

            CU.setStage(CompileUnit::Stage::UpdateDependenciesCompleteness);
          }
        } break;

        case CompileUnit::Stage::UpdateDependenciesCompleteness:
#ifndef NDEBUG
          CU.verifyDependencies();
#endif

          if (ArtificialTypeUnit) {
            if (Error Err =
                    CU.assignTypeNames(ArtificialTypeUnit->getTypePool()))
              return std::move(Err);
          }
          CU.setStage(CompileUnit::Stage::TypeNamesAssigned);
          break;

        case CompileUnit::Stage::TypeNamesAssigned:
          // Clone input compile unit.
          if (CU.isClangModule() ||
              GlobalData.getOptions().UpdateIndexTablesOnly ||
              CU.getContainingFile().Addresses->hasValidRelocs()) {
            if (Error Err = CU.cloneAndEmit(GlobalData.getTargetTriple(),
                                            ArtificialTypeUnit))
              return std::move(Err);
          }

          CU.setStage(CompileUnit::Stage::Cloned);
          break;

        case CompileUnit::Stage::Cloned:
          // Update DIEs referencies.
          CU.updateDieRefPatchesWithClonedOffsets();

          // Later than cloning, so that the offsets are final, and no later,
          // because a unit which got this far can no longer be skipped and have
          // its section dropped from the output.
          if (CU.isClangModule())
            CU.noteModuleAnchors();

          CU.setStage(CompileUnit::Stage::PatchesUpdated);
          break;

        case CompileUnit::Stage::PatchesUpdated:
          // Cleanup resources.
          CU.cleanupDataAfterClonning();
          CU.setStage(CompileUnit::Stage::Cleaned);
          break;

        case CompileUnit::Stage::Cleaned:
          assert(false);
          break;

        case CompileUnit::Stage::Skipped:
          // Nothing to do.
          break;
        }

        return true;
      })) {
    CU.error(std::move(Err));
    CU.cleanupDataAfterClonning();
    CU.setStage(CompileUnit::Stage::Skipped);
  }
}

Error DWARFLinkerImpl::LinkContext::emitInvariantSections() {
  if (!GlobalData.getTargetTriple().has_value())
    return Error::success();

  getOrCreateSectionDescriptor(DebugSectionKind::DebugLoc).OS
      << InputDWARFFile.Dwarf->getDWARFObj().getLocSection().Data;
  getOrCreateSectionDescriptor(DebugSectionKind::DebugLocLists).OS
      << InputDWARFFile.Dwarf->getDWARFObj().getLoclistsSection().Data;
  getOrCreateSectionDescriptor(DebugSectionKind::DebugRange).OS
      << InputDWARFFile.Dwarf->getDWARFObj().getRangesSection().Data;
  getOrCreateSectionDescriptor(DebugSectionKind::DebugRngLists).OS
      << InputDWARFFile.Dwarf->getDWARFObj().getRnglistsSection().Data;
  getOrCreateSectionDescriptor(DebugSectionKind::DebugARanges).OS
      << InputDWARFFile.Dwarf->getDWARFObj().getArangesSection();
  getOrCreateSectionDescriptor(DebugSectionKind::DebugFrame).OS
      << InputDWARFFile.Dwarf->getDWARFObj().getFrameSection().Data;
  getOrCreateSectionDescriptor(DebugSectionKind::DebugAddr).OS
      << InputDWARFFile.Dwarf->getDWARFObj().getAddrSection().Data;

  return Error::success();
}

Error DWARFLinkerImpl::LinkContext::scanFrameData() {
  if (GlobalData.getOptions().UpdateIndexTablesOnly)
    return Error::success();
  if (!GlobalData.getTargetTriple().has_value())
    return Error::success();

  if (InputDWARFFile.Dwarf == nullptr)
    return Error::success();
  if (CompileUnits.empty())
    return Error::success();

  const DWARFObject &InputDWARFObj = InputDWARFFile.Dwarf->getDWARFObj();

  StringRef OrigFrameData = InputDWARFObj.getFrameSection().Data;
  if (OrigFrameData.empty())
    return Error::success();

  auto Scan = std::make_unique<FrameScanResult>();
  Scan->FrameData = OrigFrameData;
  Scan->AddressSize = InputDWARFObj.getAddressSize();

  RangesTy AllUnitsRanges;
  for (std::unique_ptr<CompileUnit> &Unit : CompileUnits) {
    for (auto CurRange : Unit->getFunctionRanges())
      AllUnitsRanges.insert(CurRange.Range, CurRange.Value);
  }

  StringRef FrameBytes = Scan->FrameData;
  DataExtractor Data(FrameBytes, InputDWARFObj.isLittleEndian());
  uint64_t InputOffset = 0;
  const unsigned SrcAddrSize = Scan->AddressSize;
  // Width of the CIE_pointer field at the start of every FDE (and of the
  // CIE_id sentinel at the start of every CIE) in DWARF32 .debug_frame.
  constexpr unsigned CIEPointerSize = 4;

  // CIEs defined in this input, keyed by their input offsets.
  DenseMap<uint64_t, StringRef> LocalCIEs;
  DenseSet<uint64_t> AddedCIEs;

  while (Data.isValidOffset(InputOffset)) {
    uint64_t EntryOffset = InputOffset;
    uint32_t InitialLength = Data.getU32(&InputOffset);
    if (InitialLength == 0xFFFFFFFF)
      return createFileError(InputDWARFFile.FileName,
                             createStringError(std::errc::invalid_argument,
                                               "Dwarf64 bits not supported"));

    // Reject lengths that don't fit in the input section. substr() saturates
    // silently, which would otherwise let a malformed length poison the
    // CIE bytes used as the registry key.
    if (InitialLength > FrameBytes.size() - InputOffset)
      return createFileError(
          InputDWARFFile.FileName,
          createStringError(std::errc::invalid_argument,
                            "Truncated .debug_frame entry."));

    uint32_t CIEId = Data.getU32(&InputOffset);
    if (CIEId == 0xFFFFFFFF) {
      // This is a CIE, store it.
      StringRef CIEData = FrameBytes.substr(EntryOffset, InitialLength + 4);
      LocalCIEs[EntryOffset] = CIEData;
      // The -4 is to account for the CIEId we just read.
      InputOffset += InitialLength - 4;
      continue;
    }

    uint64_t Loc = Data.getUnsigned(&InputOffset, SrcAddrSize);

    // Some compilers seem to emit frame info that doesn't start at
    // the function entry point, thus we can't just lookup the address
    // in the debug map. Use the AddressInfo's range map to see if the FDE
    // describes something that we can relocate.
    std::optional<AddressRangeValuePair> Range =
        AllUnitsRanges.getRangeThatContains(Loc);
    if (!Range) {
      // The +4 is to account for the size of the InitialLength field itself.
      InputOffset = EntryOffset + InitialLength + 4;
      continue;
    }

    // This is an FDE, and we have a mapping.
    StringRef CIEData = LocalCIEs.lookup(CIEId);
    if (CIEData.empty())
      return createFileError(
          InputDWARFFile.FileName,
          createStringError(std::errc::invalid_argument,
                            "Inconsistent debug_frame content. Dropping."));

    // Reject FDEs whose length doesn't even cover the CIE_pointer and
    // initial_location fields; otherwise the unsigned subtraction below
    // would wrap and substr() would saturate to a giant garbage blob.
    if (InitialLength < CIEPointerSize + SrcAddrSize)
      return createFileError(InputDWARFFile.FileName,
                             createStringError(std::errc::invalid_argument,
                                               "Truncated .debug_frame FDE."));

    // Promote each CIE on first reference; CIEs no FDE references are
    // dropped from the output.
    if (AddedCIEs.insert(CIEId).second)
      Scan->CIEs.push_back(CIEData);

    unsigned FDERemainingBytes = InitialLength - (CIEPointerSize + SrcAddrSize);
    Scan->FDEs.push_back({CIEData, Loc + Range->Value,
                          FrameBytes.substr(InputOffset, FDERemainingBytes)});
    InputOffset += FDERemainingBytes;
  }

  FrameScan = std::move(Scan);
  return Error::success();
}

void DWARFLinkerImpl::LinkContext::registerCIEs(CIERegistry &CIEs) {
  assert(FrameScan && "registerCIEs called without FrameScan");
  SectionDescriptor &OutSection =
      getOrCreateSectionDescriptor(DebugSectionKind::DebugFrame);

  uint32_t NextLocalOffset = 0;
  for (StringRef CIEBytes : FrameScan->CIEs) {
    auto [It, Inserted] =
        CIEs.try_emplace(CIEBytes, CIELocation{&OutSection, NextLocalOffset});
    if (Inserted) {
      FrameScan->OwnedCIEs.push_back(CIEBytes);
      NextLocalOffset += static_cast<uint32_t>(CIEBytes.size());
    }
  }
}

Error DWARFLinkerImpl::LinkContext::emitDebugFrame(const CIERegistry &CIEs) {
  assert(FrameScan && "emitDebugFrame called without FrameScan");
  SectionDescriptor &OutSection =
      getSectionDescriptor(DebugSectionKind::DebugFrame);

  // Emit owned CIEs at the offsets registerCIEs reserved for them.
  for (StringRef CIEBytes : FrameScan->OwnedCIEs)
    OutSection.OS << CIEBytes;

  const dwarf::FormParams FP = OutSection.getFormParams();
  const unsigned SrcAddrSize = FrameScan->AddressSize;

  for (const FrameScanResult::FDE &FDE : FrameScan->FDEs) {
    auto It = CIEs.find(FDE.CIEBytes);
    assert(It != CIEs.end() && "CIE missing from registry");
    SectionDescriptor *CIEOwnerSection = It->second.OwnerSection;
    const uint32_t CIELocalOffset = It->second.LocalOffset;

    const uint64_t FDEPos = OutSection.OS.tell();
    // Note: this guards against a single context's section exceeding the
    // DWARF32 limit. It does NOT catch the post-glue overflow that would
    // happen if the concatenated .debug_frame across all contexts pushes
    // past 4 GB; that case slips through silently because StartOffset is
    // not yet assigned. A post-glue check would belong in the patch
    // resolver in OutputSections.cpp.
    if (FDEPos > FP.getDwarfMaxOffset())
      return createFileError(
          InputDWARFFile.FileName,
          createStringError(".debug_frame section offset "
                            "0x" +
                            Twine::utohexstr(FDEPos) + " exceeds the " +
                            dwarf::FormatString(FP.Format) + " limit"));

    // CIE_pointer field follows the 4-byte initial_length.
    OutSection.notePatch(DebugOffsetPatch{FDEPos + 4, CIEOwnerSection, true});

    emitFDE(CIELocalOffset, SrcAddrSize, FDE.Address, FDE.Instructions,
            OutSection);
  }

  FrameScan.reset();
  return Error::success();
}

Error DWARFLinkerImpl::LinkContext::unloadInput() {
  // Scan the input's .debug_frame now, while the DWARFContext is still
  // loaded, so the later (post-pool) emission pass can run against the
  // scan result alone.
  Error ScanErr = scanFrameData();
  InputDWARFFile.unload();
  return ScanErr;
}

/// Emit a FDE into the debug_frame section. \p FDEBytes
/// contains the FDE data without the length, CIE offset and address
/// which will be replaced with the parameter values.
void DWARFLinkerImpl::LinkContext::emitFDE(uint32_t CIEOffset,
                                           uint32_t AddrSize, uint64_t Address,
                                           StringRef FDEBytes,
                                           SectionDescriptor &Section) {
  Section.emitIntVal(FDEBytes.size() + 4 + AddrSize, 4);
  Section.emitIntVal(CIEOffset, 4);
  Section.emitIntVal(Address, AddrSize);
  Section.OS.write(FDEBytes.data(), FDEBytes.size());
}

void DWARFLinkerImpl::glueCompileUnitsAndWriteToTheOutput() {
  if (!GlobalData.getTargetTriple().has_value())
    return;
  assert(SectionHandler);

  // Go through all object files, all compile units and assign
  // offsets to them.
  assignOffsets();

  forEachCompileUnit([&](CompileUnit *CU) { CU->reportUnitOffsets(); });

  // Patch size/offsets fields according to the assigned CU offsets.
  patchOffsetsAndSizes();

  // Emit common sections and write debug tables from all object files/compile
  // units into the resulting file.
  emitCommonSectionsAndWriteCompileUnitsToTheOutput();

  if (ArtificialTypeUnit != nullptr)
    ArtificialTypeUnit.reset();

  // Write common debug sections into the resulting file.
  writeCommonSectionsToTheOutput();

  // Cleanup data.
  cleanupDataAfterDWARFOutputIsWritten();

  if (GlobalData.getOptions().Statistics)
    printStatistic();
}

void DWARFLinkerImpl::printStatistic() {

  // For each object file map how many bytes were emitted.
  StringMap<DebugInfoSize> SizeByObject;

  for (const std::unique_ptr<LinkContext> &Context : ObjectContexts) {
    uint64_t AllDebugInfoSectionsSize = 0;

    for (std::unique_ptr<CompileUnit> &CU : Context->CompileUnits)
      if (std::optional<SectionDescriptor *> DebugInfo =
              CU->tryGetSectionDescriptor(DebugSectionKind::DebugInfo))
        AllDebugInfoSectionsSize += (*DebugInfo)->getContents().size();

    auto &Size = SizeByObject[Context->InputDWARFFile.FileName];
    Size.Input = Context->OriginalDebugInfoSize;
    Size.Output = AllDebugInfoSectionsSize;
  }

  // Create a vector sorted in descending order by output size.
  std::vector<std::pair<StringRef, DebugInfoSize>> Sorted;
  for (auto &E : SizeByObject)
    Sorted.emplace_back(E.first(), E.second);
  llvm::sort(Sorted, [](auto &LHS, auto &RHS) {
    return LHS.second.Output > RHS.second.Output;
  });

  auto ComputePercentange = [](int64_t Input, int64_t Output) -> float {
    const float Difference = Output - Input;
    const float Sum = Input + Output;
    if (Sum == 0)
      return 0;
    return (Difference / (Sum / 2));
  };

  int64_t InputTotal = 0;
  int64_t OutputTotal = 0;
  const char *FormatStr = "{0,-45} {1,10}b  {2,10}b {3,8:P}\n";

  // Print header.
  outs() << ".debug_info section size (in bytes)\n";
  outs() << "----------------------------------------------------------------"
            "---------------\n";
  outs() << "Filename                                           Object       "
            "  dSYM   Change\n";
  outs() << "----------------------------------------------------------------"
            "---------------\n";

  // Print body.
  for (auto &E : Sorted) {
    InputTotal += E.second.Input;
    OutputTotal += E.second.Output;
    llvm::outs() << formatv(
        FormatStr, sys::path::filename(E.first).take_back(45), E.second.Input,
        E.second.Output, ComputePercentange(E.second.Input, E.second.Output));
  }
  // Print total and footer.
  outs() << "----------------------------------------------------------------"
            "---------------\n";
  llvm::outs() << formatv(FormatStr, "Total", InputTotal, OutputTotal,
                          ComputePercentange(InputTotal, OutputTotal));
  outs() << "----------------------------------------------------------------"
            "---------------\n\n";
}

void DWARFLinkerImpl::assignOffsets() {
  llvm::parallel::TaskGroup TGroup;
  TGroup.spawn([&]() { assignOffsetsToStrings(); });
  TGroup.spawn([&]() { assignOffsetsToSections(); });
}

void DWARFLinkerImpl::assignOffsetsToStrings() {
  size_t CurDebugStrIndex = 1; // start from 1 to take into account zero entry.
  uint64_t CurDebugStrOffset =
      1; // start from 1 to take into account zero entry.
  size_t CurDebugLineStrIndex = 0;
  uint64_t CurDebugLineStrOffset = 0;

  // Enumerates all strings, add them into the DwarfStringPoolEntry map,
  // assign offset and index to the string if it is not indexed yet.
  forEachOutputString([&](StringDestinationKind Kind,
                          const StringEntry *String) {
    switch (Kind) {
    case StringDestinationKind::DebugStr: {
      DwarfStringPoolEntryWithExtString *Entry = DebugStrStrings.add(String);
      assert(Entry != nullptr);

      if (!Entry->isIndexed()) {
        Entry->Offset = CurDebugStrOffset;
        CurDebugStrOffset += Entry->String.size() + 1;
        Entry->Index = CurDebugStrIndex++;
      }
    } break;
    case StringDestinationKind::DebugLineStr: {
      DwarfStringPoolEntryWithExtString *Entry =
          DebugLineStrStrings.add(String);
      assert(Entry != nullptr);

      if (!Entry->isIndexed()) {
        Entry->Offset = CurDebugLineStrOffset;
        CurDebugLineStrOffset += Entry->String.size() + 1;
        Entry->Index = CurDebugLineStrIndex++;
      }
    } break;
    }
  });
}

void DWARFLinkerImpl::assignOffsetsToSections() {
  std::array<uint64_t, SectionKindsNum> SectionSizesAccumulator = {0};

  forEachObjectSectionsSet([&](OutputSections &UnitSections) {
    UnitSections.assignSectionsOffsetAndAccumulateSize(SectionSizesAccumulator);
  });
}

void DWARFLinkerImpl::forEachOutputString(
    function_ref<void(StringDestinationKind Kind, const StringEntry *String)>
        StringHandler) {
  // To save space we do not create any separate string table.
  // We use already allocated string patches and accelerator entries:
  // enumerate them in natural order and assign offsets.
  // ASSUMPTION: strings should be stored into .debug_str/.debug_line_str
  // sections in the same order as they were assigned offsets.
  forEachCompileUnit([&](CompileUnit *CU) {
    CU->forEach([&](SectionDescriptor &OutSection) {
      OutSection.ListDebugStrPatch.forEach([&](DebugStrPatch &Patch) {
        StringHandler(StringDestinationKind::DebugStr, Patch.String);
      });

      OutSection.ListDebugLineStrPatch.forEach([&](DebugLineStrPatch &Patch) {
        StringHandler(StringDestinationKind::DebugLineStr, Patch.String);
      });
    });

    CU->forEachAcceleratorRecord([&](DwarfUnit::AccelInfo &Info) {
      StringHandler(DebugStr, Info.String);
    });
  });

  if (ArtificialTypeUnit != nullptr) {
    ArtificialTypeUnit->forEach([&](SectionDescriptor &OutSection) {
      OutSection.ListDebugStrPatch.forEach([&](DebugStrPatch &Patch) {
        StringHandler(StringDestinationKind::DebugStr, Patch.String);
      });

      OutSection.ListDebugLineStrPatch.forEach([&](DebugLineStrPatch &Patch) {
        StringHandler(StringDestinationKind::DebugLineStr, Patch.String);
      });

      OutSection.ListDebugTypeStrPatch.forEach([&](DebugTypeStrPatch &Patch) {
        if (Patch.Die == nullptr)
          return;

        TypeEntryBody *TypeEntry = Patch.TypeName->getValue().load();
        if (&TypeEntry->getFinalDie() != Patch.Die)
          return;

        StringHandler(StringDestinationKind::DebugStr, Patch.String);
      });

      OutSection.ListDebugTypeLineStrPatch.forEach(
          [&](DebugTypeLineStrPatch &Patch) {
            if (Patch.Die == nullptr)
              return;

            TypeEntryBody *TypeEntry = Patch.TypeName->getValue().load();
            if (&TypeEntry->getFinalDie() != Patch.Die)
              return;

            StringHandler(StringDestinationKind::DebugStr, Patch.String);
          });
    });
  }
}

void DWARFLinkerImpl::forEachObjectSectionsSet(
    function_ref<void(OutputSections &)> SectionsSetHandler) {
  // Handle artificial type unit first.
  if (ArtificialTypeUnit != nullptr)
    SectionsSetHandler(*ArtificialTypeUnit);

  // Then all modules(before regular compilation units).
  for (const std::unique_ptr<LinkContext> &Context : ObjectContexts)
    for (std::unique_ptr<CompileUnit> &ModuleUnit :
         Context->ModulesCompileUnits)
      if (ModuleUnit->getStage() != CompileUnit::Stage::Skipped)
        SectionsSetHandler(*ModuleUnit);

  // Finally all compilation units.
  for (const std::unique_ptr<LinkContext> &Context : ObjectContexts) {
    // Handle object file common sections.
    SectionsSetHandler(*Context);

    // Handle compilation units.
    for (std::unique_ptr<CompileUnit> &CU : Context->CompileUnits)
      if (CU->getStage() != CompileUnit::Stage::Skipped)
        SectionsSetHandler(*CU);
  }
}

void DWARFLinkerImpl::forEachCompileAndTypeUnit(
    function_ref<void(DwarfUnit *CU)> UnitHandler) {
  if (ArtificialTypeUnit != nullptr)
    UnitHandler(ArtificialTypeUnit.get());

  // Enumerate module units.
  for (const std::unique_ptr<LinkContext> &Context : ObjectContexts)
    for (std::unique_ptr<CompileUnit> &ModuleUnit :
         Context->ModulesCompileUnits)
      if (ModuleUnit->getStage() != CompileUnit::Stage::Skipped)
        UnitHandler(ModuleUnit.get());

  // Enumerate compile units.
  for (const std::unique_ptr<LinkContext> &Context : ObjectContexts)
    for (std::unique_ptr<CompileUnit> &CU : Context->CompileUnits)
      if (CU->getStage() != CompileUnit::Stage::Skipped)
        UnitHandler(CU.get());
}

void DWARFLinkerImpl::forEachCompileUnit(
    function_ref<void(CompileUnit *CU)> UnitHandler) {
  // Enumerate module units.
  for (const std::unique_ptr<LinkContext> &Context : ObjectContexts)
    for (std::unique_ptr<CompileUnit> &ModuleUnit :
         Context->ModulesCompileUnits)
      if (ModuleUnit->getStage() != CompileUnit::Stage::Skipped)
        UnitHandler(ModuleUnit.get());

  // Enumerate compile units.
  for (const std::unique_ptr<LinkContext> &Context : ObjectContexts)
    for (std::unique_ptr<CompileUnit> &CU : Context->CompileUnits)
      if (CU->getStage() != CompileUnit::Stage::Skipped)
        UnitHandler(CU.get());
}

void DWARFLinkerImpl::patchOffsetsAndSizes() {
  forEachObjectSectionsSet([&](OutputSections &SectionsSet) {
    SectionsSet.forEach([&](SectionDescriptor &OutSection) {
      SectionsSet.applyPatches(OutSection, DebugStrStrings, DebugLineStrStrings,
                               ArtificialTypeUnit.get());
    });
  });
}

void DWARFLinkerImpl::emitCommonSectionsAndWriteCompileUnitsToTheOutput() {
  llvm::parallel::TaskGroup TG;

  // Create section descriptors ahead if they are not exist at the moment.
  // SectionDescriptors container is not thread safe. Thus we should be sure
  // that descriptors would not be created in following parallel tasks.

  CommonSections.getOrCreateSectionDescriptor(DebugSectionKind::DebugStr);
  CommonSections.getOrCreateSectionDescriptor(DebugSectionKind::DebugLineStr);

  if (llvm::is_contained(GlobalData.Options.AccelTables,
                         AccelTableKind::Apple)) {
    CommonSections.getOrCreateSectionDescriptor(DebugSectionKind::AppleNames);
    CommonSections.getOrCreateSectionDescriptor(
        DebugSectionKind::AppleNamespaces);
    CommonSections.getOrCreateSectionDescriptor(DebugSectionKind::AppleObjC);
    CommonSections.getOrCreateSectionDescriptor(DebugSectionKind::AppleTypes);
  }

  if (llvm::is_contained(GlobalData.Options.AccelTables,
                         AccelTableKind::DebugNames))
    CommonSections.getOrCreateSectionDescriptor(DebugSectionKind::DebugNames);

  // Emit .debug_str and .debug_line_str sections.
  TG.spawn([&]() { emitStringSections(); });

  // Spawn writing the compile units to the output file early, so it overlaps
  // with building the Apple tables (collected on this thread).
  TG.spawn([&]() { writeCompileUnitsToTheOutput(); });

  // Emit apple accelerator sections. This collects the records in this
  // thread, before the .debug_names task below may enumerate them too.
  if (llvm::is_contained(GlobalData.Options.AccelTables, AccelTableKind::Apple))
    emitAppleAcceleratorSections(TG);

  if (llvm::is_contained(GlobalData.Options.AccelTables,
                         AccelTableKind::DebugNames)) {
    // Emit .debug_names section.
    TG.spawn([&]() {
      emitDWARFv5DebugNamesSection((*GlobalData.getTargetTriple()).get());
    });
  }
}

void DWARFLinkerImpl::emitStringSections() {
  uint64_t DebugStrNextOffset = 0;
  uint64_t DebugLineStrNextOffset = 0;

  // Emit zero length string. Accelerator tables does not work correctly
  // if the first string is not zero length string.
  CommonSections.getSectionDescriptor(DebugSectionKind::DebugStr)
      .emitInplaceString("");
  DebugStrNextOffset++;

  forEachOutputString(
      [&](StringDestinationKind Kind, const StringEntry *String) {
        switch (Kind) {
        case StringDestinationKind::DebugStr: {
          DwarfStringPoolEntryWithExtString *StringToEmit =
              DebugStrStrings.getExistingEntry(String);
          assert(StringToEmit->isIndexed());

          // Strings may be repeated. Use accumulated DebugStrNextOffset
          // to understand whether corresponding string is already emitted.
          // Skip string if its offset less than accumulated offset.
          if (StringToEmit->Offset >= DebugStrNextOffset) {
            DebugStrNextOffset =
                StringToEmit->Offset + StringToEmit->String.size() + 1;
            // Emit the string itself.
            CommonSections.getSectionDescriptor(DebugSectionKind::DebugStr)
                .emitInplaceString(StringToEmit->String);
          }
        } break;
        case StringDestinationKind::DebugLineStr: {
          DwarfStringPoolEntryWithExtString *StringToEmit =
              DebugLineStrStrings.getExistingEntry(String);
          assert(StringToEmit->isIndexed());

          // Strings may be repeated. Use accumulated DebugLineStrStrings
          // to understand whether corresponding string is already emitted.
          // Skip string if its offset less than accumulated offset.
          if (StringToEmit->Offset >= DebugLineStrNextOffset) {
            DebugLineStrNextOffset =
                StringToEmit->Offset + StringToEmit->String.size() + 1;
            // Emit the string itself.
            CommonSections.getSectionDescriptor(DebugSectionKind::DebugLineStr)
                .emitInplaceString(StringToEmit->String);
          }
        } break;
        }
      });
}

namespace {
/// Appends values in the section endianness to the contents of a section.
/// The section stream is unbuffered, so its buffer is always up to date and
/// values are appended to it directly.
class SectionWriter {
public:
  SectionWriter(SectionDescriptor &Section)
      : Buffer(Section.OS.buffer()), Start(Buffer.size()),
        Endianness(Section.getEndianess()) {}

  template <typename T> void write(T Value) {
    support::endian::write(Buffer, Value, Endianness);
  }
  /// \returns the number of bytes written so far.
  uint64_t size() const { return Buffer.size() - Start; }

private:
  SmallVectorImpl<char> &Buffer;
  uint64_t Start;
  llvm::endianness Endianness;
};

/// \returns the number of distinct hash values of \p Entries.
template <typename EntryT>
uint32_t countUniqueHashes(const std::vector<EntryT> &Entries) {
  std::vector<uint32_t> Hashes;
  Hashes.reserve(Entries.size());
  for (const EntryT &E : Entries)
    Hashes.push_back(E.HashValue);
  std::sort(Hashes.begin(), Hashes.end());
  return std::unique(Hashes.begin(), Hashes.end()) - Hashes.begin();
}

/// \returns the indexes of \p Entries ordered by bucket, then by hash value,
/// then by name: the order of the sorted buckets of
/// AccelTableBase::finalize().
template <typename EntryT>
std::vector<uint32_t> orderByBucket(const std::vector<EntryT> &Entries,
                                    uint32_t BucketCount) {
  std::vector<uint32_t> BucketStart(BucketCount + 1, 0);
  for (const EntryT &E : Entries)
    ++BucketStart[E.HashValue % BucketCount + 1];
  for (uint32_t Bucket = 0; Bucket < BucketCount; ++Bucket)
    BucketStart[Bucket + 1] += BucketStart[Bucket];

  std::vector<uint32_t> Order(Entries.size());
  {
    std::vector<uint32_t> Next(BucketStart.begin(), BucketStart.end() - 1);
    for (uint32_t Idx = 0, End = Entries.size(); Idx != End; ++Idx)
      Order[Next[Entries[Idx].HashValue % BucketCount]++] = Idx;
  }

  // Buckets are small: insertion sort by hash value, then by name.
  auto Less = [&](uint32_t LHS, uint32_t RHS) {
    const EntryT &L = Entries[LHS];
    const EntryT &R = Entries[RHS];
    if (L.HashValue != R.HashValue)
      return L.HashValue < R.HashValue;
    return L.Name->String < R.Name->String;
  };
  for (uint32_t Bucket = 0; Bucket < BucketCount; ++Bucket) {
    for (uint32_t I = BucketStart[Bucket] + 1; I < BucketStart[Bucket + 1];
         ++I) {
      uint32_t Idx = Order[I];
      uint32_t J = I;
      for (; J > BucketStart[Bucket] && Less(Idx, Order[J - 1]); --J)
        Order[J] = Order[J - 1];
      Order[J] = Idx;
    }
  }
  return Order;
}
} // anonymous namespace

namespace {
/// Builds one Apple accelerator table and writes its section contents
/// directly. The output is byte-identical to filling an AccelTable and
/// emitting it through AppleAccelTableWriter, without the MC layer.
class AppleAccelTableBuilder {
public:
  struct Value {
    uint32_t Offset;
    uint32_t QualifiedNameHash;
    uint16_t Tag;
    bool ObjCClassIsImplementation;
  };

  AppleAccelTableBuilder(bool IsTypeTable) : IsTypeTable(IsTypeTable) {}

  void reserve(size_t NumRecords) { Records.reserve(NumRecords); }

  void addName(const StringEntry *Key,
               const DwarfStringPoolEntryWithExtString &Name, Value V) {
    auto [It, Inserted] = Index.try_emplace(Key, Entries.size());
    if (Inserted)
      Entries.push_back({&Name, djbHash(Name.String), 0, 0});
    ++Entries[It->second].NumValues;
    Records.push_back({It->second, V});
  }

  void emit(SectionDescriptor &OutSection);

private:
  struct Entry {
    const DwarfStringPoolEntryWithExtString *Name;
    uint32_t HashValue;
    /// The values of the entry are Values[FirstValue, FirstValue+NumValues).
    uint32_t FirstValue;
    uint32_t NumValues;
  };

  struct Record {
    uint32_t EntryIdx;
    Value V;
  };

  ArrayRef<Value> getValues(const Entry &E) const {
    return ArrayRef(Values).slice(E.FirstValue, E.NumValues);
  }

  bool IsTypeTable;
  DenseMap<const StringEntry *, uint32_t> Index;
  std::vector<Entry> Entries;
  std::vector<Record> Records;
  std::vector<Value> Values;
};

void AppleAccelTableBuilder::emit(SectionDescriptor &OutSection) {
  // Group the values by entry, in insertion order, then order them as in
  // AccelTableBase::finalize().
  uint32_t FirstValue = 0;
  for (Entry &E : Entries) {
    E.FirstValue = FirstValue;
    FirstValue += E.NumValues;
  }
  Values.resize(Records.size());
  {
    std::vector<uint32_t> NextValue(Entries.size());
    for (size_t I = 0; I < Entries.size(); ++I)
      NextValue[I] = Entries[I].FirstValue;
    for (const Record &R : Records)
      Values[NextValue[R.EntryIdx]++] = R.V;
    std::vector<Record>().swap(Records);
  }
  for (Entry &E : Entries)
    std::stable_sort(Values.begin() + E.FirstValue,
                     Values.begin() + E.FirstValue + E.NumValues,
                     [](const Value &LHS, const Value &RHS) {
                       return LHS.Offset < RHS.Offset;
                     });

  uint32_t UniqueHashCount = countUniqueHashes(Entries);
  uint32_t BucketCount = dwarf::getDebugNamesBucketCount(UniqueHashCount);

  // Entries ordered by bucket, then by hash value, then by name: the order of
  // the sorted buckets in AccelTableBase::finalize().
  std::vector<uint32_t> Order = orderByBucket(Entries, BucketCount);

  ArrayRef<AppleAccelTableData::Atom> Atoms =
      IsTypeTable ? ArrayRef<AppleAccelTableData::Atom>(
                        AppleAccelTableStaticTypeData::Atoms)
                  : ArrayRef<AppleAccelTableData::Atom>(
                        AppleAccelTableStaticOffsetData::Atoms);
  const uint32_t ValueSize = IsTypeTable ? 4 + 2 + 1 + 4 : 4;

  SectionWriter W(OutSection);

  // Header.
  W.write<uint32_t>(0x48415348); // 'HASH'
  W.write<uint16_t>(1);
  W.write<uint16_t>(dwarf::DW_hash_function_djb);
  W.write<uint32_t>(BucketCount);
  W.write<uint32_t>(UniqueHashCount);
  W.write<uint32_t>(8 + Atoms.size() * 4);

  // Header data.
  W.write<uint32_t>(0);
  W.write<uint32_t>(Atoms.size());
  for (const AppleAccelTableData::Atom &A : Atoms) {
    W.write<uint16_t>(A.Type);
    W.write<uint16_t>(A.Form);
  }

  // Buckets: index of the first hash of each bucket.
  constexpr uint64_t NoHash = std::numeric_limits<uint64_t>::max();
  {
    size_t Pos = 0;
    uint32_t HashIndex = 0;
    for (uint32_t Bucket = 0; Bucket < BucketCount; ++Bucket) {
      if (Pos < Order.size() &&
          Entries[Order[Pos]].HashValue % BucketCount == Bucket)
        W.write<uint32_t>(HashIndex);
      else
        W.write<uint32_t>(std::numeric_limits<uint32_t>::max());
      uint64_t PrevHash = NoHash;
      for (; Pos < Order.size() &&
             Entries[Order[Pos]].HashValue % BucketCount == Bucket;
           ++Pos) {
        uint32_t HashValue = Entries[Order[Pos]].HashValue;
        if (PrevHash != HashValue)
          ++HashIndex;
        PrevHash = HashValue;
      }
    }
  }

  // Hashes, skipping identical consecutive values.
  {
    uint64_t PrevHash = NoHash;
    for (uint32_t Idx : Order) {
      uint32_t HashValue = Entries[Idx].HashValue;
      if (PrevHash == HashValue)
        continue;
      W.write<uint32_t>(HashValue);
      PrevHash = HashValue;
    }
  }

  // Offsets of the data of each unique hash, relative to the section start.
  uint64_t DataStart = W.size() + UniqueHashCount * 4;
  {
    uint64_t PrevHash = NoHash;
    uint64_t DataOffset = DataStart;
    uint32_t PrevBucket = 0;
    bool First = true;
    for (uint32_t Idx : Order) {
      const Entry &E = Entries[Idx];
      uint32_t Bucket = E.HashValue % BucketCount;
      if (!First && Bucket != PrevBucket) {
        // End marker of the previous bucket.
        DataOffset += 4;
        PrevHash = NoHash;
      } else if (PrevHash != NoHash && PrevHash != E.HashValue) {
        // Terminator of the previous hash.
        DataOffset += 4;
      }
      if (First || PrevHash != E.HashValue)
        W.write<uint32_t>(DataOffset);
      DataOffset += 4 + 4 + E.NumValues * ValueSize;
      PrevHash = E.HashValue;
      PrevBucket = Bucket;
      First = false;
    }
  }

  // Data.
  assert(W.size() == DataStart);
  {
    uint64_t PrevHash = NoHash;
    uint32_t PrevBucket = 0;
    bool First = true;
    for (uint32_t Idx : Order) {
      const Entry &E = Entries[Idx];
      uint32_t Bucket = E.HashValue % BucketCount;
      if (!First && Bucket != PrevBucket) {
        W.write<uint32_t>(0);
        PrevHash = NoHash;
      } else if (PrevHash != NoHash && PrevHash != E.HashValue) {
        W.write<uint32_t>(0);
      }
      W.write<uint32_t>(E.Name->Offset);
      W.write<uint32_t>(E.NumValues);
      for (const Value &V : getValues(E)) {
        W.write<uint32_t>(V.Offset);
        if (IsTypeTable) {
          W.write<uint16_t>(V.Tag);
          W.write<uint8_t>(V.ObjCClassIsImplementation
                               ? dwarf::DW_FLAG_type_implementation
                               : 0);
          W.write<uint32_t>(V.QualifiedNameHash);
        }
      }
      PrevHash = E.HashValue;
      PrevBucket = Bucket;
      First = false;
    }
    if (!First)
      W.write<uint32_t>(0);
  }
}
} // anonymous namespace

void DWARFLinkerImpl::emitAppleAcceleratorSections(
    llvm::parallel::TaskGroup &TG) {

  // Collect the records first: forEachAcceleratorRecord() of the type unit
  // updates the records, so it must not run concurrently.
  struct Record {
    const StringEntry *String;
    AppleAccelTableBuilder::Value V;
  };
  enum AppleTableIndex : unsigned { Namespaces, Names, ObjC, Types };
  std::array<std::vector<Record>, Types + 1> Records;
  auto TableIndexOf = [](DwarfUnit::AccelType Type) {
    switch (Type) {
    case DwarfUnit::AccelType::Namespace:
      return Namespaces;
    case DwarfUnit::AccelType::Name:
      return Names;
    case DwarfUnit::AccelType::ObjC:
      return ObjC;
    case DwarfUnit::AccelType::Type:
      return Types;
    case DwarfUnit::AccelType::None:
      break;
    }
    llvm_unreachable("Unknown accelerator record");
  };
  auto SectionKindOf = [](AppleTableIndex Table) {
    switch (Table) {
    case Namespaces:
      return DebugSectionKind::AppleNamespaces;
    case Names:
      return DebugSectionKind::AppleNames;
    case ObjC:
      return DebugSectionKind::AppleObjC;
    case Types:
      return DebugSectionKind::AppleTypes;
    }
    llvm_unreachable("Unknown Apple accelerator table");
  };

  forEachCompileAndTypeUnit([&](DwarfUnit *CU) {
    uint64_t StartOffset =
        CU->getSectionDescriptor(DebugSectionKind::DebugInfo).StartOffset;
    CU->forEachAcceleratorRecord([&](const DwarfUnit::AccelInfo &Info) {
      AppleTableIndex Table = TableIndexOf(Info.Type);
      AppleAccelTableBuilder::Value V{
          static_cast<uint32_t>(StartOffset + Info.OutOffset), 0, 0, false};
      if (Table == Types) {
        V.QualifiedNameHash = Info.QualifiedNameHash;
        V.Tag = Info.Tag;
        V.ObjCClassIsImplementation = Info.ObjcClassImplementation;
      }
      Records[Table].push_back({Info.String, V});
    });
  });

  for (AppleTableIndex Table : {Namespaces, Names, ObjC, Types})
    TG.spawn([this, Table, SectionKind = SectionKindOf(Table),
              TableRecords = std::move(Records[Table])]() {
      AppleAccelTableBuilder Builder(/*IsTypeTable=*/Table == Types);
      Builder.reserve(TableRecords.size());
      for (const Record &R : TableRecords)
        Builder.addName(R.String, *DebugStrStrings.getExistingEntry(R.String),
                        R.V);
      Builder.emit(CommonSections.getSectionDescriptor(SectionKind));
    });
}

void DWARFLinkerImpl::emitDWARFv5DebugNamesSection(const Triple &TargetTriple) {
  std::unique_ptr<DWARF5AccelTable> DebugNames;

  DebugNamesUnitsOffsets CompUnits;
  CompUnitIDToIdx CUidToIdx;

  unsigned Id = 0;

  forEachCompileAndTypeUnit([&](DwarfUnit *CU) {
    bool HasRecords = false;
    CU->forEachAcceleratorRecord([&](const DwarfUnit::AccelInfo &Info) {
      if (DebugNames == nullptr)
        DebugNames = std::make_unique<DWARF5AccelTable>();

      HasRecords = true;
      switch (Info.Type) {
      case DwarfUnit::AccelType::Name:
      case DwarfUnit::AccelType::Namespace:
      case DwarfUnit::AccelType::Type: {
        DebugNames->addName(*DebugStrStrings.getExistingEntry(Info.String),
                            Info.OutOffset, Info.ParentOffset, Info.Tag,
                            CU->getUniqueID(),
                            CU->getTag() == dwarf::DW_TAG_type_unit);
      } break;

      default:
        break; // Nothing to do.
      };
    });

    if (HasRecords) {
      CompUnits.push_back(
          CU->getOrCreateSectionDescriptor(DebugSectionKind::DebugInfo)
              .StartOffset);
      CUidToIdx[CU->getUniqueID()] = Id++;
    }
  });

  if (DebugNames != nullptr) {
    // FIXME: we use AsmPrinter to emit accelerator sections.
    // It might be beneficial to directly emit accelerator data
    // to the raw_svector_ostream.
    SectionDescriptor &OutSection =
        CommonSections.getSectionDescriptor(DebugSectionKind::DebugNames);
    DwarfEmitterImpl Emitter(DWARFLinker::OutputFileType::Object,
                             OutSection.OS);
    if (Error Err = Emitter.init(TargetTriple, "__DWARF")) {
      consumeError(std::move(Err));
      return;
    }

    // Emit table.
    Emitter.emitDebugNames(*DebugNames, CompUnits, CUidToIdx);
    Emitter.finish();

    // Set start offset ans size for output section.
    OutSection.setSizesForSectionCreatedByAsmPrinter();
  }
}

void DWARFLinkerImpl::cleanupDataAfterDWARFOutputIsWritten() {
  GlobalData.getStringPool().clear();
  DebugStrStrings.clear();
  DebugLineStrStrings.clear();
}

void DWARFLinkerImpl::writeCompileUnitsToTheOutput() {
  // Enumerate all sections and store them into the final emitter.
  forEachObjectSectionsSet([&](OutputSections &Sections) {
    Sections.forEach([&](std::shared_ptr<SectionDescriptor> OutSection) {
      // Emit section content.
      SectionHandler(OutSection);
    });
  });
}

void DWARFLinkerImpl::writeCommonSectionsToTheOutput() {
  CommonSections.forEach([&](std::shared_ptr<SectionDescriptor> OutSection) {
    SectionHandler(OutSection);
  });
}
