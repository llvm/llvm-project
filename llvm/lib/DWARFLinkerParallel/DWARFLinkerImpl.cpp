//=== DWARFLinkerImpl.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DWARFLinkerImpl.h"
#include "DIEGenerator.h"
#include "DependencyTracker.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/ThreadPool.h"

namespace llvm {
namespace dwarflinker_parallel {

Error DWARFLinkerImpl::createEmitter(const Triple &TheTriple,
                                     OutputFileType FileType,
                                     raw_pwrite_stream &OutFile) {

  TheDwarfEmitter = std::make_unique<DwarfEmitterImpl>(FileType, OutFile);

  return TheDwarfEmitter->init(TheTriple, "__DWARF");
}

ExtraDwarfEmitter *DWARFLinkerImpl::getEmitter() {
  return TheDwarfEmitter.get();
}

void DWARFLinkerImpl::addObjectFile(DWARFFile &File, ObjFileLoaderTy Loader,
                                    CompileUnitHandlerTy OnCUDieLoaded) {
  ObjectContexts.emplace_back(std::make_unique<LinkContext>(
      GlobalData, File, ClangModules, UniqueUnitID,
      (TheDwarfEmitter.get() == nullptr ? std::optional<Triple>(std::nullopt)
                                        : TheDwarfEmitter->getTargetTriple())));

  if (ObjectContexts.back()->InputDWARFFile.Dwarf) {
    for (const std::unique_ptr<DWARFUnit> &CU :
         ObjectContexts.back()->InputDWARFFile.Dwarf->compile_units()) {
      DWARFDie CUDie = CU->getUnitDIE();
      OverallNumberOfCU++;

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

Error DWARFLinkerImpl::link() {
  // reset compile unit unique ID counter.
  UniqueUnitID = 0;

  if (Error Err = validateAndUpdateOptions())
    return Err;

  dwarf::FormParams GlobalFormat = {GlobalData.getOptions().TargetDWARFVersion,
                                    0, dwarf::DwarfFormat::DWARF32};
  support::endianness GlobalEndianness = support::endian::system_endianness();

  if (TheDwarfEmitter) {
    GlobalEndianness = TheDwarfEmitter->getTargetTriple().isLittleEndian()
                           ? support::endianness::little
                           : support::endianness::big;
  }

  for (std::unique_ptr<LinkContext> &Context : ObjectContexts) {
    if (Context->InputDWARFFile.Dwarf.get() == nullptr) {
      Context->setOutputFormat(Context->getFormParams(), GlobalEndianness);
      continue;
    }

    if (GlobalData.getOptions().Verbose) {
      outs() << "OBJECT: " << Context->InputDWARFFile.FileName << "\n";

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

    if (!TheDwarfEmitter)
      GlobalEndianness = Context->getEndianness();
    GlobalFormat.AddrSize =
        std::max(GlobalFormat.AddrSize, Context->getFormParams().AddrSize);

    Context->setOutputFormat(Context->getFormParams(), GlobalEndianness);
  }

  if (GlobalFormat.AddrSize == 0) {
    if (TheDwarfEmitter)
      GlobalFormat.AddrSize =
          TheDwarfEmitter->getTargetTriple().isArch32Bit() ? 4 : 8;
    else
      GlobalFormat.AddrSize = 8;
  }

  CommonSections.setOutputFormat(GlobalFormat, GlobalEndianness);

  // Set parallel options.
  if (GlobalData.getOptions().Threads == 0)
    parallel::strategy = optimal_concurrency(OverallNumberOfCU);
  else
    parallel::strategy = hardware_concurrency(GlobalData.getOptions().Threads);

  // Link object files.
  if (GlobalData.getOptions().Threads == 1) {
    for (std::unique_ptr<LinkContext> &Context : ObjectContexts) {
      // Link object file.
      if (Error Err = Context->link())
        GlobalData.error(std::move(Err), Context->InputDWARFFile.FileName);

      Context->InputDWARFFile.unload();
    }
  } else {
    ThreadPool Pool(parallel::strategy);
    for (std::unique_ptr<LinkContext> &Context : ObjectContexts)
      Pool.async([&]() {
        // Link object file.
        if (Error Err = Context->link())
          GlobalData.error(std::move(Err), Context->InputDWARFFile.FileName);

        Context->InputDWARFFile.unload();
      });

    Pool.wait();
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

  GlobalData.Options.NoOutput = TheDwarfEmitter.get() == nullptr;

  if (GlobalData.getOptions().Verbose && GlobalData.getOptions().Threads != 1) {
    GlobalData.Options.Threads = 1;
    GlobalData.warn(
        "set number of threads to 1 to make --verbose to work properly.", "");
  }

  return Error::success();
}

/// Resolve the relative path to a build artifact referenced by DWARF by
/// applying DW_AT_comp_dir.
static void resolveRelativeObjectPath(SmallVectorImpl<char> &Buf, DWARFDie CU) {
  sys::path::append(Buf, dwarf::toString(CU.find(dwarf::DW_AT_comp_dir), ""));
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
                              DWARFLinker::ObjectPrefixMapTy *ObjectPrefixMap) {
  std::string PCMFile = dwarf::toString(
      CUDie.find({dwarf::DW_AT_dwo_name, dwarf::DW_AT_GNU_dwo_name}), "");

  if (PCMFile.empty())
    return PCMFile;

  if (ObjectPrefixMap)
    PCMFile = remapPath(PCMFile, *ObjectPrefixMap);

  return PCMFile;
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
  std::string PCMFile =
      getPCMFile(CUDie, GlobalData.getOptions().ObjectPrefixMap);
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
    resolveRelativeObjectPath(Path, CUDie);
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
    ModulesCompileUnits.emplace_back(RefModuleUnit{*ErrOrObj, std::move(Unit)});
    // Preload line table, as it can't be loaded asynchronously.
    ModulesCompileUnits.back().Unit->loadLineTable();
  }

  return Error::success();
}

Error DWARFLinkerImpl::LinkContext::link() {
  InterCUProcessingStarted = false;
  if (InputDWARFFile.Warnings.empty()) {
    if (!InputDWARFFile.Dwarf)
      return Error::success();

    // Preload macro tables, as they can't be loaded asynchronously.
    InputDWARFFile.Dwarf->getDebugMacinfo();
    InputDWARFFile.Dwarf->getDebugMacro();

    // Link modules compile units first.
    parallelForEach(ModulesCompileUnits, [&](RefModuleUnit &RefModule) {
      linkSingleCompileUnit(*RefModule.Unit);
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
    for (const auto &OrigCU : InputDWARFFile.Dwarf->compile_units()) {
      // Load only unit DIE at this stage.
      auto CUDie = OrigCU->getUnitDIE();
      std::string PCMFile =
          getPCMFile(CUDie, GlobalData.getOptions().ObjectPrefixMap);

      // The !isClangModuleRef condition effectively skips over fully resolved
      // skeleton units.
      if (!CUDie || GlobalData.getOptions().UpdateIndexTablesOnly ||
          !isClangModuleRef(CUDie, PCMFile, 0, true).first) {
        CompileUnits.emplace_back(std::make_unique<CompileUnit>(
            GlobalData, *OrigCU, UniqueUnitID.fetch_add(1), "", InputDWARFFile,
            getUnitForOffset, OrigCU->getFormParams(), getEndianness()));

        // Preload line table, as it can't be loaded asynchronously.
        CompileUnits.back()->loadLineTable();
      }
    };

    HasNewInterconnectedCUs = false;

    // Link self-sufficient compile units and discover inter-connected compile
    // units.
    parallelForEach(CompileUnits, [&](std::unique_ptr<CompileUnit> &CU) {
      linkSingleCompileUnit(*CU);
    });

    // Link all inter-connected units.
    if (HasNewInterconnectedCUs) {
      InterCUProcessingStarted = true;

      do {
        HasNewInterconnectedCUs = false;

        // Load inter-connected units.
        parallelForEach(CompileUnits, [&](std::unique_ptr<CompileUnit> &CU) {
          if (CU->isInterconnectedCU()) {
            CU->maybeResetToLoadedStage();
            linkSingleCompileUnit(*CU, CompileUnit::Stage::Loaded);
          }
        });

        // Do liveness analysis for inter-connected units.
        parallelForEach(CompileUnits, [&](std::unique_ptr<CompileUnit> &CU) {
          linkSingleCompileUnit(*CU, CompileUnit::Stage::LivenessAnalysisDone);
        });
      } while (HasNewInterconnectedCUs);

      // Clone inter-connected units.
      parallelForEach(CompileUnits, [&](std::unique_ptr<CompileUnit> &CU) {
        linkSingleCompileUnit(*CU, CompileUnit::Stage::Cloned);
      });

      // Update patches for inter-connected units.
      parallelForEach(CompileUnits, [&](std::unique_ptr<CompileUnit> &CU) {
        linkSingleCompileUnit(*CU, CompileUnit::Stage::PatchesUpdated);
      });

      // Release data.
      parallelForEach(CompileUnits, [&](std::unique_ptr<CompileUnit> &CU) {
        linkSingleCompileUnit(*CU, CompileUnit::Stage::Cleaned);
      });
    }
  }

  if (!InputDWARFFile.Warnings.empty()) {
    // Create compile unit with paper trail warnings.

    Error ResultErr = Error::success();
    // We use task group here as PerThreadBumpPtrAllocator should be called from
    // the threads created by ThreadPoolExecutor.
    parallel::TaskGroup TGroup;
    TGroup.spawn([&]() {
      if (Error Err = cloneAndEmitPaperTrails())
        ResultErr = std::move(Err);
    });
    return ResultErr;
  } else if (GlobalData.getOptions().UpdateIndexTablesOnly) {
    // Emit Invariant sections.

    if (Error Err = emitInvariantSections())
      return Err;
  } else if (!CompileUnits.empty()) {
    // Emit .debug_frame section.

    Error ResultErr = Error::success();
    parallel::TaskGroup TGroup;
    // We use task group here as PerThreadBumpPtrAllocator should be called from
    // the threads created by ThreadPoolExecutor.
    TGroup.spawn([&]() {
      if (Error Err = cloneAndEmitDebugFrame())
        ResultErr = std::move(Err);
    });
    return ResultErr;
  }

  return Error::success();
}

void DWARFLinkerImpl::LinkContext::linkSingleCompileUnit(
    CompileUnit &CU, enum CompileUnit::Stage DoUntilStage) {
  while (CU.getStage() < DoUntilStage) {
    if (InterCUProcessingStarted != CU.isInterconnectedCU())
      return;

    switch (CU.getStage()) {
    case CompileUnit::Stage::CreatedNotLoaded: {
      // Load input compilation unit DIEs.
      // Analyze properties of DIEs.
      if (!CU.loadInputDIEs()) {
        // We do not need to do liveness analysis for invalud compilation unit.
        CU.setStage(CompileUnit::Stage::LivenessAnalysisDone);
      } else {
        CU.analyzeDWARFStructure();

        // The registerModuleReference() condition effectively skips
        // over fully resolved skeleton units. This second pass of
        // registerModuleReferences doesn't do any new work, but it
        // will collect top-level errors, which are suppressed. Module
        // warnings were already displayed in the first iteration.
        if (registerModuleReference(
                CU.getOrigUnit().getUnitDIE(), nullptr,
                [](const DWARFUnit &) {}, 0))
          CU.setStage(CompileUnit::Stage::PatchesUpdated);
        else
          CU.setStage(CompileUnit::Stage::Loaded);
      }
    } break;

    case CompileUnit::Stage::Loaded: {
      // Mark all the DIEs that need to be present in the generated output.
      // If ODR requested, build type names.
      if (!DependencyTracker(*this).resolveDependenciesAndMarkLiveness(CU)) {
        assert(HasNewInterconnectedCUs);
        return;
      }

      CU.setStage(CompileUnit::Stage::LivenessAnalysisDone);
    } break;

    case CompileUnit::Stage::LivenessAnalysisDone:

#ifndef NDEBUG
      DependencyTracker::verifyKeepChain(CU);
#endif

      // Clone input compile unit.
      if (CU.isClangModule() || GlobalData.getOptions().UpdateIndexTablesOnly ||
          CU.getContaingFile().Addresses->hasValidRelocs()) {
        if (Error Err = CU.cloneAndEmit(TargetTriple))
          CU.error(std::move(Err));
      }

      CU.setStage(CompileUnit::Stage::Cloned);
      break;

    case CompileUnit::Stage::Cloned:
      // Update DIEs referencies.
      CU.updateDieRefPatchesWithClonedOffsets();
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
    }
  }
}

Error DWARFLinkerImpl::LinkContext::emitInvariantSections() {
  if (GlobalData.getOptions().NoOutput)
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

Error DWARFLinkerImpl::LinkContext::cloneAndEmitDebugFrame() {
  if (GlobalData.getOptions().NoOutput)
    return Error::success();

  if (InputDWARFFile.Dwarf.get() == nullptr)
    return Error::success();

  const DWARFObject &InputDWARFObj = InputDWARFFile.Dwarf->getDWARFObj();

  StringRef OrigFrameData = InputDWARFObj.getFrameSection().Data;
  if (OrigFrameData.empty())
    return Error::success();

  RangesTy AllUnitsRanges;
  for (std::unique_ptr<CompileUnit> &Unit : CompileUnits) {
    for (auto CurRange : Unit->getFunctionRanges())
      AllUnitsRanges.insert(CurRange.Range, CurRange.Value);
  }

  unsigned SrcAddrSize = InputDWARFObj.getAddressSize();

  SectionDescriptor &OutSection =
      getOrCreateSectionDescriptor(DebugSectionKind::DebugFrame);

  DataExtractor Data(OrigFrameData, InputDWARFObj.isLittleEndian(), 0);
  uint64_t InputOffset = 0;

  // Store the data of the CIEs defined in this object, keyed by their
  // offsets.
  DenseMap<uint64_t, StringRef> LocalCIES;

  /// The CIEs that have been emitted in the output section. The actual CIE
  /// data serves a the key to this StringMap.
  StringMap<uint32_t> EmittedCIEs;

  while (Data.isValidOffset(InputOffset)) {
    uint64_t EntryOffset = InputOffset;
    uint32_t InitialLength = Data.getU32(&InputOffset);
    if (InitialLength == 0xFFFFFFFF)
      return createFileError(InputDWARFObj.getFileName(),
                             createStringError(std::errc::invalid_argument,
                                               "Dwarf64 bits no supported"));

    uint32_t CIEId = Data.getU32(&InputOffset);
    if (CIEId == 0xFFFFFFFF) {
      // This is a CIE, store it.
      StringRef CIEData = OrigFrameData.substr(EntryOffset, InitialLength + 4);
      LocalCIES[EntryOffset] = CIEData;
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
    // Have we already emitted a corresponding CIE?
    StringRef CIEData = LocalCIES[CIEId];
    if (CIEData.empty())
      return createFileError(
          InputDWARFObj.getFileName(),
          createStringError(std::errc::invalid_argument,
                            "Inconsistent debug_frame content. Dropping."));

    uint64_t OffsetToCIERecord = OutSection.OS.tell();

    // Look if we already emitted a CIE that corresponds to the
    // referenced one (the CIE data is the key of that lookup).
    auto IteratorInserted =
        EmittedCIEs.insert(std::make_pair(CIEData, OffsetToCIERecord));
    OffsetToCIERecord = IteratorInserted.first->getValue();

    // Emit CIE for this ID if it is not emitted yet.
    if (IteratorInserted.second)
      OutSection.OS << CIEData;

    // Remember offset to the FDE record, so that we might update
    // field referencing CIE record(containing OffsetToCIERecord),
    // when final offsets are known. OffsetToCIERecord(which is written later)
    // is local to the current .debug_frame section, it should be updated
    // with final offset of the .debug_frame section.
    OutSection.notePatch(
        DebugOffsetPatch{OutSection.OS.tell() + 4, &OutSection, true});

    // Emit the FDE with updated address and CIE pointer.
    // (4 + AddrSize) is the size of the CIEId + initial_location
    // fields that will get reconstructed by emitFDE().
    unsigned FDERemainingBytes = InitialLength - (4 + SrcAddrSize);
    emitFDE(OffsetToCIERecord, SrcAddrSize, Loc + Range->Value,
            OrigFrameData.substr(InputOffset, FDERemainingBytes), OutSection);
    InputOffset += FDERemainingBytes;
  }

  return Error::success();
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

Error DWARFLinkerImpl::LinkContext::cloneAndEmitPaperTrails() {
  CompileUnits.emplace_back(std::make_unique<CompileUnit>(
      GlobalData, UniqueUnitID.fetch_add(1), "", InputDWARFFile,
      getUnitForOffset, Format, Endianness));

  CompileUnit &CU = *CompileUnits.back();

  BumpPtrAllocator Allocator;

  DIEGenerator ParentGenerator(Allocator, CU);

  SectionDescriptor &DebugInfoSection =
      CU.getOrCreateSectionDescriptor(DebugSectionKind::DebugInfo);
  OffsetsPtrVector PatchesOffsets;

  uint64_t CurrentOffset = CU.getDebugInfoHeaderSize();
  DIE *CUDie =
      ParentGenerator.createDIE(dwarf::DW_TAG_compile_unit, CurrentOffset);
  CU.setOutUnitDIE(CUDie);

  DebugInfoSection.notePatchWithOffsetUpdate(
      DebugStrPatch{{CurrentOffset},
                    GlobalData.getStringPool().insert("dsymutil").first},
      PatchesOffsets);
  CurrentOffset += ParentGenerator
                       .addStringPlaceholderAttribute(dwarf::DW_AT_producer,
                                                      dwarf::DW_FORM_strp)
                       .second;

  CurrentOffset +=
      ParentGenerator
          .addInplaceString(dwarf::DW_AT_name, InputDWARFFile.FileName)
          .second;

  size_t SizeAbbrevNumber = ParentGenerator.finalizeAbbreviations(true);
  CurrentOffset += SizeAbbrevNumber;
  for (uint64_t *OffsetPtr : PatchesOffsets)
    *OffsetPtr += SizeAbbrevNumber;
  for (const auto &Warning : CU.getContaingFile().Warnings) {
    PatchesOffsets.clear();
    DIEGenerator ChildGenerator(Allocator, CU);

    DIE *ChildDie =
        ChildGenerator.createDIE(dwarf::DW_TAG_constant, CurrentOffset);
    ParentGenerator.addChild(ChildDie);

    DebugInfoSection.notePatchWithOffsetUpdate(
        DebugStrPatch{
            {CurrentOffset},
            GlobalData.getStringPool().insert("dsymutil_warning").first},
        PatchesOffsets);
    CurrentOffset += ChildGenerator
                         .addStringPlaceholderAttribute(dwarf::DW_AT_name,
                                                        dwarf::DW_FORM_strp)
                         .second;

    CurrentOffset +=
        ChildGenerator
            .addScalarAttribute(dwarf::DW_AT_artificial, dwarf::DW_FORM_flag, 1)
            .second;

    DebugInfoSection.notePatchWithOffsetUpdate(
        DebugStrPatch{{CurrentOffset},
                      GlobalData.getStringPool().insert(Warning).first},
        PatchesOffsets);
    CurrentOffset += ChildGenerator
                         .addStringPlaceholderAttribute(
                             dwarf::DW_AT_const_value, dwarf::DW_FORM_strp)
                         .second;

    SizeAbbrevNumber = ChildGenerator.finalizeAbbreviations(false);

    CurrentOffset += SizeAbbrevNumber;
    for (uint64_t *OffsetPtr : PatchesOffsets)
      *OffsetPtr += SizeAbbrevNumber;

    ChildDie->setSize(CurrentOffset - ChildDie->getOffset());
  }

  CurrentOffset += 1; // End of children
  CUDie->setSize(CurrentOffset - CUDie->getOffset());

  uint64_t UnitSize = 0;
  UnitSize += CU.getDebugInfoHeaderSize();
  UnitSize += CUDie->getSize();
  CU.setUnitSize(UnitSize);

  if (GlobalData.getOptions().NoOutput)
    return Error::success();

  if (Error Err = CU.emitDebugInfo(*TargetTriple))
    return Err;

  return CU.emitAbbreviations();
}

void DWARFLinkerImpl::glueCompileUnitsAndWriteToTheOutput() {
  if (GlobalData.getOptions().NoOutput)
    return;

  // Go through all object files, all compile units and assign
  // offsets to them.
  assignOffsets();

  // Patch size/offsets fields according to the assigned CU offsets.
  patchOffsetsAndSizes();

  // FIXME: Build accelerator tables.

  // Emit common sections.
  emitCommonSections();

  // Cleanup data.
  cleanupDataAfterOutputSectionsAreGenerated();

  // Write debug tables from all object files/compile units into the
  // resulting file.
  writeDWARFToTheOutput();

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
              CU->getSectionDescriptor(DebugSectionKind::DebugInfo))
        AllDebugInfoSectionsSize += (*DebugInfo)->getContents().size();

    SizeByObject[Context->InputDWARFFile.FileName].Input =
        Context->OriginalDebugInfoSize;
    SizeByObject[Context->InputDWARFFile.FileName].Output =
        AllDebugInfoSectionsSize;
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
  parallel::TaskGroup TGroup;
  TGroup.spawn([&]() { assignOffsetsToStrings(); });
  TGroup.spawn([&]() { assignOffsetsToSections(); });
}

void DWARFLinkerImpl::assignOffsetsToStrings() {
  size_t CurDebugStrIndex = 1; // start from 1 to take into account zero entry.
  uint64_t CurDebugStrOffset =
      1; // start from 1 to take into account zero entry.
  size_t CurDebugLineStrIndex = 0;
  uint64_t CurDebugLineStrOffset = 0;

  // To save space we do not create any separate string table.
  // We use already allocated string patches and assign offsets
  // to them in the natural order.
  // ASSUMPTION: strings should be stored into .debug_str/.debug_line_str
  // sections in the same order as they were assigned offsets.

  forEachObjectSectionsSet([&](OutputSections &SectionsSet) {
    SectionsSet.forEach([&](SectionDescriptor &OutSection) {
      assignOffsetsToStringsImpl(OutSection.ListDebugStrPatch, CurDebugStrIndex,
                                 CurDebugStrOffset, DebugStrStrings);

      assignOffsetsToStringsImpl(OutSection.ListDebugLineStrPatch,
                                 CurDebugLineStrIndex, CurDebugLineStrOffset,
                                 DebugLineStrStrings);
    });
  });
}

template <typename PatchTy>
void DWARFLinkerImpl::assignOffsetsToStringsImpl(
    ArrayList<PatchTy> &Patches, size_t &IndexAccumulator,
    uint64_t &OffsetAccumulator,
    StringEntryToDwarfStringPoolEntryMap &StringsForEmission) {

  // Enumerates all patches, adds string into the
  // StringEntry->DwarfStringPoolEntry map, assign offset and index to the
  // string if it is not indexed yet.
  Patches.forEach([&](PatchTy &Patch) {
    DwarfStringPoolEntryWithExtString *Entry =
        StringsForEmission.add(Patch.String);
    assert(Entry != nullptr);

    if (!Entry->isIndexed()) {
      Entry->Offset = OffsetAccumulator;
      OffsetAccumulator += Entry->String.size() + 1;
      Entry->Index = IndexAccumulator++;
    }
  });
}

void DWARFLinkerImpl::assignOffsetsToSections() {
  std::array<uint64_t, SectionKindsNum> SectionSizesAccumulator = {0};

  forEachObjectSectionsSet([&](OutputSections &UnitSections) {
    UnitSections.assignSectionsOffsetAndAccumulateSize(SectionSizesAccumulator);
  });
}

void DWARFLinkerImpl::forEachObjectSectionsSet(
    function_ref<void(OutputSections &)> SectionsSetHandler) {
  // Handle all modules first(before regular compilation units).
  for (const std::unique_ptr<LinkContext> &Context : ObjectContexts)
    for (LinkContext::RefModuleUnit &ModuleUnit : Context->ModulesCompileUnits)
      SectionsSetHandler(*ModuleUnit.Unit);

  for (const std::unique_ptr<LinkContext> &Context : ObjectContexts) {
    // Handle object file common sections.
    SectionsSetHandler(*Context);

    // Handle compilation units.
    for (std::unique_ptr<CompileUnit> &CU : Context->CompileUnits)
      SectionsSetHandler(*CU);
  }
}

void DWARFLinkerImpl::patchOffsetsAndSizes() {
  forEachObjectSectionsSet([&](OutputSections &SectionsSet) {
    SectionsSet.forEach([&](SectionDescriptor &OutSection) {
      SectionsSet.applyPatches(OutSection, DebugStrStrings,
                               DebugLineStrStrings);
    });
  });
}

template <typename PatchTy>
void DWARFLinkerImpl::emitStringsImpl(
    ArrayList<PatchTy> &StringPatches,
    const StringEntryToDwarfStringPoolEntryMap &Strings, uint64_t &NextOffset,
    SectionDescriptor &OutSection) {
  // Enumerate all string patches and write strings into the destination
  // section. We enumerate patches to have a predictable order of strings(i.e.
  // strings are emitted in the order as they appear in the patches).
  StringPatches.forEach([&](const PatchTy &Patch) {
    DwarfStringPoolEntryWithExtString *StringToEmit =
        Strings.getExistingEntry(Patch.String);
    assert(StringToEmit->isIndexed());

    // Patches can refer the same strings. We use accumulated NextOffset
    // to understand whether corresponding string is already emitted.
    // Skip patch if string is already emitted.
    if (StringToEmit->Offset >= NextOffset) {
      NextOffset = StringToEmit->Offset + StringToEmit->String.size() + 1;
      // Emit the string itself.
      OutSection.emitInplaceString(StringToEmit->String);
    }
  });
}

void DWARFLinkerImpl::emitCommonSections() {
  parallel::TaskGroup TG;

  SectionDescriptor &OutDebugStrSection =
      CommonSections.getOrCreateSectionDescriptor(DebugSectionKind::DebugStr);
  SectionDescriptor &OutDebugLineStrSection =
      CommonSections.getOrCreateSectionDescriptor(
          DebugSectionKind::DebugLineStr);

  // Emit .debug_str section.
  TG.spawn([&]() {
    uint64_t DebugStrNextOffset = 0;

    // Emit zero length string. Accelerator tables does not work correctly
    // if the first string is not zero length string.
    OutDebugStrSection.emitInplaceString("");
    DebugStrNextOffset++;

    forEachObjectSectionsSet([&](OutputSections &Sections) {
      Sections.forEach([&](SectionDescriptor &Section) {
        emitStringsImpl(Section.ListDebugStrPatch, DebugStrStrings,
                        DebugStrNextOffset, OutDebugStrSection);
      });
    });
  });

  // Emit .debug_line_str section.
  TG.spawn([&]() {
    uint64_t DebugLineStrNextOffset = 0;

    forEachObjectSectionsSet([&](OutputSections &Sections) {
      Sections.forEach([&](SectionDescriptor &Section) {
        emitStringsImpl(Section.ListDebugLineStrPatch, DebugLineStrStrings,
                        DebugLineStrNextOffset, OutDebugLineStrSection);
      });
    });
  });
}

void DWARFLinkerImpl::cleanupDataAfterOutputSectionsAreGenerated() {
  GlobalData.getStringPool().clear();
  DebugStrStrings.clear();
  DebugLineStrStrings.clear();
}

void DWARFLinkerImpl::writeDWARFToTheOutput() {
  bool HasAbbreviations = false;

  forEachObjectSectionsSet([&](OutputSections &Sections) {
    Sections.forEach([&](SectionDescriptor &OutSection) {
      if (!HasAbbreviations && !OutSection.getContents().empty() &&
          OutSection.getKind() == DebugSectionKind::DebugAbbrev)
        HasAbbreviations = true;

      // Emit section content.
      TheDwarfEmitter->emitSectionContents(OutSection.getContents(),
                                           OutSection.getName());
      OutSection.erase();
    });
  });

  CommonSections.forEach([&](SectionDescriptor &OutSection) {
    // Emit section content.
    TheDwarfEmitter->emitSectionContents(OutSection.getContents(),
                                         OutSection.getName());
    OutSection.erase();
  });

  if (!HasAbbreviations) {
    const SmallVector<std::unique_ptr<DIEAbbrev>> Abbreviations;
    TheDwarfEmitter->emitAbbrevs(Abbreviations, 3);
  }
}

} // end of namespace dwarflinker_parallel
} // namespace llvm
