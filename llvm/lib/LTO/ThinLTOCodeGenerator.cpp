//===-ThinLTOCodeGenerator.cpp - LLVM Link Time Optimizer -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Thin Link Time Optimization library. This library is
// intended to be used by linker to optimize code at link time.
//
//===----------------------------------------------------------------------===//

#include "llvm/LTO/ThinLTOCodeGenerator.h"

#ifdef HAVE_LLVM_REVISION
#include "LLVMLTORevision.h"
#endif

#include "UpdateCompilerUsed.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/ModuleSummaryAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/ExecutionEngine/ObjectMemoryBuffer.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Object/IRObjectFile.h"
#include "llvm/Object/ModuleSummaryIndexObjectFile.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/CachePruning.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA1.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/FunctionImport.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/ObjCARC.h"
#include "llvm/Transforms/Utils/FunctionImportUtils.h"

using namespace llvm;

#define DEBUG_TYPE "thinlto"

namespace llvm {
// Flags -discard-value-names, defined in LTOCodeGenerator.cpp
extern cl::opt<bool> LTODiscardValueNames;
}

namespace {

static cl::opt<int> ThreadCount("threads",
                                cl::init(std::thread::hardware_concurrency()));

static void diagnosticHandler(const DiagnosticInfo &DI) {
  DiagnosticPrinterRawOStream DP(errs());
  DI.print(DP);
  errs() << '\n';
}

// Simple helper to load a module from bitcode
static std::unique_ptr<Module>
loadModuleFromBuffer(const MemoryBufferRef &Buffer, LLVMContext &Context,
                     bool Lazy) {
  SMDiagnostic Err;
  ErrorOr<std::unique_ptr<Module>> ModuleOrErr(nullptr);
  if (Lazy) {
    ModuleOrErr =
        getLazyBitcodeModule(MemoryBuffer::getMemBuffer(Buffer, false), Context,
                             /* ShouldLazyLoadMetadata */ Lazy);
  } else {
    ModuleOrErr = parseBitcodeFile(Buffer, Context);
  }
  if (std::error_code EC = ModuleOrErr.getError()) {
    Err = SMDiagnostic(Buffer.getBufferIdentifier(), SourceMgr::DK_Error,
                       EC.message());
    Err.print("ThinLTO", errs());
    report_fatal_error("Can't load module, abort.");
  }
  return std::move(ModuleOrErr.get());
}

// Simple helper to save temporary files for debug.
static void saveTempBitcode(const Module &TheModule, StringRef TempDir,
                            unsigned count, StringRef Suffix) {
  if (TempDir.empty())
    return;
  // User asked to save temps, let dump the bitcode file after import.
  auto SaveTempPath = TempDir + llvm::utostr(count) + Suffix;
  std::error_code EC;
  raw_fd_ostream OS(SaveTempPath.str(), EC, sys::fs::F_None);
  if (EC)
    report_fatal_error(Twine("Failed to open ") + SaveTempPath +
                       " to save optimized bitcode\n");
  WriteBitcodeToFile(&TheModule, OS, /* ShouldPreserveUseListOrder */ true);
}

bool IsFirstDefinitionForLinker(const GlobalValueInfoList &GVInfo,
                                const ModuleSummaryIndex &Index,
                                StringRef ModulePath) {
  // Get the first *linker visible* definition for this global in the summary
  // list.
  auto FirstDefForLinker = llvm::find_if(
      GVInfo, [](const std::unique_ptr<GlobalValueInfo> &FuncInfo) {
        auto Linkage = FuncInfo->summary()->linkage();
        return !GlobalValue::isAvailableExternallyLinkage(Linkage);
      });
  // If \p GV is not the first definition, give up...
  if ((*FirstDefForLinker)->summary()->modulePath() != ModulePath)
    return false;
  // If there is any strong definition anywhere, do not bother emitting this.
  if (llvm::any_of(
          GVInfo, [](const std::unique_ptr<GlobalValueInfo> &FuncInfo) {
            auto Linkage = FuncInfo->summary()->linkage();
            return !GlobalValue::isAvailableExternallyLinkage(Linkage) &&
                   !GlobalValue::isWeakForLinker(Linkage);
          }))
    return false;
  return true;
}

static GlobalValue::LinkageTypes
ResolveODR(const ModuleSummaryIndex &Index,
           const FunctionImporter::ExportSetTy &ExportList,
           StringRef ModuleIdentifier, GlobalValue::GUID GUID,
           const GlobalValueSummary &GV) {
  auto HasMultipleCopies =
      [&](const GlobalValueInfoList &GVInfo) { return GVInfo.size() > 1; };

  auto OriginalLinkage = GV.linkage();
  switch (OriginalLinkage) {
  case GlobalValue::ExternalLinkage:
  case GlobalValue::AvailableExternallyLinkage:
  case GlobalValue::AppendingLinkage:
  case GlobalValue::InternalLinkage:
  case GlobalValue::PrivateLinkage:
  case GlobalValue::ExternalWeakLinkage:
  case GlobalValue::CommonLinkage:
  case GlobalValue::LinkOnceAnyLinkage:
  case GlobalValue::WeakAnyLinkage:
    break;
  case GlobalValue::LinkOnceODRLinkage:
  case GlobalValue::WeakODRLinkage: {
    auto &GVInfo = Index.findGlobalValueInfoList(GUID)->second;
    // We need to emit only one of these, the first module will keep
    // it, but turned into a weak while the others will drop it.
    if (!HasMultipleCopies(GVInfo)) {
      // Exported LinkonceODR needs to be promoted to not be discarded
      if (GlobalValue::isDiscardableIfUnused(OriginalLinkage) &&
          ExportList.count(GUID))
        return GlobalValue::WeakODRLinkage;
      break;
    }
    if (IsFirstDefinitionForLinker(GVInfo, Index, ModuleIdentifier))
      return GlobalValue::WeakODRLinkage;
    else if (isa<AliasSummary>(&GV))
      // Alias can't be turned into available_externally.
      return OriginalLinkage;
    return GlobalValue::AvailableExternallyLinkage;
  }
  }
  return OriginalLinkage;
}

/// Resolve LinkOnceODR and WeakODR.
///
/// We'd like to drop these function if they are no longer referenced in the
/// current module. However there is a chance that another module is still
/// referencing them because of the import. We make sure we always emit at least
/// one copy.
static void ResolveODR(
    const ModuleSummaryIndex &Index,
    const FunctionImporter::ExportSetTy &ExportList,
    const std::map<GlobalValue::GUID, GlobalValueSummary *> &DefinedGlobals,
    StringRef ModuleIdentifier,
    std::map<GlobalValue::GUID, GlobalValue::LinkageTypes> &ResolvedODR) {
  if (Index.modulePaths().size() == 1)
    // Nothing to do if we don't have multiple modules
    return;

  // We won't optimize the globals that are referenced by an alias for now
  // Ideally we should turn the alias into a global and duplicate the definition
  // when needed.
  DenseSet<GlobalValueSummary *> GlobalInvolvedWithAlias;
  for (auto &GA : DefinedGlobals) {
    if (auto AS = dyn_cast<AliasSummary>(GA.second))
      GlobalInvolvedWithAlias.insert(&AS->getAliasee());
  }

  for (auto &GV : DefinedGlobals) {
    if (GlobalInvolvedWithAlias.count(GV.second))
      continue;
    auto NewLinkage =
        ResolveODR(Index, ExportList, ModuleIdentifier, GV.first, *GV.second);
    if (NewLinkage != GV.second->linkage()) {
      ResolvedODR[GV.first] = NewLinkage;
    }
  }
}

/// Fixup linkage, see ResolveODR() above.
void fixupODR(
    Module &TheModule,
    const std::map<GlobalValue::GUID, GlobalValue::LinkageTypes> &ResolvedODR) {
  // Process functions and global now
  for (auto &GV : TheModule) {
    auto NewLinkage = ResolvedODR.find(GV.getGUID());
    if (NewLinkage == ResolvedODR.end())
      continue;
    DEBUG(dbgs() << "ODR fixing up linkage for `" << GV.getName() << "` from "
                 << GV.getLinkage() << " to " << NewLinkage->second << "\n");
    GV.setLinkage(NewLinkage->second);
  }
  for (auto &GV : TheModule.globals()) {
    auto NewLinkage = ResolvedODR.find(GV.getGUID());
    if (NewLinkage == ResolvedODR.end())
      continue;
    DEBUG(dbgs() << "ODR fixing up linkage for `" << GV.getName() << "` from "
                 << GV.getLinkage() << " to " << NewLinkage->second << "\n");
    GV.setLinkage(NewLinkage->second);
  }
  for (auto &GV : TheModule.aliases()) {
    auto NewLinkage = ResolvedODR.find(GV.getGUID());
    if (NewLinkage == ResolvedODR.end())
      continue;
    DEBUG(dbgs() << "ODR fixing up linkage for `" << GV.getName() << "` from "
                 << GV.getLinkage() << " to " << NewLinkage->second << "\n");
    GV.setLinkage(NewLinkage->second);
  }
}

static StringMap<MemoryBufferRef>
generateModuleMap(const std::vector<MemoryBufferRef> &Modules) {
  StringMap<MemoryBufferRef> ModuleMap;
  for (auto &ModuleBuffer : Modules) {
    assert(ModuleMap.find(ModuleBuffer.getBufferIdentifier()) ==
               ModuleMap.end() &&
           "Expect unique Buffer Identifier");
    ModuleMap[ModuleBuffer.getBufferIdentifier()] = ModuleBuffer;
  }
  return ModuleMap;
}

/// Provide a "loader" for the FunctionImporter to access function from other
/// modules.
class ModuleLoader {
  /// The context that will be used for importing.
  LLVMContext &Context;

  /// Map from Module identifier to MemoryBuffer. Used by clients like the
  /// FunctionImported to request loading a Module.
  StringMap<MemoryBufferRef> &ModuleMap;

public:
  ModuleLoader(LLVMContext &Context, StringMap<MemoryBufferRef> &ModuleMap)
      : Context(Context), ModuleMap(ModuleMap) {}

  /// Load a module on demand.
  std::unique_ptr<Module> operator()(StringRef Identifier) {
    return loadModuleFromBuffer(ModuleMap[Identifier], Context, /*Lazy*/ true);
  }
};

static void promoteModule(Module &TheModule, const ModuleSummaryIndex &Index) {
  if (renameModuleForThinLTO(TheModule, Index))
    report_fatal_error("renameModuleForThinLTO failed");
}

static void
crossImportIntoModule(Module &TheModule, const ModuleSummaryIndex &Index,
                      StringMap<MemoryBufferRef> &ModuleMap,
                      const FunctionImporter::ImportMapTy &ImportList) {
  ModuleLoader Loader(TheModule.getContext(), ModuleMap);
  FunctionImporter Importer(Index, Loader);
  Importer.importFunctions(TheModule, ImportList);
}

static void optimizeModule(Module &TheModule, TargetMachine &TM) {
  // Populate the PassManager
  PassManagerBuilder PMB;
  PMB.LibraryInfo = new TargetLibraryInfoImpl(TM.getTargetTriple());
  PMB.Inliner = createFunctionInliningPass();
  // FIXME: should get it from the bitcode?
  PMB.OptLevel = 3;
  PMB.LoopVectorize = true;
  PMB.SLPVectorize = true;
  PMB.VerifyInput = true;
  PMB.VerifyOutput = false;

  legacy::PassManager PM;

  // Add the TTI (required to inform the vectorizer about register size for
  // instance)
  PM.add(createTargetTransformInfoWrapperPass(TM.getTargetIRAnalysis()));

  // Add optimizations
  PMB.populateThinLTOPassManager(PM);

  PM.run(TheModule);
}

// Create a DenseSet of GlobalValue to be used with the Internalizer.
static DenseSet<const GlobalValue *> computePreservedSymbolsForModule(
    Module &TheModule, const DenseSet<GlobalValue::GUID> &GUIDPreservedSymbols,
    const FunctionImporter::ExportSetTy &ExportList) {
  DenseSet<const GlobalValue *> PreservedGV;
  if (GUIDPreservedSymbols.empty())
    // Early exit: internalize is disabled when there is nothing to preserve.
    return PreservedGV;

  auto AddPreserveGV = [&](const GlobalValue &GV) {
    auto GUID = GV.getGUID();
    if (GUIDPreservedSymbols.count(GUID) || ExportList.count(GUID))
      PreservedGV.insert(&GV);
  };

  for (auto &GV : TheModule)
    AddPreserveGV(GV);
  for (auto &GV : TheModule.globals())
    AddPreserveGV(GV);
  for (auto &GV : TheModule.aliases())
    AddPreserveGV(GV);

  return PreservedGV;
}

// Run internalization on \p TheModule
static void
doInternalizeModule(Module &TheModule, const TargetMachine &TM,
                    const DenseSet<const GlobalValue *> &PreservedGV) {
  if (PreservedGV.empty()) {
    // Be friendly and don't nuke totally the module when the client didn't
    // supply anything to preserve.
    return;
  }

  // Parse inline ASM and collect the list of symbols that are not defined in
  // the current module.
  StringSet<> AsmUndefinedRefs;
  object::IRObjectFile::CollectAsmUndefinedRefs(
      Triple(TheModule.getTargetTriple()), TheModule.getModuleInlineAsm(),
      [&AsmUndefinedRefs](StringRef Name, object::BasicSymbolRef::Flags Flags) {
        if (Flags & object::BasicSymbolRef::SF_Undefined)
          AsmUndefinedRefs.insert(Name);
      });

  // Update the llvm.compiler_used globals to force preserving libcalls and
  // symbols referenced from asm
  UpdateCompilerUsed(TheModule, TM, AsmUndefinedRefs);

  // Declare a callback for the internalize pass that will ask for every
  // candidate GlobalValue if it can be internalized or not.
  auto MustPreserveGV =
      [&](const GlobalValue &GV) -> bool { return PreservedGV.count(&GV); };

  llvm::internalizeModule(TheModule, MustPreserveGV);
}

// Convert the PreservedSymbols map from "Name" based to "GUID" based.
static DenseSet<GlobalValue::GUID>
computeGUIDPreservedSymbols(const StringSet<> &PreservedSymbols,
                            const Triple &TheTriple) {
  DenseSet<GlobalValue::GUID> GUIDPreservedSymbols(PreservedSymbols.size());
  for (auto &Entry : PreservedSymbols) {
    StringRef Name = Entry.first();
    if (TheTriple.isOSBinFormatMachO() && Name.size() > 0 && Name[0] == '_')
      Name = Name.drop_front();
    GUIDPreservedSymbols.insert(GlobalValue::getGUID(Name));
  }
  return GUIDPreservedSymbols;
}

std::unique_ptr<MemoryBuffer> codegenModule(Module &TheModule,
                                            TargetMachine &TM) {
  SmallVector<char, 128> OutputBuffer;

  // CodeGen
  {
    raw_svector_ostream OS(OutputBuffer);
    legacy::PassManager PM;

    // If the bitcode files contain ARC code and were compiled with optimization,
    // the ObjCARCContractPass must be run, so do it unconditionally here.
    PM.add(createObjCARCContractPass());

    // Setup the codegen now.
    if (TM.addPassesToEmitFile(PM, OS, TargetMachine::CGFT_ObjectFile,
                               /* DisableVerify */ true))
      report_fatal_error("Failed to setup codegen");

    // Run codegen now. resulting binary is in OutputBuffer.
    PM.run(TheModule);
  }
  return make_unique<ObjectMemoryBuffer>(std::move(OutputBuffer));
}

/// Manage caching for a single Module.
class ModuleCacheEntry {
  SmallString<128> EntryPath;

public:
  // Create a cache entry. This compute a unique hash for the Module considering
  // the current list of export/import, and offer an interface to query to
  // access the content in the cache.
  ModuleCacheEntry(
      StringRef CachePath, const ModuleSummaryIndex &Index, StringRef ModuleID,
      const FunctionImporter::ImportMapTy &ImportList,
      const FunctionImporter::ExportSetTy &ExportList,
      const std::map<GlobalValue::GUID, GlobalValue::LinkageTypes> &ResolvedODR,
      const std::map<GlobalValue::GUID, GlobalValueSummary *> &DefinedFunctions,
      const DenseSet<GlobalValue::GUID> &PreservedSymbols) {
    if (CachePath.empty())
      return;

    // Compute the unique hash for this entry
    // This is based on the current compiler version, the module itself, the
    // export list, the hash for every single module in the import list, the
    // list of ResolvedODR for the module, and the list of preserved symbols.

    SHA1 Hasher;

    // Start with the compiler revision
    Hasher.update(LLVM_VERSION_STRING);
#ifdef HAVE_LLVM_REVISION
    Hasher.update(LLVM_REVISION);
#endif

    // Include the hash for the current module
    auto ModHash = Index.getModuleHash(ModuleID);
    Hasher.update(ArrayRef<uint8_t>((uint8_t *)&ModHash[0], sizeof(ModHash)));
    for (auto F : ExportList)
      // The export list can impact the internalization, be conservative here
      Hasher.update(ArrayRef<uint8_t>((uint8_t *)&F, sizeof(F)));

    // Include the hash for every module we import functions from
    for (auto &Entry : ImportList) {
      auto ModHash = Index.getModuleHash(Entry.first());
      Hasher.update(ArrayRef<uint8_t>((uint8_t *)&ModHash[0], sizeof(ModHash)));
    }

    // Include the hash for the resolved ODR.
    for (auto &Entry : ResolvedODR) {
      Hasher.update(ArrayRef<uint8_t>((uint8_t *)&Entry.first,
                                      sizeof(GlobalValue::GUID)));
      Hasher.update(ArrayRef<uint8_t>((uint8_t *)&Entry.second,
                                      sizeof(GlobalValue::LinkageTypes)));
    }

    // Include the hash for the preserved symbols.
    for (auto &Entry : PreservedSymbols) {
      if (DefinedFunctions.count(Entry))
        Hasher.update(
            ArrayRef<uint8_t>((uint8_t *)&Entry, sizeof(GlobalValue::GUID)));
    }

    sys::path::append(EntryPath, CachePath, toHex(Hasher.result()));
  }

  // Access the path to this entry in the cache.
  StringRef getEntryPath() { return EntryPath; }

  // Try loading the buffer for this cache entry.
  ErrorOr<std::unique_ptr<MemoryBuffer>> tryLoadingBuffer() {
    if (EntryPath.empty())
      return std::error_code();
    return MemoryBuffer::getFile(EntryPath);
  }

  // Cache the Produced object file
  void write(MemoryBufferRef OutputBuffer) {
    if (EntryPath.empty())
      return;

    // Write to a temporary to avoid race condition
    SmallString<128> TempFilename;
    int TempFD;
    std::error_code EC =
        sys::fs::createTemporaryFile("Thin", "tmp.o", TempFD, TempFilename);
    if (EC) {
      errs() << "Error: " << EC.message() << "\n";
      report_fatal_error("ThinLTO: Can't get a temporary file");
    }
    {
      raw_fd_ostream OS(TempFD, /* ShouldClose */ true);
      OS << OutputBuffer.getBuffer();
    }
    // Rename to final destination (hopefully race condition won't matter here)
    sys::fs::rename(TempFilename, EntryPath);
  }
};

static std::unique_ptr<MemoryBuffer> ProcessThinLTOModule(
    Module &TheModule, const ModuleSummaryIndex &Index,
    StringMap<MemoryBufferRef> &ModuleMap, TargetMachine &TM,
    const FunctionImporter::ImportMapTy &ImportList,
    const FunctionImporter::ExportSetTy &ExportList,
    const DenseSet<GlobalValue::GUID> &GUIDPreservedSymbols,
    std::map<GlobalValue::GUID, GlobalValue::LinkageTypes> &ResolvedODR,
    ThinLTOCodeGenerator::CachingOptions CacheOptions, bool DisableCodeGen,
    StringRef SaveTempsDir, unsigned count) {

  // Save temps: after IPO.
  saveTempBitcode(TheModule, SaveTempsDir, count, ".1.IPO.bc");

  // Prepare for internalization by computing the set of symbols to preserve.
  // We need to compute the list of symbols to preserve during internalization
  // before doing any promotion because after renaming we won't (easily) match
  // to the original name.
  auto PreservedGV = computePreservedSymbolsForModule(
      TheModule, GUIDPreservedSymbols, ExportList);

  // "Benchmark"-like optimization: single-source case
  bool SingleModule = (ModuleMap.size() == 1);

  if (!SingleModule) {
    promoteModule(TheModule, Index);

    // Resolve the LinkOnce/Weak ODR, trying to turn them into
    // "available_externally" when possible.
    // This is a compile-time optimization.
    fixupODR(TheModule, ResolvedODR);

    // Save temps: after promotion.
    saveTempBitcode(TheModule, SaveTempsDir, count, ".2.promoted.bc");
  }

  // Internalization
  doInternalizeModule(TheModule, TM, PreservedGV);

  // Save internalized bitcode
  saveTempBitcode(TheModule, SaveTempsDir, count, ".3.internalized.bc");

  if (!SingleModule) {
    crossImportIntoModule(TheModule, Index, ModuleMap, ImportList);

    // Save temps: after cross-module import.
    saveTempBitcode(TheModule, SaveTempsDir, count, ".4.imported.bc");
  }

  optimizeModule(TheModule, TM);

  saveTempBitcode(TheModule, SaveTempsDir, count, ".5.opt.bc");

  if (DisableCodeGen) {
    // Configured to stop before CodeGen, serialize the bitcode and return.
    SmallVector<char, 128> OutputBuffer;
    {
      raw_svector_ostream OS(OutputBuffer);
      ModuleSummaryIndexBuilder IndexBuilder(&TheModule);
      WriteBitcodeToFile(&TheModule, OS, true, &IndexBuilder.getIndex());
    }
    return make_unique<ObjectMemoryBuffer>(std::move(OutputBuffer));
  }

  return codegenModule(TheModule, TM);
}

// Initialize the TargetMachine builder for a given Triple
static void initTMBuilder(TargetMachineBuilder &TMBuilder,
                          const Triple &TheTriple) {
  // Set a default CPU for Darwin triples (copied from LTOCodeGenerator).
  // FIXME this looks pretty terrible...
  if (TMBuilder.MCpu.empty() && TheTriple.isOSDarwin()) {
    if (TheTriple.getArch() == llvm::Triple::x86_64)
      TMBuilder.MCpu = "core2";
    else if (TheTriple.getArch() == llvm::Triple::x86)
      TMBuilder.MCpu = "yonah";
    else if (TheTriple.getArch() == llvm::Triple::aarch64)
      TMBuilder.MCpu = "cyclone";
  }
  TMBuilder.TheTriple = std::move(TheTriple);
}

} // end anonymous namespace

void ThinLTOCodeGenerator::addModule(StringRef Identifier, StringRef Data) {
  MemoryBufferRef Buffer(Data, Identifier);
  if (Modules.empty()) {
    // First module added, so initialize the triple and some options
    LLVMContext Context;
    Triple TheTriple(getBitcodeTargetTriple(Buffer, Context));
    initTMBuilder(TMBuilder, Triple(TheTriple));
  }
#ifndef NDEBUG
  else {
    LLVMContext Context;
    assert(TMBuilder.TheTriple.str() ==
               getBitcodeTargetTriple(Buffer, Context) &&
           "ThinLTO modules with different triple not supported");
  }
#endif
  Modules.push_back(Buffer);
}

void ThinLTOCodeGenerator::preserveSymbol(StringRef Name) {
  PreservedSymbols.insert(Name);
}

void ThinLTOCodeGenerator::crossReferenceSymbol(StringRef Name) {
  // FIXME: At the moment, we don't take advantage of this extra information,
  // we're conservatively considering cross-references as preserved.
  //  CrossReferencedSymbols.insert(Name);
  PreservedSymbols.insert(Name);
}

// TargetMachine factory
std::unique_ptr<TargetMachine> TargetMachineBuilder::create() const {
  std::string ErrMsg;
  const Target *TheTarget =
      TargetRegistry::lookupTarget(TheTriple.str(), ErrMsg);
  if (!TheTarget) {
    report_fatal_error("Can't load target for this Triple: " + ErrMsg);
  }

  // Use MAttr as the default set of features.
  SubtargetFeatures Features(MAttr);
  Features.getDefaultSubtargetFeatures(TheTriple);
  std::string FeatureStr = Features.getString();
  return std::unique_ptr<TargetMachine>(TheTarget->createTargetMachine(
      TheTriple.str(), MCpu, FeatureStr, Options, RelocModel,
      CodeModel::Default, CGOptLevel));
}

/**
 * Produce the combined summary index from all the bitcode files:
 * "thin-link".
 */
std::unique_ptr<ModuleSummaryIndex> ThinLTOCodeGenerator::linkCombinedIndex() {
  std::unique_ptr<ModuleSummaryIndex> CombinedIndex;
  uint64_t NextModuleId = 0;
  for (auto &ModuleBuffer : Modules) {
    ErrorOr<std::unique_ptr<object::ModuleSummaryIndexObjectFile>> ObjOrErr =
        object::ModuleSummaryIndexObjectFile::create(ModuleBuffer,
                                                     diagnosticHandler);
    if (std::error_code EC = ObjOrErr.getError()) {
      // FIXME diagnose
      errs() << "error: can't create ModuleSummaryIndexObjectFile for buffer: "
             << EC.message() << "\n";
      return nullptr;
    }
    auto Index = (*ObjOrErr)->takeIndex();
    if (CombinedIndex) {
      CombinedIndex->mergeFrom(std::move(Index), ++NextModuleId);
    } else {
      CombinedIndex = std::move(Index);
    }
  }
  return CombinedIndex;
}

/**
 * Perform promotion and renaming of exported internal functions.
 */
void ThinLTOCodeGenerator::promote(Module &TheModule,
                                   ModuleSummaryIndex &Index) {
  auto ModuleCount = Index.modulePaths().size();
  auto ModuleIdentifier = TheModule.getModuleIdentifier();
  // Collect for each module the list of function it defines (GUID -> Summary).
  StringMap<std::map<GlobalValue::GUID, GlobalValueSummary *>>
      ModuleToDefinedGVSummaries;
  Index.collectDefinedGVSummariesPerModule(ModuleToDefinedGVSummaries);

  // Generate import/export list
  StringMap<FunctionImporter::ImportMapTy> ImportLists(ModuleCount);
  StringMap<FunctionImporter::ExportSetTy> ExportLists(ModuleCount);
  ComputeCrossModuleImport(Index, ModuleToDefinedGVSummaries, ImportLists,
                           ExportLists);
  auto &ExportList = ExportLists[ModuleIdentifier];

  // Resolve the LinkOnceODR, trying to turn them into "available_externally"
  // where possible.
  // This is a compile-time optimization.
  // We use a std::map here to be able to have a defined ordering when
  // producing a hash for the cache entry.
  std::map<GlobalValue::GUID, GlobalValue::LinkageTypes> ResolvedODR;
  ResolveODR(Index, ExportList, ModuleToDefinedGVSummaries[ModuleIdentifier],
             ModuleIdentifier, ResolvedODR);
  fixupODR(TheModule, ResolvedODR);

  promoteModule(TheModule, Index);
}

/**
 * Perform cross-module importing for the module identified by ModuleIdentifier.
 */
void ThinLTOCodeGenerator::crossModuleImport(Module &TheModule,
                                             ModuleSummaryIndex &Index) {
  auto ModuleMap = generateModuleMap(Modules);
  auto ModuleCount = Index.modulePaths().size();

  // Collect for each module the list of function it defines (GUID -> Summary).
  StringMap<std::map<GlobalValue::GUID, GlobalValueSummary *>>
      ModuleToDefinedGVSummaries(ModuleCount);
  Index.collectDefinedGVSummariesPerModule(ModuleToDefinedGVSummaries);

  // Generate import/export list
  StringMap<FunctionImporter::ImportMapTy> ImportLists(ModuleCount);
  StringMap<FunctionImporter::ExportSetTy> ExportLists(ModuleCount);
  ComputeCrossModuleImport(Index, ModuleToDefinedGVSummaries, ImportLists,
                           ExportLists);
  auto &ImportList = ImportLists[TheModule.getModuleIdentifier()];

  crossImportIntoModule(TheModule, Index, ModuleMap, ImportList);
}

/**
 * Perform internalization.
 */
void ThinLTOCodeGenerator::internalize(Module &TheModule,
                                       ModuleSummaryIndex &Index) {
  initTMBuilder(TMBuilder, Triple(TheModule.getTargetTriple()));
  auto ModuleCount = Index.modulePaths().size();
  auto ModuleIdentifier = TheModule.getModuleIdentifier();

  // Convert the preserved symbols set from string to GUID
  auto GUIDPreservedSymbols =
      computeGUIDPreservedSymbols(PreservedSymbols, TMBuilder.TheTriple);

  // Collect for each module the list of function it defines (GUID -> Summary).
  StringMap<std::map<GlobalValue::GUID, GlobalValueSummary *>>
      ModuleToDefinedGVSummaries(ModuleCount);
  Index.collectDefinedGVSummariesPerModule(ModuleToDefinedGVSummaries);

  // Generate import/export list
  StringMap<FunctionImporter::ImportMapTy> ImportLists(ModuleCount);
  StringMap<FunctionImporter::ExportSetTy> ExportLists(ModuleCount);
  ComputeCrossModuleImport(Index, ModuleToDefinedGVSummaries, ImportLists,
                           ExportLists);
  auto &ExportList = ExportLists[ModuleIdentifier];

  // Internalization
  auto PreservedGV = computePreservedSymbolsForModule(
      TheModule, GUIDPreservedSymbols, ExportList);
  doInternalizeModule(TheModule, *TMBuilder.create(), PreservedGV);
}

/**
 * Perform post-importing ThinLTO optimizations.
 */
void ThinLTOCodeGenerator::optimize(Module &TheModule) {
  initTMBuilder(TMBuilder, Triple(TheModule.getTargetTriple()));

  // Optimize now
  optimizeModule(TheModule, *TMBuilder.create());
}

/**
 * Perform ThinLTO CodeGen.
 */
std::unique_ptr<MemoryBuffer> ThinLTOCodeGenerator::codegen(Module &TheModule) {
  initTMBuilder(TMBuilder, Triple(TheModule.getTargetTriple()));
  return codegenModule(TheModule, *TMBuilder.create());
}

// Main entry point for the ThinLTO processing
void ThinLTOCodeGenerator::run() {
  if (CodeGenOnly) {
    // Perform only parallel codegen and return.
    ThreadPool Pool;
    assert(ProducedBinaries.empty() && "The generator should not be reused");
    ProducedBinaries.resize(Modules.size());
    int count = 0;
    for (auto &ModuleBuffer : Modules) {
      Pool.async([&](int count) {
        LLVMContext Context;
        Context.setDiscardValueNames(LTODiscardValueNames);

        // Parse module now
        auto TheModule = loadModuleFromBuffer(ModuleBuffer, Context, false);

        // CodeGen
        ProducedBinaries[count] = codegen(*TheModule);
      }, count++);
    }

    return;
  }

  // Sequential linking phase
  auto Index = linkCombinedIndex();

  // Save temps: index.
  if (!SaveTempsDir.empty()) {
    auto SaveTempPath = SaveTempsDir + "index.bc";
    std::error_code EC;
    raw_fd_ostream OS(SaveTempPath, EC, sys::fs::F_None);
    if (EC)
      report_fatal_error(Twine("Failed to open ") + SaveTempPath +
                         " to save optimized bitcode\n");
    WriteIndexToFile(*Index, OS);
  }

  // Prepare the resulting object vector
  assert(ProducedBinaries.empty() && "The generator should not be reused");
  ProducedBinaries.resize(Modules.size());

  // Prepare the module map.
  auto ModuleMap = generateModuleMap(Modules);
  auto ModuleCount = Modules.size();

  // Collect for each module the list of function it defines (GUID -> Summary).
  StringMap<std::map<GlobalValue::GUID, GlobalValueSummary *>>
      ModuleToDefinedGVSummaries(ModuleCount);
  Index->collectDefinedGVSummariesPerModule(ModuleToDefinedGVSummaries);

  // Collect the import/export lists for all modules from the call-graph in the
  // combined index.
  StringMap<FunctionImporter::ImportMapTy> ImportLists(ModuleCount);
  StringMap<FunctionImporter::ExportSetTy> ExportLists(ModuleCount);
  ComputeCrossModuleImport(*Index, ModuleToDefinedGVSummaries, ImportLists,
                           ExportLists);

  // Convert the preserved symbols set from string to GUID, this is needed for
  // computing the caching hash and the internalization.
  auto GUIDPreservedSymbols =
      computeGUIDPreservedSymbols(PreservedSymbols, TMBuilder.TheTriple);

  // Parallel optimizer + codegen
  {
    ThreadPool Pool(ThreadCount);
    int count = 0;
    for (auto &ModuleBuffer : Modules) {
      Pool.async([&](int count) {
        auto ModuleIdentifier = ModuleBuffer.getBufferIdentifier();
        auto &ExportList = ExportLists[ModuleIdentifier];

        auto &DefinedFunctions = ModuleToDefinedGVSummaries[ModuleIdentifier];

        // Resolve ODR, this has to be done early because it impacts the caching
        // We use a std::map here to be able to have a defined ordering when
        // producing a hash for the cache entry.
        std::map<GlobalValue::GUID, GlobalValue::LinkageTypes> ResolvedODR;
        ResolveODR(*Index, ExportList, DefinedFunctions, ModuleIdentifier,
                   ResolvedODR);

        // The module may be cached, this helps handling it.
        ModuleCacheEntry CacheEntry(CacheOptions.Path, *Index, ModuleIdentifier,
                                    ImportLists[ModuleIdentifier], ExportList,
                                    ResolvedODR, DefinedFunctions,
                                    GUIDPreservedSymbols);

        {
          auto ErrOrBuffer = CacheEntry.tryLoadingBuffer();
          DEBUG(dbgs() << "Cache " << (ErrOrBuffer ? "hit" : "miss") << " '"
                       << CacheEntry.getEntryPath() << "' for buffer " << count
                       << " " << ModuleIdentifier << "\n");

          if (ErrOrBuffer) {
            // Cache Hit!
            ProducedBinaries[count] = std::move(ErrOrBuffer.get());
            return;
          }
        }

        LLVMContext Context;
        Context.setDiscardValueNames(LTODiscardValueNames);
        Context.enableDebugTypeODRUniquing();

        // Parse module now
        auto TheModule = loadModuleFromBuffer(ModuleBuffer, Context, false);

        // Save temps: original file.
        saveTempBitcode(*TheModule, SaveTempsDir, count, ".0.original.bc");

        auto &ImportList = ImportLists[ModuleIdentifier];
        // Run the main process now, and generates a binary
        auto OutputBuffer = ProcessThinLTOModule(
            *TheModule, *Index, ModuleMap, *TMBuilder.create(), ImportList,
            ExportList, GUIDPreservedSymbols, ResolvedODR, CacheOptions,
            DisableCodeGen, SaveTempsDir, count);

        CacheEntry.write(*OutputBuffer);
        ProducedBinaries[count] = std::move(OutputBuffer);
      }, count);
      count++;
    }
  }

  CachePruning(CacheOptions.Path)
      .setPruningInterval(CacheOptions.PruningInterval)
      .setEntryExpiration(CacheOptions.Expiration)
      .setMaxSize(CacheOptions.MaxPercentageOfAvailableSpace)
      .prune();

  // If statistics were requested, print them out now.
  if (llvm::AreStatisticsEnabled())
    llvm::PrintStatistics();
}
