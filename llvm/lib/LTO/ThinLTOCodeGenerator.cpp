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

#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
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
#include "llvm/Object/ModuleSummaryIndexObjectFile.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/FunctionImport.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/ObjCARC.h"
#include "llvm/Transforms/Utils/FunctionImportUtils.h"

using namespace llvm;

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
  WriteBitcodeToFile(&TheModule, OS, true, false);
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

static void ResolveODR(GlobalValue &GV, const ModuleSummaryIndex &Index,
                             StringRef ModulePath) {
  if (GV.isDeclaration())
    return;

  auto HasMultipleCopies =
      [&](const GlobalValueInfoList &GVInfo) { return GVInfo.size() > 1; };

  auto getGVInfo = [&](GlobalValue &GV) -> const GlobalValueInfoList *{
    auto GUID = Function::getGlobalIdentifier(GV.getName(), GV.getLinkage(),
                                              ModulePath);
    auto It = Index.findGlobalValueInfoList(GV.getName());
    if (It == Index.end())
      return nullptr;
    return &It->second;
  };

  switch (GV.getLinkage()) {
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
    auto *GVInfo = getGVInfo(GV);
    if (!GVInfo)
      break;
    // We need to emit only one of these, the first module will keep
    // it, but turned into a weak while the others will drop it.
    if (!HasMultipleCopies(*GVInfo))
      break;
    if (IsFirstDefinitionForLinker(*GVInfo, Index, ModulePath))
      GV.setLinkage(GlobalValue::WeakODRLinkage);
    else
      GV.setLinkage(GlobalValue::AvailableExternallyLinkage);
    break;
  }
  }
}

/// Resolve LinkOnceODR and WeakODR.
///
/// We'd like to drop these function if they are no longer referenced in the
/// current module. However there is a chance that another module is still
/// referencing them because of the import. We make sure we always emit at least
/// one copy.
static void ResolveODR(Module &TheModule,
                             const ModuleSummaryIndex &Index) {
  // We won't optimize the globals that are referenced by an alias for now
  // Ideally we should turn the alias into a global and duplicate the definition
  // when needed.
  DenseSet<GlobalValue *> GlobalInvolvedWithAlias;
  for (auto &GA : TheModule.aliases()) {
    auto *GO = GA.getBaseObject();
    if (auto *GV = dyn_cast<GlobalValue>(GO))
      GlobalInvolvedWithAlias.insert(GV);
  }
  // Process functions and global now
  for (auto &GV : TheModule) {
    if (!GlobalInvolvedWithAlias.count(&GV))
      ResolveODR(GV, Index, TheModule.getModuleIdentifier());
  }
  for (auto &GV : TheModule.globals()) {
    if (!GlobalInvolvedWithAlias.count(&GV))
      ResolveODR(GV, Index, TheModule.getModuleIdentifier());
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

static std::unique_ptr<MemoryBuffer>
ProcessThinLTOModule(Module &TheModule, const ModuleSummaryIndex &Index,
                     StringMap<MemoryBufferRef> &ModuleMap, TargetMachine &TM,
                     const FunctionImporter::ImportMapTy &ImportList,
                     ThinLTOCodeGenerator::CachingOptions CacheOptions,
                     bool DisableCodeGen, StringRef SaveTempsDir,
                     unsigned count) {

  // Save temps: after IPO.
  saveTempBitcode(TheModule, SaveTempsDir, count, ".1.IPO.bc");

  // "Benchmark"-like optimization: single-source case
  bool SingleModule = (ModuleMap.size() == 1);

  if (!SingleModule) {
    promoteModule(TheModule, Index);

    // Resolve the LinkOnce/Weak ODR, trying to turn them into
    // "available_externally" when possible.
    // This is a compile-time optimization.
    ResolveODR(TheModule, Index);

    // Save temps: after promotion.
    saveTempBitcode(TheModule, SaveTempsDir, count, ".2.promoted.bc");

    crossImportIntoModule(TheModule, Index, ModuleMap, ImportList);

    // Save temps: after cross-module import.
    saveTempBitcode(TheModule, SaveTempsDir, count, ".3.imported.bc");
  }

  optimizeModule(TheModule, TM);

  saveTempBitcode(TheModule, SaveTempsDir, count, ".3.opt.bc");

  if (DisableCodeGen) {
    // Configured to stop before CodeGen, serialize the bitcode and return.
    SmallVector<char, 128> OutputBuffer;
    {
      raw_svector_ostream OS(OutputBuffer);
      WriteBitcodeToFile(&TheModule, OS, true, true);
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
  CrossReferencedSymbols.insert(Name);
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
                                                     diagnosticHandler, false);
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

  // Resolve the LinkOnceODR, trying to turn them into "available_externally"
  // where possible.
  ResolveODR(TheModule, Index);

  promoteModule(TheModule, Index);
}

/**
 * Perform cross-module importing for the module identified by ModuleIdentifier.
 */
void ThinLTOCodeGenerator::crossModuleImport(Module &TheModule,
                                             ModuleSummaryIndex &Index) {
  auto ModuleMap = generateModuleMap(Modules);

  // Generate import/export list
  auto ModuleCount = Index.modulePaths().size();
  StringMap<FunctionImporter::ImportMapTy> ImportLists(ModuleCount);
  StringMap<FunctionImporter::ExportSetTy> ExportLists(ModuleCount);
  ComputeCrossModuleImport(Index, ImportLists, ExportLists);
  auto &ImportList = ImportLists[TheModule.getModuleIdentifier()];

  crossImportIntoModule(TheModule, Index, ModuleMap, ImportList);
}

/**
 * Perform post-importing ThinLTO optimizations.
 */
void ThinLTOCodeGenerator::optimize(Module &TheModule) {
  initTMBuilder(TMBuilder, Triple(TheModule.getTargetTriple()));
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

  // Collect the import/export lists for all modules from the call-graph in the
  // combined index.
  StringMap<FunctionImporter::ImportMapTy> ImportLists(ModuleCount);
  StringMap<FunctionImporter::ExportSetTy> ExportLists(ModuleCount);
  ComputeCrossModuleImport(*Index, ImportLists, ExportLists);

  // Parallel optimizer + codegen
  {
    ThreadPool Pool(ThreadCount);
    int count = 0;
    for (auto &ModuleBuffer : Modules) {
      Pool.async([&](int count) {
        LLVMContext Context;
        Context.setDiscardValueNames(LTODiscardValueNames);

        // Parse module now
        auto TheModule = loadModuleFromBuffer(ModuleBuffer, Context, false);

        // Save temps: original file.
        if (!SaveTempsDir.empty()) {
          saveTempBitcode(*TheModule, SaveTempsDir, count, ".0.original.bc");
        }

        auto &ImportList = ImportLists[TheModule->getModuleIdentifier()];
        ProducedBinaries[count] = ProcessThinLTOModule(
            *TheModule, *Index, ModuleMap, *TMBuilder.create(), ImportList,
            CacheOptions, DisableCodeGen, SaveTempsDir, count);
      }, count);
      count++;
    }
  }

  // If statistics were requested, print them out now.
  if (llvm::AreStatisticsEnabled())
    llvm::PrintStatistics();
}
