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

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/ExecutionEngine/ObjectMemoryBuffer.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Object/FunctionIndexObjectFile.h"
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
                             /* ShouldLazyLoadMetadata */ false);
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
  StringMap<MemoryBufferRef> &ModuleMap;

public:
  ModuleLoader(LLVMContext &Context, StringMap<MemoryBufferRef> &ModuleMap)
      : Context(Context), ModuleMap(ModuleMap) {}

  /// Load a module on demand.
  std::unique_ptr<Module> operator()(StringRef Identifier) {
    return loadModuleFromBuffer(ModuleMap[Identifier], Context, /*Lazy*/ true);
  }
};

static void promoteModule(Module &TheModule, const FunctionInfoIndex &Index) {
  if (renameModuleForThinLTO(TheModule, &Index))
    report_fatal_error("renameModuleForThinLTO failed");
}

static void crossImportIntoModule(Module &TheModule,
                                 const FunctionInfoIndex &Index,
                                 StringMap<MemoryBufferRef> &ModuleMap) {
  ModuleLoader Loader(TheModule.getContext(), ModuleMap);
  FunctionImporter Importer(Index, Loader);
  Importer.importFunctions(TheModule);
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
  PM.add(createObjCARCContractPass());

  PM.run(TheModule);
}

std::unique_ptr<MemoryBuffer> codegenModule(Module &TheModule,
                                            TargetMachine &TM) {
  SmallVector<char, 128> OutputBuffer;

  // CodeGen
  {
    raw_svector_ostream OS(OutputBuffer);
    legacy::PassManager PM;
    if (TM.addPassesToEmitFile(PM, OS, TargetMachine::CGFT_ObjectFile,
                               /* DisableVerify */ true))
      report_fatal_error("Failed to setup codegen");

    // Run codegen now. resulting binary is in OutputBuffer.
    PM.run(TheModule);
  }
  return make_unique<ObjectMemoryBuffer>(std::move(OutputBuffer));
}

static std::unique_ptr<MemoryBuffer>
ProcessThinLTOModule(Module &TheModule, const FunctionInfoIndex &Index,
                     StringMap<MemoryBufferRef> &ModuleMap, TargetMachine &TM,
                     CachingOptions CacheOptions, StringRef SaveTempsDir,
                     unsigned count) {

  // Save temps: after IPO.
  saveTempBitcode(TheModule, SaveTempsDir, count, ".1.IPO.bc");

  // "Benchmark"-like optimization: single-source case
  bool SingleModule = (ModuleMap.size() == 1);

  if (!SingleModule) {
    promoteModule(TheModule, Index);

    // Save temps: after promotion.
    saveTempBitcode(TheModule, SaveTempsDir, count, ".2.promoted.bc");

    crossImportIntoModule(TheModule, Index, ModuleMap);

    // Save temps: after cross-module import.
    saveTempBitcode(TheModule, SaveTempsDir, count, ".3.imported.bc");
  }

  optimizeModule(TheModule, TM);

  saveTempBitcode(TheModule, SaveTempsDir, count, ".3.opt.bc");

  return codegenModule(TheModule, TM);
}

} // end anonymous namespace

void ThinLTOCodeGenerator::addModule(StringRef Identifier, StringRef Data) {
  MemoryBufferRef Buffer(Data, Identifier);
  if (Modules.empty()) {
    // First module added, so initialize the triple and some options
    LLVMContext Context;
    Triple TheTriple(getBitcodeTargetTriple(Buffer, Context));

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
    report_fatal_error("Can't load target for this Triple");
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
 * Produce the combined function index from all the bitcode files:
 * "thin-link".
 */
std::unique_ptr<FunctionInfoIndex> ThinLTOCodeGenerator::linkCombinedIndex() {
  std::unique_ptr<FunctionInfoIndex> CombinedIndex;
  uint64_t NextModuleId = 0;
  for (auto &ModuleBuffer : Modules) {
    ErrorOr<std::unique_ptr<object::FunctionIndexObjectFile>> ObjOrErr =
        object::FunctionIndexObjectFile::create(ModuleBuffer, diagnosticHandler,
                                                false);
    if (std::error_code EC = ObjOrErr.getError()) {
      // FIXME diagnose
      errs() << "error: can't create FunctionIndexObjectFile for buffer: "
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
                                   FunctionInfoIndex &Index) {
  promoteModule(TheModule, Index);
}

/**
 * Perform cross-module importing for the module identified by ModuleIdentifier.
 */
void ThinLTOCodeGenerator::crossModuleImport(Module &TheModule,
                                             FunctionInfoIndex &Index) {
  auto ModuleMap = generateModuleMap(Modules);
  crossImportIntoModule(TheModule, Index, ModuleMap);
}

/**
 * Perform post-importing ThinLTO optimizations.
 */
void ThinLTOCodeGenerator::optimize(Module &TheModule) {
  optimizeModule(TheModule, *TMBuilder.create());
}

/**
 * Perform ThinLTO CodeGen.
 */
std::unique_ptr<MemoryBuffer> ThinLTOCodeGenerator::codegen(Module &TheModule) {
  return codegenModule(TheModule, *TMBuilder.create());
}

// Main entry point for the ThinLTO processing
void ThinLTOCodeGenerator::run() {
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
    WriteFunctionSummaryToFile(*Index, OS);
  }

  // Prepare the resulting object vector
  assert(ProducedBinaries.empty() && "The generator should not be reused");
  ProducedBinaries.resize(Modules.size());

  // Prepare the module map.
  auto ModuleMap = generateModuleMap(Modules);

  // Parallel optimizer + codegen
  {
    ThreadPool Pool(ThreadCount);
    int count = 0;
    for (auto &ModuleBuffer : Modules) {
      Pool.async([&](int count) {
        LLVMContext Context;

        // Save temps: original file.
        if (!SaveTempsDir.empty()) {
          auto TheModule = loadModuleFromBuffer(ModuleBuffer, Context, false);
          saveTempBitcode(*TheModule, SaveTempsDir, count, ".0.original.bc");
        }

        // Parse module now
        auto TheModule =
            loadModuleFromBuffer(ModuleMap[ModuleBuffer.getBufferIdentifier()],
                                 Context, /*Lazy*/ false);

        ProducedBinaries[count] = ProcessThinLTOModule(
            *TheModule, *Index, ModuleMap, *TMBuilder.create(), CacheOptions,
            SaveTempsDir, count);
      }, count);
      count++;
    }
  }

  // If statistics were requested, print them out now.
  if (llvm::AreStatisticsEnabled())
    llvm::PrintStatistics();
}
