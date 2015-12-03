//===- FunctionImport.cpp - ThinLTO Summary-based Function Import ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements Function import based on summaries.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/FunctionImport.h"

#include "llvm/ADT/StringSet.h"
#include "llvm/IR/AutoUpgrade.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Object/FunctionIndexObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SourceMgr.h"
using namespace llvm;

#define DEBUG_TYPE "function-import"

/// Limit on instruction count of imported functions.
static cl::opt<unsigned> ImportInstrLimit(
    "import-instr-limit", cl::init(100), cl::Hidden, cl::value_desc("N"),
    cl::desc("Only import functions with less than N instructions"));

// Load lazily a module from \p FileName in \p Context.
static std::unique_ptr<Module> loadFile(const std::string &FileName,
                                        LLVMContext &Context) {
  SMDiagnostic Err;
  DEBUG(dbgs() << "Loading '" << FileName << "'\n");
  std::unique_ptr<Module> Result = getLazyIRFileModule(FileName, Err, Context);
  if (!Result) {
    Err.print("function-import", errs());
    return nullptr;
  }

  Result->materializeMetadata();
  UpgradeDebugInfo(*Result);

  return Result;
}

// Get a Module for \p FileName from the cache, or load it lazily.
Module &ModuleLazyLoaderCache::operator()(StringRef FileName) {
  auto &Module = ModuleMap[FileName];
  if (!Module)
    Module = loadFile(FileName, Context);
  return *Module;
}

/// Walk through the instructions in \p F looking for external
/// calls not already in the \p CalledFunctions set. If any are
/// found they are added to the \p Worklist for importing.
static void findExternalCalls(const Function &F, StringSet<> &CalledFunctions,
                              SmallVector<StringRef, 64> &Worklist) {
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (isa<CallInst>(I)) {
        auto CalledFunction = cast<CallInst>(I).getCalledFunction();
        // Insert any new external calls that have not already been
        // added to set/worklist.
        if (CalledFunction && CalledFunction->hasName() &&
            CalledFunction->isDeclaration() &&
            !CalledFunctions.count(CalledFunction->getName())) {
          CalledFunctions.insert(CalledFunction->getName());
          Worklist.push_back(CalledFunction->getName());
        }
      }
    }
  }
}

// Helper function: given a worklist and an index, will process all the worklist
// and import them based on the summary information
static unsigned ProcessImportWorklist(
    Module &DestModule, SmallVector<StringRef, 64> &Worklist,
    StringSet<> &CalledFunctions, Linker &TheLinker,
    const FunctionInfoIndex &Index,
    std::function<Module &(StringRef FileName)> &LazyModuleLoader) {
  unsigned ImportCount = 0;
  while (!Worklist.empty()) {
    auto CalledFunctionName = Worklist.pop_back_val();
    DEBUG(dbgs() << "Process import for " << CalledFunctionName << "\n");

    // Try to get a summary for this function call.
    auto InfoList = Index.findFunctionInfoList(CalledFunctionName);
    if (InfoList == Index.end()) {
      DEBUG(dbgs() << "No summary for " << CalledFunctionName
                   << " Ignoring.\n");
      continue;
    }
    assert(!InfoList->second.empty() && "No summary, error at import?");

    // Comdat can have multiple entries, FIXME: what do we do with them?
    auto &Info = InfoList->second[0];
    assert(Info && "Nullptr in list, error importing summaries?\n");

    auto *Summary = Info->functionSummary();
    if (!Summary) {
      // FIXME: in case we are lazyloading summaries, we can do it now.
      DEBUG(dbgs() << "Missing summary for  " << CalledFunctionName
                   << ", error at import?\n");
      llvm_unreachable("Missing summary");
    }

    if (Summary->instCount() > ImportInstrLimit) {
      DEBUG(dbgs() << "Skip import of " << CalledFunctionName << " with "
                   << Summary->instCount() << " instructions (limit "
                   << ImportInstrLimit << ")\n");
      continue;
    }

    // Get the module path from the summary.
    auto FileName = Summary->modulePath();
    DEBUG(dbgs() << "Importing " << CalledFunctionName << " from " << FileName
                 << "\n");

    // Get the module for the import (potentially from the cache).
    auto &Module = LazyModuleLoader(FileName);
    assert(&Module.getContext() == &DestModule.getContext());

    // The function that we will import!
    GlobalValue *SGV = Module.getNamedValue(CalledFunctionName);
    StringRef ImportFunctionName = CalledFunctionName;
    if (!SGV) {
      // Might be local in source Module, promoted/renamed in DestModule.
      std::pair<StringRef, StringRef> Split =
          CalledFunctionName.split(".llvm.");
      SGV = Module.getNamedValue(Split.first);
#ifndef NDEBUG
      // Assert that Split.second is module id
      uint64_t ModuleId;
      assert(!Split.second.getAsInteger(10, ModuleId));
      assert(ModuleId == Index.getModuleId(FileName));
#endif
    }
    Function *F = dyn_cast<Function>(SGV);
    if (!F && isa<GlobalAlias>(SGV)) {
      auto *SGA = dyn_cast<GlobalAlias>(SGV);
      F = dyn_cast<Function>(SGA->getBaseObject());
      ImportFunctionName = F->getName();
    }
    if (!F) {
      errs() << "Can't load function '" << CalledFunctionName << "' in Module '"
             << FileName << "', error in the summary?\n";
      llvm_unreachable("Can't load function in Module");
    }

    // We cannot import weak_any functions/aliases without possibly affecting
    // the order they are seen and selected by the linker, changing program
    // semantics.
    if (SGV->hasWeakAnyLinkage()) {
      DEBUG(dbgs() << "Ignoring import request for weak-any "
                   << (isa<Function>(SGV) ? "function " : "alias ")
                   << CalledFunctionName << " from " << FileName << "\n");
      continue;
    }

    // Link in the specified function.
    DenseSet<const GlobalValue *> FunctionsToImport;
    FunctionsToImport.insert(F);
    if (TheLinker.linkInModule(Module, Linker::Flags::None, &Index,
                               &FunctionsToImport))
      report_fatal_error("Function Import: link error");

    // Process the newly imported function and add callees to the worklist.
    GlobalValue *NewGV = DestModule.getNamedValue(ImportFunctionName);
    assert(NewGV);
    Function *NewF = dyn_cast<Function>(NewGV);
    assert(NewF);
    findExternalCalls(*NewF, CalledFunctions, Worklist);
    ++ImportCount;
  }
  return ImportCount;
}

// Automatically import functions in Module \p DestModule based on the summaries
// index.
//
// The current implementation imports every called functions that exists in the
// summaries index.
bool FunctionImporter::importFunctions(Module &DestModule) {
  DEBUG(errs() << "Starting import for Module "
               << DestModule.getModuleIdentifier() << "\n");
  unsigned ImportedCount = 0;

  /// First step is collecting the called external functions.
  StringSet<> CalledFunctions;
  SmallVector<StringRef, 64> Worklist;
  for (auto &F : DestModule) {
    if (F.isDeclaration() || F.hasFnAttribute(Attribute::OptimizeNone))
      continue;
    findExternalCalls(F, CalledFunctions, Worklist);
  }
  if (Worklist.empty())
    return false;

  /// Second step: for every call to an external function, try to import it.

  // Linker that will be used for importing function
  Linker TheLinker(DestModule, DiagnosticHandler);

  ImportedCount += ProcessImportWorklist(DestModule, Worklist, CalledFunctions,
                                         TheLinker, Index, getLazyModule);

  DEBUG(errs() << "Imported " << ImportedCount << " functions for Module "
               << DestModule.getModuleIdentifier() << "\n");
  return ImportedCount;
}

/// Summary file to use for function importing when using -function-import from
/// the command line.
static cl::opt<std::string>
    SummaryFile("summary-file",
                cl::desc("The summary file to use for function importing."));

static void diagnosticHandler(const DiagnosticInfo &DI) {
  raw_ostream &OS = errs();
  DiagnosticPrinterRawOStream DP(OS);
  DI.print(DP);
  OS << '\n';
}

/// Parse the function index out of an IR file and return the function
/// index object if found, or nullptr if not.
static std::unique_ptr<FunctionInfoIndex>
getFunctionIndexForFile(StringRef Path, std::string &Error,
                        DiagnosticHandlerFunction DiagnosticHandler) {
  std::unique_ptr<MemoryBuffer> Buffer;
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
      MemoryBuffer::getFile(Path);
  if (std::error_code EC = BufferOrErr.getError()) {
    Error = EC.message();
    return nullptr;
  }
  Buffer = std::move(BufferOrErr.get());
  ErrorOr<std::unique_ptr<object::FunctionIndexObjectFile>> ObjOrErr =
      object::FunctionIndexObjectFile::create(Buffer->getMemBufferRef(),
                                              DiagnosticHandler);
  if (std::error_code EC = ObjOrErr.getError()) {
    Error = EC.message();
    return nullptr;
  }
  return (*ObjOrErr)->takeIndex();
}

/// Pass that performs cross-module function import provided a summary file.
class FunctionImportPass : public ModulePass {

public:
  /// Pass identification, replacement for typeid
  static char ID;

  explicit FunctionImportPass() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    if (SummaryFile.empty()) {
      report_fatal_error("error: -function-import requires -summary-file\n");
    }
    std::string Error;
    std::unique_ptr<FunctionInfoIndex> Index =
        getFunctionIndexForFile(SummaryFile, Error, diagnosticHandler);
    if (!Index) {
      errs() << "Error loading file '" << SummaryFile << "': " << Error << "\n";
      return false;
    }

    // Perform the import now.
    ModuleLazyLoaderCache Loader(M.getContext());
    FunctionImporter Importer(*Index, diagnosticHandler,
                              [&](StringRef Name)
                                  -> Module &{ return Loader(Name); });
    return Importer.importFunctions(M);

    return false;
  }
};

char FunctionImportPass::ID = 0;
INITIALIZE_PASS_BEGIN(FunctionImportPass, "function-import",
                      "Summary Based Function Import", false, false)
INITIALIZE_PASS_END(FunctionImportPass, "function-import",
                    "Summary Based Function Import", false, false)

namespace llvm {
Pass *createFunctionImportPass() { return new FunctionImportPass(); }
}
