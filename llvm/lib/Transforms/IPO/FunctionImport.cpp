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

#include <map>

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

namespace {
/// Helper to load on demand a Module from file and cache it for subsequent
/// queries. It can be used with the FunctionImporter.
class ModuleLazyLoaderCache {
  /// Cache of lazily loaded module for import.
  StringMap<std::unique_ptr<Module>> ModuleMap;

  /// Retrieve a Module from the cache or lazily load it on demand.
  std::function<std::unique_ptr<Module>(StringRef FileName)> createLazyModule;

public:
  /// Create the loader, Module will be initialized in \p Context.
  ModuleLazyLoaderCache(std::function<
      std::unique_ptr<Module>(StringRef FileName)> createLazyModule)
      : createLazyModule(createLazyModule) {}

  /// Retrieve a Module from the cache or lazily load it on demand.
  Module &operator()(StringRef FileName);
};

// Get a Module for \p FileName from the cache, or load it lazily.
Module &ModuleLazyLoaderCache::operator()(StringRef Identifier) {
  auto &Module = ModuleMap[Identifier];
  if (!Module)
    Module = createLazyModule(Identifier);
  return *Module;
}
} // anonymous namespace

/// Walk through the instructions in \p F looking for external
/// calls not already in the \p CalledFunctions set. If any are
/// found they are added to the \p Worklist for importing.
static void findExternalCalls(const Module &DestModule, Function &F,
                              const FunctionInfoIndex &Index,
                              StringSet<> &CalledFunctions,
                              SmallVector<StringRef, 64> &Worklist) {
  // We need to suffix internal function calls imported from other modules,
  // prepare the suffix ahead of time.
  std::string Suffix;
  if (F.getParent() != &DestModule)
    Suffix =
        (Twine(".llvm.") +
         Twine(Index.getModuleId(F.getParent()->getModuleIdentifier()))).str();

  for (auto &BB : F) {
    for (auto &I : BB) {
      if (isa<CallInst>(I)) {
        auto CalledFunction = cast<CallInst>(I).getCalledFunction();
        // Insert any new external calls that have not already been
        // added to set/worklist.
        if (!CalledFunction || !CalledFunction->hasName())
          continue;
        // Ignore intrinsics early
        if (CalledFunction->isIntrinsic()) {
          assert(CalledFunction->getIntrinsicID() != 0);
          continue;
        }
        auto ImportedName = CalledFunction->getName();
        auto Renamed = (ImportedName + Suffix).str();
        // Rename internal functions
        if (CalledFunction->hasInternalLinkage()) {
          ImportedName = Renamed;
        }
        auto It = CalledFunctions.insert(ImportedName);
        if (!It.second) {
          // This is a call to a function we already considered, skip.
          continue;
        }
        // Ignore functions already present in the destination module
        auto *SrcGV = DestModule.getNamedValue(ImportedName);
        if (SrcGV) {
          assert(isa<Function>(SrcGV) && "Name collision during import");
          if (!cast<Function>(SrcGV)->isDeclaration()) {
            DEBUG(dbgs() << DestModule.getModuleIdentifier() << ": Ignoring "
                         << ImportedName << " already in DestinationModule\n");
            continue;
          }
        }

        Worklist.push_back(It.first->getKey());
        DEBUG(dbgs() << DestModule.getModuleIdentifier()
                     << ": Adding callee for : " << ImportedName << " : "
                     << F.getName() << "\n");
      }
    }
  }
}

// Helper function: given a worklist and an index, will process all the worklist
// and decide what to import based on the summary information.
//
// Nothing is actually imported, functions are materialized in their source
// module and analyzed there.
//
// \p ModuleToFunctionsToImportMap is filled with the set of Function to import
// per Module.
static void GetImportList(
    Module &DestModule, SmallVector<StringRef, 64> &Worklist,
    StringSet<> &CalledFunctions,
    std::map<StringRef, std::pair<Module *, DenseSet<const GlobalValue *>>> &
        ModuleToFunctionsToImportMap,
    const FunctionInfoIndex &Index, ModuleLazyLoaderCache &ModuleLoaderCache) {
  while (!Worklist.empty()) {
    auto CalledFunctionName = Worklist.pop_back_val();
    DEBUG(dbgs() << DestModule.getModuleIdentifier() << ": Process import for "
                 << CalledFunctionName << "\n");

    // Try to get a summary for this function call.
    auto InfoList = Index.findFunctionInfoList(CalledFunctionName);
    if (InfoList == Index.end()) {
      DEBUG(dbgs() << DestModule.getModuleIdentifier() << ": No summary for "
                   << CalledFunctionName << " Ignoring.\n");
      continue;
    }
    assert(!InfoList->second.empty() && "No summary, error at import?");

    // Comdat can have multiple entries, FIXME: what do we do with them?
    auto &Info = InfoList->second[0];
    assert(Info && "Nullptr in list, error importing summaries?\n");

    auto *Summary = Info->functionSummary();
    if (!Summary) {
      // FIXME: in case we are lazyloading summaries, we can do it now.
      DEBUG(dbgs() << DestModule.getModuleIdentifier()
                   << ": Missing summary for  " << CalledFunctionName
                   << ", error at import?\n");
      llvm_unreachable("Missing summary");
    }

    if (Summary->instCount() > ImportInstrLimit) {
      DEBUG(dbgs() << DestModule.getModuleIdentifier() << ": Skip import of "
                   << CalledFunctionName << " with " << Summary->instCount()
                   << " instructions (limit " << ImportInstrLimit << ")\n");
      continue;
    }

    // Get the module path from the summary.
    auto ModuleIdentifier = Summary->modulePath();
    DEBUG(dbgs() << DestModule.getModuleIdentifier() << ": Importing "
                 << CalledFunctionName << " from " << ModuleIdentifier << "\n");

    auto &SrcModule = ModuleLoaderCache(ModuleIdentifier);

    // The function that we will import!
    GlobalValue *SGV = SrcModule.getNamedValue(CalledFunctionName);

    if (!SGV) {
      // The destination module is referencing function using their renamed name
      // when importing a function that was originally local in the source
      // module. The source module we have might not have been renamed so we try
      // to remove the suffix added during the renaming to recover the original
      // name in the source module.
      std::pair<StringRef, StringRef> Split =
          CalledFunctionName.split(".llvm.");
      SGV = SrcModule.getNamedValue(Split.first);
      assert(SGV && "Can't find function to import in source module");
    }
    if (!SGV) {
      report_fatal_error(Twine("Can't load function '") + CalledFunctionName +
                         "' in Module '" + SrcModule.getModuleIdentifier() +
                         "', error in the summary?\n");
    }

    Function *F = dyn_cast<Function>(SGV);
    if (!F && isa<GlobalAlias>(SGV)) {
      auto *SGA = dyn_cast<GlobalAlias>(SGV);
      F = dyn_cast<Function>(SGA->getBaseObject());
      CalledFunctionName = F->getName();
    }
    assert(F && "Imported Function is ... not a Function");

    // We cannot import weak_any functions/aliases without possibly affecting
    // the order they are seen and selected by the linker, changing program
    // semantics.
    if (SGV->hasWeakAnyLinkage()) {
      DEBUG(dbgs() << DestModule.getModuleIdentifier()
                   << ": Ignoring import request for weak-any "
                   << (isa<Function>(SGV) ? "function " : "alias ")
                   << CalledFunctionName << " from "
                   << SrcModule.getModuleIdentifier() << "\n");
      continue;
    }

    // Add the function to the import list
    auto &Entry = ModuleToFunctionsToImportMap[SrcModule.getModuleIdentifier()];
    Entry.first = &SrcModule;
    Entry.second.insert(F);

    // Process the newly imported functions and add callees to the worklist.
    F->materialize();
    findExternalCalls(DestModule, *F, Index, CalledFunctions, Worklist);
  }
}

// Automatically import functions in Module \p DestModule based on the summaries
// index.
//
// The current implementation imports every called functions that exists in the
// summaries index.
bool FunctionImporter::importFunctions(Module &DestModule) {
  DEBUG(dbgs() << "Starting import for Module "
               << DestModule.getModuleIdentifier() << "\n");
  unsigned ImportedCount = 0;

  /// First step is collecting the called external functions.
  StringSet<> CalledFunctions;
  SmallVector<StringRef, 64> Worklist;
  for (auto &F : DestModule) {
    if (F.isDeclaration() || F.hasFnAttribute(Attribute::OptimizeNone))
      continue;
    findExternalCalls(DestModule, F, Index, CalledFunctions, Worklist);
  }
  if (Worklist.empty())
    return false;

  /// Second step: for every call to an external function, try to import it.

  // Linker that will be used for importing function
  Linker TheLinker(DestModule, DiagnosticHandler);

  // Map of Module -> List of Function to import from the Module
  std::map<StringRef, std::pair<Module *, DenseSet<const GlobalValue *>>>
      ModuleToFunctionsToImportMap;

  // Analyze the summaries and get the list of functions to import by
  // populating ModuleToFunctionsToImportMap
  ModuleLazyLoaderCache ModuleLoaderCache(ModuleLoader);
  GetImportList(DestModule, Worklist, CalledFunctions,
                ModuleToFunctionsToImportMap, Index, ModuleLoaderCache);
  assert(Worklist.empty() && "Worklist hasn't been flushed in GetImportList");

  // Do the actual import of functions now, one Module at a time
  for (auto &FunctionsToImportPerModule : ModuleToFunctionsToImportMap) {
    // Get the module for the import
    auto &FunctionsToImport = FunctionsToImportPerModule.second.second;
    auto *SrcModule = FunctionsToImportPerModule.second.first;
    assert(&DestModule.getContext() == &SrcModule->getContext() &&
           "Context mismatch");

    // Link in the specified functions.
    if (TheLinker.linkInModule(*SrcModule, Linker::Flags::None, &Index,
                               &FunctionsToImport))
      report_fatal_error("Function Import: link error");

    ImportedCount += FunctionsToImport.size();
  }
  DEBUG(dbgs() << "Imported " << ImportedCount << " functions for Module "
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
  /// Optional function summary index to use for importing, otherwise
  /// the summary-file option must be specified.
  const FunctionInfoIndex *Index;

public:
  /// Pass identification, replacement for typeid
  static char ID;

  /// Specify pass name for debug output
  const char *getPassName() const override {
    return "Function Importing";
  }

  explicit FunctionImportPass(const FunctionInfoIndex *Index = nullptr)
      : ModulePass(ID), Index(Index) {}

  bool runOnModule(Module &M) override {
    if (SummaryFile.empty() && !Index)
      report_fatal_error("error: -function-import requires -summary-file or "
                         "file from frontend\n");
    std::unique_ptr<FunctionInfoIndex> IndexPtr;
    if (!SummaryFile.empty()) {
      if (Index)
        report_fatal_error("error: -summary-file and index from frontend\n");
      std::string Error;
      IndexPtr = getFunctionIndexForFile(SummaryFile, Error, diagnosticHandler);
      if (!IndexPtr) {
        errs() << "Error loading file '" << SummaryFile << "': " << Error
               << "\n";
        return false;
      }
      Index = IndexPtr.get();
    }

    // Perform the import now.
    auto ModuleLoader = [&M](StringRef Identifier) {
      return loadFile(Identifier, M.getContext());
    };
    FunctionImporter Importer(*Index, diagnosticHandler, ModuleLoader);
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
Pass *createFunctionImportPass(const FunctionInfoIndex *Index = nullptr) {
  return new FunctionImportPass(Index);
}
}
