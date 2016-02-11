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
#include "llvm/Transforms/Utils/FunctionImportUtils.h"

#include <map>

using namespace llvm;

#define DEBUG_TYPE "function-import"

/// Limit on instruction count of imported functions.
static cl::opt<unsigned> ImportInstrLimit(
    "import-instr-limit", cl::init(100), cl::Hidden, cl::value_desc("N"),
    cl::desc("Only import functions with less than N instructions"));

static cl::opt<float>
    ImportInstrFactor("import-instr-evolution-factor", cl::init(0.7),
                      cl::Hidden, cl::value_desc("x"),
                      cl::desc("As we import functions, multiply the "
                               "`import-instr-limit` threshold by this factor "
                               "before processing newly imported functions"));

// Load lazily a module from \p FileName in \p Context.
static std::unique_ptr<Module> loadFile(const std::string &FileName,
                                        LLVMContext &Context) {
  SMDiagnostic Err;
  DEBUG(dbgs() << "Loading '" << FileName << "'\n");
  // Metadata isn't loaded until functions are imported, to minimize
  // the memory overhead.
  std::unique_ptr<Module> Result =
      getLazyIRFileModule(FileName, Err, Context,
                          /* ShouldLazyLoadMetadata = */ true);
  if (!Result) {
    Err.print("function-import", errs());
    return nullptr;
  }

  return Result;
}

namespace {

/// Track functions already seen using a map that record the current
/// Threshold and the importing decision. Since the traversal of the call graph
/// is DFS, we can revisit a function a second time with a higher threshold. In
/// this case and if the function was not imported the first time, it is added
/// back to the worklist with the new threshold
using VisitedFunctionTrackerTy = StringMap<std::pair<unsigned, bool>>;

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

  std::unique_ptr<Module> takeModule(StringRef FileName) {
    auto I = ModuleMap.find(FileName);
    assert(I != ModuleMap.end());
    std::unique_ptr<Module> Ret = std::move(I->second);
    ModuleMap.erase(I);
    return Ret;
  }
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
/// calls not already in the \p VisitedFunctions map. If any are
/// found they are added to the \p Worklist for importing.
static void findExternalCalls(
    const Module &DestModule, Function &F, const FunctionInfoIndex &Index,
    VisitedFunctionTrackerTy &VisitedFunctions, unsigned Threshold,
    SmallVectorImpl<std::pair<StringRef, unsigned>> &Worklist) {
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
        // Compute the global identifier used in the function index.
        auto CalledFunctionGlobalID = Function::getGlobalIdentifier(
            CalledFunction->getName(), CalledFunction->getLinkage(),
            CalledFunction->getParent()->getSourceFileName());

        auto CalledFunctionInfo = std::make_pair(Threshold, false);
        auto It = VisitedFunctions.insert(
            std::make_pair(CalledFunctionGlobalID, CalledFunctionInfo));
        if (!It.second) {
          // This is a call to a function we already considered, if the function
          // has been imported the first time, or if the current threshold is
          // not higher, skip it.
          auto &FunctionInfo = It.first->second;
          if (FunctionInfo.second || FunctionInfo.first >= Threshold)
            continue;
          It.first->second = CalledFunctionInfo;
        }
        // Ignore functions already present in the destination module
        auto *SrcGV = DestModule.getNamedValue(ImportedName);
        if (SrcGV) {
          if (GlobalAlias *SGA = dyn_cast<GlobalAlias>(SrcGV))
            SrcGV = SGA->getBaseObject();
          assert(isa<Function>(SrcGV) && "Name collision during import");
          if (!cast<Function>(SrcGV)->isDeclaration()) {
            DEBUG(dbgs() << DestModule.getModuleIdentifier() << ": Ignoring "
                         << ImportedName << " already in DestinationModule\n");
            continue;
          }
        }

        Worklist.push_back(std::make_pair(It.first->getKey(), Threshold));
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
static void
GetImportList(Module &DestModule,
              SmallVectorImpl<std::pair<StringRef, unsigned>> &Worklist,
              VisitedFunctionTrackerTy &VisitedFunctions,
              std::map<StringRef, DenseSet<const GlobalValue *>> &
                  ModuleToFunctionsToImportMap,
              const FunctionInfoIndex &Index,
              ModuleLazyLoaderCache &ModuleLoaderCache) {
  while (!Worklist.empty()) {
    StringRef CalledFunctionName;
    unsigned Threshold;
    std::tie(CalledFunctionName, Threshold) = Worklist.pop_back_val();
    DEBUG(dbgs() << DestModule.getModuleIdentifier() << ": Process import for "
                 << CalledFunctionName << " with Threshold " << Threshold
                 << "\n");

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

    if (Summary->instCount() > Threshold) {
      DEBUG(dbgs() << DestModule.getModuleIdentifier() << ": Skip import of "
                   << CalledFunctionName << " with " << Summary->instCount()
                   << " instructions (limit " << Threshold << ")\n");
      continue;
    }

    // Mark the function as imported in the VisitedFunctions tracker
    assert(VisitedFunctions.count(CalledFunctionName));
    VisitedFunctions[CalledFunctionName].second = true;

    // Get the module path from the summary.
    auto ModuleIdentifier = Summary->modulePath();
    DEBUG(dbgs() << DestModule.getModuleIdentifier() << ": Importing "
                 << CalledFunctionName << " from " << ModuleIdentifier << "\n");

    auto &SrcModule = ModuleLoaderCache(ModuleIdentifier);

    // The function that we will import!
    GlobalValue *SGV = SrcModule.getNamedValue(CalledFunctionName);

    if (!SGV) {
      // The function is referenced by a global identifier, which has the
      // source file name prepended for functions that were originally local
      // in the source module. Strip any prepended name to recover the original
      // name in the source module.
      std::pair<StringRef, StringRef> Split = CalledFunctionName.rsplit(':');
      SGV = SrcModule.getNamedValue(Split.second);
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
    Entry.insert(F);

    // Process the newly imported functions and add callees to the worklist.
    // Adjust the threshold
    Threshold = Threshold * ImportInstrFactor;
    F->materialize();
    findExternalCalls(DestModule, *F, Index, VisitedFunctions, Threshold,
                      Worklist);
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

  // First step is collecting the called external functions.
  // We keep the function name as well as the import threshold for its callees.
  VisitedFunctionTrackerTy VisitedFunctions;
  SmallVector<std::pair<StringRef, unsigned>, 64> Worklist;
  for (auto &F : DestModule) {
    if (F.isDeclaration() || F.hasFnAttribute(Attribute::OptimizeNone))
      continue;
    findExternalCalls(DestModule, F, Index, VisitedFunctions, ImportInstrLimit,
                      Worklist);
  }
  if (Worklist.empty())
    return false;

  /// Second step: for every call to an external function, try to import it.

  // Linker that will be used for importing function
  Linker TheLinker(DestModule);

  // Map of Module -> List of Function to import from the Module
  std::map<StringRef, DenseSet<const GlobalValue *>>
      ModuleToFunctionsToImportMap;

  // Analyze the summaries and get the list of functions to import by
  // populating ModuleToFunctionsToImportMap
  ModuleLazyLoaderCache ModuleLoaderCache(ModuleLoader);
  GetImportList(DestModule, Worklist, VisitedFunctions,
                ModuleToFunctionsToImportMap, Index, ModuleLoaderCache);
  assert(Worklist.empty() && "Worklist hasn't been flushed in GetImportList");

  // Do the actual import of functions now, one Module at a time
  for (auto &FunctionsToImportPerModule : ModuleToFunctionsToImportMap) {
    // Get the module for the import
    auto &FunctionsToImport = FunctionsToImportPerModule.second;
    std::unique_ptr<Module> SrcModule =
        ModuleLoaderCache.takeModule(FunctionsToImportPerModule.first);
    assert(&DestModule.getContext() == &SrcModule->getContext() &&
           "Context mismatch");

    // If modules were created with lazy metadata loading, materialize it
    // now, before linking it (otherwise this will be a noop).
    SrcModule->materializeMetadata();
    UpgradeDebugInfo(*SrcModule);

    // Link in the specified functions.
    if (TheLinker.linkInModule(std::move(SrcModule), Linker::Flags::None,
                               &Index, &FunctionsToImport))
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

namespace {
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

    // First we need to promote to global scope and rename any local values that
    // are potentially exported to other modules.
    if (renameModuleForThinLTO(M, Index)) {
      errs() << "Error renaming module\n";
      return false;
    }

    // Perform the import now.
    auto ModuleLoader = [&M](StringRef Identifier) {
      return loadFile(Identifier, M.getContext());
    };
    FunctionImporter Importer(*Index, ModuleLoader);
    return Importer.importFunctions(M);
  }
};
} // anonymous namespace

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
