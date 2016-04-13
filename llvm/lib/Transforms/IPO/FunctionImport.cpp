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

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IR/AutoUpgrade.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Object/ModuleSummaryIndexObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Utils/FunctionImportUtils.h"

#define DEBUG_TYPE "function-import"

using namespace llvm;

STATISTIC(NumImported, "Number of functions imported");

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

static cl::opt<bool> PrintImports("print-imports", cl::init(false), cl::Hidden,
                                  cl::desc("Print imported functions"));

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
    report_fatal_error("Abort");
  }

  return Result;
}

namespace {

/// Given a list of possible callee implementation for a call site, select one
/// that fits the \p Threshold.
///
/// FIXME: select "best" instead of first that fits. But what is "best"?
/// - The smallest: more likely to be inlined.
/// - The one with the least outgoing edges (already well optimized).
/// - One from a module already being imported from in order to reduce the
///   number of source modules parsed/linked.
/// - One that has PGO data attached.
/// - [insert you fancy metric here]
static const FunctionSummary *
selectCallee(const GlobalValueInfoList &CalleeInfoList, unsigned Threshold) {
  auto It = llvm::find_if(
      CalleeInfoList, [&](const std::unique_ptr<GlobalValueInfo> &GlobInfo) {
        assert(GlobInfo->summary() &&
               "We should not have a Global Info without summary");
        auto *Summary = cast<FunctionSummary>(GlobInfo->summary());

        if (GlobalValue::isWeakAnyLinkage(Summary->linkage()))
          return false;

        if (Summary->instCount() > Threshold)
          return false;

        return true;
      });
  if (It == CalleeInfoList.end())
    return nullptr;

  return cast<FunctionSummary>((*It)->summary());
}

/// Return the summary for the function \p GUID that fits the \p Threshold, or
/// null if there's no match.
static const FunctionSummary *selectCallee(GlobalValue::GUID GUID,
                                           unsigned Threshold,
                                           const ModuleSummaryIndex &Index) {
  auto CalleeInfoList = Index.findGlobalValueInfoList(GUID);
  if (CalleeInfoList == Index.end()) {
    return nullptr; // This function does not have a summary
  }
  return selectCallee(CalleeInfoList->second, Threshold);
}

/// Return true if the global \p GUID is exported by module \p ExportModulePath.
static bool isGlobalExported(const ModuleSummaryIndex &Index,
                             StringRef ExportModulePath,
                             GlobalValue::GUID GUID) {
  auto CalleeInfoList = Index.findGlobalValueInfoList(GUID);
  if (CalleeInfoList == Index.end())
    // This global does not have a summary, it is not part of the ThinLTO
    // process
    return false;
  auto DefinedInCalleeModule = llvm::find_if(
      CalleeInfoList->second,
      [&](const std::unique_ptr<GlobalValueInfo> &GlobInfo) {
        auto *Summary = GlobInfo->summary();
        assert(Summary && "Unexpected GlobalValueInfo without summary");
        return Summary->modulePath() == ExportModulePath;
      });
  return (DefinedInCalleeModule != CalleeInfoList->second.end());
}

using EdgeInfo = std::pair<const FunctionSummary *, unsigned /* Threshold */>;

/// Compute the list of functions to import for a given caller. Mark these
/// imported functions and the symbols they reference in their source module as
/// exported from their source module.
static void computeImportForFunction(
    StringRef ModulePath, const FunctionSummary &Summary,
    const ModuleSummaryIndex &Index, unsigned Threshold,
    const std::map<GlobalValue::GUID, FunctionSummary *> &DefinedFunctions,
    SmallVectorImpl<EdgeInfo> &Worklist,
    FunctionImporter::ImportMapTy &ImportsForModule,
    StringMap<FunctionImporter::ExportSetTy> &ExportLists) {
  for (auto &Edge : Summary.calls()) {
    auto GUID = Edge.first;
    DEBUG(dbgs() << " edge -> " << GUID << " Threshold:" << Threshold << "\n");

    if (DefinedFunctions.count(GUID)) {
      DEBUG(dbgs() << "ignored! Target already in destination module.\n");
      continue;
    }

    auto *CalleeSummary = selectCallee(GUID, Threshold, Index);
    if (!CalleeSummary) {
      DEBUG(dbgs() << "ignored! No qualifying callee with summary found.\n");
      continue;
    }
    assert(CalleeSummary->instCount() <= Threshold &&
           "selectCallee() didn't honor the threshold");

    auto &ProcessedThreshold =
        ImportsForModule[CalleeSummary->modulePath()][GUID];
    /// Since the traversal of the call graph is DFS, we can revisit a function
    /// a second time with a higher threshold. In this case, it is added back to
    /// the worklist with the new threshold.
    if (ProcessedThreshold && ProcessedThreshold > Threshold) {
      DEBUG(dbgs() << "ignored! Target was already seen with Threshold "
                   << ProcessedThreshold << "\n");
      continue;
    }
    // Mark this function as imported in this module, with the current Threshold
    ProcessedThreshold = Threshold;

    // Make exports in the source module.
    auto ExportModulePath = CalleeSummary->modulePath();
    auto ExportList = ExportLists[ExportModulePath];
    ExportList.insert(GUID);
    // Mark all functions and globals referenced by this function as exported to
    // the outside if they are defined in the same source module.
    for (auto &Edge : CalleeSummary->calls()) {
      auto CalleeGUID = Edge.first;
      if (isGlobalExported(Index, ExportModulePath, CalleeGUID))
        ExportList.insert(CalleeGUID);
    }
    for (auto &GUID : CalleeSummary->refs()) {
      if (isGlobalExported(Index, ExportModulePath, GUID))
        ExportList.insert(GUID);
    }

    // Insert the newly imported function to the worklist.
    Worklist.push_back(std::make_pair(CalleeSummary, Threshold));
  }
}

/// Given the list of globals defined in a module, compute the list of imports
/// as well as the list of "exports", i.e. the list of symbols referenced from
/// another module (that may require promotion).
static void ComputeImportForModule(
    StringRef ModulePath,
    const std::map<GlobalValue::GUID, FunctionSummary *> &DefinedFunctions,
    const ModuleSummaryIndex &Index,
    FunctionImporter::ImportMapTy &ImportsForModule,
    StringMap<FunctionImporter::ExportSetTy> &ExportLists) {
  // Worklist contains the list of function imported in this module, for which
  // we will analyse the callees and may import further down the callgraph.
  SmallVector<EdgeInfo, 128> Worklist;

  // Populate the worklist with the import for the functions in the current
  // module
  for (auto &FuncInfo : DefinedFunctions) {
    auto *Summary = FuncInfo.second;
    DEBUG(dbgs() << "Initalize import for " << FuncInfo.first << "\n");
    computeImportForFunction(ModulePath, *Summary, Index, ImportInstrLimit,
                             DefinedFunctions, Worklist, ImportsForModule,
                             ExportLists);
  }

  while (!Worklist.empty()) {
    auto FuncInfo = Worklist.pop_back_val();
    auto *Summary = FuncInfo.first;
    auto Threshold = FuncInfo.second;

    // Process the newly imported functions and add callees to the worklist.
    // Adjust the threshold
    Threshold = Threshold * ImportInstrFactor;

    computeImportForFunction(ModulePath, *Summary, Index, Threshold,
                             DefinedFunctions, Worklist, ImportsForModule,
                             ExportLists);
  }
}

} // anonymous namespace

/// Compute all the import and export for every module in the Index.
void llvm::ComputeCrossModuleImport(
    const ModuleSummaryIndex &Index,
    StringMap<FunctionImporter::ImportMapTy> &ImportLists,
    StringMap<FunctionImporter::ExportSetTy> &ExportLists) {
  auto ModuleCount = Index.modulePaths().size();

  // Collect for each module the list of function it defines.
  // GUID -> Summary
  StringMap<std::map<GlobalValue::GUID, FunctionSummary *>>
      Module2FunctionInfoMap(ModuleCount);

  for (auto &GlobalList : Index) {
    auto GUID = GlobalList.first;
    for (auto &GlobInfo : GlobalList.second) {
      auto *Summary = dyn_cast_or_null<FunctionSummary>(GlobInfo->summary());
      if (!Summary)
        /// Ignore global variable, focus on functions
        continue;
      DEBUG(dbgs() << "Adding definition: Module '" << Summary->modulePath()
                   << "' defines '" << GUID << "'\n");
      Module2FunctionInfoMap[Summary->modulePath()][GUID] = Summary;
    }
  }

  // For each module that has function defined, compute the import/export lists.
  for (auto &DefinedFunctions : Module2FunctionInfoMap) {
    auto &ImportsForModule = ImportLists[DefinedFunctions.first()];
    DEBUG(dbgs() << "Computing import for Module '" << DefinedFunctions.first()
                 << "'\n");
    ComputeImportForModule(DefinedFunctions.first(), DefinedFunctions.second,
                           Index, ImportsForModule, ExportLists);
  }

#ifndef NDEBUG
  DEBUG(dbgs() << "Import/Export lists for " << ImportLists.size()
               << " modules:\n");
  for (auto &ModuleImports : ImportLists) {
    auto ModName = ModuleImports.first();
    auto &Exports = ExportLists[ModName];
    DEBUG(dbgs() << "* Module " << ModName << " exports " << Exports.size()
                 << " functions. Imports from " << ModuleImports.second.size()
                 << " modules.\n");
    for (auto &Src : ModuleImports.second) {
      auto SrcModName = Src.first();
      DEBUG(dbgs() << " - " << Src.second.size() << " functions imported from "
                   << SrcModName << "\n");
    }
  }
#endif
}

// Automatically import functions in Module \p DestModule based on the summaries
// index.
//
bool FunctionImporter::importFunctions(
    Module &DestModule, const FunctionImporter::ImportMapTy &ImportList) {
  DEBUG(dbgs() << "Starting import for Module "
               << DestModule.getModuleIdentifier() << "\n");
  unsigned ImportedCount = 0;

  // Linker that will be used for importing function
  Linker TheLinker(DestModule);
  // Do the actual import of functions now, one Module at a time
  std::set<StringRef> ModuleNameOrderedList;
  for (auto &FunctionsToImportPerModule : ImportList) {
    ModuleNameOrderedList.insert(FunctionsToImportPerModule.first());
  }
  for (auto &Name : ModuleNameOrderedList) {
    // Get the module for the import
    const auto &FunctionsToImportPerModule = ImportList.find(Name);
    assert(FunctionsToImportPerModule != ImportList.end());
    std::unique_ptr<Module> SrcModule = ModuleLoader(Name);
    assert(&DestModule.getContext() == &SrcModule->getContext() &&
           "Context mismatch");

    // If modules were created with lazy metadata loading, materialize it
    // now, before linking it (otherwise this will be a noop).
    SrcModule->materializeMetadata();
    UpgradeDebugInfo(*SrcModule);

    auto &ImportGUIDs = FunctionsToImportPerModule->second;
    // Find the globals to import
    DenseSet<const GlobalValue *> GlobalsToImport;
    for (auto &GV : *SrcModule) {
      if (!GV.hasName())
        continue;
      auto GUID = GV.getGUID();
      auto Import = ImportGUIDs.count(GUID);
      DEBUG(dbgs() << (Import ? "Is" : "Not") << " importing " << GUID << " "
                   << GV.getName() << " from " << SrcModule->getSourceFileName()
                   << "\n");
      if (Import) {
        GV.materialize();
        GlobalsToImport.insert(&GV);
      }
    }
    for (auto &GV : SrcModule->aliases()) {
      if (!GV.hasName())
        continue;
      auto GUID = GV.getGUID();
      auto Import = ImportGUIDs.count(GUID);
      DEBUG(dbgs() << (Import ? "Is" : "Not") << " importing " << GUID << " "
                   << GV.getName() << " from " << SrcModule->getSourceFileName()
                   << "\n");
      if (Import) {
        // Alias can't point to "available_externally". However when we import
        // linkOnceODR the linkage does not change. So we import the alias
        // and aliasee only in this case.
        const GlobalObject *GO = GV.getBaseObject();
        if (!GO->hasLinkOnceODRLinkage())
          continue;
        GV.materialize();
        GlobalsToImport.insert(&GV);
        GlobalsToImport.insert(GO);
      }
    }
    for (auto &GV : SrcModule->globals()) {
      if (!GV.hasName())
        continue;
      auto GUID = GV.getGUID();
      auto Import = ImportGUIDs.count(GUID);
      DEBUG(dbgs() << (Import ? "Is" : "Not") << " importing " << GUID << " "
                   << GV.getName() << " from " << SrcModule->getSourceFileName()
                   << "\n");
      if (Import) {
        GV.materialize();
        GlobalsToImport.insert(&GV);
      }
    }

    // Link in the specified functions.
    if (renameModuleForThinLTO(*SrcModule, Index, &GlobalsToImport))
      return true;

    if (PrintImports) {
      for (const auto *GV : GlobalsToImport)
        dbgs() << DestModule.getSourceFileName() << ": Import " << GV->getName()
               << " from " << SrcModule->getSourceFileName() << "\n";
    }

    if (TheLinker.linkInModule(std::move(SrcModule), Linker::Flags::None,
                               &GlobalsToImport))
      report_fatal_error("Function Import: link error");

    ImportedCount += GlobalsToImport.size();
  }

  NumImported += ImportedCount;

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

/// Parse the summary index out of an IR file and return the summary
/// index object if found, or nullptr if not.
static std::unique_ptr<ModuleSummaryIndex>
getModuleSummaryIndexForFile(StringRef Path, std::string &Error,
                             DiagnosticHandlerFunction DiagnosticHandler) {
  std::unique_ptr<MemoryBuffer> Buffer;
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
      MemoryBuffer::getFile(Path);
  if (std::error_code EC = BufferOrErr.getError()) {
    Error = EC.message();
    return nullptr;
  }
  Buffer = std::move(BufferOrErr.get());
  ErrorOr<std::unique_ptr<object::ModuleSummaryIndexObjectFile>> ObjOrErr =
      object::ModuleSummaryIndexObjectFile::create(Buffer->getMemBufferRef(),
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
  /// Optional module summary index to use for importing, otherwise
  /// the summary-file option must be specified.
  const ModuleSummaryIndex *Index;

public:
  /// Pass identification, replacement for typeid
  static char ID;

  /// Specify pass name for debug output
  const char *getPassName() const override {
    return "Function Importing";
  }

  explicit FunctionImportPass(const ModuleSummaryIndex *Index = nullptr)
      : ModulePass(ID), Index(Index) {}

  bool runOnModule(Module &M) override {
    if (SummaryFile.empty() && !Index)
      report_fatal_error("error: -function-import requires -summary-file or "
                         "file from frontend\n");
    std::unique_ptr<ModuleSummaryIndex> IndexPtr;
    if (!SummaryFile.empty()) {
      if (Index)
        report_fatal_error("error: -summary-file and index from frontend\n");
      std::string Error;
      IndexPtr =
          getModuleSummaryIndexForFile(SummaryFile, Error, diagnosticHandler);
      if (!IndexPtr) {
        errs() << "Error loading file '" << SummaryFile << "': " << Error
               << "\n";
        return false;
      }
      Index = IndexPtr.get();
    }

    // First step is collecting the import/export lists
    // The export list is not used yet, but could limit the amount of renaming
    // performed in renameModuleForThinLTO()
    StringMap<FunctionImporter::ImportMapTy> ImportLists;
    StringMap<FunctionImporter::ExportSetTy> ExportLists;
    ComputeCrossModuleImport(*Index, ImportLists, ExportLists);
    auto &ImportList = ImportLists[M.getModuleIdentifier()];

    // Next we need to promote to global scope and rename any local values that
    // are potentially exported to other modules.
    if (renameModuleForThinLTO(M, *Index, nullptr)) {
      errs() << "Error renaming module\n";
      return false;
    }

    // Perform the import now.
    auto ModuleLoader = [&M](StringRef Identifier) {
      return loadFile(Identifier, M.getContext());
    };
    FunctionImporter Importer(*Index, ModuleLoader);
    return Importer.importFunctions(M, ImportList);
  }
};
} // anonymous namespace

char FunctionImportPass::ID = 0;
INITIALIZE_PASS_BEGIN(FunctionImportPass, "function-import",
                      "Summary Based Function Import", false, false)
INITIALIZE_PASS_END(FunctionImportPass, "function-import",
                    "Summary Based Function Import", false, false)

namespace llvm {
Pass *createFunctionImportPass(const ModuleSummaryIndex *Index = nullptr) {
  return new FunctionImportPass(Index);
}
}
