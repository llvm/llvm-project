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

// Temporary allows the function import pass to disable always linking
// referenced discardable symbols.
static cl::opt<bool>
    DontForceImportReferencedDiscardableSymbols("disable-force-link-odr",
                                                cl::init(false), cl::Hidden);

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
static const GlobalValueSummary *
selectCallee(const GlobalValueSummaryList &CalleeSummaryList,
             unsigned Threshold) {
  auto It = llvm::find_if(
      CalleeSummaryList,
      [&](const std::unique_ptr<GlobalValueSummary> &SummaryPtr) {
        auto *GVSummary = SummaryPtr.get();
        if (GlobalValue::isWeakAnyLinkage(GVSummary->linkage()))
          // There is no point in importing weak symbols, we can't inline them
          return false;
        if (auto *AS = dyn_cast<AliasSummary>(GVSummary)) {
          GVSummary = &AS->getAliasee();
          // Alias can't point to "available_externally". However when we import
          // linkOnceODR the linkage does not change. So we import the alias
          // and aliasee only in this case.
          // FIXME: we should import alias as available_externally *function*,
          // the destination module does need to know it is an alias.
          if (!GlobalValue::isLinkOnceODRLinkage(GVSummary->linkage()))
            return false;
        }

        auto *Summary = cast<FunctionSummary>(GVSummary);

        if (Summary->instCount() > Threshold)
          return false;

        return true;
      });
  if (It == CalleeSummaryList.end())
    return nullptr;

  return cast<GlobalValueSummary>(It->get());
}

/// Return the summary for the function \p GUID that fits the \p Threshold, or
/// null if there's no match.
static const GlobalValueSummary *selectCallee(GlobalValue::GUID GUID,
                                              unsigned Threshold,
                                              const ModuleSummaryIndex &Index) {
  auto CalleeSummaryList = Index.findGlobalValueSummaryList(GUID);
  if (CalleeSummaryList == Index.end()) {
    return nullptr; // This function does not have a summary
  }
  return selectCallee(CalleeSummaryList->second, Threshold);
}

/// Mark the global \p GUID as export by module \p ExportModulePath if found in
/// this module. If it is a GlobalVariable, we also mark any referenced global
/// in the current module as exported.
static void exportGlobalInModule(const ModuleSummaryIndex &Index,
                                 StringRef ExportModulePath,
                                 GlobalValue::GUID GUID,
                                 FunctionImporter::ExportSetTy &ExportList) {
  auto FindGlobalSummaryInModule =
      [&](GlobalValue::GUID GUID) -> GlobalValueSummary *{
        auto SummaryList = Index.findGlobalValueSummaryList(GUID);
        if (SummaryList == Index.end())
          // This global does not have a summary, it is not part of the ThinLTO
          // process
          return nullptr;
        auto SummaryIter = llvm::find_if(
            SummaryList->second,
            [&](const std::unique_ptr<GlobalValueSummary> &Summary) {
              return Summary->modulePath() == ExportModulePath;
            });
        if (SummaryIter == SummaryList->second.end())
          return nullptr;
        return SummaryIter->get();
      };

  auto *Summary = FindGlobalSummaryInModule(GUID);
  if (!Summary)
    return;
  // We found it in the current module, mark as exported
  ExportList.insert(GUID);

  auto GVS = dyn_cast<GlobalVarSummary>(Summary);
  if (!GVS)
    return;
  // FunctionImportGlobalProcessing::doPromoteLocalToGlobal() will always
  // trigger importing  the initializer for `constant unnamed addr` globals that
  // are referenced. We conservatively export all the referenced symbols for
  // every global to workaround this, so that the ExportList is accurate.
  // FIXME: with a "isConstant" flag in the summary we could be more targetted.
  for (auto &Ref : GVS->refs()) {
    auto GUID = Ref.getGUID();
    auto *RefSummary = FindGlobalSummaryInModule(GUID);
    if (RefSummary)
      // Found a ref in the current module, mark it as exported
      ExportList.insert(GUID);
  }
}

using EdgeInfo = std::pair<const FunctionSummary *, unsigned /* Threshold */>;

/// Compute the list of functions to import for a given caller. Mark these
/// imported functions and the symbols they reference in their source module as
/// exported from their source module.
static void computeImportForFunction(
    const FunctionSummary &Summary, const ModuleSummaryIndex &Index,
    unsigned Threshold, const GVSummaryMapTy &DefinedGVSummaries,
    SmallVectorImpl<EdgeInfo> &Worklist,
    FunctionImporter::ImportMapTy &ImportsForModule,
    StringMap<FunctionImporter::ExportSetTy> *ExportLists = nullptr) {
  for (auto &Edge : Summary.calls()) {
    auto GUID = Edge.first.getGUID();
    DEBUG(dbgs() << " edge -> " << GUID << " Threshold:" << Threshold << "\n");

    if (DefinedGVSummaries.count(GUID)) {
      DEBUG(dbgs() << "ignored! Target already in destination module.\n");
      continue;
    }

    auto *CalleeSummary = selectCallee(GUID, Threshold, Index);
    if (!CalleeSummary) {
      DEBUG(dbgs() << "ignored! No qualifying callee with summary found.\n");
      continue;
    }
    // "Resolve" the summary, traversing alias,
    const FunctionSummary *ResolvedCalleeSummary;
    if (isa<AliasSummary>(CalleeSummary)) {
      ResolvedCalleeSummary = cast<FunctionSummary>(
          &cast<AliasSummary>(CalleeSummary)->getAliasee());
      assert(
          GlobalValue::isLinkOnceODRLinkage(ResolvedCalleeSummary->linkage()) &&
          "Unexpected alias to a non-linkonceODR in import list");
    } else
      ResolvedCalleeSummary = cast<FunctionSummary>(CalleeSummary);

    assert(ResolvedCalleeSummary->instCount() <= Threshold &&
           "selectCallee() didn't honor the threshold");

    auto ExportModulePath = ResolvedCalleeSummary->modulePath();
    auto &ProcessedThreshold = ImportsForModule[ExportModulePath][GUID];
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
    if (ExportLists) {
      auto &ExportList = (*ExportLists)[ExportModulePath];
      ExportList.insert(GUID);
      // Mark all functions and globals referenced by this function as exported
      // to the outside if they are defined in the same source module.
      for (auto &Edge : ResolvedCalleeSummary->calls()) {
        auto CalleeGUID = Edge.first.getGUID();
        exportGlobalInModule(Index, ExportModulePath, CalleeGUID, ExportList);
      }
      for (auto &Ref : ResolvedCalleeSummary->refs()) {
        auto GUID = Ref.getGUID();
        exportGlobalInModule(Index, ExportModulePath, GUID, ExportList);
      }
    }

    // Insert the newly imported function to the worklist.
    Worklist.push_back(std::make_pair(ResolvedCalleeSummary, Threshold));
  }
}

/// Given the list of globals defined in a module, compute the list of imports
/// as well as the list of "exports", i.e. the list of symbols referenced from
/// another module (that may require promotion).
static void ComputeImportForModule(
    const GVSummaryMapTy &DefinedGVSummaries, const ModuleSummaryIndex &Index,
    FunctionImporter::ImportMapTy &ImportsForModule,
    StringMap<FunctionImporter::ExportSetTy> *ExportLists = nullptr) {
  // Worklist contains the list of function imported in this module, for which
  // we will analyse the callees and may import further down the callgraph.
  SmallVector<EdgeInfo, 128> Worklist;

  // Populate the worklist with the import for the functions in the current
  // module
  for (auto &GVSummary : DefinedGVSummaries) {
    auto *Summary = GVSummary.second;
    if (auto *AS = dyn_cast<AliasSummary>(Summary))
      Summary = &AS->getAliasee();
    auto *FuncSummary = dyn_cast<FunctionSummary>(Summary);
    if (!FuncSummary)
      // Skip import for global variables
      continue;
    DEBUG(dbgs() << "Initalize import for " << GVSummary.first << "\n");
    computeImportForFunction(*FuncSummary, Index, ImportInstrLimit,
                             DefinedGVSummaries, Worklist, ImportsForModule,
                             ExportLists);
  }

  while (!Worklist.empty()) {
    auto FuncInfo = Worklist.pop_back_val();
    auto *Summary = FuncInfo.first;
    auto Threshold = FuncInfo.second;

    // Process the newly imported functions and add callees to the worklist.
    // Adjust the threshold
    Threshold = Threshold * ImportInstrFactor;

    computeImportForFunction(*Summary, Index, Threshold, DefinedGVSummaries,
                             Worklist, ImportsForModule, ExportLists);
  }
}

} // anonymous namespace

/// Compute all the import and export for every module using the Index.
void llvm::ComputeCrossModuleImport(
    const ModuleSummaryIndex &Index,
    const StringMap<GVSummaryMapTy> &ModuleToDefinedGVSummaries,
    StringMap<FunctionImporter::ImportMapTy> &ImportLists,
    StringMap<FunctionImporter::ExportSetTy> &ExportLists) {
  // For each module that has function defined, compute the import/export lists.
  for (auto &DefinedGVSummaries : ModuleToDefinedGVSummaries) {
    auto &ImportsForModule = ImportLists[DefinedGVSummaries.first()];
    DEBUG(dbgs() << "Computing import for Module '"
                 << DefinedGVSummaries.first() << "'\n");
    ComputeImportForModule(DefinedGVSummaries.second, Index, ImportsForModule,
                           &ExportLists);
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

/// Compute all the imports for the given module in the Index.
void llvm::ComputeCrossModuleImportForModule(
    StringRef ModulePath, const ModuleSummaryIndex &Index,
    FunctionImporter::ImportMapTy &ImportList) {

  // Collect the list of functions this module defines.
  // GUID -> Summary
  GVSummaryMapTy FunctionSummaryMap;
  Index.collectDefinedFunctionsForModule(ModulePath, FunctionSummaryMap);

  // Compute the import list for this module.
  DEBUG(dbgs() << "Computing import for Module '" << ModulePath << "'\n");
  ComputeImportForModule(FunctionSummaryMap, Index, ImportList);

#ifndef NDEBUG
  DEBUG(dbgs() << "* Module " << ModulePath << " imports from "
               << ImportList.size() << " modules.\n");
  for (auto &Src : ImportList) {
    auto SrcModName = Src.first();
    DEBUG(dbgs() << " - " << Src.second.size() << " functions imported from "
                 << SrcModName << "\n");
  }
#endif
}

// Automatically import functions in Module \p DestModule based on the summaries
// index.
//
bool FunctionImporter::importFunctions(
    Module &DestModule, const FunctionImporter::ImportMapTy &ImportList,
    bool ForceImportReferencedDiscardableSymbols) {
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
      DEBUG(dbgs() << (Import ? "Is" : "Not") << " importing function " << GUID
                   << " " << GV.getName() << " from "
                   << SrcModule->getSourceFileName() << "\n");
      if (Import) {
        GV.materialize();
        GlobalsToImport.insert(&GV);
      }
    }
    for (auto &GV : SrcModule->globals()) {
      if (!GV.hasName())
        continue;
      auto GUID = GV.getGUID();
      auto Import = ImportGUIDs.count(GUID);
      DEBUG(dbgs() << (Import ? "Is" : "Not") << " importing global " << GUID
                   << " " << GV.getName() << " from "
                   << SrcModule->getSourceFileName() << "\n");
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
      DEBUG(dbgs() << (Import ? "Is" : "Not") << " importing alias " << GUID
                   << " " << GV.getName() << " from "
                   << SrcModule->getSourceFileName() << "\n");
      if (Import) {
        // Alias can't point to "available_externally". However when we import
        // linkOnceODR the linkage does not change. So we import the alias
        // and aliasee only in this case. This has been handled by
        // computeImportForFunction()
        GlobalObject *GO = GV.getBaseObject();
        assert(GO->hasLinkOnceODRLinkage() &&
               "Unexpected alias to a non-linkonceODR in import list");
#ifndef NDEBUG
        if (!GlobalsToImport.count(GO))
          DEBUG(dbgs() << " alias triggers importing aliasee " << GO->getGUID()
                       << " " << GO->getName() << " from "
                       << SrcModule->getSourceFileName() << "\n");
#endif
        GO->materialize();
        GlobalsToImport.insert(GO);
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

    // Instruct the linker that the client will take care of linkonce resolution
    unsigned Flags = Linker::Flags::None;
    if (!ForceImportReferencedDiscardableSymbols)
      Flags |= Linker::Flags::DontForceLinkLinkonceODR;

    if (TheLinker.linkInModule(std::move(SrcModule), Flags, &GlobalsToImport))
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
  const char *getPassName() const override { return "Function Importing"; }

  explicit FunctionImportPass(const ModuleSummaryIndex *Index = nullptr)
      : ModulePass(ID), Index(Index) {}

  bool runOnModule(Module &M) override {
    if (skipModule(M))
      return false;

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

    // First step is collecting the import list.
    FunctionImporter::ImportMapTy ImportList;
    ComputeCrossModuleImportForModule(M.getModuleIdentifier(), *Index,
                                      ImportList);

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
    return Importer.importFunctions(
        M, ImportList, !DontForceImportReferencedDiscardableSymbols);
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
