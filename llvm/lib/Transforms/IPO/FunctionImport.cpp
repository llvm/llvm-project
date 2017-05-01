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
#include "llvm/ADT/Triple.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/AutoUpgrade.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Object/IRObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/Utils/FunctionImportUtils.h"

#define DEBUG_TYPE "function-import"

using namespace llvm;

STATISTIC(NumImportedFunctions, "Number of functions imported");
STATISTIC(NumImportedModules, "Number of modules imported from");
STATISTIC(NumDeadSymbols, "Number of dead stripped symbols in index");
STATISTIC(NumLiveSymbols, "Number of live symbols in index");

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

static cl::opt<float> ImportHotInstrFactor(
    "import-hot-evolution-factor", cl::init(1.0), cl::Hidden,
    cl::value_desc("x"),
    cl::desc("As we import functions called from hot callsite, multiply the "
             "`import-instr-limit` threshold by this factor "
             "before processing newly imported functions"));

static cl::opt<float> ImportHotMultiplier(
    "import-hot-multiplier", cl::init(3.0), cl::Hidden, cl::value_desc("x"),
    cl::desc("Multiply the `import-instr-limit` threshold for hot callsites"));

// FIXME: This multiplier was not really tuned up.
static cl::opt<float> ImportColdMultiplier(
    "import-cold-multiplier", cl::init(0), cl::Hidden, cl::value_desc("N"),
    cl::desc("Multiply the `import-instr-limit` threshold for cold callsites"));

static cl::opt<bool> PrintImports("print-imports", cl::init(false), cl::Hidden,
                                  cl::desc("Print imported functions"));

static cl::opt<bool> ComputeDead("compute-dead", cl::init(true), cl::Hidden,
                                 cl::desc("Compute dead symbols"));

static cl::opt<bool> EnableImportMetadata(
    "enable-import-metadata", cl::init(
#if !defined(NDEBUG)
                                  true /*Enabled with asserts.*/
#else
                                  false
#endif
                                  ),
    cl::Hidden, cl::desc("Enable import metadata like 'thinlto_src_module'"));

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
selectCallee(const ModuleSummaryIndex &Index,
             const GlobalValueSummaryList &CalleeSummaryList,
             unsigned Threshold, StringRef CallerModulePath) {
  auto It = llvm::find_if(
      CalleeSummaryList,
      [&](const std::unique_ptr<GlobalValueSummary> &SummaryPtr) {
        auto *GVSummary = SummaryPtr.get();
        if (GlobalValue::isInterposableLinkage(GVSummary->linkage()))
          // There is no point in importing these, we can't inline them
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

        // If this is a local function, make sure we import the copy
        // in the caller's module. The only time a local function can
        // share an entry in the index is if there is a local with the same name
        // in another module that had the same source file name (in a different
        // directory), where each was compiled in their own directory so there
        // was not distinguishing path.
        // However, do the import from another module if there is only one
        // entry in the list - in that case this must be a reference due
        // to indirect call profile data, since a function pointer can point to
        // a local in another module.
        if (GlobalValue::isLocalLinkage(Summary->linkage()) &&
            CalleeSummaryList.size() > 1 &&
            Summary->modulePath() != CallerModulePath)
          return false;

        if (Summary->instCount() > Threshold)
          return false;

        if (Summary->notEligibleToImport())
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
                                              const ModuleSummaryIndex &Index,
                                              StringRef CallerModulePath) {
  auto CalleeSummaryList = Index.findGlobalValueSummaryList(GUID);
  if (CalleeSummaryList == Index.end())
    return nullptr; // This function does not have a summary
  return selectCallee(Index, CalleeSummaryList->second, Threshold,
                      CallerModulePath);
}

using EdgeInfo = std::tuple<const FunctionSummary *, unsigned /* Threshold */,
                            GlobalValue::GUID>;

/// Compute the list of functions to import for a given caller. Mark these
/// imported functions and the symbols they reference in their source module as
/// exported from their source module.
static void computeImportForFunction(
    const FunctionSummary &Summary, const ModuleSummaryIndex &Index,
    const unsigned Threshold, const GVSummaryMapTy &DefinedGVSummaries,
    SmallVectorImpl<EdgeInfo> &Worklist,
    FunctionImporter::ImportMapTy &ImportList,
    StringMap<FunctionImporter::ExportSetTy> *ExportLists = nullptr) {
  for (auto &Edge : Summary.calls()) {
    auto GUID = Edge.first.getGUID();
    DEBUG(dbgs() << " edge -> " << GUID << " Threshold:" << Threshold << "\n");

    if (Index.findGlobalValueSummaryList(GUID) == Index.end()) {
      // For SamplePGO, the indirect call targets for local functions will
      // have its original name annotated in profile. We try to find the
      // corresponding PGOFuncName as the GUID.
      GUID = Index.getGUIDFromOriginalID(GUID);
      if (GUID == 0)
        continue;
    }

    if (DefinedGVSummaries.count(GUID)) {
      DEBUG(dbgs() << "ignored! Target already in destination module.\n");
      continue;
    }

    auto GetBonusMultiplier = [](CalleeInfo::HotnessType Hotness) -> float {
      if (Hotness == CalleeInfo::HotnessType::Hot)
        return ImportHotMultiplier;
      if (Hotness == CalleeInfo::HotnessType::Cold)
        return ImportColdMultiplier;
      return 1.0;
    };

    const auto NewThreshold =
        Threshold * GetBonusMultiplier(Edge.second.Hotness);

    auto *CalleeSummary =
        selectCallee(GUID, NewThreshold, Index, Summary.modulePath());
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

    assert(ResolvedCalleeSummary->instCount() <= NewThreshold &&
           "selectCallee() didn't honor the threshold");

    auto GetAdjustedThreshold = [](unsigned Threshold, bool IsHotCallsite) {
      // Adjust the threshold for next level of imported functions.
      // The threshold is different for hot callsites because we can then
      // inline chains of hot calls.
      if (IsHotCallsite)
        return Threshold * ImportHotInstrFactor;
      return Threshold * ImportInstrFactor;
    };

    bool IsHotCallsite = Edge.second.Hotness == CalleeInfo::HotnessType::Hot;
    const auto AdjThreshold = GetAdjustedThreshold(Threshold, IsHotCallsite);

    auto ExportModulePath = ResolvedCalleeSummary->modulePath();
    auto &ProcessedThreshold = ImportList[ExportModulePath][GUID];
    /// Since the traversal of the call graph is DFS, we can revisit a function
    /// a second time with a higher threshold. In this case, it is added back to
    /// the worklist with the new threshold.
    if (ProcessedThreshold && ProcessedThreshold >= AdjThreshold) {
      DEBUG(dbgs() << "ignored! Target was already seen with Threshold "
                   << ProcessedThreshold << "\n");
      continue;
    }
    bool PreviouslyImported = ProcessedThreshold != 0;
    // Mark this function as imported in this module, with the current Threshold
    ProcessedThreshold = AdjThreshold;

    // Make exports in the source module.
    if (ExportLists) {
      auto &ExportList = (*ExportLists)[ExportModulePath];
      ExportList.insert(GUID);
      if (!PreviouslyImported) {
        // This is the first time this function was exported from its source
        // module, so mark all functions and globals it references as exported
        // to the outside if they are defined in the same source module.
        // For efficiency, we unconditionally add all the referenced GUIDs
        // to the ExportList for this module, and will prune out any not
        // defined in the module later in a single pass.
        for (auto &Edge : ResolvedCalleeSummary->calls()) {
          auto CalleeGUID = Edge.first.getGUID();
          ExportList.insert(CalleeGUID);
        }
        for (auto &Ref : ResolvedCalleeSummary->refs()) {
          auto GUID = Ref.getGUID();
          ExportList.insert(GUID);
        }
      }
    }

    // Insert the newly imported function to the worklist.
    Worklist.emplace_back(ResolvedCalleeSummary, AdjThreshold, GUID);
  }
}

/// Given the list of globals defined in a module, compute the list of imports
/// as well as the list of "exports", i.e. the list of symbols referenced from
/// another module (that may require promotion).
static void ComputeImportForModule(
    const GVSummaryMapTy &DefinedGVSummaries, const ModuleSummaryIndex &Index,
    FunctionImporter::ImportMapTy &ImportList,
    StringMap<FunctionImporter::ExportSetTy> *ExportLists = nullptr,
    const DenseSet<GlobalValue::GUID> *DeadSymbols = nullptr) {
  // Worklist contains the list of function imported in this module, for which
  // we will analyse the callees and may import further down the callgraph.
  SmallVector<EdgeInfo, 128> Worklist;

  // Populate the worklist with the import for the functions in the current
  // module
  for (auto &GVSummary : DefinedGVSummaries) {
    if (DeadSymbols && DeadSymbols->count(GVSummary.first)) {
      DEBUG(dbgs() << "Ignores Dead GUID: " << GVSummary.first << "\n");
      continue;
    }
    auto *Summary = GVSummary.second;
    if (auto *AS = dyn_cast<AliasSummary>(Summary))
      Summary = &AS->getAliasee();
    auto *FuncSummary = dyn_cast<FunctionSummary>(Summary);
    if (!FuncSummary)
      // Skip import for global variables
      continue;
    DEBUG(dbgs() << "Initalize import for " << GVSummary.first << "\n");
    computeImportForFunction(*FuncSummary, Index, ImportInstrLimit,
                             DefinedGVSummaries, Worklist, ImportList,
                             ExportLists);
  }

  // Process the newly imported functions and add callees to the worklist.
  while (!Worklist.empty()) {
    auto FuncInfo = Worklist.pop_back_val();
    auto *Summary = std::get<0>(FuncInfo);
    auto Threshold = std::get<1>(FuncInfo);
    auto GUID = std::get<2>(FuncInfo);

    // Check if we later added this summary with a higher threshold.
    // If so, skip this entry.
    auto ExportModulePath = Summary->modulePath();
    auto &LatestProcessedThreshold = ImportList[ExportModulePath][GUID];
    if (LatestProcessedThreshold > Threshold)
      continue;

    computeImportForFunction(*Summary, Index, Threshold, DefinedGVSummaries,
                             Worklist, ImportList, ExportLists);
  }
}

} // anonymous namespace

/// Compute all the import and export for every module using the Index.
void llvm::ComputeCrossModuleImport(
    const ModuleSummaryIndex &Index,
    const StringMap<GVSummaryMapTy> &ModuleToDefinedGVSummaries,
    StringMap<FunctionImporter::ImportMapTy> &ImportLists,
    StringMap<FunctionImporter::ExportSetTy> &ExportLists,
    const DenseSet<GlobalValue::GUID> *DeadSymbols) {
  // For each module that has function defined, compute the import/export lists.
  for (auto &DefinedGVSummaries : ModuleToDefinedGVSummaries) {
    auto &ImportList = ImportLists[DefinedGVSummaries.first()];
    DEBUG(dbgs() << "Computing import for Module '"
                 << DefinedGVSummaries.first() << "'\n");
    ComputeImportForModule(DefinedGVSummaries.second, Index, ImportList,
                           &ExportLists, DeadSymbols);
  }

  // When computing imports we added all GUIDs referenced by anything
  // imported from the module to its ExportList. Now we prune each ExportList
  // of any not defined in that module. This is more efficient than checking
  // while computing imports because some of the summary lists may be long
  // due to linkonce (comdat) copies.
  for (auto &ELI : ExportLists) {
    const auto &DefinedGVSummaries =
        ModuleToDefinedGVSummaries.lookup(ELI.first());
    for (auto EI = ELI.second.begin(); EI != ELI.second.end();) {
      if (!DefinedGVSummaries.count(*EI))
        EI = ELI.second.erase(EI);
      else
        ++EI;
    }
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

DenseSet<GlobalValue::GUID> llvm::computeDeadSymbols(
    const ModuleSummaryIndex &Index,
    const DenseSet<GlobalValue::GUID> &GUIDPreservedSymbols) {
  if (!ComputeDead)
    return DenseSet<GlobalValue::GUID>();
  if (GUIDPreservedSymbols.empty())
    // Don't do anything when nothing is live, this is friendly with tests.
    return DenseSet<GlobalValue::GUID>();
  DenseSet<GlobalValue::GUID> LiveSymbols = GUIDPreservedSymbols;
  SmallVector<GlobalValue::GUID, 128> Worklist;
  Worklist.reserve(LiveSymbols.size() * 2);
  for (auto GUID : LiveSymbols) {
    DEBUG(dbgs() << "Live root: " << GUID << "\n");
    Worklist.push_back(GUID);
  }
  // Add values flagged in the index as live roots to the worklist.
  for (const auto &Entry : Index) {
    bool IsLiveRoot = llvm::any_of(
        Entry.second,
        [&](const std::unique_ptr<llvm::GlobalValueSummary> &Summary) {
          return Summary->liveRoot();
        });
    if (!IsLiveRoot)
      continue;
    DEBUG(dbgs() << "Live root (summary): " << Entry.first << "\n");
    Worklist.push_back(Entry.first);
  }

  while (!Worklist.empty()) {
    auto GUID = Worklist.pop_back_val();
    auto It = Index.findGlobalValueSummaryList(GUID);
    if (It == Index.end()) {
      DEBUG(dbgs() << "Not in index: " << GUID << "\n");
      continue;
    }

    // FIXME: we should only make the prevailing copy live here
    for (auto &Summary : It->second) {
      for (auto Ref : Summary->refs()) {
        auto RefGUID = Ref.getGUID();
        if (LiveSymbols.insert(RefGUID).second) {
          DEBUG(dbgs() << "Marking live (ref): " << RefGUID << "\n");
          Worklist.push_back(RefGUID);
        }
      }
      if (auto *FS = dyn_cast<FunctionSummary>(Summary.get())) {
        for (auto Call : FS->calls()) {
          auto CallGUID = Call.first.getGUID();
          if (LiveSymbols.insert(CallGUID).second) {
            DEBUG(dbgs() << "Marking live (call): " << CallGUID << "\n");
            Worklist.push_back(CallGUID);
          }
        }
      }
      if (auto *AS = dyn_cast<AliasSummary>(Summary.get())) {
        auto AliaseeGUID = AS->getAliasee().getOriginalName();
        if (LiveSymbols.insert(AliaseeGUID).second) {
          DEBUG(dbgs() << "Marking live (alias): " << AliaseeGUID << "\n");
          Worklist.push_back(AliaseeGUID);
        }
      }
    }
  }
  DenseSet<GlobalValue::GUID> DeadSymbols;
  DeadSymbols.reserve(
      std::min(Index.size(), Index.size() - LiveSymbols.size()));
  for (auto &Entry : Index) {
    auto GUID = Entry.first;
    if (!LiveSymbols.count(GUID)) {
      DEBUG(dbgs() << "Marking dead: " << GUID << "\n");
      DeadSymbols.insert(GUID);
    }
  }
  DEBUG(dbgs() << LiveSymbols.size() << " symbols Live, and "
               << DeadSymbols.size() << " symbols Dead \n");
  NumDeadSymbols += DeadSymbols.size();
  NumLiveSymbols += LiveSymbols.size();
  return DeadSymbols;
}

/// Compute the set of summaries needed for a ThinLTO backend compilation of
/// \p ModulePath.
void llvm::gatherImportedSummariesForModule(
    StringRef ModulePath,
    const StringMap<GVSummaryMapTy> &ModuleToDefinedGVSummaries,
    const FunctionImporter::ImportMapTy &ImportList,
    std::map<std::string, GVSummaryMapTy> &ModuleToSummariesForIndex) {
  // Include all summaries from the importing module.
  ModuleToSummariesForIndex[ModulePath] =
      ModuleToDefinedGVSummaries.lookup(ModulePath);
  // Include summaries for imports.
  for (auto &ILI : ImportList) {
    auto &SummariesForIndex = ModuleToSummariesForIndex[ILI.first()];
    const auto &DefinedGVSummaries =
        ModuleToDefinedGVSummaries.lookup(ILI.first());
    for (auto &GI : ILI.second) {
      const auto &DS = DefinedGVSummaries.find(GI.first);
      assert(DS != DefinedGVSummaries.end() &&
             "Expected a defined summary for imported global value");
      SummariesForIndex[GI.first] = DS->second;
    }
  }
}

/// Emit the files \p ModulePath will import from into \p OutputFilename.
std::error_code
llvm::EmitImportsFiles(StringRef ModulePath, StringRef OutputFilename,
                       const FunctionImporter::ImportMapTy &ModuleImports) {
  std::error_code EC;
  raw_fd_ostream ImportsOS(OutputFilename, EC, sys::fs::OpenFlags::F_None);
  if (EC)
    return EC;
  for (auto &ILI : ModuleImports)
    ImportsOS << ILI.first() << "\n";
  return std::error_code();
}

/// Fixup WeakForLinker linkages in \p TheModule based on summary analysis.
void llvm::thinLTOResolveWeakForLinkerModule(
    Module &TheModule, const GVSummaryMapTy &DefinedGlobals) {
  auto ConvertToDeclaration = [](GlobalValue &GV) {
    DEBUG(dbgs() << "Converting to a declaration: `" << GV.getName() << "\n");
    if (Function *F = dyn_cast<Function>(&GV)) {
      F->deleteBody();
      F->clearMetadata();
    } else if (GlobalVariable *V = dyn_cast<GlobalVariable>(&GV)) {
      V->setInitializer(nullptr);
      V->setLinkage(GlobalValue::ExternalLinkage);
      V->clearMetadata();
    } else
      // For now we don't resolve or drop aliases. Once we do we'll
      // need to add support here for creating either a function or
      // variable declaration, and return the new GlobalValue* for
      // the caller to use.
      llvm_unreachable("Expected function or variable");
  };

  auto updateLinkage = [&](GlobalValue &GV) {
    if (!GlobalValue::isWeakForLinker(GV.getLinkage()))
      return;
    // See if the global summary analysis computed a new resolved linkage.
    const auto &GS = DefinedGlobals.find(GV.getGUID());
    if (GS == DefinedGlobals.end())
      return;
    auto NewLinkage = GS->second->linkage();
    if (NewLinkage == GV.getLinkage())
      return;
    // Check for a non-prevailing def that has interposable linkage
    // (e.g. non-odr weak or linkonce). In that case we can't simply
    // convert to available_externally, since it would lose the
    // interposable property and possibly get inlined. Simply drop
    // the definition in that case.
    if (GlobalValue::isAvailableExternallyLinkage(NewLinkage) &&
        GlobalValue::isInterposableLinkage(GV.getLinkage()))
      ConvertToDeclaration(GV);
    else {
      DEBUG(dbgs() << "ODR fixing up linkage for `" << GV.getName() << "` from "
                   << GV.getLinkage() << " to " << NewLinkage << "\n");
      GV.setLinkage(NewLinkage);
    }
    // Remove declarations from comdats, including available_externally
    // as this is a declaration for the linker, and will be dropped eventually.
    // It is illegal for comdats to contain declarations.
    auto *GO = dyn_cast_or_null<GlobalObject>(&GV);
    if (GO && GO->isDeclarationForLinker() && GO->hasComdat())
      GO->setComdat(nullptr);
  };

  // Process functions and global now
  for (auto &GV : TheModule)
    updateLinkage(GV);
  for (auto &GV : TheModule.globals())
    updateLinkage(GV);
  for (auto &GV : TheModule.aliases())
    updateLinkage(GV);
}

/// Run internalization on \p TheModule based on symmary analysis.
void llvm::thinLTOInternalizeModule(Module &TheModule,
                                    const GVSummaryMapTy &DefinedGlobals) {
  // Parse inline ASM and collect the list of symbols that are not defined in
  // the current module.
  StringSet<> AsmUndefinedRefs;
  ModuleSymbolTable::CollectAsmSymbols(
      TheModule,
      [&AsmUndefinedRefs](StringRef Name, object::BasicSymbolRef::Flags Flags) {
        if (Flags & object::BasicSymbolRef::SF_Undefined)
          AsmUndefinedRefs.insert(Name);
      });

  // Declare a callback for the internalize pass that will ask for every
  // candidate GlobalValue if it can be internalized or not.
  auto MustPreserveGV = [&](const GlobalValue &GV) -> bool {
    // Can't be internalized if referenced in inline asm.
    if (AsmUndefinedRefs.count(GV.getName()))
      return true;

    // Lookup the linkage recorded in the summaries during global analysis.
    const auto &GS = DefinedGlobals.find(GV.getGUID());
    GlobalValue::LinkageTypes Linkage;
    if (GS == DefinedGlobals.end()) {
      // Must have been promoted (possibly conservatively). Find original
      // name so that we can access the correct summary and see if it can
      // be internalized again.
      // FIXME: Eventually we should control promotion instead of promoting
      // and internalizing again.
      StringRef OrigName =
          ModuleSummaryIndex::getOriginalNameBeforePromote(GV.getName());
      std::string OrigId = GlobalValue::getGlobalIdentifier(
          OrigName, GlobalValue::InternalLinkage,
          TheModule.getSourceFileName());
      const auto &GS = DefinedGlobals.find(GlobalValue::getGUID(OrigId));
      if (GS == DefinedGlobals.end()) {
        // Also check the original non-promoted non-globalized name. In some
        // cases a preempted weak value is linked in as a local copy because
        // it is referenced by an alias (IRLinker::linkGlobalValueProto).
        // In that case, since it was originally not a local value, it was
        // recorded in the index using the original name.
        // FIXME: This may not be needed once PR27866 is fixed.
        const auto &GS = DefinedGlobals.find(GlobalValue::getGUID(OrigName));
        assert(GS != DefinedGlobals.end());
        Linkage = GS->second->linkage();
      } else {
        Linkage = GS->second->linkage();
      }
    } else
      Linkage = GS->second->linkage();
    return !GlobalValue::isLocalLinkage(Linkage);
  };

  // FIXME: See if we can just internalize directly here via linkage changes
  // based on the index, rather than invoking internalizeModule.
  llvm::internalizeModule(TheModule, MustPreserveGV);
}

// Automatically import functions in Module \p DestModule based on the summaries
// index.
//
Expected<bool> FunctionImporter::importFunctions(
    Module &DestModule, const FunctionImporter::ImportMapTy &ImportList) {
  DEBUG(dbgs() << "Starting import for Module "
               << DestModule.getModuleIdentifier() << "\n");
  unsigned ImportedCount = 0;

  IRMover Mover(DestModule);
  // Do the actual import of functions now, one Module at a time
  std::set<StringRef> ModuleNameOrderedList;
  for (auto &FunctionsToImportPerModule : ImportList) {
    ModuleNameOrderedList.insert(FunctionsToImportPerModule.first());
  }
  for (auto &Name : ModuleNameOrderedList) {
    // Get the module for the import
    const auto &FunctionsToImportPerModule = ImportList.find(Name);
    assert(FunctionsToImportPerModule != ImportList.end());
    Expected<std::unique_ptr<Module>> SrcModuleOrErr = ModuleLoader(Name);
    if (!SrcModuleOrErr)
      return SrcModuleOrErr.takeError();
    std::unique_ptr<Module> SrcModule = std::move(*SrcModuleOrErr);
    assert(&DestModule.getContext() == &SrcModule->getContext() &&
           "Context mismatch");

    // If modules were created with lazy metadata loading, materialize it
    // now, before linking it (otherwise this will be a noop).
    if (Error Err = SrcModule->materializeMetadata())
      return std::move(Err);

    auto &ImportGUIDs = FunctionsToImportPerModule->second;
    // Find the globals to import
    SetVector<GlobalValue *> GlobalsToImport;
    for (Function &F : *SrcModule) {
      if (!F.hasName())
        continue;
      auto GUID = F.getGUID();
      auto Import = ImportGUIDs.count(GUID);
      DEBUG(dbgs() << (Import ? "Is" : "Not") << " importing function " << GUID
                   << " " << F.getName() << " from "
                   << SrcModule->getSourceFileName() << "\n");
      if (Import) {
        if (Error Err = F.materialize())
          return std::move(Err);
        if (EnableImportMetadata) {
          // Add 'thinlto_src_module' metadata for statistics and debugging.
          F.setMetadata(
              "thinlto_src_module",
              llvm::MDNode::get(
                  DestModule.getContext(),
                  {llvm::MDString::get(DestModule.getContext(),
                                       SrcModule->getSourceFileName())}));
        }
        GlobalsToImport.insert(&F);
      }
    }
    for (GlobalVariable &GV : SrcModule->globals()) {
      if (!GV.hasName())
        continue;
      auto GUID = GV.getGUID();
      auto Import = ImportGUIDs.count(GUID);
      DEBUG(dbgs() << (Import ? "Is" : "Not") << " importing global " << GUID
                   << " " << GV.getName() << " from "
                   << SrcModule->getSourceFileName() << "\n");
      if (Import) {
        if (Error Err = GV.materialize())
          return std::move(Err);
        GlobalsToImport.insert(&GV);
      }
    }
    for (GlobalAlias &GA : SrcModule->aliases()) {
      // FIXME: This should eventually be controlled entirely by the summary.
      if (FunctionImportGlobalProcessing::doImportAsDefinition(
              &GA, &GlobalsToImport)) {
        GlobalsToImport.insert(&GA);
        continue;
      }

      if (!GA.hasName())
        continue;
      auto GUID = GA.getGUID();
      auto Import = ImportGUIDs.count(GUID);
      DEBUG(dbgs() << (Import ? "Is" : "Not") << " importing alias " << GUID
                   << " " << GA.getName() << " from "
                   << SrcModule->getSourceFileName() << "\n");
      if (Import) {
        // Alias can't point to "available_externally". However when we import
        // linkOnceODR the linkage does not change. So we import the alias
        // and aliasee only in this case. This has been handled by
        // computeImportForFunction()
        GlobalObject *GO = GA.getBaseObject();
        assert(GO->hasLinkOnceODRLinkage() &&
               "Unexpected alias to a non-linkonceODR in import list");
#ifndef NDEBUG
        if (!GlobalsToImport.count(GO))
          DEBUG(dbgs() << " alias triggers importing aliasee " << GO->getGUID()
                       << " " << GO->getName() << " from "
                       << SrcModule->getSourceFileName() << "\n");
#endif
        if (Error Err = GO->materialize())
          return std::move(Err);
        GlobalsToImport.insert(GO);
        if (Error Err = GA.materialize())
          return std::move(Err);
        GlobalsToImport.insert(&GA);
      }
    }

    // Upgrade debug info after we're done materializing all the globals and we
    // have loaded all the required metadata!
    UpgradeDebugInfo(*SrcModule);

    // Link in the specified functions.
    if (renameModuleForThinLTO(*SrcModule, Index, &GlobalsToImport))
      return true;

    if (PrintImports) {
      for (const auto *GV : GlobalsToImport)
        dbgs() << DestModule.getSourceFileName() << ": Import " << GV->getName()
               << " from " << SrcModule->getSourceFileName() << "\n";
    }

    if (Mover.move(std::move(SrcModule), GlobalsToImport.getArrayRef(),
                   [](GlobalValue &, IRMover::ValueAdder) {},
                   /*IsPerformingImport=*/true))
      report_fatal_error("Function Import: link error");

    ImportedCount += GlobalsToImport.size();
    NumImportedModules++;
  }

  NumImportedFunctions += ImportedCount;

  DEBUG(dbgs() << "Imported " << ImportedCount << " functions for Module "
               << DestModule.getModuleIdentifier() << "\n");
  return ImportedCount;
}

/// Summary file to use for function importing when using -function-import from
/// the command line.
static cl::opt<std::string>
    SummaryFile("summary-file",
                cl::desc("The summary file to use for function importing."));

static bool doImportingForModule(Module &M) {
  if (SummaryFile.empty())
    report_fatal_error("error: -function-import requires -summary-file\n");
  Expected<std::unique_ptr<ModuleSummaryIndex>> IndexPtrOrErr =
      getModuleSummaryIndexForFile(SummaryFile);
  if (!IndexPtrOrErr) {
    logAllUnhandledErrors(IndexPtrOrErr.takeError(), errs(),
                          "Error loading file '" + SummaryFile + "': ");
    return false;
  }
  std::unique_ptr<ModuleSummaryIndex> Index = std::move(*IndexPtrOrErr);

  // First step is collecting the import list.
  FunctionImporter::ImportMapTy ImportList;
  ComputeCrossModuleImportForModule(M.getModuleIdentifier(), *Index,
                                    ImportList);

  // Conservatively mark all internal values as promoted. This interface is
  // only used when doing importing via the function importing pass. The pass
  // is only enabled when testing importing via the 'opt' tool, which does
  // not do the ThinLink that would normally determine what values to promote.
  for (auto &I : *Index) {
    for (auto &S : I.second) {
      if (GlobalValue::isLocalLinkage(S->linkage()))
        S->setLinkage(GlobalValue::ExternalLinkage);
    }
  }

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
  Expected<bool> Result = Importer.importFunctions(M, ImportList);

  // FIXME: Probably need to propagate Errors through the pass manager.
  if (!Result) {
    logAllUnhandledErrors(Result.takeError(), errs(),
                          "Error importing module: ");
    return false;
  }

  return *Result;
}

namespace {
/// Pass that performs cross-module function import provided a summary file.
class FunctionImportLegacyPass : public ModulePass {
public:
  /// Pass identification, replacement for typeid
  static char ID;

  /// Specify pass name for debug output
  StringRef getPassName() const override { return "Function Importing"; }

  explicit FunctionImportLegacyPass() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    if (skipModule(M))
      return false;

    return doImportingForModule(M);
  }
};
} // anonymous namespace

PreservedAnalyses FunctionImportPass::run(Module &M,
                                          ModuleAnalysisManager &AM) {
  if (!doImportingForModule(M))
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}

char FunctionImportLegacyPass::ID = 0;
INITIALIZE_PASS(FunctionImportLegacyPass, "function-import",
                "Summary Based Function Import", false, false)

namespace llvm {
Pass *createFunctionImportPass() {
  return new FunctionImportLegacyPass();
}
}
