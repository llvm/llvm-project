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
#include "llvm/IR/AutoUpgrade.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Object/IRObjectFile.h"
#include "llvm/Object/ModuleSummaryIndexObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/IPO/Internalize.h"
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
static cl::opt<float> ImportHotMultiplier(
    "import-hot-multiplier", cl::init(3.0), cl::Hidden, cl::value_desc("x"),
    cl::ZeroOrMore, cl::desc("Multiply the `import-instr-limit` threshold for "
                             "hot callsites"));

static cl::opt<bool> PrintImports("print-imports", cl::init(false), cl::Hidden,
                                  cl::desc("Print imported functions"));

// Temporary allows the function import pass to disable always linking
// referenced discardable symbols.
static cl::opt<bool>
    DontForceImportReferencedDiscardableSymbols("disable-force-link-odr",
                                                cl::init(false), cl::Hidden);

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

// Return true if the Summary describes a GlobalValue that can be externally
// referenced, i.e. it does not need renaming (linkage is not local) or renaming
// is possible (does not have a section for instance).
static bool canBeExternallyReferenced(const GlobalValueSummary &Summary) {
  if (!Summary.needsRenaming())
    return true;

  if (Summary.hasSection())
    // Can't rename a global that needs renaming if has a section.
    return false;

  return true;
}

// Return true if \p GUID describes a GlobalValue that can be externally
// referenced, i.e. it does not need renaming (linkage is not local) or
// renaming is possible (does not have a section for instance).
static bool canBeExternallyReferenced(const ModuleSummaryIndex &Index,
                                      GlobalValue::GUID GUID) {
  auto Summaries = Index.findGlobalValueSummaryList(GUID);
  if (Summaries == Index.end())
    return true;
  if (Summaries->second.size() != 1)
    // If there are multiple globals with this GUID, then we know it is
    // not a local symbol, and it is necessarily externally referenced.
    return true;

  // We don't need to check for the module path, because if it can't be
  // externally referenced and we call it, it is necessarilly in the same
  // module
  return canBeExternallyReferenced(**Summaries->second.begin());
}

// Return true if the global described by \p Summary can be imported in another
// module.
static bool eligibleForImport(const ModuleSummaryIndex &Index,
                              const GlobalValueSummary &Summary) {
  if (!canBeExternallyReferenced(Summary))
    // Can't import a global that needs renaming if has a section for instance.
    // FIXME: we may be able to import it by copying it without promotion.
    return false;

  // Don't import functions that are not viable to inline.
  if (Summary.isNotViableToInline())
    return false;

  // Check references (and potential calls) in the same module. If the current
  // value references a global that can't be externally referenced it is not
  // eligible for import.
  bool AllRefsCanBeExternallyReferenced =
      llvm::all_of(Summary.refs(), [&](const ValueInfo &VI) {
        return canBeExternallyReferenced(Index, VI.getGUID());
      });
  if (!AllRefsCanBeExternallyReferenced)
    return false;

  if (auto *FuncSummary = dyn_cast<FunctionSummary>(&Summary)) {
    bool AllCallsCanBeExternallyReferenced = llvm::all_of(
        FuncSummary->calls(), [&](const FunctionSummary::EdgeTy &Edge) {
          return canBeExternallyReferenced(Index, Edge.first.getGUID());
        });
    if (!AllCallsCanBeExternallyReferenced)
      return false;
  }
  return true;
}

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
             unsigned Threshold) {
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

        if (Summary->instCount() > Threshold)
          return false;

        if (!eligibleForImport(Index, *Summary))
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
  if (CalleeSummaryList == Index.end())
    return nullptr; // This function does not have a summary
  return selectCallee(Index, CalleeSummaryList->second, Threshold);
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
    const unsigned Threshold, const GVSummaryMapTy &DefinedGVSummaries,
    SmallVectorImpl<EdgeInfo> &Worklist,
    FunctionImporter::ImportMapTy &ImportList,
    StringMap<FunctionImporter::ExportSetTy> *ExportLists = nullptr) {
  for (auto &Edge : Summary.calls()) {
    auto GUID = Edge.first.getGUID();
    DEBUG(dbgs() << " edge -> " << GUID << " Threshold:" << Threshold << "\n");

    if (DefinedGVSummaries.count(GUID)) {
      DEBUG(dbgs() << "ignored! Target already in destination module.\n");
      continue;
    }

    // FIXME: Also lower the threshold for cold callsites.
    const auto NewThreshold =
        Edge.second.Hotness == CalleeInfo::HotnessType::Hot
            ? Threshold * ImportHotMultiplier
            : Threshold;
    auto *CalleeSummary = selectCallee(GUID, NewThreshold, Index);
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

    auto ExportModulePath = ResolvedCalleeSummary->modulePath();
    auto &ProcessedThreshold = ImportList[ExportModulePath][GUID];
    /// Since the traversal of the call graph is DFS, we can revisit a function
    /// a second time with a higher threshold. In this case, it is added back to
    /// the worklist with the new threshold.
    if (ProcessedThreshold && ProcessedThreshold >= Threshold) {
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
    FunctionImporter::ImportMapTy &ImportList,
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
                             DefinedGVSummaries, Worklist, ImportList,
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
                             Worklist, ImportList, ExportLists);
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
    auto &ImportList = ImportLists[DefinedGVSummaries.first()];
    DEBUG(dbgs() << "Computing import for Module '"
                 << DefinedGVSummaries.first() << "'\n");
    ComputeImportForModule(DefinedGVSummaries.second, Index, ImportList,
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
    DEBUG(dbgs() << "ODR fixing up linkage for `" << GV.getName() << "` from "
                 << GV.getLinkage() << " to " << NewLinkage << "\n");
    GV.setLinkage(NewLinkage);
    // Remove functions converted to available_externally from comdats,
    // as this is a declaration for the linker, and will be dropped eventually.
    // It is illegal for comdats to contain declarations.
    auto *GO = dyn_cast_or_null<GlobalObject>(&GV);
    if (GO && GO->isDeclarationForLinker() && GO->hasComdat()) {
      assert(GO->hasAvailableExternallyLinkage() &&
             "Expected comdat on definition (possibly available external)");
      GO->setComdat(nullptr);
    }
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
  object::IRObjectFile::CollectAsmUndefinedRefs(
      Triple(TheModule.getTargetTriple()), TheModule.getModuleInlineAsm(),
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
    for (Function &F : *SrcModule) {
      if (!F.hasName())
        continue;
      auto GUID = F.getGUID();
      auto Import = ImportGUIDs.count(GUID);
      DEBUG(dbgs() << (Import ? "Is" : "Not") << " importing function " << GUID
                   << " " << F.getName() << " from "
                   << SrcModule->getSourceFileName() << "\n");
      if (Import) {
        F.materialize();
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
        GV.materialize();
        GlobalsToImport.insert(&GV);
      }
    }
    for (GlobalAlias &GA : SrcModule->aliases()) {
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
        GO->materialize();
        GlobalsToImport.insert(GO);
        GA.materialize();
        GlobalsToImport.insert(&GA);
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
static std::unique_ptr<ModuleSummaryIndex> getModuleSummaryIndexForFile(
    StringRef Path, std::string &Error,
    const DiagnosticHandlerFunction &DiagnosticHandler) {
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

static bool doImportingForModule(Module &M, const ModuleSummaryIndex *Index) {
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
      errs() << "Error loading file '" << SummaryFile << "': " << Error << "\n";
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
  return Importer.importFunctions(M, ImportList,
                                  !DontForceImportReferencedDiscardableSymbols);
}

namespace {
/// Pass that performs cross-module function import provided a summary file.
class FunctionImportLegacyPass : public ModulePass {
  /// Optional module summary index to use for importing, otherwise
  /// the summary-file option must be specified.
  const ModuleSummaryIndex *Index;

public:
  /// Pass identification, replacement for typeid
  static char ID;

  /// Specify pass name for debug output
  const char *getPassName() const override { return "Function Importing"; }

  explicit FunctionImportLegacyPass(const ModuleSummaryIndex *Index = nullptr)
      : ModulePass(ID), Index(Index) {}

  bool runOnModule(Module &M) override {
    if (skipModule(M))
      return false;

    return doImportingForModule(M, Index);
  }
};
} // anonymous namespace

PreservedAnalyses FunctionImportPass::run(Module &M,
                                          ModuleAnalysisManager &AM) {
  if (!doImportingForModule(M, Index))
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}

char FunctionImportLegacyPass::ID = 0;
INITIALIZE_PASS(FunctionImportLegacyPass, "function-import",
                "Summary Based Function Import", false, false)

namespace llvm {
Pass *createFunctionImportPass(const ModuleSummaryIndex *Index = nullptr) {
  return new FunctionImportLegacyPass(Index);
}
}
