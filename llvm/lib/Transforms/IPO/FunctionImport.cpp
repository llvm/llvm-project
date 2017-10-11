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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/AutoUpgrade.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalObject.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/IRMover.h"
#include "llvm/Object/ModuleSymbolTable.h"
#include "llvm/Object/SymbolicFile.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/Utils/FunctionImportUtils.h"
#include <cassert>
#include <memory>
#include <set>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "function-import"

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
    "import-hot-multiplier", cl::init(10.0), cl::Hidden, cl::value_desc("x"),
    cl::desc("Multiply the `import-instr-limit` threshold for hot callsites"));

static cl::opt<float> ImportCriticalMultiplier(
    "import-critical-multiplier", cl::init(100.0), cl::Hidden,
    cl::value_desc("x"),
    cl::desc(
        "Multiply the `import-instr-limit` threshold for critical callsites"));

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

/// Summary file to use for function importing when using -function-import from
/// the command line.
static cl::opt<std::string>
    SummaryFile("summary-file",
                cl::desc("The summary file to use for function importing."));

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
             ArrayRef<std::unique_ptr<GlobalValueSummary>> CalleeSummaryList,
             unsigned Threshold, StringRef CallerModulePath) {
  auto It = llvm::find_if(
      CalleeSummaryList,
      [&](const std::unique_ptr<GlobalValueSummary> &SummaryPtr) {
        auto *GVSummary = SummaryPtr.get();
        // For SamplePGO, in computeImportForFunction the OriginalId
        // may have been used to locate the callee summary list (See
        // comment there).
        // The mapping from OriginalId to GUID may return a GUID
        // that corresponds to a static variable. Filter it out here.
        // This can happen when
        // 1) There is a call to a library function which is not defined
        // in the index.
        // 2) There is a static variable with the  OriginalGUID identical
        // to the GUID of the library function in 1);
        // When this happens, the logic for SamplePGO kicks in and
        // the static variable in 2) will be found, which needs to be
        // filtered out.
        if (GVSummary->getSummaryKind() == GlobalValueSummary::GlobalVarKind)
          return false;
        if (GlobalValue::isInterposableLinkage(GVSummary->linkage()))
          // There is no point in importing these, we can't inline them
          return false;
        if (isa<AliasSummary>(GVSummary))
          // Aliases can't point to "available_externally".
          // FIXME: we should import alias as available_externally *function*,
          // the destination module does not need to know it is an alias.
          return false;

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

namespace {

using EdgeInfo = std::tuple<const FunctionSummary *, unsigned /* Threshold */,
                            GlobalValue::GUID>;

} // anonymous namespace

static ValueInfo
updateValueInfoForIndirectCalls(const ModuleSummaryIndex &Index, ValueInfo VI) {
  if (!VI.getSummaryList().empty())
    return VI;
  // For SamplePGO, the indirect call targets for local functions will
  // have its original name annotated in profile. We try to find the
  // corresponding PGOFuncName as the GUID.
  // FIXME: Consider updating the edges in the graph after building
  // it, rather than needing to perform this mapping on each walk.
  auto GUID = Index.getGUIDFromOriginalID(VI.getGUID());
  if (GUID == 0)
    return nullptr;
  return Index.getValueInfo(GUID);
}

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
    ValueInfo VI = Edge.first;
    DEBUG(dbgs() << " edge -> " << VI.getGUID() << " Threshold:" << Threshold
                 << "\n");

    VI = updateValueInfoForIndirectCalls(Index, VI);
    if (!VI)
      continue;

    if (DefinedGVSummaries.count(VI.getGUID())) {
      DEBUG(dbgs() << "ignored! Target already in destination module.\n");
      continue;
    }

    auto GetBonusMultiplier = [](CalleeInfo::HotnessType Hotness) -> float {
      if (Hotness == CalleeInfo::HotnessType::Hot)
        return ImportHotMultiplier;
      if (Hotness == CalleeInfo::HotnessType::Cold)
        return ImportColdMultiplier;
      if (Hotness == CalleeInfo::HotnessType::Critical)
        return ImportCriticalMultiplier;
      return 1.0;
    };

    const auto NewThreshold =
        Threshold * GetBonusMultiplier(Edge.second.Hotness);

    auto *CalleeSummary = selectCallee(Index, VI.getSummaryList(), NewThreshold,
                                       Summary.modulePath());
    if (!CalleeSummary) {
      DEBUG(dbgs() << "ignored! No qualifying callee with summary found.\n");
      continue;
    }

    // "Resolve" the summary
    assert(!isa<AliasSummary>(CalleeSummary) &&
           "Unexpected alias in import list");
    const auto *ResolvedCalleeSummary = cast<FunctionSummary>(CalleeSummary);

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
    auto &ProcessedThreshold = ImportList[ExportModulePath][VI.getGUID()];
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
      ExportList.insert(VI.getGUID());
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
    Worklist.emplace_back(ResolvedCalleeSummary, AdjThreshold, VI.getGUID());
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
    if (!Index.isGlobalValueLive(GVSummary.second)) {
      DEBUG(dbgs() << "Ignores Dead GUID: " << GVSummary.first << "\n");
      continue;
    }
    auto *FuncSummary =
        dyn_cast<FunctionSummary>(GVSummary.second->getBaseObject());
    if (!FuncSummary)
      // Skip import for global variables
      continue;
    DEBUG(dbgs() << "Initialize import for " << GVSummary.first << "\n");
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

void llvm::computeDeadSymbols(
    ModuleSummaryIndex &Index,
    const DenseSet<GlobalValue::GUID> &GUIDPreservedSymbols) {
  assert(!Index.withGlobalValueDeadStripping());
  if (!ComputeDead)
    return;
  if (GUIDPreservedSymbols.empty())
    // Don't do anything when nothing is live, this is friendly with tests.
    return;
  unsigned LiveSymbols = 0;
  SmallVector<ValueInfo, 128> Worklist;
  Worklist.reserve(GUIDPreservedSymbols.size() * 2);
  for (auto GUID : GUIDPreservedSymbols) {
    ValueInfo VI = Index.getValueInfo(GUID);
    if (!VI)
      continue;
    for (auto &S : VI.getSummaryList())
      S->setLive(true);
  }

  // Add values flagged in the index as live roots to the worklist.
  for (const auto &Entry : Index)
    for (auto &S : Entry.second.SummaryList)
      if (S->isLive()) {
        DEBUG(dbgs() << "Live root: " << Entry.first << "\n");
        Worklist.push_back(ValueInfo(&Entry));
        ++LiveSymbols;
        break;
      }

  // Make value live and add it to the worklist if it was not live before.
  // FIXME: we should only make the prevailing copy live here
  auto visit = [&](ValueInfo VI) {
    // FIXME: If we knew which edges were created for indirect call profiles,
    // we could skip them here. Any that are live should be reached via
    // other edges, e.g. reference edges. Otherwise, using a profile collected
    // on a slightly different binary might provoke preserving, importing
    // and ultimately promoting calls to functions not linked into this
    // binary, which increases the binary size unnecessarily. Note that
    // if this code changes, the importer needs to change so that edges
    // to functions marked dead are skipped.
    VI = updateValueInfoForIndirectCalls(Index, VI);
    if (!VI)
      return;
    for (auto &S : VI.getSummaryList())
      if (S->isLive())
        return;
    for (auto &S : VI.getSummaryList())
      S->setLive(true);
    ++LiveSymbols;
    Worklist.push_back(VI);
  };

  while (!Worklist.empty()) {
    auto VI = Worklist.pop_back_val();
    for (auto &Summary : VI.getSummaryList()) {
      GlobalValueSummary *Base = Summary->getBaseObject();
      for (auto Ref : Base->refs())
        visit(Ref);
      if (auto *FS = dyn_cast<FunctionSummary>(Base))
        for (auto Call : FS->calls())
          visit(Call.first);
    }
  }
  Index.setWithGlobalValueDeadStripping();

  unsigned DeadSymbols = Index.size() - LiveSymbols;
  DEBUG(dbgs() << LiveSymbols << " symbols Live, and " << DeadSymbols
               << " symbols Dead \n");
  NumDeadSymbols += DeadSymbols;
  NumLiveSymbols += LiveSymbols;
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
    // See if the global summary analysis computed a new resolved linkage.
    const auto &GS = DefinedGlobals.find(GV.getGUID());
    if (GS == DefinedGlobals.end())
      return;
    auto NewLinkage = GS->second->linkage();
    if (NewLinkage == GV.getLinkage())
      return;

    // Switch the linkage to weakany if asked for, e.g. we do this for
    // linker redefined symbols (via --wrap or --defsym).
    // We record that the visibility should be changed here in `addThinLTO`
    // as we need access to the resolution vectors for each input file in
    // order to find which symbols have been redefined.
    // We may consider reorganizing this code and moving the linkage recording
    // somewhere else, e.g. in thinLTOResolveWeakForLinkerInIndex.
    if (NewLinkage == GlobalValue::WeakAnyLinkage) {
      GV.setLinkage(NewLinkage);
      return;
    }

    if (!GlobalValue::isWeakForLinker(GV.getLinkage()))
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
    auto GS = DefinedGlobals.find(GV.getGUID());
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
      GS = DefinedGlobals.find(GlobalValue::getGUID(OrigId));
      if (GS == DefinedGlobals.end()) {
        // Also check the original non-promoted non-globalized name. In some
        // cases a preempted weak value is linked in as a local copy because
        // it is referenced by an alias (IRLinker::linkGlobalValueProto).
        // In that case, since it was originally not a local value, it was
        // recorded in the index using the original name.
        // FIXME: This may not be needed once PR27866 is fixed.
        GS = DefinedGlobals.find(GlobalValue::getGUID(OrigName));
        assert(GS != DefinedGlobals.end());
      }
    }
    return !GlobalValue::isLocalLinkage(GS->second->linkage());
  };

  // FIXME: See if we can just internalize directly here via linkage changes
  // based on the index, rather than invoking internalizeModule.
  internalizeModule(TheModule, MustPreserveGV);
}

// Automatically import functions in Module \p DestModule based on the summaries
// index.
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
              MDNode::get(DestModule.getContext(),
                          {MDString::get(DestModule.getContext(),
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
#ifndef NDEBUG
    for (GlobalAlias &GA : SrcModule->aliases()) {
      if (!GA.hasName())
        continue;
      auto GUID = GA.getGUID();
      assert(!ImportGUIDs.count(GUID) && "Unexpected alias in import list");
      DEBUG(dbgs() << "Not importing alias " << GUID
                   << " " << GA.getName() << " from "
                   << SrcModule->getSourceFileName() << "\n");
    }
#endif

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
    for (auto &S : I.second.SummaryList) {
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

  explicit FunctionImportLegacyPass() : ModulePass(ID) {}

  /// Specify pass name for debug output
  StringRef getPassName() const override { return "Function Importing"; }

  bool runOnModule(Module &M) override {
    if (skipModule(M))
      return false;

    return doImportingForModule(M);
  }
};

} // end anonymous namespace

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

} // end namespace llvm
