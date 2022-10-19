//===- FunctionImport.cpp - ThinLTO Summary-based Function Import ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "llvm/ADT/StringRef.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/AutoUpgrade.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalObject.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/InitializePasses.h"
#include "llvm/Linker/IRMover.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/FunctionImportUtils.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <cassert>
#include <memory>
#include <set>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "function-import"

STATISTIC(NumImportedFunctionsThinLink,
          "Number of functions thin link decided to import");
STATISTIC(NumImportedHotFunctionsThinLink,
          "Number of hot functions thin link decided to import");
STATISTIC(NumImportedCriticalFunctionsThinLink,
          "Number of critical functions thin link decided to import");
STATISTIC(NumImportedGlobalVarsThinLink,
          "Number of global variables thin link decided to import");
STATISTIC(NumImportedFunctions, "Number of functions imported in backend");
STATISTIC(NumImportedGlobalVars,
          "Number of global variables imported in backend");
STATISTIC(NumImportedModules, "Number of modules imported from");
STATISTIC(NumDeadSymbols, "Number of dead stripped symbols in index");
STATISTIC(NumLiveSymbols, "Number of live symbols in index");

/// Limit on instruction count of imported functions.
static cl::opt<unsigned> ImportInstrLimit(
    "import-instr-limit", cl::init(100), cl::Hidden, cl::value_desc("N"),
    cl::desc("Only import functions with less than N instructions"));

static cl::opt<int> ImportCutoff(
    "import-cutoff", cl::init(-1), cl::Hidden, cl::value_desc("N"),
    cl::desc("Only import first N functions if N>=0 (default -1)"));

static cl::opt<bool>
    ForceImportAll("force-import-all", cl::init(false), cl::Hidden,
                   cl::desc("Import functions with noinline attribute"));

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

static cl::opt<bool> PrintImportFailures(
    "print-import-failures", cl::init(false), cl::Hidden,
    cl::desc("Print information for functions rejected for importing"));

static cl::opt<bool> ComputeDead("compute-dead", cl::init(true), cl::Hidden,
                                 cl::desc("Compute dead symbols"));

static cl::opt<bool> EnableImportMetadata(
    "enable-import-metadata", cl::init(false), cl::Hidden,
    cl::desc("Enable import metadata like 'thinlto_src_module'"));

/// Summary file to use for function importing when using -function-import from
/// the command line.
static cl::opt<std::string>
    SummaryFile("summary-file",
                cl::desc("The summary file to use for function importing."));

/// Used when testing importing from distributed indexes via opt
// -function-import.
static cl::opt<bool>
    ImportAllIndex("import-all-index",
                   cl::desc("Import all external functions in index."));

// Load lazily a module from \p FileName in \p Context.
static std::unique_ptr<Module> loadFile(const std::string &FileName,
                                        LLVMContext &Context) {
  SMDiagnostic Err;
  LLVM_DEBUG(dbgs() << "Loading '" << FileName << "'\n");
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
             unsigned Threshold, StringRef CallerModulePath,
             FunctionImporter::ImportFailureReason &Reason,
             GlobalValue::GUID GUID) {
  Reason = FunctionImporter::ImportFailureReason::None;
  auto It = llvm::find_if(
      CalleeSummaryList,
      [&](const std::unique_ptr<GlobalValueSummary> &SummaryPtr) {
        auto *GVSummary = SummaryPtr.get();
        if (!Index.isGlobalValueLive(GVSummary)) {
          Reason = FunctionImporter::ImportFailureReason::NotLive;
          return false;
        }

        if (GlobalValue::isInterposableLinkage(GVSummary->linkage())) {
          Reason = FunctionImporter::ImportFailureReason::InterposableLinkage;
          // There is no point in importing these, we can't inline them
          return false;
        }

        auto *Summary = cast<FunctionSummary>(GVSummary->getBaseObject());

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
            Summary->modulePath() != CallerModulePath) {
          Reason =
              FunctionImporter::ImportFailureReason::LocalLinkageNotInModule;
          return false;
        }

        if ((Summary->instCount() > Threshold) &&
            !Summary->fflags().AlwaysInline && !ForceImportAll) {
          Reason = FunctionImporter::ImportFailureReason::TooLarge;
          return false;
        }

        // Skip if it isn't legal to import (e.g. may reference unpromotable
        // locals).
        if (Summary->notEligibleToImport()) {
          Reason = FunctionImporter::ImportFailureReason::NotEligible;
          return false;
        }

        // Don't bother importing if we can't inline it anyway.
        if (Summary->fflags().NoInline && !ForceImportAll) {
          Reason = FunctionImporter::ImportFailureReason::NoInline;
          return false;
        }

        return true;
      });
  if (It == CalleeSummaryList.end())
    return nullptr;

  return cast<GlobalValueSummary>(It->get());
}

namespace {

using EdgeInfo =
    std::tuple<const GlobalValueSummary *, unsigned /* Threshold */>;

} // anonymous namespace

static bool shouldImportGlobal(const ValueInfo &VI,
                               const GVSummaryMapTy &DefinedGVSummaries) {
  const auto &GVS = DefinedGVSummaries.find(VI.getGUID());
  if (GVS == DefinedGVSummaries.end())
    return true;
  // We should not skip import if the module contains a definition with
  // interposable linkage type. This is required for correctness in
  // the situation with two following conditions:
  // * the def with interposable linkage is non-prevailing,
  // * there is a prevailing def available for import and marked read-only.
  // In this case, the non-prevailing def will be converted to a declaration,
  // while the prevailing one becomes internal, thus no definitions will be
  // available for linking. In order to prevent undefined symbol link error,
  // the prevailing definition must be imported.
  // FIXME: Consider adding a check that the suitable prevailing definition
  // exists and marked read-only.
  if (VI.getSummaryList().size() > 1 &&
      GlobalValue::isInterposableLinkage(GVS->second->linkage()))
    return true;

  return false;
}

static void computeImportForReferencedGlobals(
    const GlobalValueSummary &Summary, const ModuleSummaryIndex &Index,
    const GVSummaryMapTy &DefinedGVSummaries,
    SmallVectorImpl<EdgeInfo> &Worklist,
    FunctionImporter::ImportMapTy &ImportList,
    StringMap<FunctionImporter::ExportSetTy> *ExportLists) {
  for (const auto &VI : Summary.refs()) {
    if (!shouldImportGlobal(VI, DefinedGVSummaries)) {
      LLVM_DEBUG(
          dbgs() << "Ref ignored! Target already in destination module.\n");
      continue;
    }

    LLVM_DEBUG(dbgs() << " ref -> " << VI << "\n");

    // If this is a local variable, make sure we import the copy
    // in the caller's module. The only time a local variable can
    // share an entry in the index is if there is a local with the same name
    // in another module that had the same source file name (in a different
    // directory), where each was compiled in their own directory so there
    // was not distinguishing path.
    auto LocalNotInModule = [&](const GlobalValueSummary *RefSummary) -> bool {
      return GlobalValue::isLocalLinkage(RefSummary->linkage()) &&
             RefSummary->modulePath() != Summary.modulePath();
    };

    for (const auto &RefSummary : VI.getSummaryList())
      if (isa<GlobalVarSummary>(RefSummary.get()) &&
          Index.canImportGlobalVar(RefSummary.get(), /* AnalyzeRefs */ true) &&
          !LocalNotInModule(RefSummary.get())) {
        auto ILI = ImportList[RefSummary->modulePath()].insert(VI.getGUID());
        // Only update stat and exports if we haven't already imported this
        // variable.
        if (!ILI.second)
          break;
        NumImportedGlobalVarsThinLink++;
        // Any references made by this variable will be marked exported later,
        // in ComputeCrossModuleImport, after import decisions are complete,
        // which is more efficient than adding them here.
        if (ExportLists)
          (*ExportLists)[RefSummary->modulePath()].insert(VI);

        // If variable is not writeonly we attempt to recursively analyze
        // its references in order to import referenced constants.
        if (!Index.isWriteOnly(cast<GlobalVarSummary>(RefSummary.get())))
          Worklist.emplace_back(RefSummary.get(), 0);
        break;
      }
  }
}

static const char *
getFailureName(FunctionImporter::ImportFailureReason Reason) {
  switch (Reason) {
  case FunctionImporter::ImportFailureReason::None:
    return "None";
  case FunctionImporter::ImportFailureReason::GlobalVar:
    return "GlobalVar";
  case FunctionImporter::ImportFailureReason::NotLive:
    return "NotLive";
  case FunctionImporter::ImportFailureReason::TooLarge:
    return "TooLarge";
  case FunctionImporter::ImportFailureReason::InterposableLinkage:
    return "InterposableLinkage";
  case FunctionImporter::ImportFailureReason::LocalLinkageNotInModule:
    return "LocalLinkageNotInModule";
  case FunctionImporter::ImportFailureReason::NotEligible:
    return "NotEligible";
  case FunctionImporter::ImportFailureReason::NoInline:
    return "NoInline";
  }
  llvm_unreachable("invalid reason");
}

/// Compute the list of functions to import for a given caller. Mark these
/// imported functions and the symbols they reference in their source module as
/// exported from their source module.
static void computeImportForFunction(
    const FunctionSummary &Summary, const ModuleSummaryIndex &Index,
    const unsigned Threshold, const GVSummaryMapTy &DefinedGVSummaries,
    SmallVectorImpl<EdgeInfo> &Worklist,
    FunctionImporter::ImportMapTy &ImportList,
    StringMap<FunctionImporter::ExportSetTy> *ExportLists,
    FunctionImporter::ImportThresholdsTy &ImportThresholds) {
  computeImportForReferencedGlobals(Summary, Index, DefinedGVSummaries,
                                    Worklist, ImportList, ExportLists);
  static int ImportCount = 0;
  for (const auto &Edge : Summary.calls()) {
    ValueInfo VI = Edge.first;
    LLVM_DEBUG(dbgs() << " edge -> " << VI << " Threshold:" << Threshold
                      << "\n");

    if (ImportCutoff >= 0 && ImportCount >= ImportCutoff) {
      LLVM_DEBUG(dbgs() << "ignored! import-cutoff value of " << ImportCutoff
                        << " reached.\n");
      continue;
    }

    if (DefinedGVSummaries.count(VI.getGUID())) {
      // FIXME: Consider not skipping import if the module contains
      // a non-prevailing def with interposable linkage. The prevailing copy
      // can safely be imported (see shouldImportGlobal()).
      LLVM_DEBUG(dbgs() << "ignored! Target already in destination module.\n");
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
        Threshold * GetBonusMultiplier(Edge.second.getHotness());

    auto IT = ImportThresholds.insert(std::make_pair(
        VI.getGUID(), std::make_tuple(NewThreshold, nullptr, nullptr)));
    bool PreviouslyVisited = !IT.second;
    auto &ProcessedThreshold = std::get<0>(IT.first->second);
    auto &CalleeSummary = std::get<1>(IT.first->second);
    auto &FailureInfo = std::get<2>(IT.first->second);

    bool IsHotCallsite =
        Edge.second.getHotness() == CalleeInfo::HotnessType::Hot;
    bool IsCriticalCallsite =
        Edge.second.getHotness() == CalleeInfo::HotnessType::Critical;

    const FunctionSummary *ResolvedCalleeSummary = nullptr;
    if (CalleeSummary) {
      assert(PreviouslyVisited);
      // Since the traversal of the call graph is DFS, we can revisit a function
      // a second time with a higher threshold. In this case, it is added back
      // to the worklist with the new threshold (so that its own callee chains
      // can be considered with the higher threshold).
      if (NewThreshold <= ProcessedThreshold) {
        LLVM_DEBUG(
            dbgs() << "ignored! Target was already imported with Threshold "
                   << ProcessedThreshold << "\n");
        continue;
      }
      // Update with new larger threshold.
      ProcessedThreshold = NewThreshold;
      ResolvedCalleeSummary = cast<FunctionSummary>(CalleeSummary);
    } else {
      // If we already rejected importing a callee at the same or higher
      // threshold, don't waste time calling selectCallee.
      if (PreviouslyVisited && NewThreshold <= ProcessedThreshold) {
        LLVM_DEBUG(
            dbgs() << "ignored! Target was already rejected with Threshold "
            << ProcessedThreshold << "\n");
        if (PrintImportFailures) {
          assert(FailureInfo &&
                 "Expected FailureInfo for previously rejected candidate");
          FailureInfo->Attempts++;
        }
        continue;
      }

      FunctionImporter::ImportFailureReason Reason;
      CalleeSummary = selectCallee(Index, VI.getSummaryList(), NewThreshold,
                                   Summary.modulePath(), Reason, VI.getGUID());
      if (!CalleeSummary) {
        // Update with new larger threshold if this was a retry (otherwise
        // we would have already inserted with NewThreshold above). Also
        // update failure info if requested.
        if (PreviouslyVisited) {
          ProcessedThreshold = NewThreshold;
          if (PrintImportFailures) {
            assert(FailureInfo &&
                   "Expected FailureInfo for previously rejected candidate");
            FailureInfo->Reason = Reason;
            FailureInfo->Attempts++;
            FailureInfo->MaxHotness =
                std::max(FailureInfo->MaxHotness, Edge.second.getHotness());
          }
        } else if (PrintImportFailures) {
          assert(!FailureInfo &&
                 "Expected no FailureInfo for newly rejected candidate");
          FailureInfo = std::make_unique<FunctionImporter::ImportFailureInfo>(
              VI, Edge.second.getHotness(), Reason, 1);
        }
        if (ForceImportAll) {
          std::string Msg = std::string("Failed to import function ") +
                            VI.name().str() + " due to " +
                            getFailureName(Reason);
          auto Error = make_error<StringError>(
              Msg, make_error_code(errc::not_supported));
          logAllUnhandledErrors(std::move(Error), errs(),
                                "Error importing module: ");
          break;
        } else {
          LLVM_DEBUG(dbgs()
                     << "ignored! No qualifying callee with summary found.\n");
          continue;
        }
      }

      // "Resolve" the summary
      CalleeSummary = CalleeSummary->getBaseObject();
      ResolvedCalleeSummary = cast<FunctionSummary>(CalleeSummary);

      assert((ResolvedCalleeSummary->fflags().AlwaysInline || ForceImportAll ||
              (ResolvedCalleeSummary->instCount() <= NewThreshold)) &&
             "selectCallee() didn't honor the threshold");

      auto ExportModulePath = ResolvedCalleeSummary->modulePath();
      auto ILI = ImportList[ExportModulePath].insert(VI.getGUID());
      // We previously decided to import this GUID definition if it was already
      // inserted in the set of imports from the exporting module.
      bool PreviouslyImported = !ILI.second;
      if (!PreviouslyImported) {
        NumImportedFunctionsThinLink++;
        if (IsHotCallsite)
          NumImportedHotFunctionsThinLink++;
        if (IsCriticalCallsite)
          NumImportedCriticalFunctionsThinLink++;
      }

      // Any calls/references made by this function will be marked exported
      // later, in ComputeCrossModuleImport, after import decisions are
      // complete, which is more efficient than adding them here.
      if (ExportLists)
        (*ExportLists)[ExportModulePath].insert(VI);
    }

    auto GetAdjustedThreshold = [](unsigned Threshold, bool IsHotCallsite) {
      // Adjust the threshold for next level of imported functions.
      // The threshold is different for hot callsites because we can then
      // inline chains of hot calls.
      if (IsHotCallsite)
        return Threshold * ImportHotInstrFactor;
      return Threshold * ImportInstrFactor;
    };

    const auto AdjThreshold = GetAdjustedThreshold(Threshold, IsHotCallsite);

    ImportCount++;

    // Insert the newly imported function to the worklist.
    Worklist.emplace_back(ResolvedCalleeSummary, AdjThreshold);
  }
}

/// Given the list of globals defined in a module, compute the list of imports
/// as well as the list of "exports", i.e. the list of symbols referenced from
/// another module (that may require promotion).
static void ComputeImportForModule(
    const GVSummaryMapTy &DefinedGVSummaries, const ModuleSummaryIndex &Index,
    StringRef ModName, FunctionImporter::ImportMapTy &ImportList,
    StringMap<FunctionImporter::ExportSetTy> *ExportLists = nullptr) {
  // Worklist contains the list of function imported in this module, for which
  // we will analyse the callees and may import further down the callgraph.
  SmallVector<EdgeInfo, 128> Worklist;
  FunctionImporter::ImportThresholdsTy ImportThresholds;

  // Populate the worklist with the import for the functions in the current
  // module
  for (const auto &GVSummary : DefinedGVSummaries) {
#ifndef NDEBUG
    // FIXME: Change the GVSummaryMapTy to hold ValueInfo instead of GUID
    // so this map look up (and possibly others) can be avoided.
    auto VI = Index.getValueInfo(GVSummary.first);
#endif
    if (!Index.isGlobalValueLive(GVSummary.second)) {
      LLVM_DEBUG(dbgs() << "Ignores Dead GUID: " << VI << "\n");
      continue;
    }
    auto *FuncSummary =
        dyn_cast<FunctionSummary>(GVSummary.second->getBaseObject());
    if (!FuncSummary)
      // Skip import for global variables
      continue;
    LLVM_DEBUG(dbgs() << "Initialize import for " << VI << "\n");
    computeImportForFunction(*FuncSummary, Index, ImportInstrLimit,
                             DefinedGVSummaries, Worklist, ImportList,
                             ExportLists, ImportThresholds);
  }

  // Process the newly imported functions and add callees to the worklist.
  while (!Worklist.empty()) {
    auto GVInfo = Worklist.pop_back_val();
    auto *Summary = std::get<0>(GVInfo);
    auto Threshold = std::get<1>(GVInfo);

    if (auto *FS = dyn_cast<FunctionSummary>(Summary))
      computeImportForFunction(*FS, Index, Threshold, DefinedGVSummaries,
                               Worklist, ImportList, ExportLists,
                               ImportThresholds);
    else
      computeImportForReferencedGlobals(*Summary, Index, DefinedGVSummaries,
                                        Worklist, ImportList, ExportLists);
  }

  // Print stats about functions considered but rejected for importing
  // when requested.
  if (PrintImportFailures) {
    dbgs() << "Missed imports into module " << ModName << "\n";
    for (auto &I : ImportThresholds) {
      auto &ProcessedThreshold = std::get<0>(I.second);
      auto &CalleeSummary = std::get<1>(I.second);
      auto &FailureInfo = std::get<2>(I.second);
      if (CalleeSummary)
        continue; // We are going to import.
      assert(FailureInfo);
      FunctionSummary *FS = nullptr;
      if (!FailureInfo->VI.getSummaryList().empty())
        FS = dyn_cast<FunctionSummary>(
            FailureInfo->VI.getSummaryList()[0]->getBaseObject());
      dbgs() << FailureInfo->VI
             << ": Reason = " << getFailureName(FailureInfo->Reason)
             << ", Threshold = " << ProcessedThreshold
             << ", Size = " << (FS ? (int)FS->instCount() : -1)
             << ", MaxHotness = " << getHotnessName(FailureInfo->MaxHotness)
             << ", Attempts = " << FailureInfo->Attempts << "\n";
    }
  }
}

#ifndef NDEBUG
static bool isGlobalVarSummary(const ModuleSummaryIndex &Index, ValueInfo VI) {
  auto SL = VI.getSummaryList();
  return SL.empty()
             ? false
             : SL[0]->getSummaryKind() == GlobalValueSummary::GlobalVarKind;
}

static bool isGlobalVarSummary(const ModuleSummaryIndex &Index,
                               GlobalValue::GUID G) {
  if (const auto &VI = Index.getValueInfo(G))
    return isGlobalVarSummary(Index, VI);
  return false;
}

template <class T>
static unsigned numGlobalVarSummaries(const ModuleSummaryIndex &Index,
                                      T &Cont) {
  unsigned NumGVS = 0;
  for (auto &V : Cont)
    if (isGlobalVarSummary(Index, V))
      ++NumGVS;
  return NumGVS;
}
#endif

#ifndef NDEBUG
static bool
checkVariableImport(const ModuleSummaryIndex &Index,
                    StringMap<FunctionImporter::ImportMapTy> &ImportLists,
                    StringMap<FunctionImporter::ExportSetTy> &ExportLists) {

  DenseSet<GlobalValue::GUID> FlattenedImports;

  for (auto &ImportPerModule : ImportLists)
    for (auto &ExportPerModule : ImportPerModule.second)
      FlattenedImports.insert(ExportPerModule.second.begin(),
                              ExportPerModule.second.end());

  // Checks that all GUIDs of read/writeonly vars we see in export lists
  // are also in the import lists. Otherwise we my face linker undefs,
  // because readonly and writeonly vars are internalized in their
  // source modules.
  auto IsReadOrWriteOnlyVar = [&](StringRef ModulePath, const ValueInfo &VI) {
    auto *GVS = dyn_cast_or_null<GlobalVarSummary>(
        Index.findSummaryInModule(VI, ModulePath));
    return GVS && (Index.isReadOnly(GVS) || Index.isWriteOnly(GVS));
  };

  for (auto &ExportPerModule : ExportLists)
    for (auto &VI : ExportPerModule.second)
      if (!FlattenedImports.count(VI.getGUID()) &&
          IsReadOrWriteOnlyVar(ExportPerModule.first(), VI))
        return false;

  return true;
}
#endif

/// Compute all the import and export for every module using the Index.
void llvm::ComputeCrossModuleImport(
    const ModuleSummaryIndex &Index,
    const StringMap<GVSummaryMapTy> &ModuleToDefinedGVSummaries,
    StringMap<FunctionImporter::ImportMapTy> &ImportLists,
    StringMap<FunctionImporter::ExportSetTy> &ExportLists) {
  // For each module that has function defined, compute the import/export lists.
  for (const auto &DefinedGVSummaries : ModuleToDefinedGVSummaries) {
    auto &ImportList = ImportLists[DefinedGVSummaries.first()];
    LLVM_DEBUG(dbgs() << "Computing import for Module '"
                      << DefinedGVSummaries.first() << "'\n");
    ComputeImportForModule(DefinedGVSummaries.second, Index,
                           DefinedGVSummaries.first(), ImportList,
                           &ExportLists);
  }

  // When computing imports we only added the variables and functions being
  // imported to the export list. We also need to mark any references and calls
  // they make as exported as well. We do this here, as it is more efficient
  // since we may import the same values multiple times into different modules
  // during the import computation.
  for (auto &ELI : ExportLists) {
    FunctionImporter::ExportSetTy NewExports;
    const auto &DefinedGVSummaries =
        ModuleToDefinedGVSummaries.lookup(ELI.first());
    for (auto &EI : ELI.second) {
      // Find the copy defined in the exporting module so that we can mark the
      // values it references in that specific definition as exported.
      // Below we will add all references and called values, without regard to
      // whether they are also defined in this module. We subsequently prune the
      // list to only include those defined in the exporting module, see comment
      // there as to why.
      auto DS = DefinedGVSummaries.find(EI.getGUID());
      // Anything marked exported during the import computation must have been
      // defined in the exporting module.
      assert(DS != DefinedGVSummaries.end());
      auto *S = DS->getSecond();
      S = S->getBaseObject();
      if (auto *GVS = dyn_cast<GlobalVarSummary>(S)) {
        // Export referenced functions and variables. We don't export/promote
        // objects referenced by writeonly variable initializer, because
        // we convert such variables initializers to "zeroinitializer".
        // See processGlobalForThinLTO.
        if (!Index.isWriteOnly(GVS))
          for (const auto &VI : GVS->refs())
            NewExports.insert(VI);
      } else {
        auto *FS = cast<FunctionSummary>(S);
        for (const auto &Edge : FS->calls())
          NewExports.insert(Edge.first);
        for (const auto &Ref : FS->refs())
          NewExports.insert(Ref);
      }
    }
    // Prune list computed above to only include values defined in the exporting
    // module. We do this after the above insertion since we may hit the same
    // ref/call target multiple times in above loop, and it is more efficient to
    // avoid a set lookup each time.
    for (auto EI = NewExports.begin(); EI != NewExports.end();) {
      if (!DefinedGVSummaries.count(EI->getGUID()))
        NewExports.erase(EI++);
      else
        ++EI;
    }
    ELI.second.insert(NewExports.begin(), NewExports.end());
  }

  assert(checkVariableImport(Index, ImportLists, ExportLists));
#ifndef NDEBUG
  LLVM_DEBUG(dbgs() << "Import/Export lists for " << ImportLists.size()
                    << " modules:\n");
  for (auto &ModuleImports : ImportLists) {
    auto ModName = ModuleImports.first();
    auto &Exports = ExportLists[ModName];
    unsigned NumGVS = numGlobalVarSummaries(Index, Exports);
    LLVM_DEBUG(dbgs() << "* Module " << ModName << " exports "
                      << Exports.size() - NumGVS << " functions and " << NumGVS
                      << " vars. Imports from " << ModuleImports.second.size()
                      << " modules.\n");
    for (auto &Src : ModuleImports.second) {
      auto SrcModName = Src.first();
      unsigned NumGVSPerMod = numGlobalVarSummaries(Index, Src.second);
      LLVM_DEBUG(dbgs() << " - " << Src.second.size() - NumGVSPerMod
                        << " functions imported from " << SrcModName << "\n");
      LLVM_DEBUG(dbgs() << " - " << NumGVSPerMod
                        << " global vars imported from " << SrcModName << "\n");
    }
  }
#endif
}

#ifndef NDEBUG
static void dumpImportListForModule(const ModuleSummaryIndex &Index,
                                    StringRef ModulePath,
                                    FunctionImporter::ImportMapTy &ImportList) {
  LLVM_DEBUG(dbgs() << "* Module " << ModulePath << " imports from "
                    << ImportList.size() << " modules.\n");
  for (auto &Src : ImportList) {
    auto SrcModName = Src.first();
    unsigned NumGVSPerMod = numGlobalVarSummaries(Index, Src.second);
    LLVM_DEBUG(dbgs() << " - " << Src.second.size() - NumGVSPerMod
                      << " functions imported from " << SrcModName << "\n");
    LLVM_DEBUG(dbgs() << " - " << NumGVSPerMod << " vars imported from "
                      << SrcModName << "\n");
  }
}
#endif

/// Compute all the imports for the given module in the Index.
void llvm::ComputeCrossModuleImportForModule(
    StringRef ModulePath, const ModuleSummaryIndex &Index,
    FunctionImporter::ImportMapTy &ImportList) {
  // Collect the list of functions this module defines.
  // GUID -> Summary
  GVSummaryMapTy FunctionSummaryMap;
  Index.collectDefinedFunctionsForModule(ModulePath, FunctionSummaryMap);

  // Compute the import list for this module.
  LLVM_DEBUG(dbgs() << "Computing import for Module '" << ModulePath << "'\n");
  ComputeImportForModule(FunctionSummaryMap, Index, ModulePath, ImportList);

#ifndef NDEBUG
  dumpImportListForModule(Index, ModulePath, ImportList);
#endif
}

// Mark all external summaries in Index for import into the given module.
// Used for distributed builds using a distributed index.
void llvm::ComputeCrossModuleImportForModuleFromIndex(
    StringRef ModulePath, const ModuleSummaryIndex &Index,
    FunctionImporter::ImportMapTy &ImportList) {
  for (const auto &GlobalList : Index) {
    // Ignore entries for undefined references.
    if (GlobalList.second.SummaryList.empty())
      continue;

    auto GUID = GlobalList.first;
    assert(GlobalList.second.SummaryList.size() == 1 &&
           "Expected individual combined index to have one summary per GUID");
    auto &Summary = GlobalList.second.SummaryList[0];
    // Skip the summaries for the importing module. These are included to
    // e.g. record required linkage changes.
    if (Summary->modulePath() == ModulePath)
      continue;
    // Add an entry to provoke importing by thinBackend.
    ImportList[Summary->modulePath()].insert(GUID);
  }
#ifndef NDEBUG
  dumpImportListForModule(Index, ModulePath, ImportList);
#endif
}

// For SamplePGO, the indirect call targets for local functions will
// have its original name annotated in profile. We try to find the
// corresponding PGOFuncName as the GUID, and fix up the edges
// accordingly.
void updateValueInfoForIndirectCalls(ModuleSummaryIndex &Index,
                                     FunctionSummary *FS) {
  for (auto &EI : FS->mutableCalls()) {
    if (!EI.first.getSummaryList().empty())
      continue;
    auto GUID = Index.getGUIDFromOriginalID(EI.first.getGUID());
    if (GUID == 0)
      continue;
    // Update the edge to point directly to the correct GUID.
    auto VI = Index.getValueInfo(GUID);
    if (llvm::any_of(
            VI.getSummaryList(),
            [&](const std::unique_ptr<GlobalValueSummary> &SummaryPtr) {
              // The mapping from OriginalId to GUID may return a GUID
              // that corresponds to a static variable. Filter it out here.
              // This can happen when
              // 1) There is a call to a library function which is not defined
              // in the index.
              // 2) There is a static variable with the  OriginalGUID identical
              // to the GUID of the library function in 1);
              // When this happens the static variable in 2) will be found,
              // which needs to be filtered out.
              return SummaryPtr->getSummaryKind() ==
                     GlobalValueSummary::GlobalVarKind;
            }))
      continue;
    EI.first = VI;
  }
}

void llvm::updateIndirectCalls(ModuleSummaryIndex &Index) {
  for (const auto &Entry : Index) {
    for (const auto &S : Entry.second.SummaryList) {
      if (auto *FS = dyn_cast<FunctionSummary>(S.get()))
        updateValueInfoForIndirectCalls(Index, FS);
    }
  }
}

void llvm::computeDeadSymbolsAndUpdateIndirectCalls(
    ModuleSummaryIndex &Index,
    const DenseSet<GlobalValue::GUID> &GUIDPreservedSymbols,
    function_ref<PrevailingType(GlobalValue::GUID)> isPrevailing) {
  assert(!Index.withGlobalValueDeadStripping());
  if (!ComputeDead ||
      // Don't do anything when nothing is live, this is friendly with tests.
      GUIDPreservedSymbols.empty()) {
    // Still need to update indirect calls.
    updateIndirectCalls(Index);
    return;
  }
  unsigned LiveSymbols = 0;
  SmallVector<ValueInfo, 128> Worklist;
  Worklist.reserve(GUIDPreservedSymbols.size() * 2);
  for (auto GUID : GUIDPreservedSymbols) {
    ValueInfo VI = Index.getValueInfo(GUID);
    if (!VI)
      continue;
    for (const auto &S : VI.getSummaryList())
      S->setLive(true);
  }

  // Add values flagged in the index as live roots to the worklist.
  for (const auto &Entry : Index) {
    auto VI = Index.getValueInfo(Entry);
    for (const auto &S : Entry.second.SummaryList) {
      if (auto *FS = dyn_cast<FunctionSummary>(S.get()))
        updateValueInfoForIndirectCalls(Index, FS);
      if (S->isLive()) {
        LLVM_DEBUG(dbgs() << "Live root: " << VI << "\n");
        Worklist.push_back(VI);
        ++LiveSymbols;
        break;
      }
    }
  }

  // Make value live and add it to the worklist if it was not live before.
  auto visit = [&](ValueInfo VI, bool IsAliasee) {
    // FIXME: If we knew which edges were created for indirect call profiles,
    // we could skip them here. Any that are live should be reached via
    // other edges, e.g. reference edges. Otherwise, using a profile collected
    // on a slightly different binary might provoke preserving, importing
    // and ultimately promoting calls to functions not linked into this
    // binary, which increases the binary size unnecessarily. Note that
    // if this code changes, the importer needs to change so that edges
    // to functions marked dead are skipped.

    if (llvm::any_of(VI.getSummaryList(),
                     [](const std::unique_ptr<llvm::GlobalValueSummary> &S) {
                       return S->isLive();
                     }))
      return;

    // We only keep live symbols that are known to be non-prevailing if any are
    // available_externally, linkonceodr, weakodr. Those symbols are discarded
    // later in the EliminateAvailableExternally pass and setting them to
    // not-live could break downstreams users of liveness information (PR36483)
    // or limit optimization opportunities.
    if (isPrevailing(VI.getGUID()) == PrevailingType::No) {
      bool KeepAliveLinkage = false;
      bool Interposable = false;
      for (const auto &S : VI.getSummaryList()) {
        if (S->linkage() == GlobalValue::AvailableExternallyLinkage ||
            S->linkage() == GlobalValue::WeakODRLinkage ||
            S->linkage() == GlobalValue::LinkOnceODRLinkage)
          KeepAliveLinkage = true;
        else if (GlobalValue::isInterposableLinkage(S->linkage()))
          Interposable = true;
      }

      if (!IsAliasee) {
        if (!KeepAliveLinkage)
          return;

        if (Interposable)
          report_fatal_error(
              "Interposable and available_externally/linkonce_odr/weak_odr "
              "symbol");
      }
    }

    for (const auto &S : VI.getSummaryList())
      S->setLive(true);
    ++LiveSymbols;
    Worklist.push_back(VI);
  };

  while (!Worklist.empty()) {
    auto VI = Worklist.pop_back_val();
    for (const auto &Summary : VI.getSummaryList()) {
      if (auto *AS = dyn_cast<AliasSummary>(Summary.get())) {
        // If this is an alias, visit the aliasee VI to ensure that all copies
        // are marked live and it is added to the worklist for further
        // processing of its references.
        visit(AS->getAliaseeVI(), true);
        continue;
      }
      for (auto Ref : Summary->refs())
        visit(Ref, false);
      if (auto *FS = dyn_cast<FunctionSummary>(Summary.get()))
        for (auto Call : FS->calls())
          visit(Call.first, false);
    }
  }
  Index.setWithGlobalValueDeadStripping();

  unsigned DeadSymbols = Index.size() - LiveSymbols;
  LLVM_DEBUG(dbgs() << LiveSymbols << " symbols Live, and " << DeadSymbols
                    << " symbols Dead \n");
  NumDeadSymbols += DeadSymbols;
  NumLiveSymbols += LiveSymbols;
}

// Compute dead symbols and propagate constants in combined index.
void llvm::computeDeadSymbolsWithConstProp(
    ModuleSummaryIndex &Index,
    const DenseSet<GlobalValue::GUID> &GUIDPreservedSymbols,
    function_ref<PrevailingType(GlobalValue::GUID)> isPrevailing,
    bool ImportEnabled) {
  computeDeadSymbolsAndUpdateIndirectCalls(Index, GUIDPreservedSymbols,
                                           isPrevailing);
  if (ImportEnabled)
    Index.propagateAttributes(GUIDPreservedSymbols);
}

/// Compute the set of summaries needed for a ThinLTO backend compilation of
/// \p ModulePath.
void llvm::gatherImportedSummariesForModule(
    StringRef ModulePath,
    const StringMap<GVSummaryMapTy> &ModuleToDefinedGVSummaries,
    const FunctionImporter::ImportMapTy &ImportList,
    std::map<std::string, GVSummaryMapTy> &ModuleToSummariesForIndex) {
  // Include all summaries from the importing module.
  ModuleToSummariesForIndex[std::string(ModulePath)] =
      ModuleToDefinedGVSummaries.lookup(ModulePath);
  // Include summaries for imports.
  for (const auto &ILI : ImportList) {
    auto &SummariesForIndex =
        ModuleToSummariesForIndex[std::string(ILI.first())];
    const auto &DefinedGVSummaries =
        ModuleToDefinedGVSummaries.lookup(ILI.first());
    for (const auto &GI : ILI.second) {
      const auto &DS = DefinedGVSummaries.find(GI);
      assert(DS != DefinedGVSummaries.end() &&
             "Expected a defined summary for imported global value");
      SummariesForIndex[GI] = DS->second;
    }
  }
}

/// Emit the files \p ModulePath will import from into \p OutputFilename.
std::error_code llvm::EmitImportsFiles(
    StringRef ModulePath, StringRef OutputFilename,
    const std::map<std::string, GVSummaryMapTy> &ModuleToSummariesForIndex) {
  std::error_code EC;
  raw_fd_ostream ImportsOS(OutputFilename, EC, sys::fs::OpenFlags::OF_None);
  if (EC)
    return EC;
  for (const auto &ILI : ModuleToSummariesForIndex)
    // The ModuleToSummariesForIndex map includes an entry for the current
    // Module (needed for writing out the index files). We don't want to
    // include it in the imports file, however, so filter it out.
    if (ILI.first != ModulePath)
      ImportsOS << ILI.first << "\n";
  return std::error_code();
}

bool llvm::convertToDeclaration(GlobalValue &GV) {
  LLVM_DEBUG(dbgs() << "Converting to a declaration: `" << GV.getName()
                    << "\n");
  if (Function *F = dyn_cast<Function>(&GV)) {
    F->deleteBody();
    F->clearMetadata();
    F->setComdat(nullptr);
  } else if (GlobalVariable *V = dyn_cast<GlobalVariable>(&GV)) {
    V->setInitializer(nullptr);
    V->setLinkage(GlobalValue::ExternalLinkage);
    V->clearMetadata();
    V->setComdat(nullptr);
  } else {
    GlobalValue *NewGV;
    if (GV.getValueType()->isFunctionTy())
      NewGV =
          Function::Create(cast<FunctionType>(GV.getValueType()),
                           GlobalValue::ExternalLinkage, GV.getAddressSpace(),
                           "", GV.getParent());
    else
      NewGV =
          new GlobalVariable(*GV.getParent(), GV.getValueType(),
                             /*isConstant*/ false, GlobalValue::ExternalLinkage,
                             /*init*/ nullptr, "",
                             /*insertbefore*/ nullptr, GV.getThreadLocalMode(),
                             GV.getType()->getAddressSpace());
    NewGV->takeName(&GV);
    GV.replaceAllUsesWith(NewGV);
    return false;
  }
  if (!GV.isImplicitDSOLocal())
    GV.setDSOLocal(false);
  return true;
}

void llvm::thinLTOFinalizeInModule(Module &TheModule,
                                   const GVSummaryMapTy &DefinedGlobals,
                                   bool PropagateAttrs) {
  auto FinalizeInModule = [&](GlobalValue &GV, bool Propagate = false) {
    // See if the global summary analysis computed a new resolved linkage.
    const auto &GS = DefinedGlobals.find(GV.getGUID());
    if (GS == DefinedGlobals.end())
      return;

    if (Propagate)
      if (FunctionSummary *FS = dyn_cast<FunctionSummary>(GS->second)) {
        if (Function *F = dyn_cast<Function>(&GV)) {
          // TODO: propagate ReadNone and ReadOnly.
          if (FS->fflags().ReadNone && !F->doesNotAccessMemory())
            F->setDoesNotAccessMemory();

          if (FS->fflags().ReadOnly && !F->onlyReadsMemory())
            F->setOnlyReadsMemory();

          if (FS->fflags().NoRecurse && !F->doesNotRecurse())
            F->setDoesNotRecurse();

          if (FS->fflags().NoUnwind && !F->doesNotThrow())
            F->setDoesNotThrow();
        }
      }

    auto NewLinkage = GS->second->linkage();
    if (GlobalValue::isLocalLinkage(GV.getLinkage()) ||
        // Don't internalize anything here, because the code below
        // lacks necessary correctness checks. Leave this job to
        // LLVM 'internalize' pass.
        GlobalValue::isLocalLinkage(NewLinkage) ||
        // In case it was dead and already converted to declaration.
        GV.isDeclaration())
      return;

    // Set the potentially more constraining visibility computed from summaries.
    // The DefaultVisibility condition is because older GlobalValueSummary does
    // not record DefaultVisibility and we don't want to change protected/hidden
    // to default.
    if (GS->second->getVisibility() != GlobalValue::DefaultVisibility)
      GV.setVisibility(GS->second->getVisibility());

    if (NewLinkage == GV.getLinkage())
      return;

    // Check for a non-prevailing def that has interposable linkage
    // (e.g. non-odr weak or linkonce). In that case we can't simply
    // convert to available_externally, since it would lose the
    // interposable property and possibly get inlined. Simply drop
    // the definition in that case.
    if (GlobalValue::isAvailableExternallyLinkage(NewLinkage) &&
        GlobalValue::isInterposableLinkage(GV.getLinkage())) {
      if (!convertToDeclaration(GV))
        // FIXME: Change this to collect replaced GVs and later erase
        // them from the parent module once thinLTOResolvePrevailingGUID is
        // changed to enable this for aliases.
        llvm_unreachable("Expected GV to be converted");
    } else {
      // If all copies of the original symbol had global unnamed addr and
      // linkonce_odr linkage, or if all of them had local unnamed addr linkage
      // and are constants, then it should be an auto hide symbol. In that case
      // the thin link would have marked it as CanAutoHide. Add hidden
      // visibility to the symbol to preserve the property.
      if (NewLinkage == GlobalValue::WeakODRLinkage &&
          GS->second->canAutoHide()) {
        assert(GV.canBeOmittedFromSymbolTable());
        GV.setVisibility(GlobalValue::HiddenVisibility);
      }

      LLVM_DEBUG(dbgs() << "ODR fixing up linkage for `" << GV.getName()
                        << "` from " << GV.getLinkage() << " to " << NewLinkage
                        << "\n");
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
    FinalizeInModule(GV, PropagateAttrs);
  for (auto &GV : TheModule.globals())
    FinalizeInModule(GV);
  for (auto &GV : TheModule.aliases())
    FinalizeInModule(GV);
}

/// Run internalization on \p TheModule based on symmary analysis.
void llvm::thinLTOInternalizeModule(Module &TheModule,
                                    const GVSummaryMapTy &DefinedGlobals) {
  // Declare a callback for the internalize pass that will ask for every
  // candidate GlobalValue if it can be internalized or not.
  auto MustPreserveGV = [&](const GlobalValue &GV) -> bool {
    // It may be the case that GV is on a chain of an ifunc, its alias and
    // subsequent aliases. In this case, the summary for the value is not
    // available.
    if (isa<GlobalIFunc>(&GV) ||
        (isa<GlobalAlias>(&GV) &&
         isa<GlobalIFunc>(cast<GlobalAlias>(&GV)->getAliaseeObject())))
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

/// Make alias a clone of its aliasee.
static Function *replaceAliasWithAliasee(Module *SrcModule, GlobalAlias *GA) {
  Function *Fn = cast<Function>(GA->getAliaseeObject());

  ValueToValueMapTy VMap;
  Function *NewFn = CloneFunction(Fn, VMap);
  // Clone should use the original alias's linkage, visibility and name, and we
  // ensure all uses of alias instead use the new clone (casted if necessary).
  NewFn->setLinkage(GA->getLinkage());
  NewFn->setVisibility(GA->getVisibility());
  GA->replaceAllUsesWith(ConstantExpr::getBitCast(NewFn, GA->getType()));
  NewFn->takeName(GA);
  return NewFn;
}

// Internalize values that we marked with specific attribute
// in processGlobalForThinLTO.
static void internalizeGVsAfterImport(Module &M) {
  for (auto &GV : M.globals())
    // Skip GVs which have been converted to declarations
    // by dropDeadSymbols.
    if (!GV.isDeclaration() && GV.hasAttribute("thinlto-internalize")) {
      GV.setLinkage(GlobalValue::InternalLinkage);
      GV.setVisibility(GlobalValue::DefaultVisibility);
    }
}

// Automatically import functions in Module \p DestModule based on the summaries
// index.
Expected<bool> FunctionImporter::importFunctions(
    Module &DestModule, const FunctionImporter::ImportMapTy &ImportList) {
  LLVM_DEBUG(dbgs() << "Starting import for Module "
                    << DestModule.getModuleIdentifier() << "\n");
  unsigned ImportedCount = 0, ImportedGVCount = 0;

  IRMover Mover(DestModule);
  // Do the actual import of functions now, one Module at a time
  std::set<StringRef> ModuleNameOrderedList;
  for (const auto &FunctionsToImportPerModule : ImportList) {
    ModuleNameOrderedList.insert(FunctionsToImportPerModule.first());
  }
  for (const auto &Name : ModuleNameOrderedList) {
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
      LLVM_DEBUG(dbgs() << (Import ? "Is" : "Not") << " importing function "
                        << GUID << " " << F.getName() << " from "
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
      LLVM_DEBUG(dbgs() << (Import ? "Is" : "Not") << " importing global "
                        << GUID << " " << GV.getName() << " from "
                        << SrcModule->getSourceFileName() << "\n");
      if (Import) {
        if (Error Err = GV.materialize())
          return std::move(Err);
        ImportedGVCount += GlobalsToImport.insert(&GV);
      }
    }
    for (GlobalAlias &GA : SrcModule->aliases()) {
      if (!GA.hasName() || isa<GlobalIFunc>(GA.getAliaseeObject()))
        continue;
      auto GUID = GA.getGUID();
      auto Import = ImportGUIDs.count(GUID);
      LLVM_DEBUG(dbgs() << (Import ? "Is" : "Not") << " importing alias "
                        << GUID << " " << GA.getName() << " from "
                        << SrcModule->getSourceFileName() << "\n");
      if (Import) {
        if (Error Err = GA.materialize())
          return std::move(Err);
        // Import alias as a copy of its aliasee.
        GlobalObject *GO = GA.getAliaseeObject();
        if (Error Err = GO->materialize())
          return std::move(Err);
        auto *Fn = replaceAliasWithAliasee(SrcModule.get(), &GA);
        LLVM_DEBUG(dbgs() << "Is importing aliasee fn " << GO->getGUID() << " "
                          << GO->getName() << " from "
                          << SrcModule->getSourceFileName() << "\n");
        if (EnableImportMetadata) {
          // Add 'thinlto_src_module' metadata for statistics and debugging.
          Fn->setMetadata(
              "thinlto_src_module",
              MDNode::get(DestModule.getContext(),
                          {MDString::get(DestModule.getContext(),
                                         SrcModule->getSourceFileName())}));
        }
        GlobalsToImport.insert(Fn);
      }
    }

    // Upgrade debug info after we're done materializing all the globals and we
    // have loaded all the required metadata!
    UpgradeDebugInfo(*SrcModule);

    // Set the partial sample profile ratio in the profile summary module flag
    // of the imported source module, if applicable, so that the profile summary
    // module flag will match with that of the destination module when it's
    // imported.
    SrcModule->setPartialSampleProfileRatio(Index);

    // Link in the specified functions.
    if (renameModuleForThinLTO(*SrcModule, Index, ClearDSOLocalOnDeclarations,
                               &GlobalsToImport))
      return true;

    if (PrintImports) {
      for (const auto *GV : GlobalsToImport)
        dbgs() << DestModule.getSourceFileName() << ": Import " << GV->getName()
               << " from " << SrcModule->getSourceFileName() << "\n";
    }

    if (Error Err = Mover.move(std::move(SrcModule),
                               GlobalsToImport.getArrayRef(), nullptr,
                               /*IsPerformingImport=*/true))
      report_fatal_error(Twine("Function Import: link error: ") +
                         toString(std::move(Err)));

    ImportedCount += GlobalsToImport.size();
    NumImportedModules++;
  }

  internalizeGVsAfterImport(DestModule);

  NumImportedFunctions += (ImportedCount - ImportedGVCount);
  NumImportedGlobalVars += ImportedGVCount;

  LLVM_DEBUG(dbgs() << "Imported " << ImportedCount - ImportedGVCount
                    << " functions for Module "
                    << DestModule.getModuleIdentifier() << "\n");
  LLVM_DEBUG(dbgs() << "Imported " << ImportedGVCount
                    << " global variables for Module "
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
  // If requested, simply import all functions in the index. This is used
  // when testing distributed backend handling via the opt tool, when
  // we have distributed indexes containing exactly the summaries to import.
  if (ImportAllIndex)
    ComputeCrossModuleImportForModuleFromIndex(M.getModuleIdentifier(), *Index,
                                               ImportList);
  else
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
  if (renameModuleForThinLTO(M, *Index, /*ClearDSOLocalOnDeclarations=*/false,
                             /*GlobalsToImport=*/nullptr)) {
    errs() << "Error renaming module\n";
    return false;
  }

  // Perform the import now.
  auto ModuleLoader = [&M](StringRef Identifier) {
    return loadFile(std::string(Identifier), M.getContext());
  };
  FunctionImporter Importer(*Index, ModuleLoader,
                            /*ClearDSOLocalOnDeclarations=*/false);
  Expected<bool> Result = Importer.importFunctions(M, ImportList);

  // FIXME: Probably need to propagate Errors through the pass manager.
  if (!Result) {
    logAllUnhandledErrors(Result.takeError(), errs(),
                          "Error importing module: ");
    return false;
  }

  return *Result;
}

PreservedAnalyses FunctionImportPass::run(Module &M,
                                          ModuleAnalysisManager &AM) {
  if (!doImportingForModule(M))
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}
