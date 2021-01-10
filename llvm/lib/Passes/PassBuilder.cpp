//===- Parsing, selection, and construction of pass pipelines -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file provides the implementation of the PassBuilder based on our
/// static pass registry as well as related functionality. It also provides
/// helpers to aid in analyzing, debugging, and testing passes and pass
/// pipelines.
///
//===----------------------------------------------------------------------===//

#include "llvm/Passes/PassBuilder.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Analysis/AliasAnalysisEvaluator.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/CFGPrinter.h"
#include "llvm/Analysis/CFLAndersAliasAnalysis.h"
#include "llvm/Analysis/CFLSteensAliasAnalysis.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/DDG.h"
#include "llvm/Analysis/DDGPrinter.h"
#include "llvm/Analysis/Delinearization.h"
#include "llvm/Analysis/DemandedBits.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/DivergenceAnalysis.h"
#include "llvm/Analysis/DominanceFrontier.h"
#include "llvm/Analysis/FunctionPropertiesAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/IRSimilarityIdentifier.h"
#include "llvm/Analysis/IVUsers.h"
#include "llvm/Analysis/InlineAdvisor.h"
#include "llvm/Analysis/InlineSizeEstimatorAnalysis.h"
#include "llvm/Analysis/InstCount.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/Analysis/LazyValueInfo.h"
#include "llvm/Analysis/Lint.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/LoopCacheAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopNestAnalysis.h"
#include "llvm/Analysis/MemDerefPrinter.h"
#include "llvm/Analysis/MemoryDependenceAnalysis.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/ModuleDebugInfoPrinter.h"
#include "llvm/Analysis/ModuleSummaryAnalysis.h"
#include "llvm/Analysis/MustExecute.h"
#include "llvm/Analysis/ObjCARCAliasAnalysis.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/PhiValues.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionAliasAnalysis.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/Analysis/StackLifetime.h"
#include "llvm/Analysis/StackSafetyAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/TypeBasedAliasAnalysis.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PrintPasses.h"
#include "llvm/IR/SafepointIRVerifier.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Regex.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/AggressiveInstCombine/AggressiveInstCombine.h"
#include "llvm/Transforms/Coroutines/CoroCleanup.h"
#include "llvm/Transforms/Coroutines/CoroEarly.h"
#include "llvm/Transforms/Coroutines/CoroElide.h"
#include "llvm/Transforms/Coroutines/CoroSplit.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/Annotation2Metadata.h"
#include "llvm/Transforms/IPO/ArgumentPromotion.h"
#include "llvm/Transforms/IPO/Attributor.h"
#include "llvm/Transforms/IPO/BlockExtractor.h"
#include "llvm/Transforms/IPO/CalledValuePropagation.h"
#include "llvm/Transforms/IPO/ConstantMerge.h"
#include "llvm/Transforms/IPO/CrossDSOCFI.h"
#include "llvm/Transforms/IPO/DeadArgumentElimination.h"
#include "llvm/Transforms/IPO/ElimAvailExtern.h"
#include "llvm/Transforms/IPO/ForceFunctionAttrs.h"
#include "llvm/Transforms/IPO/FunctionAttrs.h"
#include "llvm/Transforms/IPO/FunctionImport.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/IPO/GlobalOpt.h"
#include "llvm/Transforms/IPO/GlobalSplit.h"
#include "llvm/Transforms/IPO/HotColdSplitting.h"
#include "llvm/Transforms/IPO/IROutliner.h"
#include "llvm/Transforms/IPO/InferFunctionAttrs.h"
#include "llvm/Transforms/IPO/Inliner.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/IPO/LoopExtractor.h"
#include "llvm/Transforms/IPO/LowerTypeTests.h"
#include "llvm/Transforms/IPO/MergeFunctions.h"
#include "llvm/Transforms/IPO/OpenMPOpt.h"
#include "llvm/Transforms/IPO/PartialInlining.h"
#include "llvm/Transforms/IPO/SCCP.h"
#include "llvm/Transforms/IPO/SampleProfile.h"
#include "llvm/Transforms/IPO/SampleProfileProbe.h"
#include "llvm/Transforms/IPO/StripDeadPrototypes.h"
#include "llvm/Transforms/IPO/StripSymbols.h"
#include "llvm/Transforms/IPO/SyntheticCountsPropagation.h"
#include "llvm/Transforms/IPO/WholeProgramDevirt.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizer.h"
#include "llvm/Transforms/Instrumentation/BoundsChecking.h"
#include "llvm/Transforms/Instrumentation/CGProfile.h"
#include "llvm/Transforms/Instrumentation/ControlHeightReduction.h"
#include "llvm/Transforms/Instrumentation/DataFlowSanitizer.h"
#include "llvm/Transforms/Instrumentation/GCOVProfiler.h"
#include "llvm/Transforms/Instrumentation/HWAddressSanitizer.h"
#include "llvm/Transforms/Instrumentation/InstrOrderFile.h"
#include "llvm/Transforms/Instrumentation/InstrProfiling.h"
#include "llvm/Transforms/Instrumentation/MemProfiler.h"
#include "llvm/Transforms/Instrumentation/MemorySanitizer.h"
#include "llvm/Transforms/Instrumentation/PGOInstrumentation.h"
#include "llvm/Transforms/Instrumentation/PoisonChecking.h"
#include "llvm/Transforms/Instrumentation/SanitizerCoverage.h"
#include "llvm/Transforms/Instrumentation/ThreadSanitizer.h"
#include "llvm/Transforms/ObjCARC.h"
#include "llvm/Transforms/Scalar/ADCE.h"
#include "llvm/Transforms/Scalar/AlignmentFromAssumptions.h"
#include "llvm/Transforms/Scalar/AnnotationRemarks.h"
#include "llvm/Transforms/Scalar/BDCE.h"
#include "llvm/Transforms/Scalar/CallSiteSplitting.h"
#include "llvm/Transforms/Scalar/ConstantHoisting.h"
#include "llvm/Transforms/Scalar/ConstraintElimination.h"
#include "llvm/Transforms/Scalar/CorrelatedValuePropagation.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Scalar/DeadStoreElimination.h"
#include "llvm/Transforms/Scalar/DivRemPairs.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/Float2Int.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/GuardWidening.h"
#include "llvm/Transforms/Scalar/IVUsersPrinter.h"
#include "llvm/Transforms/Scalar/IndVarSimplify.h"
#include "llvm/Transforms/Scalar/InductiveRangeCheckElimination.h"
#include "llvm/Transforms/Scalar/InferAddressSpaces.h"
#include "llvm/Transforms/Scalar/InstSimplifyPass.h"
#include "llvm/Transforms/Scalar/JumpThreading.h"
#include "llvm/Transforms/Scalar/LICM.h"
#include "llvm/Transforms/Scalar/LoopAccessAnalysisPrinter.h"
#include "llvm/Transforms/Scalar/LoopDataPrefetch.h"
#include "llvm/Transforms/Scalar/LoopDeletion.h"
#include "llvm/Transforms/Scalar/LoopDistribute.h"
#include "llvm/Transforms/Scalar/LoopFlatten.h"
#include "llvm/Transforms/Scalar/LoopFuse.h"
#include "llvm/Transforms/Scalar/LoopIdiomRecognize.h"
#include "llvm/Transforms/Scalar/LoopInstSimplify.h"
#include "llvm/Transforms/Scalar/LoopInterchange.h"
#include "llvm/Transforms/Scalar/LoopLoadElimination.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include "llvm/Transforms/Scalar/LoopPredication.h"
#include "llvm/Transforms/Scalar/LoopReroll.h"
#include "llvm/Transforms/Scalar/LoopRotation.h"
#include "llvm/Transforms/Scalar/LoopSimplifyCFG.h"
#include "llvm/Transforms/Scalar/LoopSink.h"
#include "llvm/Transforms/Scalar/LoopStrengthReduce.h"
#include "llvm/Transforms/Scalar/LoopUnrollAndJamPass.h"
#include "llvm/Transforms/Scalar/LoopUnrollPass.h"
#include "llvm/Transforms/Scalar/LoopVersioningLICM.h"
#include "llvm/Transforms/Scalar/LowerAtomic.h"
#include "llvm/Transforms/Scalar/LowerConstantIntrinsics.h"
#include "llvm/Transforms/Scalar/LowerExpectIntrinsic.h"
#include "llvm/Transforms/Scalar/LowerGuardIntrinsic.h"
#include "llvm/Transforms/Scalar/LowerMatrixIntrinsics.h"
#include "llvm/Transforms/Scalar/LowerWidenableCondition.h"
#include "llvm/Transforms/Scalar/MakeGuardsExplicit.h"
#include "llvm/Transforms/Scalar/MemCpyOptimizer.h"
#include "llvm/Transforms/Scalar/MergeICmps.h"
#include "llvm/Transforms/Scalar/MergedLoadStoreMotion.h"
#include "llvm/Transforms/Scalar/NaryReassociate.h"
#include "llvm/Transforms/Scalar/NewGVN.h"
#include "llvm/Transforms/Scalar/PartiallyInlineLibCalls.h"
#include "llvm/Transforms/Scalar/Reassociate.h"
#include "llvm/Transforms/Scalar/Reg2Mem.h"
#include "llvm/Transforms/Scalar/RewriteStatepointsForGC.h"
#include "llvm/Transforms/Scalar/SCCP.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Scalar/ScalarizeMaskedMemIntrin.h"
#include "llvm/Transforms/Scalar/Scalarizer.h"
#include "llvm/Transforms/Scalar/SeparateConstOffsetFromGEP.h"
#include "llvm/Transforms/Scalar/SimpleLoopUnswitch.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Scalar/Sink.h"
#include "llvm/Transforms/Scalar/SpeculateAroundPHIs.h"
#include "llvm/Transforms/Scalar/SpeculativeExecution.h"
#include "llvm/Transforms/Scalar/StraightLineStrengthReduce.h"
#include "llvm/Transforms/Scalar/StructurizeCFG.h"
#include "llvm/Transforms/Scalar/TailRecursionElimination.h"
#include "llvm/Transforms/Scalar/WarnMissedTransforms.h"
#include "llvm/Transforms/Utils/AddDiscriminators.h"
#include "llvm/Transforms/Utils/AssumeBundleBuilder.h"
#include "llvm/Transforms/Utils/BreakCriticalEdges.h"
#include "llvm/Transforms/Utils/CanonicalizeAliases.h"
#include "llvm/Transforms/Utils/CanonicalizeFreezeInLoops.h"
#include "llvm/Transforms/Utils/EntryExitInstrumenter.h"
#include "llvm/Transforms/Utils/FixIrreducible.h"
#include "llvm/Transforms/Utils/HelloWorld.h"
#include "llvm/Transforms/Utils/InjectTLIMappings.h"
#include "llvm/Transforms/Utils/InstructionNamer.h"
#include "llvm/Transforms/Utils/LCSSA.h"
#include "llvm/Transforms/Utils/LibCallsShrinkWrap.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/LoopVersioning.h"
#include "llvm/Transforms/Utils/LowerInvoke.h"
#include "llvm/Transforms/Utils/LowerSwitch.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"
#include "llvm/Transforms/Utils/MetaRenamer.h"
#include "llvm/Transforms/Utils/NameAnonGlobals.h"
#include "llvm/Transforms/Utils/StripGCRelocates.h"
#include "llvm/Transforms/Utils/StripNonLineTableDebugInfo.h"
#include "llvm/Transforms/Utils/SymbolRewriter.h"
#include "llvm/Transforms/Utils/UnifyFunctionExitNodes.h"
#include "llvm/Transforms/Utils/UnifyLoopExits.h"
#include "llvm/Transforms/Utils/UniqueInternalLinkageNames.h"
#include "llvm/Transforms/Vectorize/LoadStoreVectorizer.h"
#include "llvm/Transforms/Vectorize/LoopVectorize.h"
#include "llvm/Transforms/Vectorize/SLPVectorizer.h"
#include "llvm/Transforms/Vectorize/VectorCombine.h"

using namespace llvm;

extern cl::opt<unsigned> MaxDevirtIterations;

static cl::opt<InliningAdvisorMode> UseInlineAdvisor(
    "enable-ml-inliner", cl::init(InliningAdvisorMode::Default), cl::Hidden,
    cl::desc("Enable ML policy for inliner. Currently trained for -Oz only"),
    cl::values(clEnumValN(InliningAdvisorMode::Default, "default",
                          "Heuristics-based inliner version."),
               clEnumValN(InliningAdvisorMode::Development, "development",
                          "Use development mode (runtime-loadable model)."),
               clEnumValN(InliningAdvisorMode::Release, "release",
                          "Use release mode (AOT-compiled model).")));

static cl::opt<bool> EnableSyntheticCounts(
    "enable-npm-synthetic-counts", cl::init(false), cl::Hidden, cl::ZeroOrMore,
    cl::desc("Run synthetic function entry count generation "
             "pass"));

static const Regex DefaultAliasRegex(
    "^(default|thinlto-pre-link|thinlto|lto-pre-link|lto)<(O[0123sz])>$");

/// Flag to enable inline deferral during PGO.
static cl::opt<bool>
    EnablePGOInlineDeferral("enable-npm-pgo-inline-deferral", cl::init(true),
                            cl::Hidden,
                            cl::desc("Enable inline deferral during PGO"));

static cl::opt<bool> EnableMemProfiler("enable-mem-prof", cl::init(false),
                                       cl::Hidden, cl::ZeroOrMore,
                                       cl::desc("Enable memory profiler"));

static cl::opt<bool> PerformMandatoryInliningsFirst(
    "mandatory-inlining-first", cl::init(true), cl::Hidden, cl::ZeroOrMore,
    cl::desc("Perform mandatory inlinings module-wide, before performing "
             "inlining."));

static cl::opt<bool> EnableO3NonTrivialUnswitching(
    "enable-npm-O3-nontrivial-unswitch", cl::init(true), cl::Hidden,
    cl::ZeroOrMore, cl::desc("Enable non-trivial loop unswitching for -O3"));

PipelineTuningOptions::PipelineTuningOptions() {
  LoopInterleaving = true;
  LoopVectorization = true;
  SLPVectorization = false;
  LoopUnrolling = true;
  ForgetAllSCEVInLoopUnroll = ForgetSCEVInLoopUnroll;
  Coroutines = false;
  LicmMssaOptCap = SetLicmMssaOptCap;
  LicmMssaNoAccForPromotionCap = SetLicmMssaNoAccForPromotionCap;
  CallGraphProfile = true;
  MergeFunctions = false;
  UniqueLinkageNames = false;
}
extern cl::opt<bool> ExtraVectorizerPasses;

extern cl::opt<bool> EnableConstraintElimination;
extern cl::opt<bool> EnableGVNHoist;
extern cl::opt<bool> EnableGVNSink;
extern cl::opt<bool> EnableHotColdSplit;
extern cl::opt<bool> EnableIROutliner;
extern cl::opt<bool> EnableOrderFileInstrumentation;
extern cl::opt<bool> EnableCHR;
extern cl::opt<bool> EnableUnrollAndJam;
extern cl::opt<bool> EnableLoopFlatten;
extern cl::opt<bool> RunNewGVN;
extern cl::opt<bool> RunPartialInlining;

extern cl::opt<bool> FlattenedProfileUsed;

extern cl::opt<AttributorRunOption> AttributorRun;
extern cl::opt<bool> EnableKnowledgeRetention;

extern cl::opt<bool> EnableMatrix;

extern cl::opt<bool> DisablePreInliner;
extern cl::opt<int> PreInlineThreshold;

const PassBuilder::OptimizationLevel PassBuilder::OptimizationLevel::O0 = {
    /*SpeedLevel*/ 0,
    /*SizeLevel*/ 0};
const PassBuilder::OptimizationLevel PassBuilder::OptimizationLevel::O1 = {
    /*SpeedLevel*/ 1,
    /*SizeLevel*/ 0};
const PassBuilder::OptimizationLevel PassBuilder::OptimizationLevel::O2 = {
    /*SpeedLevel*/ 2,
    /*SizeLevel*/ 0};
const PassBuilder::OptimizationLevel PassBuilder::OptimizationLevel::O3 = {
    /*SpeedLevel*/ 3,
    /*SizeLevel*/ 0};
const PassBuilder::OptimizationLevel PassBuilder::OptimizationLevel::Os = {
    /*SpeedLevel*/ 2,
    /*SizeLevel*/ 1};
const PassBuilder::OptimizationLevel PassBuilder::OptimizationLevel::Oz = {
    /*SpeedLevel*/ 2,
    /*SizeLevel*/ 2};

namespace {

// The following passes/analyses have custom names, otherwise their name will
// include `(anonymous namespace)`. These are special since they are only for
// testing purposes and don't live in a header file.

/// No-op module pass which does nothing.
struct NoOpModulePass : PassInfoMixin<NoOpModulePass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    return PreservedAnalyses::all();
  }

  static StringRef name() { return "NoOpModulePass"; }
};

/// No-op module analysis.
class NoOpModuleAnalysis : public AnalysisInfoMixin<NoOpModuleAnalysis> {
  friend AnalysisInfoMixin<NoOpModuleAnalysis>;
  static AnalysisKey Key;

public:
  struct Result {};
  Result run(Module &, ModuleAnalysisManager &) { return Result(); }
  static StringRef name() { return "NoOpModuleAnalysis"; }
};

/// No-op CGSCC pass which does nothing.
struct NoOpCGSCCPass : PassInfoMixin<NoOpCGSCCPass> {
  PreservedAnalyses run(LazyCallGraph::SCC &C, CGSCCAnalysisManager &,
                        LazyCallGraph &, CGSCCUpdateResult &UR) {
    return PreservedAnalyses::all();
  }
  static StringRef name() { return "NoOpCGSCCPass"; }
};

/// No-op CGSCC analysis.
class NoOpCGSCCAnalysis : public AnalysisInfoMixin<NoOpCGSCCAnalysis> {
  friend AnalysisInfoMixin<NoOpCGSCCAnalysis>;
  static AnalysisKey Key;

public:
  struct Result {};
  Result run(LazyCallGraph::SCC &, CGSCCAnalysisManager &, LazyCallGraph &G) {
    return Result();
  }
  static StringRef name() { return "NoOpCGSCCAnalysis"; }
};

/// No-op function pass which does nothing.
struct NoOpFunctionPass : PassInfoMixin<NoOpFunctionPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    return PreservedAnalyses::all();
  }
  static StringRef name() { return "NoOpFunctionPass"; }
};

/// No-op function analysis.
class NoOpFunctionAnalysis : public AnalysisInfoMixin<NoOpFunctionAnalysis> {
  friend AnalysisInfoMixin<NoOpFunctionAnalysis>;
  static AnalysisKey Key;

public:
  struct Result {};
  Result run(Function &, FunctionAnalysisManager &) { return Result(); }
  static StringRef name() { return "NoOpFunctionAnalysis"; }
};

/// No-op loop pass which does nothing.
struct NoOpLoopPass : PassInfoMixin<NoOpLoopPass> {
  PreservedAnalyses run(Loop &L, LoopAnalysisManager &,
                        LoopStandardAnalysisResults &, LPMUpdater &) {
    return PreservedAnalyses::all();
  }
  static StringRef name() { return "NoOpLoopPass"; }
};

/// No-op loop analysis.
class NoOpLoopAnalysis : public AnalysisInfoMixin<NoOpLoopAnalysis> {
  friend AnalysisInfoMixin<NoOpLoopAnalysis>;
  static AnalysisKey Key;

public:
  struct Result {};
  Result run(Loop &, LoopAnalysisManager &, LoopStandardAnalysisResults &) {
    return Result();
  }
  static StringRef name() { return "NoOpLoopAnalysis"; }
};

AnalysisKey NoOpModuleAnalysis::Key;
AnalysisKey NoOpCGSCCAnalysis::Key;
AnalysisKey NoOpFunctionAnalysis::Key;
AnalysisKey NoOpLoopAnalysis::Key;

/// Whether or not we should populate a PassInstrumentationCallbacks's class to
/// pass name map.
///
/// This is for optimization purposes so we don't populate it if we never use
/// it. This should be updated if new pass instrumentation wants to use the map.
/// We currently only use this for --print-before/after.
bool shouldPopulateClassToPassNames() {
  return !printBeforePasses().empty() || !printAfterPasses().empty();
}

} // namespace

PassBuilder::PassBuilder(bool DebugLogging, TargetMachine *TM,
                         PipelineTuningOptions PTO, Optional<PGOOptions> PGOOpt,
                         PassInstrumentationCallbacks *PIC)
    : DebugLogging(DebugLogging), TM(TM), PTO(PTO), PGOOpt(PGOOpt), PIC(PIC) {
  if (TM)
    TM->registerPassBuilderCallbacks(*this, DebugLogging);
  if (PIC && shouldPopulateClassToPassNames()) {
#define MODULE_PASS(NAME, CREATE_PASS)                                         \
  PIC->addClassToPassName(decltype(CREATE_PASS)::name(), NAME);
#define MODULE_ANALYSIS(NAME, CREATE_PASS)                                     \
  PIC->addClassToPassName(decltype(CREATE_PASS)::name(), NAME);
#define FUNCTION_PASS(NAME, CREATE_PASS)                                       \
  PIC->addClassToPassName(decltype(CREATE_PASS)::name(), NAME);
#define FUNCTION_ANALYSIS(NAME, CREATE_PASS)                                   \
  PIC->addClassToPassName(decltype(CREATE_PASS)::name(), NAME);
#define LOOP_PASS(NAME, CREATE_PASS)                                           \
  PIC->addClassToPassName(decltype(CREATE_PASS)::name(), NAME);
#define LOOP_ANALYSIS(NAME, CREATE_PASS)                                       \
  PIC->addClassToPassName(decltype(CREATE_PASS)::name(), NAME);
#define CGSCC_PASS(NAME, CREATE_PASS)                                          \
  PIC->addClassToPassName(decltype(CREATE_PASS)::name(), NAME);
#define CGSCC_ANALYSIS(NAME, CREATE_PASS)                                      \
  PIC->addClassToPassName(decltype(CREATE_PASS)::name(), NAME);
#include "PassRegistry.def"
  }
}

void PassBuilder::invokePeepholeEPCallbacks(
    FunctionPassManager &FPM, PassBuilder::OptimizationLevel Level) {
  for (auto &C : PeepholeEPCallbacks)
    C(FPM, Level);
}

void PassBuilder::registerModuleAnalyses(ModuleAnalysisManager &MAM) {
#define MODULE_ANALYSIS(NAME, CREATE_PASS)                                     \
  MAM.registerPass([&] { return CREATE_PASS; });
#include "PassRegistry.def"

  for (auto &C : ModuleAnalysisRegistrationCallbacks)
    C(MAM);
}

void PassBuilder::registerCGSCCAnalyses(CGSCCAnalysisManager &CGAM) {
#define CGSCC_ANALYSIS(NAME, CREATE_PASS)                                      \
  CGAM.registerPass([&] { return CREATE_PASS; });
#include "PassRegistry.def"

  for (auto &C : CGSCCAnalysisRegistrationCallbacks)
    C(CGAM);
}

void PassBuilder::registerFunctionAnalyses(FunctionAnalysisManager &FAM) {
#define FUNCTION_ANALYSIS(NAME, CREATE_PASS)                                   \
  FAM.registerPass([&] { return CREATE_PASS; });
#include "PassRegistry.def"

  for (auto &C : FunctionAnalysisRegistrationCallbacks)
    C(FAM);
}

void PassBuilder::registerLoopAnalyses(LoopAnalysisManager &LAM) {
#define LOOP_ANALYSIS(NAME, CREATE_PASS)                                       \
  LAM.registerPass([&] { return CREATE_PASS; });
#include "PassRegistry.def"

  for (auto &C : LoopAnalysisRegistrationCallbacks)
    C(LAM);
}

// Helper to add AnnotationRemarksPass.
static void addAnnotationRemarksPass(ModulePassManager &MPM) {
  FunctionPassManager FPM;
  FPM.addPass(AnnotationRemarksPass());
  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
}

// Helper to check if the current compilation phase is preparing for LTO
static bool isLTOPreLink(ThinOrFullLTOPhase Phase) {
  return Phase == ThinOrFullLTOPhase::ThinLTOPreLink ||
         Phase == ThinOrFullLTOPhase::ThinLTOPreLink;
}

// TODO: Investigate the cost/benefit of tail call elimination on debugging.
FunctionPassManager
PassBuilder::buildO1FunctionSimplificationPipeline(OptimizationLevel Level,
                                                   ThinOrFullLTOPhase Phase) {

  FunctionPassManager FPM(DebugLogging);

  // Form SSA out of local memory accesses after breaking apart aggregates into
  // scalars.
  FPM.addPass(SROA());

  // Catch trivial redundancies
  FPM.addPass(EarlyCSEPass(true /* Enable mem-ssa. */));

  // Hoisting of scalars and load expressions.
  FPM.addPass(SimplifyCFGPass());
  FPM.addPass(InstCombinePass());

  FPM.addPass(LibCallsShrinkWrapPass());

  invokePeepholeEPCallbacks(FPM, Level);

  FPM.addPass(SimplifyCFGPass());

  // Form canonically associated expression trees, and simplify the trees using
  // basic mathematical properties. For example, this will form (nearly)
  // minimal multiplication trees.
  FPM.addPass(ReassociatePass());

  // Add the primary loop simplification pipeline.
  // FIXME: Currently this is split into two loop pass pipelines because we run
  // some function passes in between them. These can and should be removed
  // and/or replaced by scheduling the loop pass equivalents in the correct
  // positions. But those equivalent passes aren't powerful enough yet.
  // Specifically, `SimplifyCFGPass` and `InstCombinePass` are currently still
  // used. We have `LoopSimplifyCFGPass` which isn't yet powerful enough yet to
  // fully replace `SimplifyCFGPass`, and the closest to the other we have is
  // `LoopInstSimplify`.
  LoopPassManager LPM1(DebugLogging), LPM2(DebugLogging);

  // Simplify the loop body. We do this initially to clean up after other loop
  // passes run, either when iterating on a loop or on inner loops with
  // implications on the outer loop.
  LPM1.addPass(LoopInstSimplifyPass());
  LPM1.addPass(LoopSimplifyCFGPass());

  LPM1.addPass(LoopRotatePass(/* Disable header duplication */ true,
                              isLTOPreLink(Phase)));
  // TODO: Investigate promotion cap for O1.
  LPM1.addPass(LICMPass(PTO.LicmMssaOptCap, PTO.LicmMssaNoAccForPromotionCap));
  LPM1.addPass(SimpleLoopUnswitchPass());

  LPM2.addPass(LoopIdiomRecognizePass());
  LPM2.addPass(IndVarSimplifyPass());

  for (auto &C : LateLoopOptimizationsEPCallbacks)
    C(LPM2, Level);

  LPM2.addPass(LoopDeletionPass());
  // Do not enable unrolling in PreLinkThinLTO phase during sample PGO
  // because it changes IR to makes profile annotation in back compile
  // inaccurate. The normal unroller doesn't pay attention to forced full unroll
  // attributes so we need to make sure and allow the full unroll pass to pay
  // attention to it.
  if (Phase != ThinOrFullLTOPhase::ThinLTOPreLink || !PGOOpt ||
      PGOOpt->Action != PGOOptions::SampleUse)
    LPM2.addPass(LoopFullUnrollPass(Level.getSpeedupLevel(),
                                    /* OnlyWhenForced= */ !PTO.LoopUnrolling,
                                    PTO.ForgetAllSCEVInLoopUnroll));

  for (auto &C : LoopOptimizerEndEPCallbacks)
    C(LPM2, Level);

  // We provide the opt remark emitter pass for LICM to use. We only need to do
  // this once as it is immutable.
  FPM.addPass(
      RequireAnalysisPass<OptimizationRemarkEmitterAnalysis, Function>());
  FPM.addPass(createFunctionToLoopPassAdaptor(
      std::move(LPM1), EnableMSSALoopDependency, /*UseBlockFrequencyInfo=*/true,
      DebugLogging));
  FPM.addPass(SimplifyCFGPass());
  FPM.addPass(InstCombinePass());
  if (EnableLoopFlatten)
    FPM.addPass(LoopFlattenPass());
  // The loop passes in LPM2 (LoopFullUnrollPass) do not preserve MemorySSA.
  // *All* loop passes must preserve it, in order to be able to use it.
  FPM.addPass(createFunctionToLoopPassAdaptor(
      std::move(LPM2), /*UseMemorySSA=*/false, /*UseBlockFrequencyInfo=*/false,
      DebugLogging));

  // Delete small array after loop unroll.
  FPM.addPass(SROA());

  // Specially optimize memory movement as it doesn't look like dataflow in SSA.
  FPM.addPass(MemCpyOptPass());

  // Sparse conditional constant propagation.
  // FIXME: It isn't clear why we do this *after* loop passes rather than
  // before...
  FPM.addPass(SCCPPass());

  // Delete dead bit computations (instcombine runs after to fold away the dead
  // computations, and then ADCE will run later to exploit any new DCE
  // opportunities that creates).
  FPM.addPass(BDCEPass());

  // Run instcombine after redundancy and dead bit elimination to exploit
  // opportunities opened up by them.
  FPM.addPass(InstCombinePass());
  invokePeepholeEPCallbacks(FPM, Level);

  if (PTO.Coroutines)
    FPM.addPass(CoroElidePass());

  for (auto &C : ScalarOptimizerLateEPCallbacks)
    C(FPM, Level);

  // Finally, do an expensive DCE pass to catch all the dead code exposed by
  // the simplifications and basic cleanup after all the simplifications.
  // TODO: Investigate if this is too expensive.
  FPM.addPass(ADCEPass());
  FPM.addPass(SimplifyCFGPass());
  FPM.addPass(InstCombinePass());
  invokePeepholeEPCallbacks(FPM, Level);

  return FPM;
}

FunctionPassManager
PassBuilder::buildFunctionSimplificationPipeline(OptimizationLevel Level,
                                                 ThinOrFullLTOPhase Phase) {
  assert(Level != OptimizationLevel::O0 && "Must request optimizations!");

  // The O1 pipeline has a separate pipeline creation function to simplify
  // construction readability.
  if (Level.getSpeedupLevel() == 1)
    return buildO1FunctionSimplificationPipeline(Level, Phase);

  FunctionPassManager FPM(DebugLogging);

  // Form SSA out of local memory accesses after breaking apart aggregates into
  // scalars.
  FPM.addPass(SROA());

  // Catch trivial redundancies
  FPM.addPass(EarlyCSEPass(true /* Enable mem-ssa. */));
  if (EnableKnowledgeRetention)
    FPM.addPass(AssumeSimplifyPass());

  // Hoisting of scalars and load expressions.
  if (EnableGVNHoist)
    FPM.addPass(GVNHoistPass());

  // Global value numbering based sinking.
  if (EnableGVNSink) {
    FPM.addPass(GVNSinkPass());
    FPM.addPass(SimplifyCFGPass());
  }

  if (EnableConstraintElimination)
    FPM.addPass(ConstraintEliminationPass());

  // Speculative execution if the target has divergent branches; otherwise nop.
  FPM.addPass(SpeculativeExecutionPass(/* OnlyIfDivergentTarget =*/true));

  // Optimize based on known information about branches, and cleanup afterward.
  FPM.addPass(JumpThreadingPass());
  FPM.addPass(CorrelatedValuePropagationPass());

  FPM.addPass(SimplifyCFGPass());
  if (Level == OptimizationLevel::O3)
    FPM.addPass(AggressiveInstCombinePass());
  FPM.addPass(InstCombinePass());

  if (!Level.isOptimizingForSize())
    FPM.addPass(LibCallsShrinkWrapPass());

  invokePeepholeEPCallbacks(FPM, Level);

  // For PGO use pipeline, try to optimize memory intrinsics such as memcpy
  // using the size value profile. Don't perform this when optimizing for size.
  if (PGOOpt && PGOOpt->Action == PGOOptions::IRUse &&
      !Level.isOptimizingForSize())
    FPM.addPass(PGOMemOPSizeOpt());

  FPM.addPass(TailCallElimPass());
  FPM.addPass(SimplifyCFGPass());

  // Form canonically associated expression trees, and simplify the trees using
  // basic mathematical properties. For example, this will form (nearly)
  // minimal multiplication trees.
  FPM.addPass(ReassociatePass());

  // Add the primary loop simplification pipeline.
  // FIXME: Currently this is split into two loop pass pipelines because we run
  // some function passes in between them. These can and should be removed
  // and/or replaced by scheduling the loop pass equivalents in the correct
  // positions. But those equivalent passes aren't powerful enough yet.
  // Specifically, `SimplifyCFGPass` and `InstCombinePass` are currently still
  // used. We have `LoopSimplifyCFGPass` which isn't yet powerful enough yet to
  // fully replace `SimplifyCFGPass`, and the closest to the other we have is
  // `LoopInstSimplify`.
  LoopPassManager LPM1(DebugLogging), LPM2(DebugLogging);

  // Simplify the loop body. We do this initially to clean up after other loop
  // passes run, either when iterating on a loop or on inner loops with
  // implications on the outer loop.
  LPM1.addPass(LoopInstSimplifyPass());
  LPM1.addPass(LoopSimplifyCFGPass());

  // Disable header duplication in loop rotation at -Oz.
  LPM1.addPass(
      LoopRotatePass(Level != OptimizationLevel::Oz, isLTOPreLink(Phase)));
  // TODO: Investigate promotion cap for O1.
  LPM1.addPass(LICMPass(PTO.LicmMssaOptCap, PTO.LicmMssaNoAccForPromotionCap));
  LPM1.addPass(
      SimpleLoopUnswitchPass(/* NonTrivial */ Level == OptimizationLevel::O3 &&
                             EnableO3NonTrivialUnswitching));
  LPM2.addPass(LoopIdiomRecognizePass());
  LPM2.addPass(IndVarSimplifyPass());

  for (auto &C : LateLoopOptimizationsEPCallbacks)
    C(LPM2, Level);

  LPM2.addPass(LoopDeletionPass());
  // Do not enable unrolling in PreLinkThinLTO phase during sample PGO
  // because it changes IR to makes profile annotation in back compile
  // inaccurate. The normal unroller doesn't pay attention to forced full unroll
  // attributes so we need to make sure and allow the full unroll pass to pay
  // attention to it.
  if (Phase != ThinOrFullLTOPhase::ThinLTOPreLink || !PGOOpt ||
      PGOOpt->Action != PGOOptions::SampleUse)
    LPM2.addPass(LoopFullUnrollPass(Level.getSpeedupLevel(),
                                    /* OnlyWhenForced= */ !PTO.LoopUnrolling,
                                    PTO.ForgetAllSCEVInLoopUnroll));

  for (auto &C : LoopOptimizerEndEPCallbacks)
    C(LPM2, Level);

  // We provide the opt remark emitter pass for LICM to use. We only need to do
  // this once as it is immutable.
  FPM.addPass(
      RequireAnalysisPass<OptimizationRemarkEmitterAnalysis, Function>());
  FPM.addPass(createFunctionToLoopPassAdaptor(
      std::move(LPM1), EnableMSSALoopDependency, /*UseBlockFrequencyInfo=*/true,
      DebugLogging));
  FPM.addPass(SimplifyCFGPass());
  FPM.addPass(InstCombinePass());
  if (EnableLoopFlatten)
    FPM.addPass(LoopFlattenPass());
  // The loop passes in LPM2 (LoopIdiomRecognizePass, IndVarSimplifyPass,
  // LoopDeletionPass and LoopFullUnrollPass) do not preserve MemorySSA.
  // *All* loop passes must preserve it, in order to be able to use it.
  FPM.addPass(createFunctionToLoopPassAdaptor(
      std::move(LPM2), /*UseMemorySSA=*/false, /*UseBlockFrequencyInfo=*/false,
      DebugLogging));

  // Delete small array after loop unroll.
  FPM.addPass(SROA());

  // Eliminate redundancies.
  FPM.addPass(MergedLoadStoreMotionPass());
  if (RunNewGVN)
    FPM.addPass(NewGVNPass());
  else
    FPM.addPass(GVN());

  // Sparse conditional constant propagation.
  // FIXME: It isn't clear why we do this *after* loop passes rather than
  // before...
  FPM.addPass(SCCPPass());

  // Delete dead bit computations (instcombine runs after to fold away the dead
  // computations, and then ADCE will run later to exploit any new DCE
  // opportunities that creates).
  FPM.addPass(BDCEPass());

  // Run instcombine after redundancy and dead bit elimination to exploit
  // opportunities opened up by them.
  FPM.addPass(InstCombinePass());
  invokePeepholeEPCallbacks(FPM, Level);

  // Re-consider control flow based optimizations after redundancy elimination,
  // redo DCE, etc.
  FPM.addPass(JumpThreadingPass());
  FPM.addPass(CorrelatedValuePropagationPass());

  // Finally, do an expensive DCE pass to catch all the dead code exposed by
  // the simplifications and basic cleanup after all the simplifications.
  // TODO: Investigate if this is too expensive.
  FPM.addPass(ADCEPass());

  // Specially optimize memory movement as it doesn't look like dataflow in SSA.
  FPM.addPass(MemCpyOptPass());

  FPM.addPass(DSEPass());
  FPM.addPass(createFunctionToLoopPassAdaptor(
      LICMPass(PTO.LicmMssaOptCap, PTO.LicmMssaNoAccForPromotionCap),
      EnableMSSALoopDependency, /*UseBlockFrequencyInfo=*/true, DebugLogging));

  if (PTO.Coroutines)
    FPM.addPass(CoroElidePass());

  for (auto &C : ScalarOptimizerLateEPCallbacks)
    C(FPM, Level);

  FPM.addPass(SimplifyCFGPass());
  FPM.addPass(InstCombinePass());
  invokePeepholeEPCallbacks(FPM, Level);

  if (EnableCHR && Level == OptimizationLevel::O3 && PGOOpt &&
      (PGOOpt->Action == PGOOptions::IRUse ||
       PGOOpt->Action == PGOOptions::SampleUse))
    FPM.addPass(ControlHeightReductionPass());

  return FPM;
}

void PassBuilder::addRequiredLTOPreLinkPasses(ModulePassManager &MPM) {
  MPM.addPass(CanonicalizeAliasesPass());
  MPM.addPass(NameAnonGlobalPass());
}

void PassBuilder::addPGOInstrPasses(ModulePassManager &MPM,
                                    PassBuilder::OptimizationLevel Level,
                                    bool RunProfileGen, bool IsCS,
                                    std::string ProfileFile,
                                    std::string ProfileRemappingFile) {
  assert(Level != OptimizationLevel::O0 && "Not expecting O0 here!");
  if (!IsCS && !DisablePreInliner) {
    InlineParams IP;

    IP.DefaultThreshold = PreInlineThreshold;

    // FIXME: The hint threshold has the same value used by the regular inliner
    // when not optimzing for size. This should probably be lowered after
    // performance testing.
    // FIXME: this comment is cargo culted from the old pass manager, revisit).
    IP.HintThreshold = Level.isOptimizingForSize() ? PreInlineThreshold : 325;
    ModuleInlinerWrapperPass MIWP(IP, DebugLogging);
    CGSCCPassManager &CGPipeline = MIWP.getPM();

    FunctionPassManager FPM;
    FPM.addPass(SROA());
    FPM.addPass(EarlyCSEPass());    // Catch trivial redundancies.
    FPM.addPass(SimplifyCFGPass()); // Merge & remove basic blocks.
    FPM.addPass(InstCombinePass()); // Combine silly sequences.
    invokePeepholeEPCallbacks(FPM, Level);

    CGPipeline.addPass(createCGSCCToFunctionPassAdaptor(std::move(FPM)));

    MPM.addPass(std::move(MIWP));

    // Delete anything that is now dead to make sure that we don't instrument
    // dead code. Instrumentation can end up keeping dead code around and
    // dramatically increase code size.
    MPM.addPass(GlobalDCEPass());
  }

  if (!RunProfileGen) {
    assert(!ProfileFile.empty() && "Profile use expecting a profile file!");
    MPM.addPass(PGOInstrumentationUse(ProfileFile, ProfileRemappingFile, IsCS));
    // Cache ProfileSummaryAnalysis once to avoid the potential need to insert
    // RequireAnalysisPass for PSI before subsequent non-module passes.
    MPM.addPass(RequireAnalysisPass<ProfileSummaryAnalysis, Module>());
    return;
  }

  // Perform PGO instrumentation.
  MPM.addPass(PGOInstrumentationGen(IsCS));

  FunctionPassManager FPM;
  // Disable header duplication in loop rotation at -Oz.
  FPM.addPass(createFunctionToLoopPassAdaptor(
      LoopRotatePass(Level != OptimizationLevel::Oz), EnableMSSALoopDependency,
      /*UseBlockFrequencyInfo=*/false, DebugLogging));
  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));

  // Add the profile lowering pass.
  InstrProfOptions Options;
  if (!ProfileFile.empty())
    Options.InstrProfileOutput = ProfileFile;
  // Do counter promotion at Level greater than O0.
  Options.DoCounterPromotion = true;
  Options.UseBFIInPromotion = IsCS;
  MPM.addPass(InstrProfiling(Options, IsCS));
}

void PassBuilder::addPGOInstrPassesForO0(ModulePassManager &MPM,
                                         bool RunProfileGen, bool IsCS,
                                         std::string ProfileFile,
                                         std::string ProfileRemappingFile) {
  if (!RunProfileGen) {
    assert(!ProfileFile.empty() && "Profile use expecting a profile file!");
    MPM.addPass(PGOInstrumentationUse(ProfileFile, ProfileRemappingFile, IsCS));
    // Cache ProfileSummaryAnalysis once to avoid the potential need to insert
    // RequireAnalysisPass for PSI before subsequent non-module passes.
    MPM.addPass(RequireAnalysisPass<ProfileSummaryAnalysis, Module>());
    return;
  }

  // Perform PGO instrumentation.
  MPM.addPass(PGOInstrumentationGen(IsCS));
  // Add the profile lowering pass.
  InstrProfOptions Options;
  if (!ProfileFile.empty())
    Options.InstrProfileOutput = ProfileFile;
  // Do not do counter promotion at O0.
  Options.DoCounterPromotion = false;
  Options.UseBFIInPromotion = IsCS;
  MPM.addPass(InstrProfiling(Options, IsCS));
}

static InlineParams
getInlineParamsFromOptLevel(PassBuilder::OptimizationLevel Level) {
  return getInlineParams(Level.getSpeedupLevel(), Level.getSizeLevel());
}

ModuleInlinerWrapperPass
PassBuilder::buildInlinerPipeline(OptimizationLevel Level,
                                  ThinOrFullLTOPhase Phase) {
  InlineParams IP = getInlineParamsFromOptLevel(Level);
  if (Phase == ThinOrFullLTOPhase::ThinLTOPreLink && PGOOpt &&
      PGOOpt->Action == PGOOptions::SampleUse)
    IP.HotCallSiteThreshold = 0;

  if (PGOOpt)
    IP.EnableDeferral = EnablePGOInlineDeferral;

  ModuleInlinerWrapperPass MIWP(IP, DebugLogging,
                                PerformMandatoryInliningsFirst,
                                UseInlineAdvisor, MaxDevirtIterations);

  // Require the GlobalsAA analysis for the module so we can query it within
  // the CGSCC pipeline.
  MIWP.addRequiredModuleAnalysis<GlobalsAA>();

  // Require the ProfileSummaryAnalysis for the module so we can query it within
  // the inliner pass.
  MIWP.addRequiredModuleAnalysis<ProfileSummaryAnalysis>();

  // Now begin the main postorder CGSCC pipeline.
  // FIXME: The current CGSCC pipeline has its origins in the legacy pass
  // manager and trying to emulate its precise behavior. Much of this doesn't
  // make a lot of sense and we should revisit the core CGSCC structure.
  CGSCCPassManager &MainCGPipeline = MIWP.getPM();

  // Note: historically, the PruneEH pass was run first to deduce nounwind and
  // generally clean up exception handling overhead. It isn't clear this is
  // valuable as the inliner doesn't currently care whether it is inlining an
  // invoke or a call.

  if (AttributorRun & AttributorRunOption::CGSCC)
    MainCGPipeline.addPass(AttributorCGSCCPass());

  if (PTO.Coroutines)
    MainCGPipeline.addPass(CoroSplitPass(Level != OptimizationLevel::O0));

  // Now deduce any function attributes based in the current code.
  MainCGPipeline.addPass(PostOrderFunctionAttrsPass());

  // When at O3 add argument promotion to the pass pipeline.
  // FIXME: It isn't at all clear why this should be limited to O3.
  if (Level == OptimizationLevel::O3)
    MainCGPipeline.addPass(ArgumentPromotionPass());

  // Try to perform OpenMP specific optimizations. This is a (quick!) no-op if
  // there are no OpenMP runtime calls present in the module.
  if (Level == OptimizationLevel::O2 || Level == OptimizationLevel::O3)
    MainCGPipeline.addPass(OpenMPOptPass());

  for (auto &C : CGSCCOptimizerLateEPCallbacks)
    C(MainCGPipeline, Level);

  // Lastly, add the core function simplification pipeline nested inside the
  // CGSCC walk.
  MainCGPipeline.addPass(createCGSCCToFunctionPassAdaptor(
      buildFunctionSimplificationPipeline(Level, Phase)));

  return MIWP;
}

ModulePassManager
PassBuilder::buildModuleSimplificationPipeline(OptimizationLevel Level,
                                               ThinOrFullLTOPhase Phase) {
  ModulePassManager MPM(DebugLogging);

  // Add UniqueInternalLinkageNames Pass which renames internal linkage
  // symbols with unique names.
  if (PTO.UniqueLinkageNames)
    MPM.addPass(UniqueInternalLinkageNamesPass());

  // Place pseudo probe instrumentation as the first pass of the pipeline to
  // minimize the impact of optimization changes.
  if (PGOOpt && PGOOpt->PseudoProbeForProfiling &&
      Phase != ThinOrFullLTOPhase::ThinLTOPostLink)
    MPM.addPass(SampleProfileProbePass(TM));

  bool HasSampleProfile = PGOOpt && (PGOOpt->Action == PGOOptions::SampleUse);

  // In ThinLTO mode, when flattened profile is used, all the available
  // profile information will be annotated in PreLink phase so there is
  // no need to load the profile again in PostLink.
  bool LoadSampleProfile =
      HasSampleProfile &&
      !(FlattenedProfileUsed && Phase == ThinOrFullLTOPhase::ThinLTOPostLink);

  // During the ThinLTO backend phase we perform early indirect call promotion
  // here, before globalopt. Otherwise imported available_externally functions
  // look unreferenced and are removed. If we are going to load the sample
  // profile then defer until later.
  // TODO: See if we can move later and consolidate with the location where
  // we perform ICP when we are loading a sample profile.
  // TODO: We pass HasSampleProfile (whether there was a sample profile file
  // passed to the compile) to the SamplePGO flag of ICP. This is used to
  // determine whether the new direct calls are annotated with prof metadata.
  // Ideally this should be determined from whether the IR is annotated with
  // sample profile, and not whether the a sample profile was provided on the
  // command line. E.g. for flattened profiles where we will not be reloading
  // the sample profile in the ThinLTO backend, we ideally shouldn't have to
  // provide the sample profile file.
  if (Phase == ThinOrFullLTOPhase::ThinLTOPostLink && !LoadSampleProfile)
    MPM.addPass(PGOIndirectCallPromotion(true /* InLTO */, HasSampleProfile));

  // Do basic inference of function attributes from known properties of system
  // libraries and other oracles.
  MPM.addPass(InferFunctionAttrsPass());

  // Create an early function pass manager to cleanup the output of the
  // frontend.
  FunctionPassManager EarlyFPM(DebugLogging);
  EarlyFPM.addPass(SimplifyCFGPass());
  EarlyFPM.addPass(SROA());
  EarlyFPM.addPass(EarlyCSEPass());
  EarlyFPM.addPass(LowerExpectIntrinsicPass());
  if (PTO.Coroutines)
    EarlyFPM.addPass(CoroEarlyPass());
  if (Level == OptimizationLevel::O3)
    EarlyFPM.addPass(CallSiteSplittingPass());

  // In SamplePGO ThinLTO backend, we need instcombine before profile annotation
  // to convert bitcast to direct calls so that they can be inlined during the
  // profile annotation prepration step.
  // More details about SamplePGO design can be found in:
  // https://research.google.com/pubs/pub45290.html
  // FIXME: revisit how SampleProfileLoad/Inliner/ICP is structured.
  if (LoadSampleProfile)
    EarlyFPM.addPass(InstCombinePass());
  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(EarlyFPM)));

  if (LoadSampleProfile) {
    // Annotate sample profile right after early FPM to ensure freshness of
    // the debug info.
    MPM.addPass(SampleProfileLoaderPass(PGOOpt->ProfileFile,
                                        PGOOpt->ProfileRemappingFile, Phase));
    // Cache ProfileSummaryAnalysis once to avoid the potential need to insert
    // RequireAnalysisPass for PSI before subsequent non-module passes.
    MPM.addPass(RequireAnalysisPass<ProfileSummaryAnalysis, Module>());
    // Do not invoke ICP in the ThinLTOPrelink phase as it makes it hard
    // for the profile annotation to be accurate in the ThinLTO backend.
    if (Phase != ThinOrFullLTOPhase::ThinLTOPreLink)
      // We perform early indirect call promotion here, before globalopt.
      // This is important for the ThinLTO backend phase because otherwise
      // imported available_externally functions look unreferenced and are
      // removed.
      MPM.addPass(PGOIndirectCallPromotion(
          Phase == ThinOrFullLTOPhase::ThinLTOPostLink, true /* SamplePGO */));
  }

  if (AttributorRun & AttributorRunOption::MODULE)
    MPM.addPass(AttributorPass());

  // Lower type metadata and the type.test intrinsic in the ThinLTO
  // post link pipeline after ICP. This is to enable usage of the type
  // tests in ICP sequences.
  if (Phase == ThinOrFullLTOPhase::ThinLTOPostLink)
    MPM.addPass(LowerTypeTestsPass(nullptr, nullptr, true));

  for (auto &C : PipelineEarlySimplificationEPCallbacks)
    C(MPM, Level);

  // Interprocedural constant propagation now that basic cleanup has occurred
  // and prior to optimizing globals.
  // FIXME: This position in the pipeline hasn't been carefully considered in
  // years, it should be re-analyzed.
  MPM.addPass(IPSCCPPass());

  // Attach metadata to indirect call sites indicating the set of functions
  // they may target at run-time. This should follow IPSCCP.
  MPM.addPass(CalledValuePropagationPass());

  // Optimize globals to try and fold them into constants.
  MPM.addPass(GlobalOptPass());

  // Promote any localized globals to SSA registers.
  // FIXME: Should this instead by a run of SROA?
  // FIXME: We should probably run instcombine and simplify-cfg afterward to
  // delete control flows that are dead once globals have been folded to
  // constants.
  MPM.addPass(createModuleToFunctionPassAdaptor(PromotePass()));

  // Remove any dead arguments exposed by cleanups and constant folding
  // globals.
  MPM.addPass(DeadArgumentEliminationPass());

  // Create a small function pass pipeline to cleanup after all the global
  // optimizations.
  FunctionPassManager GlobalCleanupPM(DebugLogging);
  GlobalCleanupPM.addPass(InstCombinePass());
  invokePeepholeEPCallbacks(GlobalCleanupPM, Level);

  GlobalCleanupPM.addPass(SimplifyCFGPass());
  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(GlobalCleanupPM)));

  // Add all the requested passes for instrumentation PGO, if requested.
  if (PGOOpt && Phase != ThinOrFullLTOPhase::ThinLTOPostLink &&
      (PGOOpt->Action == PGOOptions::IRInstr ||
       PGOOpt->Action == PGOOptions::IRUse)) {
    addPGOInstrPasses(MPM, Level,
                      /* RunProfileGen */ PGOOpt->Action == PGOOptions::IRInstr,
                      /* IsCS */ false, PGOOpt->ProfileFile,
                      PGOOpt->ProfileRemappingFile);
    MPM.addPass(PGOIndirectCallPromotion(false, false));
  }
  if (PGOOpt && Phase != ThinOrFullLTOPhase::ThinLTOPostLink &&
      PGOOpt->CSAction == PGOOptions::CSIRInstr)
    MPM.addPass(PGOInstrumentationGenCreateVar(PGOOpt->CSProfileGenFile));

  // Synthesize function entry counts for non-PGO compilation.
  if (EnableSyntheticCounts && !PGOOpt)
    MPM.addPass(SyntheticCountsPropagation());

  MPM.addPass(buildInlinerPipeline(Level, Phase));

  if (EnableMemProfiler && Phase != ThinOrFullLTOPhase::ThinLTOPreLink) {
    MPM.addPass(createModuleToFunctionPassAdaptor(MemProfilerPass()));
    MPM.addPass(ModuleMemProfilerPass());
  }

  return MPM;
}

ModulePassManager
PassBuilder::buildModuleOptimizationPipeline(OptimizationLevel Level,
                                             bool LTOPreLink) {
  ModulePassManager MPM(DebugLogging);

  // Optimize globals now that the module is fully simplified.
  MPM.addPass(GlobalOptPass());
  MPM.addPass(GlobalDCEPass());

  // Run partial inlining pass to partially inline functions that have
  // large bodies.
  if (RunPartialInlining)
    MPM.addPass(PartialInlinerPass());

  // Remove avail extern fns and globals definitions since we aren't compiling
  // an object file for later LTO. For LTO we want to preserve these so they
  // are eligible for inlining at link-time. Note if they are unreferenced they
  // will be removed by GlobalDCE later, so this only impacts referenced
  // available externally globals. Eventually they will be suppressed during
  // codegen, but eliminating here enables more opportunity for GlobalDCE as it
  // may make globals referenced by available external functions dead and saves
  // running remaining passes on the eliminated functions. These should be
  // preserved during prelinking for link-time inlining decisions.
  if (!LTOPreLink)
    MPM.addPass(EliminateAvailableExternallyPass());

  if (EnableOrderFileInstrumentation)
    MPM.addPass(InstrOrderFilePass());

  // Do RPO function attribute inference across the module to forward-propagate
  // attributes where applicable.
  // FIXME: Is this really an optimization rather than a canonicalization?
  MPM.addPass(ReversePostOrderFunctionAttrsPass());

  // Do a post inline PGO instrumentation and use pass. This is a context
  // sensitive PGO pass. We don't want to do this in LTOPreLink phrase as
  // cross-module inline has not been done yet. The context sensitive
  // instrumentation is after all the inlines are done.
  if (!LTOPreLink && PGOOpt) {
    if (PGOOpt->CSAction == PGOOptions::CSIRInstr)
      addPGOInstrPasses(MPM, Level, /* RunProfileGen */ true,
                        /* IsCS */ true, PGOOpt->CSProfileGenFile,
                        PGOOpt->ProfileRemappingFile);
    else if (PGOOpt->CSAction == PGOOptions::CSIRUse)
      addPGOInstrPasses(MPM, Level, /* RunProfileGen */ false,
                        /* IsCS */ true, PGOOpt->ProfileFile,
                        PGOOpt->ProfileRemappingFile);
  }

  // Re-require GloblasAA here prior to function passes. This is particularly
  // useful as the above will have inlined, DCE'ed, and function-attr
  // propagated everything. We should at this point have a reasonably minimal
  // and richly annotated call graph. By computing aliasing and mod/ref
  // information for all local globals here, the late loop passes and notably
  // the vectorizer will be able to use them to help recognize vectorizable
  // memory operations.
  MPM.addPass(RequireAnalysisPass<GlobalsAA, Module>());

  FunctionPassManager OptimizePM(DebugLogging);
  OptimizePM.addPass(Float2IntPass());
  OptimizePM.addPass(LowerConstantIntrinsicsPass());

  if (EnableMatrix) {
    OptimizePM.addPass(LowerMatrixIntrinsicsPass());
    OptimizePM.addPass(EarlyCSEPass());
  }

  // FIXME: We need to run some loop optimizations to re-rotate loops after
  // simplify-cfg and others undo their rotation.

  // Optimize the loop execution. These passes operate on entire loop nests
  // rather than on each loop in an inside-out manner, and so they are actually
  // function passes.

  for (auto &C : VectorizerStartEPCallbacks)
    C(OptimizePM, Level);

  // First rotate loops that may have been un-rotated by prior passes.
  // Disable header duplication at -Oz.
  OptimizePM.addPass(createFunctionToLoopPassAdaptor(
      LoopRotatePass(Level != OptimizationLevel::Oz, LTOPreLink),
      EnableMSSALoopDependency,
      /*UseBlockFrequencyInfo=*/false, DebugLogging));

  // Distribute loops to allow partial vectorization.  I.e. isolate dependences
  // into separate loop that would otherwise inhibit vectorization.  This is
  // currently only performed for loops marked with the metadata
  // llvm.loop.distribute=true or when -enable-loop-distribute is specified.
  OptimizePM.addPass(LoopDistributePass());

  // Populates the VFABI attribute with the scalar-to-vector mappings
  // from the TargetLibraryInfo.
  OptimizePM.addPass(InjectTLIMappings());

  // Now run the core loop vectorizer.
  OptimizePM.addPass(LoopVectorizePass(
      LoopVectorizeOptions(!PTO.LoopInterleaving, !PTO.LoopVectorization)));

  // Eliminate loads by forwarding stores from the previous iteration to loads
  // of the current iteration.
  OptimizePM.addPass(LoopLoadEliminationPass());

  // Cleanup after the loop optimization passes.
  OptimizePM.addPass(InstCombinePass());

  if (Level.getSpeedupLevel() > 1 && ExtraVectorizerPasses) {
    // At higher optimization levels, try to clean up any runtime overlap and
    // alignment checks inserted by the vectorizer. We want to track correlated
    // runtime checks for two inner loops in the same outer loop, fold any
    // common computations, hoist loop-invariant aspects out of any outer loop,
    // and unswitch the runtime checks if possible. Once hoisted, we may have
    // dead (or speculatable) control flows or more combining opportunities.
    OptimizePM.addPass(EarlyCSEPass());
    OptimizePM.addPass(CorrelatedValuePropagationPass());
    OptimizePM.addPass(InstCombinePass());
    LoopPassManager LPM(DebugLogging);
    LPM.addPass(LICMPass(PTO.LicmMssaOptCap, PTO.LicmMssaNoAccForPromotionCap));
    LPM.addPass(
        SimpleLoopUnswitchPass(/* NonTrivial */ Level == OptimizationLevel::O3));
    OptimizePM.addPass(RequireAnalysisPass<OptimizationRemarkEmitterAnalysis, Function>());
    OptimizePM.addPass(createFunctionToLoopPassAdaptor(
        std::move(LPM), EnableMSSALoopDependency, /*UseBlockFrequencyInfo=*/true,
        DebugLogging));
    OptimizePM.addPass(SimplifyCFGPass());
    OptimizePM.addPass(InstCombinePass());
  }

  // Now that we've formed fast to execute loop structures, we do further
  // optimizations. These are run afterward as they might block doing complex
  // analyses and transforms such as what are needed for loop vectorization.

  // Cleanup after loop vectorization, etc. Simplification passes like CVP and
  // GVN, loop transforms, and others have already run, so it's now better to
  // convert to more optimized IR using more aggressive simplify CFG options.
  // The extra sinking transform can create larger basic blocks, so do this
  // before SLP vectorization.
  // FIXME: study whether hoisting and/or sinking of common instructions should
  //        be delayed until after SLP vectorizer.
  OptimizePM.addPass(SimplifyCFGPass(SimplifyCFGOptions()
                                         .forwardSwitchCondToPhi(true)
                                         .convertSwitchToLookupTable(true)
                                         .needCanonicalLoops(false)
                                         .hoistCommonInsts(true)
                                         .sinkCommonInsts(true)));

  // Optimize parallel scalar instruction chains into SIMD instructions.
  if (PTO.SLPVectorization) {
    OptimizePM.addPass(SLPVectorizerPass());
    if (Level.getSpeedupLevel() > 1 && ExtraVectorizerPasses) {
      OptimizePM.addPass(EarlyCSEPass());
    }
  }

  // Enhance/cleanup vector code.
  OptimizePM.addPass(VectorCombinePass());
  OptimizePM.addPass(InstCombinePass());

  // Unroll small loops to hide loop backedge latency and saturate any parallel
  // execution resources of an out-of-order processor. We also then need to
  // clean up redundancies and loop invariant code.
  // FIXME: It would be really good to use a loop-integrated instruction
  // combiner for cleanup here so that the unrolling and LICM can be pipelined
  // across the loop nests.
  // We do UnrollAndJam in a separate LPM to ensure it happens before unroll
  if (EnableUnrollAndJam && PTO.LoopUnrolling) {
    OptimizePM.addPass(LoopUnrollAndJamPass(Level.getSpeedupLevel()));
  }
  OptimizePM.addPass(LoopUnrollPass(LoopUnrollOptions(
      Level.getSpeedupLevel(), /*OnlyWhenForced=*/!PTO.LoopUnrolling,
      PTO.ForgetAllSCEVInLoopUnroll)));
  OptimizePM.addPass(WarnMissedTransformationsPass());
  OptimizePM.addPass(InstCombinePass());
  OptimizePM.addPass(RequireAnalysisPass<OptimizationRemarkEmitterAnalysis, Function>());
  OptimizePM.addPass(createFunctionToLoopPassAdaptor(
      LICMPass(PTO.LicmMssaOptCap, PTO.LicmMssaNoAccForPromotionCap),
      EnableMSSALoopDependency, /*UseBlockFrequencyInfo=*/true, DebugLogging));

  // Now that we've vectorized and unrolled loops, we may have more refined
  // alignment information, try to re-derive it here.
  OptimizePM.addPass(AlignmentFromAssumptionsPass());

  // Split out cold code. Splitting is done late to avoid hiding context from
  // other optimizations and inadvertently regressing performance. The tradeoff
  // is that this has a higher code size cost than splitting early.
  if (EnableHotColdSplit && !LTOPreLink)
    MPM.addPass(HotColdSplittingPass());

  // Search the code for similar regions of code. If enough similar regions can
  // be found where extracting the regions into their own function will decrease
  // the size of the program, we extract the regions, a deduplicate the
  // structurally similar regions.
  if (EnableIROutliner)
    MPM.addPass(IROutlinerPass());

  // Merge functions if requested.
  if (PTO.MergeFunctions)
    MPM.addPass(MergeFunctionsPass());

  // LoopSink pass sinks instructions hoisted by LICM, which serves as a
  // canonicalization pass that enables other optimizations. As a result,
  // LoopSink pass needs to be a very late IR pass to avoid undoing LICM
  // result too early.
  OptimizePM.addPass(LoopSinkPass());

  // And finally clean up LCSSA form before generating code.
  OptimizePM.addPass(InstSimplifyPass());

  // This hoists/decomposes div/rem ops. It should run after other sink/hoist
  // passes to avoid re-sinking, but before SimplifyCFG because it can allow
  // flattening of blocks.
  OptimizePM.addPass(DivRemPairsPass());

  // LoopSink (and other loop passes since the last simplifyCFG) might have
  // resulted in single-entry-single-exit or empty blocks. Clean up the CFG.
  OptimizePM.addPass(SimplifyCFGPass());

  // Optimize PHIs by speculating around them when profitable. Note that this
  // pass needs to be run after any PRE or similar pass as it is essentially
  // inserting redundancies into the program. This even includes SimplifyCFG.
  OptimizePM.addPass(SpeculateAroundPHIsPass());

  if (PTO.Coroutines)
    OptimizePM.addPass(CoroCleanupPass());

  // Add the core optimizing pipeline.
  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(OptimizePM)));

  for (auto &C : OptimizerLastEPCallbacks)
    C(MPM, Level);

  if (PTO.CallGraphProfile)
    MPM.addPass(CGProfilePass());

  // Now we need to do some global optimization transforms.
  // FIXME: It would seem like these should come first in the optimization
  // pipeline and maybe be the bottom of the canonicalization pipeline? Weird
  // ordering here.
  MPM.addPass(GlobalDCEPass());
  MPM.addPass(ConstantMergePass());

  return MPM;
}

ModulePassManager
PassBuilder::buildPerModuleDefaultPipeline(OptimizationLevel Level,
                                           bool LTOPreLink) {
  assert(Level != OptimizationLevel::O0 &&
         "Must request optimizations for the default pipeline!");

  ModulePassManager MPM(DebugLogging);

  // Convert @llvm.global.annotations to !annotation metadata.
  MPM.addPass(Annotation2MetadataPass());

  // Force any function attributes we want the rest of the pipeline to observe.
  MPM.addPass(ForceFunctionAttrsPass());

  // Apply module pipeline start EP callback.
  for (auto &C : PipelineStartEPCallbacks)
    C(MPM, Level);

  if (PGOOpt && PGOOpt->DebugInfoForProfiling)
    MPM.addPass(createModuleToFunctionPassAdaptor(AddDiscriminatorsPass()));

  // Add the core simplification pipeline.
  MPM.addPass(buildModuleSimplificationPipeline(
      Level, LTOPreLink ? ThinOrFullLTOPhase::FullLTOPreLink
                        : ThinOrFullLTOPhase::None));

  // Now add the optimization pipeline.
  MPM.addPass(buildModuleOptimizationPipeline(Level, LTOPreLink));

  if (PGOOpt && PGOOpt->PseudoProbeForProfiling)
    MPM.addPass(PseudoProbeUpdatePass());

  // Emit annotation remarks.
  addAnnotationRemarksPass(MPM);

  if (LTOPreLink)
    addRequiredLTOPreLinkPasses(MPM);

  return MPM;
}

ModulePassManager
PassBuilder::buildThinLTOPreLinkDefaultPipeline(OptimizationLevel Level) {
  assert(Level != OptimizationLevel::O0 &&
         "Must request optimizations for the default pipeline!");

  ModulePassManager MPM(DebugLogging);

  // Convert @llvm.global.annotations to !annotation metadata.
  MPM.addPass(Annotation2MetadataPass());

  // Force any function attributes we want the rest of the pipeline to observe.
  MPM.addPass(ForceFunctionAttrsPass());

  if (PGOOpt && PGOOpt->DebugInfoForProfiling)
    MPM.addPass(createModuleToFunctionPassAdaptor(AddDiscriminatorsPass()));

  // Apply module pipeline start EP callback.
  for (auto &C : PipelineStartEPCallbacks)
    C(MPM, Level);

  // If we are planning to perform ThinLTO later, we don't bloat the code with
  // unrolling/vectorization/... now. Just simplify the module as much as we
  // can.
  MPM.addPass(buildModuleSimplificationPipeline(
      Level, ThinOrFullLTOPhase::ThinLTOPreLink));

  // Run partial inlining pass to partially inline functions that have
  // large bodies.
  // FIXME: It isn't clear whether this is really the right place to run this
  // in ThinLTO. Because there is another canonicalization and simplification
  // phase that will run after the thin link, running this here ends up with
  // less information than will be available later and it may grow functions in
  // ways that aren't beneficial.
  if (RunPartialInlining)
    MPM.addPass(PartialInlinerPass());

  // Reduce the size of the IR as much as possible.
  MPM.addPass(GlobalOptPass());

  // Module simplification splits coroutines, but does not fully clean up
  // coroutine intrinsics. To ensure ThinLTO optimization passes don't trip up
  // on these, we schedule the cleanup here.
  if (PTO.Coroutines)
    MPM.addPass(createModuleToFunctionPassAdaptor(CoroCleanupPass()));

  if (PGOOpt && PGOOpt->PseudoProbeForProfiling)
    MPM.addPass(PseudoProbeUpdatePass());

  // Emit annotation remarks.
  addAnnotationRemarksPass(MPM);

  addRequiredLTOPreLinkPasses(MPM);

  return MPM;
}

ModulePassManager PassBuilder::buildThinLTODefaultPipeline(
    OptimizationLevel Level, const ModuleSummaryIndex *ImportSummary) {
  ModulePassManager MPM(DebugLogging);

  // Convert @llvm.global.annotations to !annotation metadata.
  MPM.addPass(Annotation2MetadataPass());

  if (ImportSummary) {
    // These passes import type identifier resolutions for whole-program
    // devirtualization and CFI. They must run early because other passes may
    // disturb the specific instruction patterns that these passes look for,
    // creating dependencies on resolutions that may not appear in the summary.
    //
    // For example, GVN may transform the pattern assume(type.test) appearing in
    // two basic blocks into assume(phi(type.test, type.test)), which would
    // transform a dependency on a WPD resolution into a dependency on a type
    // identifier resolution for CFI.
    //
    // Also, WPD has access to more precise information than ICP and can
    // devirtualize more effectively, so it should operate on the IR first.
    //
    // The WPD and LowerTypeTest passes need to run at -O0 to lower type
    // metadata and intrinsics.
    MPM.addPass(WholeProgramDevirtPass(nullptr, ImportSummary));
    MPM.addPass(LowerTypeTestsPass(nullptr, ImportSummary));
  }

  if (Level == OptimizationLevel::O0)
    return MPM;

  // Force any function attributes we want the rest of the pipeline to observe.
  MPM.addPass(ForceFunctionAttrsPass());

  // Add the core simplification pipeline.
  MPM.addPass(buildModuleSimplificationPipeline(
      Level, ThinOrFullLTOPhase::ThinLTOPostLink));

  // Now add the optimization pipeline.
  MPM.addPass(buildModuleOptimizationPipeline(Level));

  // Emit annotation remarks.
  addAnnotationRemarksPass(MPM);

  return MPM;
}

ModulePassManager
PassBuilder::buildLTOPreLinkDefaultPipeline(OptimizationLevel Level) {
  assert(Level != OptimizationLevel::O0 &&
         "Must request optimizations for the default pipeline!");
  // FIXME: We should use a customized pre-link pipeline!
  return buildPerModuleDefaultPipeline(Level,
                                       /* LTOPreLink */ true);
}

ModulePassManager
PassBuilder::buildLTODefaultPipeline(OptimizationLevel Level,
                                     ModuleSummaryIndex *ExportSummary) {
  ModulePassManager MPM(DebugLogging);

  // Convert @llvm.global.annotations to !annotation metadata.
  MPM.addPass(Annotation2MetadataPass());

  if (Level == OptimizationLevel::O0) {
    // The WPD and LowerTypeTest passes need to run at -O0 to lower type
    // metadata and intrinsics.
    MPM.addPass(WholeProgramDevirtPass(ExportSummary, nullptr));
    MPM.addPass(LowerTypeTestsPass(ExportSummary, nullptr));
    // Run a second time to clean up any type tests left behind by WPD for use
    // in ICP.
    MPM.addPass(LowerTypeTestsPass(nullptr, nullptr, true));

    // Emit annotation remarks.
    addAnnotationRemarksPass(MPM);

    return MPM;
  }

  if (PGOOpt && PGOOpt->Action == PGOOptions::SampleUse) {
    // Load sample profile before running the LTO optimization pipeline.
    MPM.addPass(SampleProfileLoaderPass(PGOOpt->ProfileFile,
                                        PGOOpt->ProfileRemappingFile,
                                        ThinOrFullLTOPhase::FullLTOPostLink));
    // Cache ProfileSummaryAnalysis once to avoid the potential need to insert
    // RequireAnalysisPass for PSI before subsequent non-module passes.
    MPM.addPass(RequireAnalysisPass<ProfileSummaryAnalysis, Module>());
  }

  // Remove unused virtual tables to improve the quality of code generated by
  // whole-program devirtualization and bitset lowering.
  MPM.addPass(GlobalDCEPass());

  // Force any function attributes we want the rest of the pipeline to observe.
  MPM.addPass(ForceFunctionAttrsPass());

  // Do basic inference of function attributes from known properties of system
  // libraries and other oracles.
  MPM.addPass(InferFunctionAttrsPass());

  if (Level.getSpeedupLevel() > 1) {
    FunctionPassManager EarlyFPM(DebugLogging);
    EarlyFPM.addPass(CallSiteSplittingPass());
    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(EarlyFPM)));

    // Indirect call promotion. This should promote all the targets that are
    // left by the earlier promotion pass that promotes intra-module targets.
    // This two-step promotion is to save the compile time. For LTO, it should
    // produce the same result as if we only do promotion here.
    MPM.addPass(PGOIndirectCallPromotion(
        true /* InLTO */, PGOOpt && PGOOpt->Action == PGOOptions::SampleUse));
    // Propagate constants at call sites into the functions they call.  This
    // opens opportunities for globalopt (and inlining) by substituting function
    // pointers passed as arguments to direct uses of functions.
    MPM.addPass(IPSCCPPass());

    // Attach metadata to indirect call sites indicating the set of functions
    // they may target at run-time. This should follow IPSCCP.
    MPM.addPass(CalledValuePropagationPass());
  }

  // Now deduce any function attributes based in the current code.
  MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(
              PostOrderFunctionAttrsPass()));

  // Do RPO function attribute inference across the module to forward-propagate
  // attributes where applicable.
  // FIXME: Is this really an optimization rather than a canonicalization?
  MPM.addPass(ReversePostOrderFunctionAttrsPass());

  // Use in-range annotations on GEP indices to split globals where beneficial.
  MPM.addPass(GlobalSplitPass());

  // Run whole program optimization of virtual call when the list of callees
  // is fixed.
  MPM.addPass(WholeProgramDevirtPass(ExportSummary, nullptr));

  // Stop here at -O1.
  if (Level == OptimizationLevel::O1) {
    // The LowerTypeTestsPass needs to run to lower type metadata and the
    // type.test intrinsics. The pass does nothing if CFI is disabled.
    MPM.addPass(LowerTypeTestsPass(ExportSummary, nullptr));
    // Run a second time to clean up any type tests left behind by WPD for use
    // in ICP (which is performed earlier than this in the regular LTO
    // pipeline).
    MPM.addPass(LowerTypeTestsPass(nullptr, nullptr, true));

    // Emit annotation remarks.
    addAnnotationRemarksPass(MPM);

    return MPM;
  }

  // Optimize globals to try and fold them into constants.
  MPM.addPass(GlobalOptPass());

  // Promote any localized globals to SSA registers.
  MPM.addPass(createModuleToFunctionPassAdaptor(PromotePass()));

  // Linking modules together can lead to duplicate global constant, only
  // keep one copy of each constant.
  MPM.addPass(ConstantMergePass());

  // Remove unused arguments from functions.
  MPM.addPass(DeadArgumentEliminationPass());

  // Reduce the code after globalopt and ipsccp.  Both can open up significant
  // simplification opportunities, and both can propagate functions through
  // function pointers.  When this happens, we often have to resolve varargs
  // calls, etc, so let instcombine do this.
  FunctionPassManager PeepholeFPM(DebugLogging);
  if (Level == OptimizationLevel::O3)
    PeepholeFPM.addPass(AggressiveInstCombinePass());
  PeepholeFPM.addPass(InstCombinePass());
  invokePeepholeEPCallbacks(PeepholeFPM, Level);

  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(PeepholeFPM)));

  // Note: historically, the PruneEH pass was run first to deduce nounwind and
  // generally clean up exception handling overhead. It isn't clear this is
  // valuable as the inliner doesn't currently care whether it is inlining an
  // invoke or a call.
  // Run the inliner now.
  MPM.addPass(ModuleInlinerWrapperPass(getInlineParamsFromOptLevel(Level),
                                       DebugLogging));

  // Optimize globals again after we ran the inliner.
  MPM.addPass(GlobalOptPass());

  // Garbage collect dead functions.
  // FIXME: Add ArgumentPromotion pass after once it's ported.
  MPM.addPass(GlobalDCEPass());

  FunctionPassManager FPM(DebugLogging);
  // The IPO Passes may leave cruft around. Clean up after them.
  FPM.addPass(InstCombinePass());
  invokePeepholeEPCallbacks(FPM, Level);

  FPM.addPass(JumpThreadingPass(/*InsertFreezeWhenUnfoldingSelect*/ true));

  // Do a post inline PGO instrumentation and use pass. This is a context
  // sensitive PGO pass.
  if (PGOOpt) {
    if (PGOOpt->CSAction == PGOOptions::CSIRInstr)
      addPGOInstrPasses(MPM, Level, /* RunProfileGen */ true,
                        /* IsCS */ true, PGOOpt->CSProfileGenFile,
                        PGOOpt->ProfileRemappingFile);
    else if (PGOOpt->CSAction == PGOOptions::CSIRUse)
      addPGOInstrPasses(MPM, Level, /* RunProfileGen */ false,
                        /* IsCS */ true, PGOOpt->ProfileFile,
                        PGOOpt->ProfileRemappingFile);
  }

  // Break up allocas
  FPM.addPass(SROA());

  // LTO provides additional opportunities for tailcall elimination due to
  // link-time inlining, and visibility of nocapture attribute.
  FPM.addPass(TailCallElimPass());

  // Run a few AA driver optimizations here and now to cleanup the code.
  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));

  MPM.addPass(
      createModuleToPostOrderCGSCCPassAdaptor(PostOrderFunctionAttrsPass()));
  // FIXME: here we run IP alias analysis in the legacy PM.

  FunctionPassManager MainFPM;

  MainFPM.addPass(createFunctionToLoopPassAdaptor(
      LICMPass(PTO.LicmMssaOptCap, PTO.LicmMssaNoAccForPromotionCap)));

  if (RunNewGVN)
    MainFPM.addPass(NewGVNPass());
  else
    MainFPM.addPass(GVN());

  // Remove dead memcpy()'s.
  MainFPM.addPass(MemCpyOptPass());

  // Nuke dead stores.
  MainFPM.addPass(DSEPass());
  MainFPM.addPass(MergedLoadStoreMotionPass());

  // More loops are countable; try to optimize them.
  if (EnableLoopFlatten && Level.getSpeedupLevel() > 1)
    MainFPM.addPass(LoopFlattenPass());

  if (EnableConstraintElimination)
    MainFPM.addPass(ConstraintEliminationPass());

  LoopPassManager LPM(DebugLogging);
  LPM.addPass(IndVarSimplifyPass());
  LPM.addPass(LoopDeletionPass());
  // FIXME: Add loop interchange.

  // Unroll small loops and perform peeling.
  LPM.addPass(LoopFullUnrollPass(Level.getSpeedupLevel(),
                                 /* OnlyWhenForced= */ !PTO.LoopUnrolling,
                                 PTO.ForgetAllSCEVInLoopUnroll));
  // The loop passes in LPM (LoopFullUnrollPass) do not preserve MemorySSA.
  // *All* loop passes must preserve it, in order to be able to use it.
  MainFPM.addPass(createFunctionToLoopPassAdaptor(
      std::move(LPM), /*UseMemorySSA=*/false, /*UseBlockFrequencyInfo=*/true,
      DebugLogging));

  MainFPM.addPass(LoopDistributePass());
  MainFPM.addPass(LoopVectorizePass(
      LoopVectorizeOptions(!PTO.LoopInterleaving, !PTO.LoopVectorization)));
  // The vectorizer may have significantly shortened a loop body; unroll again.
  MainFPM.addPass(LoopUnrollPass(LoopUnrollOptions(
      Level.getSpeedupLevel(), /*OnlyWhenForced=*/!PTO.LoopUnrolling,
      PTO.ForgetAllSCEVInLoopUnroll)));

  MainFPM.addPass(WarnMissedTransformationsPass());

  MainFPM.addPass(InstCombinePass());
  MainFPM.addPass(SimplifyCFGPass(SimplifyCFGOptions().hoistCommonInsts(true)));
  MainFPM.addPass(SCCPPass());
  MainFPM.addPass(InstCombinePass());
  MainFPM.addPass(BDCEPass());

  // More scalar chains could be vectorized due to more alias information
  if (PTO.SLPVectorization) {
    MainFPM.addPass(SLPVectorizerPass());
    if (Level.getSpeedupLevel() > 1 && ExtraVectorizerPasses) {
      MainFPM.addPass(EarlyCSEPass());
    }
  }

  MainFPM.addPass(VectorCombinePass()); // Clean up partial vectorization.

  // After vectorization, assume intrinsics may tell us more about pointer
  // alignments.
  MainFPM.addPass(AlignmentFromAssumptionsPass());

  // FIXME: Conditionally run LoadCombine here, after it's ported
  // (in case we still have this pass, given its questionable usefulness).

  MainFPM.addPass(InstCombinePass());
  invokePeepholeEPCallbacks(MainFPM, Level);
  MainFPM.addPass(JumpThreadingPass(/*InsertFreezeWhenUnfoldingSelect*/ true));
  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(MainFPM)));

  // Create a function that performs CFI checks for cross-DSO calls with
  // targets in the current module.
  MPM.addPass(CrossDSOCFIPass());

  // Lower type metadata and the type.test intrinsic. This pass supports
  // clang's control flow integrity mechanisms (-fsanitize=cfi*) and needs
  // to be run at link time if CFI is enabled. This pass does nothing if
  // CFI is disabled.
  MPM.addPass(LowerTypeTestsPass(ExportSummary, nullptr));
  // Run a second time to clean up any type tests left behind by WPD for use
  // in ICP (which is performed earlier than this in the regular LTO pipeline).
  MPM.addPass(LowerTypeTestsPass(nullptr, nullptr, true));

  // Enable splitting late in the FullLTO post-link pipeline. This is done in
  // the same stage in the old pass manager (\ref addLateLTOOptimizationPasses).
  if (EnableHotColdSplit)
    MPM.addPass(HotColdSplittingPass());

  // Add late LTO optimization passes.
  // Delete basic blocks, which optimization passes may have killed.
  MPM.addPass(createModuleToFunctionPassAdaptor(
      SimplifyCFGPass(SimplifyCFGOptions().hoistCommonInsts(true))));

  // Drop bodies of available eternally objects to improve GlobalDCE.
  MPM.addPass(EliminateAvailableExternallyPass());

  // Now that we have optimized the program, discard unreachable functions.
  MPM.addPass(GlobalDCEPass());

  if (PTO.MergeFunctions)
    MPM.addPass(MergeFunctionsPass());

  // Emit annotation remarks.
  addAnnotationRemarksPass(MPM);

  return MPM;
}

ModulePassManager PassBuilder::buildO0DefaultPipeline(OptimizationLevel Level,
                                                      bool LTOPreLink) {
  assert(Level == OptimizationLevel::O0 &&
         "buildO0DefaultPipeline should only be used with O0");

  ModulePassManager MPM(DebugLogging);

  // Add UniqueInternalLinkageNames Pass which renames internal linkage
  // symbols with unique names.
  if (PTO.UniqueLinkageNames)
    MPM.addPass(UniqueInternalLinkageNamesPass());

  if (PGOOpt && (PGOOpt->Action == PGOOptions::IRInstr ||
                 PGOOpt->Action == PGOOptions::IRUse))
    addPGOInstrPassesForO0(
        MPM,
        /* RunProfileGen */ (PGOOpt->Action == PGOOptions::IRInstr),
        /* IsCS */ false, PGOOpt->ProfileFile, PGOOpt->ProfileRemappingFile);

  for (auto &C : PipelineStartEPCallbacks)
    C(MPM, Level);
  for (auto &C : PipelineEarlySimplificationEPCallbacks)
    C(MPM, Level);

  // Build a minimal pipeline based on the semantics required by LLVM,
  // which is just that always inlining occurs. Further, disable generating
  // lifetime intrinsics to avoid enabling further optimizations during
  // code generation.
  // However, we need to insert lifetime intrinsics to avoid invalid access
  // caused by multithreaded coroutines.
  MPM.addPass(AlwaysInlinerPass(
      /*InsertLifetimeIntrinsics=*/PTO.Coroutines));

  if (PTO.MergeFunctions)
    MPM.addPass(MergeFunctionsPass());

  if (EnableMatrix)
    MPM.addPass(
        createModuleToFunctionPassAdaptor(LowerMatrixIntrinsicsPass(true)));

  if (!CGSCCOptimizerLateEPCallbacks.empty()) {
    CGSCCPassManager CGPM(DebugLogging);
    for (auto &C : CGSCCOptimizerLateEPCallbacks)
      C(CGPM, Level);
    if (!CGPM.isEmpty())
      MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM)));
  }
  if (!LateLoopOptimizationsEPCallbacks.empty()) {
    LoopPassManager LPM(DebugLogging);
    for (auto &C : LateLoopOptimizationsEPCallbacks)
      C(LPM, Level);
    if (!LPM.isEmpty()) {
      MPM.addPass(createModuleToFunctionPassAdaptor(
          createFunctionToLoopPassAdaptor(std::move(LPM))));
    }
  }
  if (!LoopOptimizerEndEPCallbacks.empty()) {
    LoopPassManager LPM(DebugLogging);
    for (auto &C : LoopOptimizerEndEPCallbacks)
      C(LPM, Level);
    if (!LPM.isEmpty()) {
      MPM.addPass(createModuleToFunctionPassAdaptor(
          createFunctionToLoopPassAdaptor(std::move(LPM))));
    }
  }
  if (!ScalarOptimizerLateEPCallbacks.empty()) {
    FunctionPassManager FPM(DebugLogging);
    for (auto &C : ScalarOptimizerLateEPCallbacks)
      C(FPM, Level);
    if (!FPM.isEmpty())
      MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
  }
  if (!VectorizerStartEPCallbacks.empty()) {
    FunctionPassManager FPM(DebugLogging);
    for (auto &C : VectorizerStartEPCallbacks)
      C(FPM, Level);
    if (!FPM.isEmpty())
      MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
  }

  if (PTO.Coroutines) {
    MPM.addPass(createModuleToFunctionPassAdaptor(CoroEarlyPass()));

    CGSCCPassManager CGPM(DebugLogging);
    CGPM.addPass(CoroSplitPass());
    CGPM.addPass(createCGSCCToFunctionPassAdaptor(CoroElidePass()));
    MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM)));

    MPM.addPass(createModuleToFunctionPassAdaptor(CoroCleanupPass()));
  }

  for (auto &C : OptimizerLastEPCallbacks)
    C(MPM, Level);

  if (LTOPreLink)
    addRequiredLTOPreLinkPasses(MPM);

  return MPM;
}

AAManager PassBuilder::buildDefaultAAPipeline() {
  AAManager AA;

  // The order in which these are registered determines their priority when
  // being queried.

  // First we register the basic alias analysis that provides the majority of
  // per-function local AA logic. This is a stateless, on-demand local set of
  // AA techniques.
  AA.registerFunctionAnalysis<BasicAA>();

  // Next we query fast, specialized alias analyses that wrap IR-embedded
  // information about aliasing.
  AA.registerFunctionAnalysis<ScopedNoAliasAA>();
  AA.registerFunctionAnalysis<TypeBasedAA>();

  // Add support for querying global aliasing information when available.
  // Because the `AAManager` is a function analysis and `GlobalsAA` is a module
  // analysis, all that the `AAManager` can do is query for any *cached*
  // results from `GlobalsAA` through a readonly proxy.
  AA.registerModuleAnalysis<GlobalsAA>();

  // Add target-specific alias analyses.
  if (TM)
    TM->registerDefaultAliasAnalyses(AA);

  return AA;
}

static Optional<int> parseRepeatPassName(StringRef Name) {
  if (!Name.consume_front("repeat<") || !Name.consume_back(">"))
    return None;
  int Count;
  if (Name.getAsInteger(0, Count) || Count <= 0)
    return None;
  return Count;
}

static Optional<int> parseDevirtPassName(StringRef Name) {
  if (!Name.consume_front("devirt<") || !Name.consume_back(">"))
    return None;
  int Count;
  if (Name.getAsInteger(0, Count) || Count < 0)
    return None;
  return Count;
}

static bool checkParametrizedPassName(StringRef Name, StringRef PassName) {
  if (!Name.consume_front(PassName))
    return false;
  // normal pass name w/o parameters == default parameters
  if (Name.empty())
    return true;
  return Name.startswith("<") && Name.endswith(">");
}

namespace {

/// This performs customized parsing of pass name with parameters.
///
/// We do not need parametrization of passes in textual pipeline very often,
/// yet on a rare occasion ability to specify parameters right there can be
/// useful.
///
/// \p Name - parameterized specification of a pass from a textual pipeline
/// is a string in a form of :
///      PassName '<' parameter-list '>'
///
/// Parameter list is being parsed by the parser callable argument, \p Parser,
/// It takes a string-ref of parameters and returns either StringError or a
/// parameter list in a form of a custom parameters type, all wrapped into
/// Expected<> template class.
///
template <typename ParametersParseCallableT>
auto parsePassParameters(ParametersParseCallableT &&Parser, StringRef Name,
                         StringRef PassName) -> decltype(Parser(StringRef{})) {
  using ParametersT = typename decltype(Parser(StringRef{}))::value_type;

  StringRef Params = Name;
  if (!Params.consume_front(PassName)) {
    assert(false &&
           "unable to strip pass name from parametrized pass specification");
  }
  if (Params.empty())
    return ParametersT{};
  if (!Params.consume_front("<") || !Params.consume_back(">")) {
    assert(false && "invalid format for parametrized pass name");
  }

  Expected<ParametersT> Result = Parser(Params);
  assert((Result || Result.template errorIsA<StringError>()) &&
         "Pass parameter parser can only return StringErrors.");
  return Result;
}

/// Parser of parameters for LoopUnroll pass.
Expected<LoopUnrollOptions> parseLoopUnrollOptions(StringRef Params) {
  LoopUnrollOptions UnrollOpts;
  while (!Params.empty()) {
    StringRef ParamName;
    std::tie(ParamName, Params) = Params.split(';');
    int OptLevel = StringSwitch<int>(ParamName)
                       .Case("O0", 0)
                       .Case("O1", 1)
                       .Case("O2", 2)
                       .Case("O3", 3)
                       .Default(-1);
    if (OptLevel >= 0) {
      UnrollOpts.setOptLevel(OptLevel);
      continue;
    }
    if (ParamName.consume_front("full-unroll-max=")) {
      int Count;
      if (ParamName.getAsInteger(0, Count))
        return make_error<StringError>(
            formatv("invalid LoopUnrollPass parameter '{0}' ", ParamName).str(),
            inconvertibleErrorCode());
      UnrollOpts.setFullUnrollMaxCount(Count);
      continue;
    }

    bool Enable = !ParamName.consume_front("no-");
    if (ParamName == "partial") {
      UnrollOpts.setPartial(Enable);
    } else if (ParamName == "peeling") {
      UnrollOpts.setPeeling(Enable);
    } else if (ParamName == "profile-peeling") {
      UnrollOpts.setProfileBasedPeeling(Enable);
    } else if (ParamName == "runtime") {
      UnrollOpts.setRuntime(Enable);
    } else if (ParamName == "upperbound") {
      UnrollOpts.setUpperBound(Enable);
    } else {
      return make_error<StringError>(
          formatv("invalid LoopUnrollPass parameter '{0}' ", ParamName).str(),
          inconvertibleErrorCode());
    }
  }
  return UnrollOpts;
}

Expected<MemorySanitizerOptions> parseMSanPassOptions(StringRef Params) {
  MemorySanitizerOptions Result;
  while (!Params.empty()) {
    StringRef ParamName;
    std::tie(ParamName, Params) = Params.split(';');

    if (ParamName == "recover") {
      Result.Recover = true;
    } else if (ParamName == "kernel") {
      Result.Kernel = true;
    } else if (ParamName.consume_front("track-origins=")) {
      if (ParamName.getAsInteger(0, Result.TrackOrigins))
        return make_error<StringError>(
            formatv("invalid argument to MemorySanitizer pass track-origins "
                    "parameter: '{0}' ",
                    ParamName)
                .str(),
            inconvertibleErrorCode());
    } else {
      return make_error<StringError>(
          formatv("invalid MemorySanitizer pass parameter '{0}' ", ParamName)
              .str(),
          inconvertibleErrorCode());
    }
  }
  return Result;
}

/// Parser of parameters for SimplifyCFG pass.
Expected<SimplifyCFGOptions> parseSimplifyCFGOptions(StringRef Params) {
  SimplifyCFGOptions Result;
  while (!Params.empty()) {
    StringRef ParamName;
    std::tie(ParamName, Params) = Params.split(';');

    bool Enable = !ParamName.consume_front("no-");
    if (ParamName == "forward-switch-cond") {
      Result.forwardSwitchCondToPhi(Enable);
    } else if (ParamName == "switch-to-lookup") {
      Result.convertSwitchToLookupTable(Enable);
    } else if (ParamName == "keep-loops") {
      Result.needCanonicalLoops(Enable);
    } else if (ParamName == "hoist-common-insts") {
      Result.hoistCommonInsts(Enable);
    } else if (ParamName == "sink-common-insts") {
      Result.sinkCommonInsts(Enable);
    } else if (Enable && ParamName.consume_front("bonus-inst-threshold=")) {
      APInt BonusInstThreshold;
      if (ParamName.getAsInteger(0, BonusInstThreshold))
        return make_error<StringError>(
            formatv("invalid argument to SimplifyCFG pass bonus-threshold "
                    "parameter: '{0}' ",
                    ParamName).str(),
            inconvertibleErrorCode());
      Result.bonusInstThreshold(BonusInstThreshold.getSExtValue());
    } else {
      return make_error<StringError>(
          formatv("invalid SimplifyCFG pass parameter '{0}' ", ParamName).str(),
          inconvertibleErrorCode());
    }
  }
  return Result;
}

/// Parser of parameters for LoopVectorize pass.
Expected<LoopVectorizeOptions> parseLoopVectorizeOptions(StringRef Params) {
  LoopVectorizeOptions Opts;
  while (!Params.empty()) {
    StringRef ParamName;
    std::tie(ParamName, Params) = Params.split(';');

    bool Enable = !ParamName.consume_front("no-");
    if (ParamName == "interleave-forced-only") {
      Opts.setInterleaveOnlyWhenForced(Enable);
    } else if (ParamName == "vectorize-forced-only") {
      Opts.setVectorizeOnlyWhenForced(Enable);
    } else {
      return make_error<StringError>(
          formatv("invalid LoopVectorize parameter '{0}' ", ParamName).str(),
          inconvertibleErrorCode());
    }
  }
  return Opts;
}

Expected<bool> parseLoopUnswitchOptions(StringRef Params) {
  bool Result = false;
  while (!Params.empty()) {
    StringRef ParamName;
    std::tie(ParamName, Params) = Params.split(';');

    bool Enable = !ParamName.consume_front("no-");
    if (ParamName == "nontrivial") {
      Result = Enable;
    } else {
      return make_error<StringError>(
          formatv("invalid LoopUnswitch pass parameter '{0}' ", ParamName)
              .str(),
          inconvertibleErrorCode());
    }
  }
  return Result;
}

Expected<bool> parseMergedLoadStoreMotionOptions(StringRef Params) {
  bool Result = false;
  while (!Params.empty()) {
    StringRef ParamName;
    std::tie(ParamName, Params) = Params.split(';');

    bool Enable = !ParamName.consume_front("no-");
    if (ParamName == "split-footer-bb") {
      Result = Enable;
    } else {
      return make_error<StringError>(
          formatv("invalid MergedLoadStoreMotion pass parameter '{0}' ",
                  ParamName)
              .str(),
          inconvertibleErrorCode());
    }
  }
  return Result;
}

Expected<GVNOptions> parseGVNOptions(StringRef Params) {
  GVNOptions Result;
  while (!Params.empty()) {
    StringRef ParamName;
    std::tie(ParamName, Params) = Params.split(';');

    bool Enable = !ParamName.consume_front("no-");
    if (ParamName == "pre") {
      Result.setPRE(Enable);
    } else if (ParamName == "load-pre") {
      Result.setLoadPRE(Enable);
    } else if (ParamName == "split-backedge-load-pre") {
      Result.setLoadPRESplitBackedge(Enable);
    } else if (ParamName == "memdep") {
      Result.setMemDep(Enable);
    } else {
      return make_error<StringError>(
          formatv("invalid GVN pass parameter '{0}' ", ParamName).str(),
          inconvertibleErrorCode());
    }
  }
  return Result;
}

Expected<StackLifetime::LivenessType>
parseStackLifetimeOptions(StringRef Params) {
  StackLifetime::LivenessType Result = StackLifetime::LivenessType::May;
  while (!Params.empty()) {
    StringRef ParamName;
    std::tie(ParamName, Params) = Params.split(';');

    if (ParamName == "may") {
      Result = StackLifetime::LivenessType::May;
    } else if (ParamName == "must") {
      Result = StackLifetime::LivenessType::Must;
    } else {
      return make_error<StringError>(
          formatv("invalid StackLifetime parameter '{0}' ", ParamName).str(),
          inconvertibleErrorCode());
    }
  }
  return Result;
}

} // namespace

/// Tests whether a pass name starts with a valid prefix for a default pipeline
/// alias.
static bool startsWithDefaultPipelineAliasPrefix(StringRef Name) {
  return Name.startswith("default") || Name.startswith("thinlto") ||
         Name.startswith("lto");
}

/// Tests whether registered callbacks will accept a given pass name.
///
/// When parsing a pipeline text, the type of the outermost pipeline may be
/// omitted, in which case the type is automatically determined from the first
/// pass name in the text. This may be a name that is handled through one of the
/// callbacks. We check this through the oridinary parsing callbacks by setting
/// up a dummy PassManager in order to not force the client to also handle this
/// type of query.
template <typename PassManagerT, typename CallbacksT>
static bool callbacksAcceptPassName(StringRef Name, CallbacksT &Callbacks) {
  if (!Callbacks.empty()) {
    PassManagerT DummyPM;
    for (auto &CB : Callbacks)
      if (CB(Name, DummyPM, {}))
        return true;
  }
  return false;
}

template <typename CallbacksT>
static bool isModulePassName(StringRef Name, CallbacksT &Callbacks) {
  // Manually handle aliases for pre-configured pipeline fragments.
  if (startsWithDefaultPipelineAliasPrefix(Name))
    return DefaultAliasRegex.match(Name);

  // Explicitly handle pass manager names.
  if (Name == "module")
    return true;
  if (Name == "cgscc")
    return true;
  if (Name == "function")
    return true;

  // Explicitly handle custom-parsed pass names.
  if (parseRepeatPassName(Name))
    return true;

#define MODULE_PASS(NAME, CREATE_PASS)                                         \
  if (Name == NAME)                                                            \
    return true;
#define MODULE_ANALYSIS(NAME, CREATE_PASS)                                     \
  if (Name == "require<" NAME ">" || Name == "invalidate<" NAME ">")           \
    return true;
#include "PassRegistry.def"

  return callbacksAcceptPassName<ModulePassManager>(Name, Callbacks);
}

template <typename CallbacksT>
static bool isCGSCCPassName(StringRef Name, CallbacksT &Callbacks) {
  // Explicitly handle pass manager names.
  if (Name == "cgscc")
    return true;
  if (Name == "function")
    return true;

  // Explicitly handle custom-parsed pass names.
  if (parseRepeatPassName(Name))
    return true;
  if (parseDevirtPassName(Name))
    return true;

#define CGSCC_PASS(NAME, CREATE_PASS)                                          \
  if (Name == NAME)                                                            \
    return true;
#define CGSCC_ANALYSIS(NAME, CREATE_PASS)                                      \
  if (Name == "require<" NAME ">" || Name == "invalidate<" NAME ">")           \
    return true;
#include "PassRegistry.def"

  return callbacksAcceptPassName<CGSCCPassManager>(Name, Callbacks);
}

template <typename CallbacksT>
static bool isFunctionPassName(StringRef Name, CallbacksT &Callbacks) {
  // Explicitly handle pass manager names.
  if (Name == "function")
    return true;
  if (Name == "loop" || Name == "loop-mssa")
    return true;

  // Explicitly handle custom-parsed pass names.
  if (parseRepeatPassName(Name))
    return true;

#define FUNCTION_PASS(NAME, CREATE_PASS)                                       \
  if (Name == NAME)                                                            \
    return true;
#define FUNCTION_PASS_WITH_PARAMS(NAME, CREATE_PASS, PARSER)                   \
  if (checkParametrizedPassName(Name, NAME))                                   \
    return true;
#define FUNCTION_ANALYSIS(NAME, CREATE_PASS)                                   \
  if (Name == "require<" NAME ">" || Name == "invalidate<" NAME ">")           \
    return true;
#include "PassRegistry.def"

  return callbacksAcceptPassName<FunctionPassManager>(Name, Callbacks);
}

template <typename CallbacksT>
static bool isLoopPassName(StringRef Name, CallbacksT &Callbacks) {
  // Explicitly handle pass manager names.
  if (Name == "loop" || Name == "loop-mssa")
    return true;

  // Explicitly handle custom-parsed pass names.
  if (parseRepeatPassName(Name))
    return true;

#define LOOP_PASS(NAME, CREATE_PASS)                                           \
  if (Name == NAME)                                                            \
    return true;
#define LOOP_PASS_WITH_PARAMS(NAME, CREATE_PASS, PARSER)                       \
  if (checkParametrizedPassName(Name, NAME))                                   \
    return true;
#define LOOP_ANALYSIS(NAME, CREATE_PASS)                                       \
  if (Name == "require<" NAME ">" || Name == "invalidate<" NAME ">")           \
    return true;
#include "PassRegistry.def"

  return callbacksAcceptPassName<LoopPassManager>(Name, Callbacks);
}

Optional<std::vector<PassBuilder::PipelineElement>>
PassBuilder::parsePipelineText(StringRef Text) {
  std::vector<PipelineElement> ResultPipeline;

  SmallVector<std::vector<PipelineElement> *, 4> PipelineStack = {
      &ResultPipeline};
  for (;;) {
    std::vector<PipelineElement> &Pipeline = *PipelineStack.back();
    size_t Pos = Text.find_first_of(",()");
    Pipeline.push_back({Text.substr(0, Pos), {}});

    // If we have a single terminating name, we're done.
    if (Pos == Text.npos)
      break;

    char Sep = Text[Pos];
    Text = Text.substr(Pos + 1);
    if (Sep == ',')
      // Just a name ending in a comma, continue.
      continue;

    if (Sep == '(') {
      // Push the inner pipeline onto the stack to continue processing.
      PipelineStack.push_back(&Pipeline.back().InnerPipeline);
      continue;
    }

    assert(Sep == ')' && "Bogus separator!");
    // When handling the close parenthesis, we greedily consume them to avoid
    // empty strings in the pipeline.
    do {
      // If we try to pop the outer pipeline we have unbalanced parentheses.
      if (PipelineStack.size() == 1)
        return None;

      PipelineStack.pop_back();
    } while (Text.consume_front(")"));

    // Check if we've finished parsing.
    if (Text.empty())
      break;

    // Otherwise, the end of an inner pipeline always has to be followed by
    // a comma, and then we can continue.
    if (!Text.consume_front(","))
      return None;
  }

  if (PipelineStack.size() > 1)
    // Unbalanced paretheses.
    return None;

  assert(PipelineStack.back() == &ResultPipeline &&
         "Wrong pipeline at the bottom of the stack!");
  return {std::move(ResultPipeline)};
}

Error PassBuilder::parseModulePass(ModulePassManager &MPM,
                                   const PipelineElement &E) {
  auto &Name = E.Name;
  auto &InnerPipeline = E.InnerPipeline;

  // First handle complex passes like the pass managers which carry pipelines.
  if (!InnerPipeline.empty()) {
    if (Name == "module") {
      ModulePassManager NestedMPM(DebugLogging);
      if (auto Err = parseModulePassPipeline(NestedMPM, InnerPipeline))
        return Err;
      MPM.addPass(std::move(NestedMPM));
      return Error::success();
    }
    if (Name == "cgscc") {
      CGSCCPassManager CGPM(DebugLogging);
      if (auto Err = parseCGSCCPassPipeline(CGPM, InnerPipeline))
        return Err;
      MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM)));
      return Error::success();
    }
    if (Name == "function") {
      FunctionPassManager FPM(DebugLogging);
      if (auto Err = parseFunctionPassPipeline(FPM, InnerPipeline))
        return Err;
      MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
      return Error::success();
    }
    if (auto Count = parseRepeatPassName(Name)) {
      ModulePassManager NestedMPM(DebugLogging);
      if (auto Err = parseModulePassPipeline(NestedMPM, InnerPipeline))
        return Err;
      MPM.addPass(createRepeatedPass(*Count, std::move(NestedMPM)));
      return Error::success();
    }

    for (auto &C : ModulePipelineParsingCallbacks)
      if (C(Name, MPM, InnerPipeline))
        return Error::success();

    // Normal passes can't have pipelines.
    return make_error<StringError>(
        formatv("invalid use of '{0}' pass as module pipeline", Name).str(),
        inconvertibleErrorCode());
    ;
  }

  // Manually handle aliases for pre-configured pipeline fragments.
  if (startsWithDefaultPipelineAliasPrefix(Name)) {
    SmallVector<StringRef, 3> Matches;
    if (!DefaultAliasRegex.match(Name, &Matches))
      return make_error<StringError>(
          formatv("unknown default pipeline alias '{0}'", Name).str(),
          inconvertibleErrorCode());

    assert(Matches.size() == 3 && "Must capture two matched strings!");

    OptimizationLevel L = StringSwitch<OptimizationLevel>(Matches[2])
                              .Case("O0", OptimizationLevel::O0)
                              .Case("O1", OptimizationLevel::O1)
                              .Case("O2", OptimizationLevel::O2)
                              .Case("O3", OptimizationLevel::O3)
                              .Case("Os", OptimizationLevel::Os)
                              .Case("Oz", OptimizationLevel::Oz);
    if (L == OptimizationLevel::O0 && Matches[1] != "thinlto" &&
        Matches[1] != "lto") {
      MPM.addPass(buildO0DefaultPipeline(L, Matches[1] == "thinlto-pre-link" ||
                                                Matches[1] == "lto-pre-link"));
      return Error::success();
    }

    // This is consistent with old pass manager invoked via opt, but
    // inconsistent with clang. Clang doesn't enable loop vectorization
    // but does enable slp vectorization at Oz.
    PTO.LoopVectorization =
        L.getSpeedupLevel() > 1 && L != OptimizationLevel::Oz;
    PTO.SLPVectorization =
        L.getSpeedupLevel() > 1 && L != OptimizationLevel::Oz;

    if (Matches[1] == "default") {
      MPM.addPass(buildPerModuleDefaultPipeline(L));
    } else if (Matches[1] == "thinlto-pre-link") {
      MPM.addPass(buildThinLTOPreLinkDefaultPipeline(L));
    } else if (Matches[1] == "thinlto") {
      MPM.addPass(buildThinLTODefaultPipeline(L, nullptr));
    } else if (Matches[1] == "lto-pre-link") {
      MPM.addPass(buildLTOPreLinkDefaultPipeline(L));
    } else {
      assert(Matches[1] == "lto" && "Not one of the matched options!");
      MPM.addPass(buildLTODefaultPipeline(L, nullptr));
    }
    return Error::success();
  }

  // Finally expand the basic registered passes from the .inc file.
#define MODULE_PASS(NAME, CREATE_PASS)                                         \
  if (Name == NAME) {                                                          \
    MPM.addPass(CREATE_PASS);                                                  \
    return Error::success();                                                   \
  }
#define MODULE_ANALYSIS(NAME, CREATE_PASS)                                     \
  if (Name == "require<" NAME ">") {                                           \
    MPM.addPass(                                                               \
        RequireAnalysisPass<                                                   \
            std::remove_reference<decltype(CREATE_PASS)>::type, Module>());    \
    return Error::success();                                                   \
  }                                                                            \
  if (Name == "invalidate<" NAME ">") {                                        \
    MPM.addPass(InvalidateAnalysisPass<                                        \
                std::remove_reference<decltype(CREATE_PASS)>::type>());        \
    return Error::success();                                                   \
  }
#define CGSCC_PASS(NAME, CREATE_PASS)                                          \
  if (Name == NAME) {                                                          \
    MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(CREATE_PASS));         \
    return Error::success();                                                   \
  }
#define FUNCTION_PASS(NAME, CREATE_PASS)                                       \
  if (Name == NAME) {                                                          \
    MPM.addPass(createModuleToFunctionPassAdaptor(CREATE_PASS));               \
    return Error::success();                                                   \
  }
#define FUNCTION_PASS_WITH_PARAMS(NAME, CREATE_PASS, PARSER)                   \
  if (checkParametrizedPassName(Name, NAME)) {                                 \
    auto Params = parsePassParameters(PARSER, Name, NAME);                     \
    if (!Params)                                                               \
      return Params.takeError();                                               \
    MPM.addPass(createModuleToFunctionPassAdaptor(CREATE_PASS(Params.get()))); \
    return Error::success();                                                   \
  }
#define LOOP_PASS(NAME, CREATE_PASS)                                           \
  if (Name == NAME) {                                                          \
    MPM.addPass(                                                               \
        createModuleToFunctionPassAdaptor(createFunctionToLoopPassAdaptor(     \
            CREATE_PASS, false, false, DebugLogging)));                        \
    return Error::success();                                                   \
  }
#define LOOP_PASS_WITH_PARAMS(NAME, CREATE_PASS, PARSER)                       \
  if (checkParametrizedPassName(Name, NAME)) {                                 \
    auto Params = parsePassParameters(PARSER, Name, NAME);                     \
    if (!Params)                                                               \
      return Params.takeError();                                               \
    MPM.addPass(                                                               \
        createModuleToFunctionPassAdaptor(createFunctionToLoopPassAdaptor(     \
            CREATE_PASS(Params.get()), false, false, DebugLogging)));          \
    return Error::success();                                                   \
  }
#include "PassRegistry.def"

  for (auto &C : ModulePipelineParsingCallbacks)
    if (C(Name, MPM, InnerPipeline))
      return Error::success();
  return make_error<StringError>(
      formatv("unknown module pass '{0}'", Name).str(),
      inconvertibleErrorCode());
}

Error PassBuilder::parseCGSCCPass(CGSCCPassManager &CGPM,
                                  const PipelineElement &E) {
  auto &Name = E.Name;
  auto &InnerPipeline = E.InnerPipeline;

  // First handle complex passes like the pass managers which carry pipelines.
  if (!InnerPipeline.empty()) {
    if (Name == "cgscc") {
      CGSCCPassManager NestedCGPM(DebugLogging);
      if (auto Err = parseCGSCCPassPipeline(NestedCGPM, InnerPipeline))
        return Err;
      // Add the nested pass manager with the appropriate adaptor.
      CGPM.addPass(std::move(NestedCGPM));
      return Error::success();
    }
    if (Name == "function") {
      FunctionPassManager FPM(DebugLogging);
      if (auto Err = parseFunctionPassPipeline(FPM, InnerPipeline))
        return Err;
      // Add the nested pass manager with the appropriate adaptor.
      CGPM.addPass(createCGSCCToFunctionPassAdaptor(std::move(FPM)));
      return Error::success();
    }
    if (auto Count = parseRepeatPassName(Name)) {
      CGSCCPassManager NestedCGPM(DebugLogging);
      if (auto Err = parseCGSCCPassPipeline(NestedCGPM, InnerPipeline))
        return Err;
      CGPM.addPass(createRepeatedPass(*Count, std::move(NestedCGPM)));
      return Error::success();
    }
    if (auto MaxRepetitions = parseDevirtPassName(Name)) {
      CGSCCPassManager NestedCGPM(DebugLogging);
      if (auto Err = parseCGSCCPassPipeline(NestedCGPM, InnerPipeline))
        return Err;
      CGPM.addPass(
          createDevirtSCCRepeatedPass(std::move(NestedCGPM), *MaxRepetitions));
      return Error::success();
    }

    for (auto &C : CGSCCPipelineParsingCallbacks)
      if (C(Name, CGPM, InnerPipeline))
        return Error::success();

    // Normal passes can't have pipelines.
    return make_error<StringError>(
        formatv("invalid use of '{0}' pass as cgscc pipeline", Name).str(),
        inconvertibleErrorCode());
  }

// Now expand the basic registered passes from the .inc file.
#define CGSCC_PASS(NAME, CREATE_PASS)                                          \
  if (Name == NAME) {                                                          \
    CGPM.addPass(CREATE_PASS);                                                 \
    return Error::success();                                                   \
  }
#define CGSCC_ANALYSIS(NAME, CREATE_PASS)                                      \
  if (Name == "require<" NAME ">") {                                           \
    CGPM.addPass(RequireAnalysisPass<                                          \
                 std::remove_reference<decltype(CREATE_PASS)>::type,           \
                 LazyCallGraph::SCC, CGSCCAnalysisManager, LazyCallGraph &,    \
                 CGSCCUpdateResult &>());                                      \
    return Error::success();                                                   \
  }                                                                            \
  if (Name == "invalidate<" NAME ">") {                                        \
    CGPM.addPass(InvalidateAnalysisPass<                                       \
                 std::remove_reference<decltype(CREATE_PASS)>::type>());       \
    return Error::success();                                                   \
  }
#define FUNCTION_PASS(NAME, CREATE_PASS)                                       \
  if (Name == NAME) {                                                          \
    CGPM.addPass(createCGSCCToFunctionPassAdaptor(CREATE_PASS));               \
    return Error::success();                                                   \
  }
#define FUNCTION_PASS_WITH_PARAMS(NAME, CREATE_PASS, PARSER)                   \
  if (checkParametrizedPassName(Name, NAME)) {                                 \
    auto Params = parsePassParameters(PARSER, Name, NAME);                     \
    if (!Params)                                                               \
      return Params.takeError();                                               \
    CGPM.addPass(createCGSCCToFunctionPassAdaptor(CREATE_PASS(Params.get()))); \
    return Error::success();                                                   \
  }
#define LOOP_PASS(NAME, CREATE_PASS)                                           \
  if (Name == NAME) {                                                          \
    CGPM.addPass(                                                              \
        createCGSCCToFunctionPassAdaptor(createFunctionToLoopPassAdaptor(      \
            CREATE_PASS, false, false, DebugLogging)));                        \
    return Error::success();                                                   \
  }
#define LOOP_PASS_WITH_PARAMS(NAME, CREATE_PASS, PARSER)                       \
  if (checkParametrizedPassName(Name, NAME)) {                                 \
    auto Params = parsePassParameters(PARSER, Name, NAME);                     \
    if (!Params)                                                               \
      return Params.takeError();                                               \
    CGPM.addPass(                                                              \
        createCGSCCToFunctionPassAdaptor(createFunctionToLoopPassAdaptor(      \
            CREATE_PASS(Params.get()), false, false, DebugLogging)));          \
    return Error::success();                                                   \
  }
#include "PassRegistry.def"

  for (auto &C : CGSCCPipelineParsingCallbacks)
    if (C(Name, CGPM, InnerPipeline))
      return Error::success();
  return make_error<StringError>(
      formatv("unknown cgscc pass '{0}'", Name).str(),
      inconvertibleErrorCode());
}

Error PassBuilder::parseFunctionPass(FunctionPassManager &FPM,
                                     const PipelineElement &E) {
  auto &Name = E.Name;
  auto &InnerPipeline = E.InnerPipeline;

  // First handle complex passes like the pass managers which carry pipelines.
  if (!InnerPipeline.empty()) {
    if (Name == "function") {
      FunctionPassManager NestedFPM(DebugLogging);
      if (auto Err = parseFunctionPassPipeline(NestedFPM, InnerPipeline))
        return Err;
      // Add the nested pass manager with the appropriate adaptor.
      FPM.addPass(std::move(NestedFPM));
      return Error::success();
    }
    if (Name == "loop" || Name == "loop-mssa") {
      LoopPassManager LPM(DebugLogging);
      if (auto Err = parseLoopPassPipeline(LPM, InnerPipeline))
        return Err;
      // Add the nested pass manager with the appropriate adaptor.
      bool UseMemorySSA = (Name == "loop-mssa");
      bool UseBFI = llvm::any_of(
          InnerPipeline, [](auto Pipeline) { return Pipeline.Name == "licm"; });
      FPM.addPass(createFunctionToLoopPassAdaptor(std::move(LPM), UseMemorySSA,
                                                  UseBFI, DebugLogging));
      return Error::success();
    }
    if (auto Count = parseRepeatPassName(Name)) {
      FunctionPassManager NestedFPM(DebugLogging);
      if (auto Err = parseFunctionPassPipeline(NestedFPM, InnerPipeline))
        return Err;
      FPM.addPass(createRepeatedPass(*Count, std::move(NestedFPM)));
      return Error::success();
    }

    for (auto &C : FunctionPipelineParsingCallbacks)
      if (C(Name, FPM, InnerPipeline))
        return Error::success();

    // Normal passes can't have pipelines.
    return make_error<StringError>(
        formatv("invalid use of '{0}' pass as function pipeline", Name).str(),
        inconvertibleErrorCode());
  }

// Now expand the basic registered passes from the .inc file.
#define FUNCTION_PASS(NAME, CREATE_PASS)                                       \
  if (Name == NAME) {                                                          \
    FPM.addPass(CREATE_PASS);                                                  \
    return Error::success();                                                   \
  }
#define FUNCTION_PASS_WITH_PARAMS(NAME, CREATE_PASS, PARSER)                   \
  if (checkParametrizedPassName(Name, NAME)) {                                 \
    auto Params = parsePassParameters(PARSER, Name, NAME);                     \
    if (!Params)                                                               \
      return Params.takeError();                                               \
    FPM.addPass(CREATE_PASS(Params.get()));                                    \
    return Error::success();                                                   \
  }
#define FUNCTION_ANALYSIS(NAME, CREATE_PASS)                                   \
  if (Name == "require<" NAME ">") {                                           \
    FPM.addPass(                                                               \
        RequireAnalysisPass<                                                   \
            std::remove_reference<decltype(CREATE_PASS)>::type, Function>());  \
    return Error::success();                                                   \
  }                                                                            \
  if (Name == "invalidate<" NAME ">") {                                        \
    FPM.addPass(InvalidateAnalysisPass<                                        \
                std::remove_reference<decltype(CREATE_PASS)>::type>());        \
    return Error::success();                                                   \
  }
// FIXME: UseMemorySSA is set to false. Maybe we could do things like:
//        bool UseMemorySSA = !("canon-freeze" || "loop-predication" ||
//                              "guard-widening");
//        The risk is that it may become obsolete if we're not careful.
#define LOOP_PASS(NAME, CREATE_PASS)                                           \
  if (Name == NAME) {                                                          \
    FPM.addPass(createFunctionToLoopPassAdaptor(CREATE_PASS, false, false,     \
                                                DebugLogging));                \
    return Error::success();                                                   \
  }
#define LOOP_PASS_WITH_PARAMS(NAME, CREATE_PASS, PARSER)                       \
  if (checkParametrizedPassName(Name, NAME)) {                                 \
    auto Params = parsePassParameters(PARSER, Name, NAME);                     \
    if (!Params)                                                               \
      return Params.takeError();                                               \
    FPM.addPass(createFunctionToLoopPassAdaptor(CREATE_PASS(Params.get()),     \
                                                false, false, DebugLogging));  \
    return Error::success();                                                   \
  }
#include "PassRegistry.def"

  for (auto &C : FunctionPipelineParsingCallbacks)
    if (C(Name, FPM, InnerPipeline))
      return Error::success();
  return make_error<StringError>(
      formatv("unknown function pass '{0}'", Name).str(),
      inconvertibleErrorCode());
}

Error PassBuilder::parseLoopPass(LoopPassManager &LPM,
                                 const PipelineElement &E) {
  StringRef Name = E.Name;
  auto &InnerPipeline = E.InnerPipeline;

  // First handle complex passes like the pass managers which carry pipelines.
  if (!InnerPipeline.empty()) {
    if (Name == "loop") {
      LoopPassManager NestedLPM(DebugLogging);
      if (auto Err = parseLoopPassPipeline(NestedLPM, InnerPipeline))
        return Err;
      // Add the nested pass manager with the appropriate adaptor.
      LPM.addPass(std::move(NestedLPM));
      return Error::success();
    }
    if (auto Count = parseRepeatPassName(Name)) {
      LoopPassManager NestedLPM(DebugLogging);
      if (auto Err = parseLoopPassPipeline(NestedLPM, InnerPipeline))
        return Err;
      LPM.addPass(createRepeatedPass(*Count, std::move(NestedLPM)));
      return Error::success();
    }

    for (auto &C : LoopPipelineParsingCallbacks)
      if (C(Name, LPM, InnerPipeline))
        return Error::success();

    // Normal passes can't have pipelines.
    return make_error<StringError>(
        formatv("invalid use of '{0}' pass as loop pipeline", Name).str(),
        inconvertibleErrorCode());
  }

// Now expand the basic registered passes from the .inc file.
#define LOOP_PASS(NAME, CREATE_PASS)                                           \
  if (Name == NAME) {                                                          \
    LPM.addPass(CREATE_PASS);                                                  \
    return Error::success();                                                   \
  }
#define LOOP_PASS_WITH_PARAMS(NAME, CREATE_PASS, PARSER)                       \
  if (checkParametrizedPassName(Name, NAME)) {                                 \
    auto Params = parsePassParameters(PARSER, Name, NAME);                     \
    if (!Params)                                                               \
      return Params.takeError();                                               \
    LPM.addPass(CREATE_PASS(Params.get()));                                    \
    return Error::success();                                                   \
  }
#define LOOP_ANALYSIS(NAME, CREATE_PASS)                                       \
  if (Name == "require<" NAME ">") {                                           \
    LPM.addPass(RequireAnalysisPass<                                           \
                std::remove_reference<decltype(CREATE_PASS)>::type, Loop,      \
                LoopAnalysisManager, LoopStandardAnalysisResults &,            \
                LPMUpdater &>());                                              \
    return Error::success();                                                   \
  }                                                                            \
  if (Name == "invalidate<" NAME ">") {                                        \
    LPM.addPass(InvalidateAnalysisPass<                                        \
                std::remove_reference<decltype(CREATE_PASS)>::type>());        \
    return Error::success();                                                   \
  }
#include "PassRegistry.def"

  for (auto &C : LoopPipelineParsingCallbacks)
    if (C(Name, LPM, InnerPipeline))
      return Error::success();
  return make_error<StringError>(formatv("unknown loop pass '{0}'", Name).str(),
                                 inconvertibleErrorCode());
}

bool PassBuilder::parseAAPassName(AAManager &AA, StringRef Name) {
#define MODULE_ALIAS_ANALYSIS(NAME, CREATE_PASS)                               \
  if (Name == NAME) {                                                          \
    AA.registerModuleAnalysis<                                                 \
        std::remove_reference<decltype(CREATE_PASS)>::type>();                 \
    return true;                                                               \
  }
#define FUNCTION_ALIAS_ANALYSIS(NAME, CREATE_PASS)                             \
  if (Name == NAME) {                                                          \
    AA.registerFunctionAnalysis<                                               \
        std::remove_reference<decltype(CREATE_PASS)>::type>();                 \
    return true;                                                               \
  }
#include "PassRegistry.def"

  for (auto &C : AAParsingCallbacks)
    if (C(Name, AA))
      return true;
  return false;
}

Error PassBuilder::parseLoopPassPipeline(LoopPassManager &LPM,
                                         ArrayRef<PipelineElement> Pipeline) {
  for (const auto &Element : Pipeline) {
    if (auto Err = parseLoopPass(LPM, Element))
      return Err;
  }
  return Error::success();
}

Error PassBuilder::parseFunctionPassPipeline(
    FunctionPassManager &FPM, ArrayRef<PipelineElement> Pipeline) {
  for (const auto &Element : Pipeline) {
    if (auto Err = parseFunctionPass(FPM, Element))
      return Err;
  }
  return Error::success();
}

Error PassBuilder::parseCGSCCPassPipeline(CGSCCPassManager &CGPM,
                                          ArrayRef<PipelineElement> Pipeline) {
  for (const auto &Element : Pipeline) {
    if (auto Err = parseCGSCCPass(CGPM, Element))
      return Err;
  }
  return Error::success();
}

void PassBuilder::crossRegisterProxies(LoopAnalysisManager &LAM,
                                       FunctionAnalysisManager &FAM,
                                       CGSCCAnalysisManager &CGAM,
                                       ModuleAnalysisManager &MAM) {
  MAM.registerPass([&] { return FunctionAnalysisManagerModuleProxy(FAM); });
  MAM.registerPass([&] { return CGSCCAnalysisManagerModuleProxy(CGAM); });
  CGAM.registerPass([&] { return ModuleAnalysisManagerCGSCCProxy(MAM); });
  FAM.registerPass([&] { return CGSCCAnalysisManagerFunctionProxy(CGAM); });
  FAM.registerPass([&] { return ModuleAnalysisManagerFunctionProxy(MAM); });
  FAM.registerPass([&] { return LoopAnalysisManagerFunctionProxy(LAM); });
  LAM.registerPass([&] { return FunctionAnalysisManagerLoopProxy(FAM); });
}

Error PassBuilder::parseModulePassPipeline(ModulePassManager &MPM,
                                           ArrayRef<PipelineElement> Pipeline) {
  for (const auto &Element : Pipeline) {
    if (auto Err = parseModulePass(MPM, Element))
      return Err;
  }
  return Error::success();
}

// Primary pass pipeline description parsing routine for a \c ModulePassManager
// FIXME: Should this routine accept a TargetMachine or require the caller to
// pre-populate the analysis managers with target-specific stuff?
Error PassBuilder::parsePassPipeline(ModulePassManager &MPM,
                                     StringRef PipelineText) {
  auto Pipeline = parsePipelineText(PipelineText);
  if (!Pipeline || Pipeline->empty())
    return make_error<StringError>(
        formatv("invalid pipeline '{0}'", PipelineText).str(),
        inconvertibleErrorCode());

  // If the first name isn't at the module layer, wrap the pipeline up
  // automatically.
  StringRef FirstName = Pipeline->front().Name;

  if (!isModulePassName(FirstName, ModulePipelineParsingCallbacks)) {
    if (isCGSCCPassName(FirstName, CGSCCPipelineParsingCallbacks)) {
      Pipeline = {{"cgscc", std::move(*Pipeline)}};
    } else if (isFunctionPassName(FirstName,
                                  FunctionPipelineParsingCallbacks)) {
      Pipeline = {{"function", std::move(*Pipeline)}};
    } else if (isLoopPassName(FirstName, LoopPipelineParsingCallbacks)) {
      Pipeline = {{"function", {{"loop", std::move(*Pipeline)}}}};
    } else {
      for (auto &C : TopLevelPipelineParsingCallbacks)
        if (C(MPM, *Pipeline, DebugLogging))
          return Error::success();

      // Unknown pass or pipeline name!
      auto &InnerPipeline = Pipeline->front().InnerPipeline;
      return make_error<StringError>(
          formatv("unknown {0} name '{1}'",
                  (InnerPipeline.empty() ? "pass" : "pipeline"), FirstName)
              .str(),
          inconvertibleErrorCode());
    }
  }

  if (auto Err = parseModulePassPipeline(MPM, *Pipeline))
    return Err;
  return Error::success();
}

// Primary pass pipeline description parsing routine for a \c CGSCCPassManager
Error PassBuilder::parsePassPipeline(CGSCCPassManager &CGPM,
                                     StringRef PipelineText) {
  auto Pipeline = parsePipelineText(PipelineText);
  if (!Pipeline || Pipeline->empty())
    return make_error<StringError>(
        formatv("invalid pipeline '{0}'", PipelineText).str(),
        inconvertibleErrorCode());

  StringRef FirstName = Pipeline->front().Name;
  if (!isCGSCCPassName(FirstName, CGSCCPipelineParsingCallbacks))
    return make_error<StringError>(
        formatv("unknown cgscc pass '{0}' in pipeline '{1}'", FirstName,
                PipelineText)
            .str(),
        inconvertibleErrorCode());

  if (auto Err = parseCGSCCPassPipeline(CGPM, *Pipeline))
    return Err;
  return Error::success();
}

// Primary pass pipeline description parsing routine for a \c
// FunctionPassManager
Error PassBuilder::parsePassPipeline(FunctionPassManager &FPM,
                                     StringRef PipelineText) {
  auto Pipeline = parsePipelineText(PipelineText);
  if (!Pipeline || Pipeline->empty())
    return make_error<StringError>(
        formatv("invalid pipeline '{0}'", PipelineText).str(),
        inconvertibleErrorCode());

  StringRef FirstName = Pipeline->front().Name;
  if (!isFunctionPassName(FirstName, FunctionPipelineParsingCallbacks))
    return make_error<StringError>(
        formatv("unknown function pass '{0}' in pipeline '{1}'", FirstName,
                PipelineText)
            .str(),
        inconvertibleErrorCode());

  if (auto Err = parseFunctionPassPipeline(FPM, *Pipeline))
    return Err;
  return Error::success();
}

// Primary pass pipeline description parsing routine for a \c LoopPassManager
Error PassBuilder::parsePassPipeline(LoopPassManager &CGPM,
                                     StringRef PipelineText) {
  auto Pipeline = parsePipelineText(PipelineText);
  if (!Pipeline || Pipeline->empty())
    return make_error<StringError>(
        formatv("invalid pipeline '{0}'", PipelineText).str(),
        inconvertibleErrorCode());

  if (auto Err = parseLoopPassPipeline(CGPM, *Pipeline))
    return Err;

  return Error::success();
}

Error PassBuilder::parseAAPipeline(AAManager &AA, StringRef PipelineText) {
  // If the pipeline just consists of the word 'default' just replace the AA
  // manager with our default one.
  if (PipelineText == "default") {
    AA = buildDefaultAAPipeline();
    return Error::success();
  }

  while (!PipelineText.empty()) {
    StringRef Name;
    std::tie(Name, PipelineText) = PipelineText.split(',');
    if (!parseAAPassName(AA, Name))
      return make_error<StringError>(
          formatv("unknown alias analysis name '{0}'", Name).str(),
          inconvertibleErrorCode());
  }

  return Error::success();
}

bool PassBuilder::isAAPassName(StringRef PassName) {
#define MODULE_ALIAS_ANALYSIS(NAME, CREATE_PASS)                               \
  if (PassName == NAME)                                                        \
    return true;
#define FUNCTION_ALIAS_ANALYSIS(NAME, CREATE_PASS)                             \
  if (PassName == NAME)                                                        \
    return true;
#include "PassRegistry.def"
  return false;
}

bool PassBuilder::isAnalysisPassName(StringRef PassName) {
#define MODULE_ANALYSIS(NAME, CREATE_PASS)                                     \
  if (PassName == NAME)                                                        \
    return true;
#define FUNCTION_ANALYSIS(NAME, CREATE_PASS)                                   \
  if (PassName == NAME)                                                        \
    return true;
#define LOOP_ANALYSIS(NAME, CREATE_PASS)                                       \
  if (PassName == NAME)                                                        \
    return true;
#define CGSCC_ANALYSIS(NAME, CREATE_PASS)                                      \
  if (PassName == NAME)                                                        \
    return true;
#define MODULE_ALIAS_ANALYSIS(NAME, CREATE_PASS)                               \
  if (PassName == NAME)                                                        \
    return true;
#define FUNCTION_ALIAS_ANALYSIS(NAME, CREATE_PASS)                             \
  if (PassName == NAME)                                                        \
    return true;
#include "PassRegistry.def"
  return false;
}

static void printPassName(StringRef PassName, raw_ostream &OS) {
  OS << "  " << PassName << "\n";
}

void PassBuilder::printPassNames(raw_ostream &OS) {
  // TODO: print pass descriptions when they are available

  OS << "Module passes:\n";
#define MODULE_PASS(NAME, CREATE_PASS) printPassName(NAME, OS);
#include "PassRegistry.def"

  OS << "Module analyses:\n";
#define MODULE_ANALYSIS(NAME, CREATE_PASS) printPassName(NAME, OS);
#include "PassRegistry.def"

  OS << "Module alias analyses:\n";
#define MODULE_ALIAS_ANALYSIS(NAME, CREATE_PASS) printPassName(NAME, OS);
#include "PassRegistry.def"

  OS << "CGSCC passes:\n";
#define CGSCC_PASS(NAME, CREATE_PASS) printPassName(NAME, OS);
#include "PassRegistry.def"

  OS << "CGSCC analyses:\n";
#define CGSCC_ANALYSIS(NAME, CREATE_PASS) printPassName(NAME, OS);
#include "PassRegistry.def"

  OS << "Function passes:\n";
#define FUNCTION_PASS(NAME, CREATE_PASS) printPassName(NAME, OS);
#include "PassRegistry.def"

  OS << "Function analyses:\n";
#define FUNCTION_ANALYSIS(NAME, CREATE_PASS) printPassName(NAME, OS);
#include "PassRegistry.def"

  OS << "Function alias analyses:\n";
#define FUNCTION_ALIAS_ANALYSIS(NAME, CREATE_PASS) printPassName(NAME, OS);
#include "PassRegistry.def"

  OS << "Loop passes:\n";
#define LOOP_PASS(NAME, CREATE_PASS) printPassName(NAME, OS);
#include "PassRegistry.def"

  OS << "Loop analyses:\n";
#define LOOP_ANALYSIS(NAME, CREATE_PASS) printPassName(NAME, OS);
#include "PassRegistry.def"
}

void PassBuilder::registerParseTopLevelPipelineCallback(
    const std::function<bool(ModulePassManager &, ArrayRef<PipelineElement>,
                             bool DebugLogging)> &C) {
  TopLevelPipelineParsingCallbacks.push_back(C);
}
