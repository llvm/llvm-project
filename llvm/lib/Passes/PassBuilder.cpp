//===- Parsing, selection, and construction of pass pipelines -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AliasAnalysisEvaluator.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BlockFrequencyInfoImpl.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/CFLAndersAliasAnalysis.h"
#include "llvm/Analysis/CFLSteensAliasAnalysis.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/DemandedBits.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/DominanceFrontier.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/IVUsers.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/Analysis/LazyValueInfo.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemoryDependenceAnalysis.h"
#include "llvm/Analysis/ModuleSummaryAnalysis.h"
#include "llvm/Analysis/OptimizationDiagnosticInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionAliasAnalysis.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/TypeBasedAliasAnalysis.h"
#include "llvm/CodeGen/PreISelIntrinsicLowering.h"
#include "llvm/CodeGen/UnreachableBlockElim.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Regex.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/GCOVProfiler.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/ConstantMerge.h"
#include "llvm/Transforms/IPO/CrossDSOCFI.h"
#include "llvm/Transforms/IPO/DeadArgumentElimination.h"
#include "llvm/Transforms/IPO/ElimAvailExtern.h"
#include "llvm/Transforms/IPO/ForceFunctionAttrs.h"
#include "llvm/Transforms/IPO/FunctionAttrs.h"
#include "llvm/Transforms/IPO/FunctionImport.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/IPO/GlobalOpt.h"
#include "llvm/Transforms/IPO/InferFunctionAttrs.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/IPO/LowerTypeTests.h"
#include "llvm/Transforms/IPO/PartialInlining.h"
#include "llvm/Transforms/IPO/SCCP.h"
#include "llvm/Transforms/IPO/StripDeadPrototypes.h"
#include "llvm/Transforms/IPO/WholeProgramDevirt.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/InstrProfiling.h"
#include "llvm/Transforms/PGOInstrumentation.h"
#include "llvm/Transforms/SampleProfile.h"
#include "llvm/Transforms/Scalar/ADCE.h"
#include "llvm/Transforms/Scalar/AlignmentFromAssumptions.h"
#include "llvm/Transforms/Scalar/BDCE.h"
#include "llvm/Transforms/Scalar/ConstantHoisting.h"
#include "llvm/Transforms/Scalar/CorrelatedValuePropagation.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Scalar/DeadStoreElimination.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/Float2Int.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/GuardWidening.h"
#include "llvm/Transforms/Scalar/IndVarSimplify.h"
#include "llvm/Transforms/Scalar/JumpThreading.h"
#include "llvm/Transforms/Scalar/LICM.h"
#include "llvm/Transforms/Scalar/LoopDataPrefetch.h"
#include "llvm/Transforms/Scalar/LoopDeletion.h"
#include "llvm/Transforms/Scalar/LoopDistribute.h"
#include "llvm/Transforms/Scalar/LoopIdiomRecognize.h"
#include "llvm/Transforms/Scalar/LoopInstSimplify.h"
#include "llvm/Transforms/Scalar/LoopRotation.h"
#include "llvm/Transforms/Scalar/LoopSimplifyCFG.h"
#include "llvm/Transforms/Scalar/LoopStrengthReduce.h"
#include "llvm/Transforms/Scalar/LoopUnrollPass.h"
#include "llvm/Transforms/Scalar/LowerAtomic.h"
#include "llvm/Transforms/Scalar/LowerExpectIntrinsic.h"
#include "llvm/Transforms/Scalar/LowerGuardIntrinsic.h"
#include "llvm/Transforms/Scalar/MemCpyOptimizer.h"
#include "llvm/Transforms/Scalar/MergedLoadStoreMotion.h"
#include "llvm/Transforms/Scalar/NaryReassociate.h"
#include "llvm/Transforms/Scalar/PartiallyInlineLibCalls.h"
#include "llvm/Transforms/Scalar/Reassociate.h"
#include "llvm/Transforms/Scalar/SCCP.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Scalar/Sink.h"
#include "llvm/Transforms/Scalar/SpeculativeExecution.h"
#include "llvm/Transforms/Scalar/TailRecursionElimination.h"
#include "llvm/Transforms/Utils/AddDiscriminators.h"
#include "llvm/Transforms/Utils/BreakCriticalEdges.h"
#include "llvm/Transforms/Utils/LCSSA.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/LowerInvoke.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"
#include "llvm/Transforms/Utils/MemorySSA.h"
#include "llvm/Transforms/Utils/NameAnonFunctions.h"
#include "llvm/Transforms/Utils/SimplifyInstructions.h"
#include "llvm/Transforms/Utils/SymbolRewriter.h"
#include "llvm/Transforms/Vectorize/LoopVectorize.h"
#include "llvm/Transforms/Vectorize/SLPVectorizer.h"

#include <type_traits>

using namespace llvm;

static Regex DefaultAliasRegex("^(default|lto-pre-link|lto)<(O[0123sz])>$");

namespace {

/// \brief No-op module pass which does nothing.
struct NoOpModulePass {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    return PreservedAnalyses::all();
  }
  static StringRef name() { return "NoOpModulePass"; }
};

/// \brief No-op module analysis.
class NoOpModuleAnalysis : public AnalysisInfoMixin<NoOpModuleAnalysis> {
  friend AnalysisInfoMixin<NoOpModuleAnalysis>;
  static char PassID;

public:
  struct Result {};
  Result run(Module &, ModuleAnalysisManager &) { return Result(); }
  static StringRef name() { return "NoOpModuleAnalysis"; }
};

/// \brief No-op CGSCC pass which does nothing.
struct NoOpCGSCCPass {
  PreservedAnalyses run(LazyCallGraph::SCC &C,
                        CGSCCAnalysisManager &) {
    return PreservedAnalyses::all();
  }
  static StringRef name() { return "NoOpCGSCCPass"; }
};

/// \brief No-op CGSCC analysis.
class NoOpCGSCCAnalysis : public AnalysisInfoMixin<NoOpCGSCCAnalysis> {
  friend AnalysisInfoMixin<NoOpCGSCCAnalysis>;
  static char PassID;

public:
  struct Result {};
  Result run(LazyCallGraph::SCC &, CGSCCAnalysisManager &) {
    return Result();
  }
  static StringRef name() { return "NoOpCGSCCAnalysis"; }
};

/// \brief No-op function pass which does nothing.
struct NoOpFunctionPass {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    return PreservedAnalyses::all();
  }
  static StringRef name() { return "NoOpFunctionPass"; }
};

/// \brief No-op function analysis.
class NoOpFunctionAnalysis : public AnalysisInfoMixin<NoOpFunctionAnalysis> {
  friend AnalysisInfoMixin<NoOpFunctionAnalysis>;
  static char PassID;

public:
  struct Result {};
  Result run(Function &, FunctionAnalysisManager &) { return Result(); }
  static StringRef name() { return "NoOpFunctionAnalysis"; }
};

/// \brief No-op loop pass which does nothing.
struct NoOpLoopPass {
  PreservedAnalyses run(Loop &L, LoopAnalysisManager &) {
    return PreservedAnalyses::all();
  }
  static StringRef name() { return "NoOpLoopPass"; }
};

/// \brief No-op loop analysis.
class NoOpLoopAnalysis : public AnalysisInfoMixin<NoOpLoopAnalysis> {
  friend AnalysisInfoMixin<NoOpLoopAnalysis>;
  static char PassID;

public:
  struct Result {};
  Result run(Loop &, LoopAnalysisManager &) { return Result(); }
  static StringRef name() { return "NoOpLoopAnalysis"; }
};

char NoOpModuleAnalysis::PassID;
char NoOpCGSCCAnalysis::PassID;
char NoOpFunctionAnalysis::PassID;
char NoOpLoopAnalysis::PassID;

} // End anonymous namespace.

void PassBuilder::registerModuleAnalyses(ModuleAnalysisManager &MAM) {
#define MODULE_ANALYSIS(NAME, CREATE_PASS)                                     \
  MAM.registerPass([&] { return CREATE_PASS; });
#include "PassRegistry.def"
}

void PassBuilder::registerCGSCCAnalyses(CGSCCAnalysisManager &CGAM) {
#define CGSCC_ANALYSIS(NAME, CREATE_PASS)                                      \
  CGAM.registerPass([&] { return CREATE_PASS; });
#include "PassRegistry.def"
}

void PassBuilder::registerFunctionAnalyses(FunctionAnalysisManager &FAM) {
#define FUNCTION_ANALYSIS(NAME, CREATE_PASS)                                   \
  FAM.registerPass([&] { return CREATE_PASS; });
#include "PassRegistry.def"
}

void PassBuilder::registerLoopAnalyses(LoopAnalysisManager &LAM) {
#define LOOP_ANALYSIS(NAME, CREATE_PASS)                                       \
  LAM.registerPass([&] { return CREATE_PASS; });
#include "PassRegistry.def"
}

void PassBuilder::addPerModuleDefaultPipeline(ModulePassManager &MPM,
                                              OptimizationLevel Level,
                                              bool DebugLogging) {
  // FIXME: Finish fleshing this out to match the legacy pipelines.
  FunctionPassManager EarlyFPM(DebugLogging);
  EarlyFPM.addPass(SimplifyCFGPass());
  EarlyFPM.addPass(SROA());
  EarlyFPM.addPass(EarlyCSEPass());
  EarlyFPM.addPass(LowerExpectIntrinsicPass());

  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(EarlyFPM)));
}

void PassBuilder::addLTOPreLinkDefaultPipeline(ModulePassManager &MPM,
                                               OptimizationLevel Level,
                                               bool DebugLogging) {
  // FIXME: We should use a customized pre-link pipeline!
  addPerModuleDefaultPipeline(MPM, Level, DebugLogging);
}

void PassBuilder::addLTODefaultPipeline(ModulePassManager &MPM,
                                        OptimizationLevel Level,
                                        bool DebugLogging) {
  // FIXME: Finish fleshing this out to match the legacy LTO pipelines.
  FunctionPassManager LateFPM(DebugLogging);
  LateFPM.addPass(InstCombinePass());
  LateFPM.addPass(SimplifyCFGPass());

  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(LateFPM)));
}

static Optional<int> parseRepeatPassName(StringRef Name) {
  if (!Name.consume_front("repeat<") || !Name.consume_back(">"))
    return None;
  int Count;
  if (Name.getAsInteger(0, Count) || Count <= 0)
    return None;
  return Count;
}

static bool isModulePassName(StringRef Name) {
  // Manually handle aliases for pre-configured pipeline fragments.
  if (Name.startswith("default") || Name.startswith("lto"))
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

  return false;
}

static bool isCGSCCPassName(StringRef Name) {
  // Explicitly handle pass manager names.
  if (Name == "cgscc")
    return true;
  if (Name == "function")
    return true;

  // Explicitly handle custom-parsed pass names.
  if (parseRepeatPassName(Name))
    return true;

#define CGSCC_PASS(NAME, CREATE_PASS)                                          \
  if (Name == NAME)                                                            \
    return true;
#define CGSCC_ANALYSIS(NAME, CREATE_PASS)                                      \
  if (Name == "require<" NAME ">" || Name == "invalidate<" NAME ">")           \
    return true;
#include "PassRegistry.def"

  return false;
}

static bool isFunctionPassName(StringRef Name) {
  // Explicitly handle pass manager names.
  if (Name == "function")
    return true;
  if (Name == "loop")
    return true;

  // Explicitly handle custom-parsed pass names.
  if (parseRepeatPassName(Name))
    return true;

#define FUNCTION_PASS(NAME, CREATE_PASS)                                       \
  if (Name == NAME)                                                            \
    return true;
#define FUNCTION_ANALYSIS(NAME, CREATE_PASS)                                   \
  if (Name == "require<" NAME ">" || Name == "invalidate<" NAME ">")           \
    return true;
#include "PassRegistry.def"

  return false;
}

static bool isLoopPassName(StringRef Name) {
  // Explicitly handle pass manager names.
  if (Name == "loop")
    return true;

  // Explicitly handle custom-parsed pass names.
  if (parseRepeatPassName(Name))
    return true;

#define LOOP_PASS(NAME, CREATE_PASS)                                           \
  if (Name == NAME)                                                            \
    return true;
#define LOOP_ANALYSIS(NAME, CREATE_PASS)                                       \
  if (Name == "require<" NAME ">" || Name == "invalidate<" NAME ">")           \
    return true;
#include "PassRegistry.def"

  return false;
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

bool PassBuilder::parseModulePass(ModulePassManager &MPM,
                                  const PipelineElement &E, bool VerifyEachPass,
                                  bool DebugLogging) {
  auto &Name = E.Name;
  auto &InnerPipeline = E.InnerPipeline;

  // First handle complex passes like the pass managers which carry pipelines.
  if (!InnerPipeline.empty()) {
    if (Name == "module") {
      ModulePassManager NestedMPM(DebugLogging);
      if (!parseModulePassPipeline(NestedMPM, InnerPipeline, VerifyEachPass,
                                   DebugLogging))
        return false;
      MPM.addPass(std::move(NestedMPM));
      return true;
    }
    if (Name == "cgscc") {
      CGSCCPassManager CGPM(DebugLogging);
      if (!parseCGSCCPassPipeline(CGPM, InnerPipeline, VerifyEachPass,
                                  DebugLogging))
        return false;
      MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM),
                                                          DebugLogging));
      return true;
    }
    if (Name == "function") {
      FunctionPassManager FPM(DebugLogging);
      if (!parseFunctionPassPipeline(FPM, InnerPipeline, VerifyEachPass,
                                     DebugLogging))
        return false;
      MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
      return true;
    }
    if (auto Count = parseRepeatPassName(Name)) {
      ModulePassManager NestedMPM(DebugLogging);
      if (!parseModulePassPipeline(NestedMPM, InnerPipeline, VerifyEachPass,
                                   DebugLogging))
        return false;
      MPM.addPass(createRepeatedPass(*Count, std::move(NestedMPM)));
      return true;
    }
    // Normal passes can't have pipelines.
    return false;
  }

  // Manually handle aliases for pre-configured pipeline fragments.
  if (Name.startswith("default") || Name.startswith("lto")) {
    SmallVector<StringRef, 3> Matches;
    if (!DefaultAliasRegex.match(Name, &Matches))
      return false;
    assert(Matches.size() == 3 && "Must capture two matched strings!");

    OptimizationLevel L = StringSwitch<OptimizationLevel>(Matches[2])
        .Case("O0", O0)
        .Case("O1", O1)
        .Case("O2", O2)
        .Case("O3", O3)
        .Case("Os", Os)
        .Case("Oz", Oz);

    if (Matches[1] == "default") {
      addPerModuleDefaultPipeline(MPM, L, DebugLogging);
    } else if (Matches[1] == "lto-pre-link") {
      addLTOPreLinkDefaultPipeline(MPM, L, DebugLogging);
    } else {
      assert(Matches[1] == "lto" && "Not one of the matched options!");
      addLTODefaultPipeline(MPM, L, DebugLogging);
    }
    return true;
  }

  // Finally expand the basic registered passes from the .inc file.
#define MODULE_PASS(NAME, CREATE_PASS)                                         \
  if (Name == NAME) {                                                          \
    MPM.addPass(CREATE_PASS);                                                  \
    return true;                                                               \
  }
#define MODULE_ANALYSIS(NAME, CREATE_PASS)                                     \
  if (Name == "require<" NAME ">") {                                           \
    MPM.addPass(                                                               \
        RequireAnalysisPass<                                                   \
            std::remove_reference<decltype(CREATE_PASS)>::type, Module>());    \
    return true;                                                               \
  }                                                                            \
  if (Name == "invalidate<" NAME ">") {                                        \
    MPM.addPass(InvalidateAnalysisPass<                                        \
                std::remove_reference<decltype(CREATE_PASS)>::type>());        \
    return true;                                                               \
  }
#include "PassRegistry.def"

  return false;
}

bool PassBuilder::parseCGSCCPass(CGSCCPassManager &CGPM,
                                 const PipelineElement &E, bool VerifyEachPass,
                                 bool DebugLogging) {
  auto &Name = E.Name;
  auto &InnerPipeline = E.InnerPipeline;

  // First handle complex passes like the pass managers which carry pipelines.
  if (!InnerPipeline.empty()) {
    if (Name == "cgscc") {
      CGSCCPassManager NestedCGPM(DebugLogging);
      if (!parseCGSCCPassPipeline(NestedCGPM, InnerPipeline, VerifyEachPass,
                                  DebugLogging))
        return false;
      // Add the nested pass manager with the appropriate adaptor.
      CGPM.addPass(std::move(NestedCGPM));
      return true;
    }
    if (Name == "function") {
      FunctionPassManager FPM(DebugLogging);
      if (!parseFunctionPassPipeline(FPM, InnerPipeline, VerifyEachPass,
                                     DebugLogging))
        return false;
      // Add the nested pass manager with the appropriate adaptor.
      CGPM.addPass(
          createCGSCCToFunctionPassAdaptor(std::move(FPM), DebugLogging));
      return true;
    }
    if (auto Count = parseRepeatPassName(Name)) {
      CGSCCPassManager NestedCGPM(DebugLogging);
      if (!parseCGSCCPassPipeline(NestedCGPM, InnerPipeline, VerifyEachPass,
                                  DebugLogging))
        return false;
      CGPM.addPass(createRepeatedPass(*Count, std::move(NestedCGPM)));
      return true;
    }
    // Normal passes can't have pipelines.
    return false;
  }

  // Now expand the basic registered passes from the .inc file.
#define CGSCC_PASS(NAME, CREATE_PASS)                                          \
  if (Name == NAME) {                                                          \
    CGPM.addPass(CREATE_PASS);                                                 \
    return true;                                                               \
  }
#define CGSCC_ANALYSIS(NAME, CREATE_PASS)                                      \
  if (Name == "require<" NAME ">") {                                           \
    CGPM.addPass(RequireAnalysisPass<                                          \
                 std::remove_reference<decltype(CREATE_PASS)>::type,           \
                 LazyCallGraph::SCC>());                                       \
    return true;                                                               \
  }                                                                            \
  if (Name == "invalidate<" NAME ">") {                                        \
    CGPM.addPass(InvalidateAnalysisPass<                                       \
                 std::remove_reference<decltype(CREATE_PASS)>::type>());       \
    return true;                                                               \
  }
#include "PassRegistry.def"

  return false;
}

bool PassBuilder::parseFunctionPass(FunctionPassManager &FPM,
                                    const PipelineElement &E,
                                    bool VerifyEachPass, bool DebugLogging) {
  auto &Name = E.Name;
  auto &InnerPipeline = E.InnerPipeline;

  // First handle complex passes like the pass managers which carry pipelines.
  if (!InnerPipeline.empty()) {
    if (Name == "function") {
      FunctionPassManager NestedFPM(DebugLogging);
      if (!parseFunctionPassPipeline(NestedFPM, InnerPipeline, VerifyEachPass,
                                     DebugLogging))
        return false;
      // Add the nested pass manager with the appropriate adaptor.
      FPM.addPass(std::move(NestedFPM));
      return true;
    }
    if (Name == "loop") {
      LoopPassManager LPM(DebugLogging);
      if (!parseLoopPassPipeline(LPM, InnerPipeline, VerifyEachPass,
                                 DebugLogging))
        return false;
      // Add the nested pass manager with the appropriate adaptor.
      FPM.addPass(createFunctionToLoopPassAdaptor(std::move(LPM)));
      return true;
    }
    if (auto Count = parseRepeatPassName(Name)) {
      FunctionPassManager NestedFPM(DebugLogging);
      if (!parseFunctionPassPipeline(NestedFPM, InnerPipeline, VerifyEachPass,
                                     DebugLogging))
        return false;
      FPM.addPass(createRepeatedPass(*Count, std::move(NestedFPM)));
      return true;
    }
    // Normal passes can't have pipelines.
    return false;
  }

  // Now expand the basic registered passes from the .inc file.
#define FUNCTION_PASS(NAME, CREATE_PASS)                                       \
  if (Name == NAME) {                                                          \
    FPM.addPass(CREATE_PASS);                                                  \
    return true;                                                               \
  }
#define FUNCTION_ANALYSIS(NAME, CREATE_PASS)                                   \
  if (Name == "require<" NAME ">") {                                           \
    FPM.addPass(                                                               \
        RequireAnalysisPass<                                                   \
            std::remove_reference<decltype(CREATE_PASS)>::type, Function>());  \
    return true;                                                               \
  }                                                                            \
  if (Name == "invalidate<" NAME ">") {                                        \
    FPM.addPass(InvalidateAnalysisPass<                                        \
                std::remove_reference<decltype(CREATE_PASS)>::type>());        \
    return true;                                                               \
  }
#include "PassRegistry.def"

  return false;
}

bool PassBuilder::parseLoopPass(LoopPassManager &LPM, const PipelineElement &E,
                                bool VerifyEachPass, bool DebugLogging) {
  StringRef Name = E.Name;
  auto &InnerPipeline = E.InnerPipeline;

  // First handle complex passes like the pass managers which carry pipelines.
  if (!InnerPipeline.empty()) {
    if (Name == "loop") {
      LoopPassManager NestedLPM(DebugLogging);
      if (!parseLoopPassPipeline(NestedLPM, InnerPipeline, VerifyEachPass,
                                 DebugLogging))
        return false;
      // Add the nested pass manager with the appropriate adaptor.
      LPM.addPass(std::move(NestedLPM));
      return true;
    }
    if (auto Count = parseRepeatPassName(Name)) {
      LoopPassManager NestedLPM(DebugLogging);
      if (!parseLoopPassPipeline(NestedLPM, InnerPipeline, VerifyEachPass,
                                 DebugLogging))
        return false;
      LPM.addPass(createRepeatedPass(*Count, std::move(NestedLPM)));
      return true;
    }
    // Normal passes can't have pipelines.
    return false;
  }

  // Now expand the basic registered passes from the .inc file.
#define LOOP_PASS(NAME, CREATE_PASS)                                           \
  if (Name == NAME) {                                                          \
    LPM.addPass(CREATE_PASS);                                                  \
    return true;                                                               \
  }
#define LOOP_ANALYSIS(NAME, CREATE_PASS)                                       \
  if (Name == "require<" NAME ">") {                                           \
    LPM.addPass(RequireAnalysisPass<                                           \
                std::remove_reference<decltype(CREATE_PASS)>::type, Loop>());  \
    return true;                                                               \
  }                                                                            \
  if (Name == "invalidate<" NAME ">") {                                        \
    LPM.addPass(InvalidateAnalysisPass<                                        \
                std::remove_reference<decltype(CREATE_PASS)>::type>());        \
    return true;                                                               \
  }
#include "PassRegistry.def"

  return false;
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

  return false;
}

bool PassBuilder::parseLoopPassPipeline(LoopPassManager &LPM,
                                        ArrayRef<PipelineElement> Pipeline,
                                        bool VerifyEachPass,
                                        bool DebugLogging) {
  for (const auto &Element : Pipeline) {
    if (!parseLoopPass(LPM, Element, VerifyEachPass, DebugLogging))
      return false;
    // FIXME: No verifier support for Loop passes!
  }
  return true;
}

bool PassBuilder::parseFunctionPassPipeline(FunctionPassManager &FPM,
                                            ArrayRef<PipelineElement> Pipeline,
                                            bool VerifyEachPass,
                                            bool DebugLogging) {
  for (const auto &Element : Pipeline) {
    if (!parseFunctionPass(FPM, Element, VerifyEachPass, DebugLogging))
      return false;
    if (VerifyEachPass)
      FPM.addPass(VerifierPass());
  }
  return true;
}

bool PassBuilder::parseCGSCCPassPipeline(CGSCCPassManager &CGPM,
                                         ArrayRef<PipelineElement> Pipeline,
                                         bool VerifyEachPass,
                                         bool DebugLogging) {
  for (const auto &Element : Pipeline) {
    if (!parseCGSCCPass(CGPM, Element, VerifyEachPass, DebugLogging))
      return false;
    // FIXME: No verifier support for CGSCC passes!
  }
  return true;
}

void PassBuilder::crossRegisterProxies(LoopAnalysisManager &LAM,
                                       FunctionAnalysisManager &FAM,
                                       CGSCCAnalysisManager &CGAM,
                                       ModuleAnalysisManager &MAM) {
  MAM.registerPass([&] { return FunctionAnalysisManagerModuleProxy(FAM); });
  MAM.registerPass([&] { return CGSCCAnalysisManagerModuleProxy(CGAM); });
  CGAM.registerPass([&] { return FunctionAnalysisManagerCGSCCProxy(FAM); });
  CGAM.registerPass([&] { return ModuleAnalysisManagerCGSCCProxy(MAM); });
  FAM.registerPass([&] { return CGSCCAnalysisManagerFunctionProxy(CGAM); });
  FAM.registerPass([&] { return ModuleAnalysisManagerFunctionProxy(MAM); });
  FAM.registerPass([&] { return LoopAnalysisManagerFunctionProxy(LAM); });
  LAM.registerPass([&] { return FunctionAnalysisManagerLoopProxy(FAM); });
}

bool PassBuilder::parseModulePassPipeline(ModulePassManager &MPM,
                                          ArrayRef<PipelineElement> Pipeline,
                                          bool VerifyEachPass,
                                          bool DebugLogging) {
  for (const auto &Element : Pipeline) {
    if (!parseModulePass(MPM, Element, VerifyEachPass, DebugLogging))
      return false;
    if (VerifyEachPass)
      MPM.addPass(VerifierPass());
  }
  return true;
}

// Primary pass pipeline description parsing routine.
// FIXME: Should this routine accept a TargetMachine or require the caller to
// pre-populate the analysis managers with target-specific stuff?
bool PassBuilder::parsePassPipeline(ModulePassManager &MPM,
                                    StringRef PipelineText, bool VerifyEachPass,
                                    bool DebugLogging) {
  auto Pipeline = parsePipelineText(PipelineText);
  if (!Pipeline || Pipeline->empty())
    return false;

  // If the first name isn't at the module layer, wrap the pipeline up
  // automatically.
  StringRef FirstName = Pipeline->front().Name;

  if (!isModulePassName(FirstName)) {
    if (isCGSCCPassName(FirstName))
      Pipeline = {{"cgscc", std::move(*Pipeline)}};
    else if (isFunctionPassName(FirstName))
      Pipeline = {{"function", std::move(*Pipeline)}};
    else if (isLoopPassName(FirstName))
      Pipeline = {{"function", {{"loop", std::move(*Pipeline)}}}};
    else
      // Unknown pass name!
      return false;
  }

  return parseModulePassPipeline(MPM, *Pipeline, VerifyEachPass, DebugLogging);
}

bool PassBuilder::parseAAPipeline(AAManager &AA, StringRef PipelineText) {
  while (!PipelineText.empty()) {
    StringRef Name;
    std::tie(Name, PipelineText) = PipelineText.split(',');
    if (!parseAAPassName(AA, Name))
      return false;
  }

  return true;
}
