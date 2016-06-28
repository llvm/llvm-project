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
#include "llvm/Analysis/CFLAliasAnalysis.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/DemandedBits.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/DominanceFrontier.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/Analysis/LazyValueInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemoryDependenceAnalysis.h"
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
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Regex.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/GCOVProfiler.h"
#include "llvm/Transforms/IPO/ConstantMerge.h"
#include "llvm/Transforms/IPO/DeadArgumentElimination.h"
#include "llvm/Transforms/IPO/ElimAvailExtern.h"
#include "llvm/Transforms/IPO/ForceFunctionAttrs.h"
#include "llvm/Transforms/IPO/FunctionAttrs.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/IPO/GlobalOpt.h"
#include "llvm/Transforms/IPO/InferFunctionAttrs.h"
#include "llvm/Transforms/IPO/Internalize.h"
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
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Scalar/DeadStoreElimination.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/Float2Int.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/GuardWidening.h"
#include "llvm/Transforms/Scalar/IndVarSimplify.h"
#include "llvm/Transforms/Scalar/JumpThreading.h"
#include "llvm/Transforms/Scalar/LoopRotation.h"
#include "llvm/Transforms/Scalar/LoopSimplifyCFG.h"
#include "llvm/Transforms/Scalar/LowerAtomic.h"
#include "llvm/Transforms/Scalar/LowerExpectIntrinsic.h"
#include "llvm/Transforms/Scalar/MemCpyOptimizer.h"
#include "llvm/Transforms/Scalar/MergedLoadStoreMotion.h"
#include "llvm/Transforms/Scalar/PartiallyInlineLibCalls.h"
#include "llvm/Transforms/Scalar/Reassociate.h"
#include "llvm/Transforms/Scalar/SCCP.h"
#include "llvm/Transforms/Scalar/SLPVectorizer.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Scalar/Sink.h"
#include "llvm/Transforms/Utils/AddDiscriminators.h"
#include "llvm/Transforms/Utils/LCSSA.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"
#include "llvm/Transforms/Utils/MemorySSA.h"

#include <type_traits>

using namespace llvm;

static Regex DefaultAliasRegex("^(default|lto-pre-link|lto)<(O[0123sz])>$");

namespace {

/// \brief No-op module pass which does nothing.
struct NoOpModulePass {
  PreservedAnalyses run(Module &M, AnalysisManager<Module> &) {
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
  Result run(Module &, AnalysisManager<Module> &) { return Result(); }
  static StringRef name() { return "NoOpModuleAnalysis"; }
};

/// \brief No-op CGSCC pass which does nothing.
struct NoOpCGSCCPass {
  PreservedAnalyses run(LazyCallGraph::SCC &C,
                        AnalysisManager<LazyCallGraph::SCC> &) {
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
  Result run(LazyCallGraph::SCC &, AnalysisManager<LazyCallGraph::SCC> &) {
    return Result();
  }
  static StringRef name() { return "NoOpCGSCCAnalysis"; }
};

/// \brief No-op function pass which does nothing.
struct NoOpFunctionPass {
  PreservedAnalyses run(Function &F, AnalysisManager<Function> &) {
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
  Result run(Function &, AnalysisManager<Function> &) { return Result(); }
  static StringRef name() { return "NoOpFunctionAnalysis"; }
};

/// \brief No-op loop pass which does nothing.
struct NoOpLoopPass {
  PreservedAnalyses run(Loop &L, AnalysisManager<Loop> &) {
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
  Result run(Loop &, AnalysisManager<Loop> &) { return Result(); }
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

#ifndef NDEBUG
static bool isModulePassName(StringRef Name) {
  // Manually handle aliases for pre-configured pipeline fragments.
  if (Name.startswith("default") || Name.startswith("lto"))
    return DefaultAliasRegex.match(Name);

#define MODULE_PASS(NAME, CREATE_PASS)                                         \
  if (Name == NAME)                                                            \
    return true;
#define MODULE_ANALYSIS(NAME, CREATE_PASS)                                     \
  if (Name == "require<" NAME ">" || Name == "invalidate<" NAME ">")           \
    return true;
#include "PassRegistry.def"

  return false;
}
#endif

static bool isCGSCCPassName(StringRef Name) {
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
#define LOOP_PASS(NAME, CREATE_PASS)                                           \
  if (Name == NAME)                                                            \
    return true;
#define LOOP_ANALYSIS(NAME, CREATE_PASS)                                       \
  if (Name == "require<" NAME ">" || Name == "invalidate<" NAME ">")           \
    return true;
#include "PassRegistry.def"

  return false;
}

bool PassBuilder::parseModulePassName(ModulePassManager &MPM, StringRef Name,
                                      bool DebugLogging) {
  // Manually handle aliases for pre-configured pipeline fragments.
  if (Name.startswith("default") || Name.startswith("lto")) {
    SmallVector<StringRef, 3> Matches;
    if (!DefaultAliasRegex.match(Name, &Matches))
      return false;
    assert(Matches.size() == 3 && "Must capture two matched strings!");

    auto L = StringSwitch<OptimizationLevel>(Matches[2])
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

#define MODULE_PASS(NAME, CREATE_PASS)                                         \
  if (Name == NAME) {                                                          \
    MPM.addPass(CREATE_PASS);                                                  \
    return true;                                                               \
  }
#define MODULE_ANALYSIS(NAME, CREATE_PASS)                                     \
  if (Name == "require<" NAME ">") {                                           \
    MPM.addPass(RequireAnalysisPass<                                           \
                std::remove_reference<decltype(CREATE_PASS)>::type>());        \
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

bool PassBuilder::parseCGSCCPassName(CGSCCPassManager &CGPM, StringRef Name) {
#define CGSCC_PASS(NAME, CREATE_PASS)                                          \
  if (Name == NAME) {                                                          \
    CGPM.addPass(CREATE_PASS);                                                 \
    return true;                                                               \
  }
#define CGSCC_ANALYSIS(NAME, CREATE_PASS)                                      \
  if (Name == "require<" NAME ">") {                                           \
    CGPM.addPass(RequireAnalysisPass<                                          \
                 std::remove_reference<decltype(CREATE_PASS)>::type>());       \
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

bool PassBuilder::parseFunctionPassName(FunctionPassManager &FPM,
                                        StringRef Name) {
#define FUNCTION_PASS(NAME, CREATE_PASS)                                       \
  if (Name == NAME) {                                                          \
    FPM.addPass(CREATE_PASS);                                                  \
    return true;                                                               \
  }
#define FUNCTION_ANALYSIS(NAME, CREATE_PASS)                                   \
  if (Name == "require<" NAME ">") {                                           \
    FPM.addPass(RequireAnalysisPass<                                           \
                std::remove_reference<decltype(CREATE_PASS)>::type>());        \
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

bool PassBuilder::parseLoopPassName(LoopPassManager &FPM, StringRef Name) {
#define LOOP_PASS(NAME, CREATE_PASS)                                           \
  if (Name == NAME) {                                                          \
    FPM.addPass(CREATE_PASS);                                                  \
    return true;                                                               \
  }
#define LOOP_ANALYSIS(NAME, CREATE_PASS)                                       \
  if (Name == "require<" NAME ">") {                                           \
    FPM.addPass(RequireAnalysisPass<                                           \
                std::remove_reference<decltype(CREATE_PASS)>::type>());        \
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
                                        StringRef &PipelineText,
                                        bool VerifyEachPass,
                                        bool DebugLogging) {
  for (;;) {
    // Parse nested pass managers by recursing.
    if (PipelineText.startswith("loop(")) {
      LoopPassManager NestedLPM(DebugLogging);

      // Parse the inner pipeline inte the nested manager.
      PipelineText = PipelineText.substr(strlen("loop("));
      if (!parseLoopPassPipeline(NestedLPM, PipelineText, VerifyEachPass,
                                 DebugLogging) ||
          PipelineText.empty())
        return false;
      assert(PipelineText[0] == ')');
      PipelineText = PipelineText.substr(1);

      // Add the nested pass manager with the appropriate adaptor.
      LPM.addPass(std::move(NestedLPM));
    } else {
      // Otherwise try to parse a pass name.
      size_t End = PipelineText.find_first_of(",)");
      if (!parseLoopPassName(LPM, PipelineText.substr(0, End)))
        return false;
      // TODO: Ideally, we would run a LoopVerifierPass() here in the
      // VerifyEachPass case, but we don't have such a verifier yet.

      PipelineText = PipelineText.substr(End);
    }

    if (PipelineText.empty() || PipelineText[0] == ')')
      return true;

    assert(PipelineText[0] == ',');
    PipelineText = PipelineText.substr(1);
  }
}

bool PassBuilder::parseFunctionPassPipeline(FunctionPassManager &FPM,
                                            StringRef &PipelineText,
                                            bool VerifyEachPass,
                                            bool DebugLogging) {
  for (;;) {
    // Parse nested pass managers by recursing.
    if (PipelineText.startswith("function(")) {
      FunctionPassManager NestedFPM(DebugLogging);

      // Parse the inner pipeline inte the nested manager.
      PipelineText = PipelineText.substr(strlen("function("));
      if (!parseFunctionPassPipeline(NestedFPM, PipelineText, VerifyEachPass,
                                     DebugLogging) ||
          PipelineText.empty())
        return false;
      assert(PipelineText[0] == ')');
      PipelineText = PipelineText.substr(1);

      // Add the nested pass manager with the appropriate adaptor.
      FPM.addPass(std::move(NestedFPM));
    } else if (PipelineText.startswith("loop(")) {
      LoopPassManager NestedLPM(DebugLogging);

      // Parse the inner pipeline inte the nested manager.
      PipelineText = PipelineText.substr(strlen("loop("));
      if (!parseLoopPassPipeline(NestedLPM, PipelineText, VerifyEachPass,
                                 DebugLogging) ||
          PipelineText.empty())
        return false;
      assert(PipelineText[0] == ')');
      PipelineText = PipelineText.substr(1);

      // Add the nested pass manager with the appropriate adaptor.
      FPM.addPass(createFunctionToLoopPassAdaptor(std::move(NestedLPM)));
    } else {
      // Otherwise try to parse a pass name.
      size_t End = PipelineText.find_first_of(",)");
      if (!parseFunctionPassName(FPM, PipelineText.substr(0, End)))
        return false;
      if (VerifyEachPass)
        FPM.addPass(VerifierPass());

      PipelineText = PipelineText.substr(End);
    }

    if (PipelineText.empty() || PipelineText[0] == ')')
      return true;

    assert(PipelineText[0] == ',');
    PipelineText = PipelineText.substr(1);
  }
}

bool PassBuilder::parseCGSCCPassPipeline(CGSCCPassManager &CGPM,
                                         StringRef &PipelineText,
                                         bool VerifyEachPass,
                                         bool DebugLogging) {
  for (;;) {
    // Parse nested pass managers by recursing.
    if (PipelineText.startswith("cgscc(")) {
      CGSCCPassManager NestedCGPM(DebugLogging);

      // Parse the inner pipeline into the nested manager.
      PipelineText = PipelineText.substr(strlen("cgscc("));
      if (!parseCGSCCPassPipeline(NestedCGPM, PipelineText, VerifyEachPass,
                                  DebugLogging) ||
          PipelineText.empty())
        return false;
      assert(PipelineText[0] == ')');
      PipelineText = PipelineText.substr(1);

      // Add the nested pass manager with the appropriate adaptor.
      CGPM.addPass(std::move(NestedCGPM));
    } else if (PipelineText.startswith("function(")) {
      FunctionPassManager NestedFPM(DebugLogging);

      // Parse the inner pipeline inte the nested manager.
      PipelineText = PipelineText.substr(strlen("function("));
      if (!parseFunctionPassPipeline(NestedFPM, PipelineText, VerifyEachPass,
                                     DebugLogging) ||
          PipelineText.empty())
        return false;
      assert(PipelineText[0] == ')');
      PipelineText = PipelineText.substr(1);

      // Add the nested pass manager with the appropriate adaptor.
      CGPM.addPass(
          createCGSCCToFunctionPassAdaptor(std::move(NestedFPM), DebugLogging));
    } else {
      // Otherwise try to parse a pass name.
      size_t End = PipelineText.find_first_of(",)");
      if (!parseCGSCCPassName(CGPM, PipelineText.substr(0, End)))
        return false;
      // FIXME: No verifier support for CGSCC passes!

      PipelineText = PipelineText.substr(End);
    }

    if (PipelineText.empty() || PipelineText[0] == ')')
      return true;

    assert(PipelineText[0] == ',');
    PipelineText = PipelineText.substr(1);
  }
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
                                          StringRef &PipelineText,
                                          bool VerifyEachPass,
                                          bool DebugLogging) {
  for (;;) {
    // Parse nested pass managers by recursing.
    if (PipelineText.startswith("module(")) {
      ModulePassManager NestedMPM(DebugLogging);

      // Parse the inner pipeline into the nested manager.
      PipelineText = PipelineText.substr(strlen("module("));
      if (!parseModulePassPipeline(NestedMPM, PipelineText, VerifyEachPass,
                                   DebugLogging) ||
          PipelineText.empty())
        return false;
      assert(PipelineText[0] == ')');
      PipelineText = PipelineText.substr(1);

      // Now add the nested manager as a module pass.
      MPM.addPass(std::move(NestedMPM));
    } else if (PipelineText.startswith("cgscc(")) {
      CGSCCPassManager NestedCGPM(DebugLogging);

      // Parse the inner pipeline inte the nested manager.
      PipelineText = PipelineText.substr(strlen("cgscc("));
      if (!parseCGSCCPassPipeline(NestedCGPM, PipelineText, VerifyEachPass,
                                  DebugLogging) ||
          PipelineText.empty())
        return false;
      assert(PipelineText[0] == ')');
      PipelineText = PipelineText.substr(1);

      // Add the nested pass manager with the appropriate adaptor.
      MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(NestedCGPM),
                                                          DebugLogging));
    } else if (PipelineText.startswith("function(")) {
      FunctionPassManager NestedFPM(DebugLogging);

      // Parse the inner pipeline inte the nested manager.
      PipelineText = PipelineText.substr(strlen("function("));
      if (!parseFunctionPassPipeline(NestedFPM, PipelineText, VerifyEachPass,
                                     DebugLogging) ||
          PipelineText.empty())
        return false;
      assert(PipelineText[0] == ')');
      PipelineText = PipelineText.substr(1);

      // Add the nested pass manager with the appropriate adaptor.
      MPM.addPass(createModuleToFunctionPassAdaptor(std::move(NestedFPM)));
    } else {
      // Otherwise try to parse a pass name.
      size_t End = PipelineText.find_first_of(",)");
      if (!parseModulePassName(MPM, PipelineText.substr(0, End), DebugLogging))
        return false;
      if (VerifyEachPass)
        MPM.addPass(VerifierPass());

      PipelineText = PipelineText.substr(End);
    }

    if (PipelineText.empty() || PipelineText[0] == ')')
      return true;

    assert(PipelineText[0] == ',');
    PipelineText = PipelineText.substr(1);
  }
}

// Primary pass pipeline description parsing routine.
// FIXME: Should this routine accept a TargetMachine or require the caller to
// pre-populate the analysis managers with target-specific stuff?
bool PassBuilder::parsePassPipeline(ModulePassManager &MPM,
                                    StringRef PipelineText, bool VerifyEachPass,
                                    bool DebugLogging) {
  // By default, try to parse the pipeline as-if it were within an implicit
  // 'module(...)' pass pipeline. If this will parse at all, it needs to
  // consume the entire string.
  if (parseModulePassPipeline(MPM, PipelineText, VerifyEachPass, DebugLogging))
    return PipelineText.empty();

  // This isn't parsable as a module pipeline, look for the end of a pass name
  // and directly drop down to that layer.
  StringRef FirstName =
      PipelineText.substr(0, PipelineText.find_first_of(",)"));
  assert(!isModulePassName(FirstName) &&
         "Already handled all module pipeline options.");

  // If this looks like a CGSCC pass, parse the whole thing as a CGSCC
  // pipeline.
  if (PipelineText.startswith("cgscc(") || isCGSCCPassName(FirstName)) {
    CGSCCPassManager CGPM(DebugLogging);
    if (!parseCGSCCPassPipeline(CGPM, PipelineText, VerifyEachPass,
                                DebugLogging) ||
        !PipelineText.empty())
      return false;
    MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM), DebugLogging));
    return true;
  }

  // Similarly, if this looks like a Function pass, parse the whole thing as
  // a Function pipelien.
  if (PipelineText.startswith("function(") || isFunctionPassName(FirstName)) {
    FunctionPassManager FPM(DebugLogging);
    if (!parseFunctionPassPipeline(FPM, PipelineText, VerifyEachPass,
                                   DebugLogging) ||
        !PipelineText.empty())
      return false;
    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
    return true;
  }

  // If this looks like a Loop pass, parse the whole thing as a Loop pipeline.
  if (PipelineText.startswith("loop(") || isLoopPassName(FirstName)) {
    LoopPassManager LPM(DebugLogging);
    if (!parseLoopPassPipeline(LPM, PipelineText, VerifyEachPass,
                               DebugLogging) ||
        !PipelineText.empty())
      return false;
    FunctionPassManager FPM(DebugLogging);
    FPM.addPass(createFunctionToLoopPassAdaptor(std::move(LPM)));
    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
    return true;
  }

  return false;
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
