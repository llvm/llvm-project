//===------ RegisterPasses.cpp - Add the Polly Passes to default passes  --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file composes the individual LLVM-IR passes provided by Polly to a
// functional polyhedral optimizer. The polyhedral optimizer is automatically
// made available to LLVM based compilers by loading the Polly shared library
// into such a compiler.
//
// The Polly optimizer is made available by executing a static constructor that
// registers the individual Polly passes in the LLVM pass manager builder. The
// passes are registered such that the default behaviour of the compiler is not
// changed, but that the flag '-polly' provided at optimization level '-O3'
// enables additional polyhedral optimizations.
//===----------------------------------------------------------------------===//

#include "polly/RegisterPasses.h"
#include "polly/Canonicalization.h"
#include "polly/CodeGen/CodeGeneration.h"
#include "polly/CodeGen/CodegenCleanup.h"
#include "polly/CodeGen/IslAst.h"
#include "polly/CodePreparation.h"
#include "polly/DependenceInfo.h"
#include "polly/ForwardOpTree.h"
#include "polly/JSONExporter.h"
#include "polly/LinkAllPasses.h"
#include "polly/PolyhedralInfo.h"
#include "polly/ScopDetection.h"
#include "polly/ScopInfo.h"
#include "polly/Simplify.h"
#include "polly/Support/DumpModulePass.h"
#include "llvm/Analysis/CFGPrinter.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

using namespace llvm;
using namespace polly;

cl::OptionCategory PollyCategory("Polly Options",
                                 "Configure the polly loop optimizer");

static cl::opt<bool>
    PollyEnabled("polly", cl::desc("Enable the polly optimizer (only at -O3)"),
                 cl::init(false), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<bool> PollyDetectOnly(
    "polly-only-scop-detection",
    cl::desc("Only run scop detection, but no other optimizations"),
    cl::init(false), cl::ZeroOrMore, cl::cat(PollyCategory));

enum PassPositionChoice {
  POSITION_EARLY,
  POSITION_AFTER_LOOPOPT,
  POSITION_BEFORE_VECTORIZER
};

enum OptimizerChoice { OPTIMIZER_NONE, OPTIMIZER_ISL };

static cl::opt<PassPositionChoice> PassPosition(
    "polly-position", cl::desc("Where to run polly in the pass pipeline"),
    cl::values(
        clEnumValN(POSITION_EARLY, "early", "Before everything"),
        clEnumValN(POSITION_AFTER_LOOPOPT, "after-loopopt",
                   "After the loop optimizer (but within the inline cycle)"),
        clEnumValN(POSITION_BEFORE_VECTORIZER, "before-vectorizer",
                   "Right before the vectorizer")),
    cl::Hidden, cl::init(POSITION_BEFORE_VECTORIZER), cl::ZeroOrMore,
    cl::cat(PollyCategory));

static cl::opt<OptimizerChoice>
    Optimizer("polly-optimizer", cl::desc("Select the scheduling optimizer"),
              cl::values(clEnumValN(OPTIMIZER_NONE, "none", "No optimizer"),
                         clEnumValN(OPTIMIZER_ISL, "isl",
                                    "The isl scheduling optimizer")),
              cl::Hidden, cl::init(OPTIMIZER_ISL), cl::ZeroOrMore,
              cl::cat(PollyCategory));

enum CodeGenChoice { CODEGEN_FULL, CODEGEN_AST, CODEGEN_NONE };
static cl::opt<CodeGenChoice> CodeGeneration(
    "polly-code-generation", cl::desc("How much code-generation to perform"),
    cl::values(clEnumValN(CODEGEN_FULL, "full", "AST and IR generation"),
               clEnumValN(CODEGEN_AST, "ast", "Only AST generation"),
               clEnumValN(CODEGEN_NONE, "none", "No code generation")),
    cl::Hidden, cl::init(CODEGEN_FULL), cl::ZeroOrMore, cl::cat(PollyCategory));

enum TargetChoice { TARGET_CPU, TARGET_GPU, TARGET_HYBRID };
static cl::opt<TargetChoice>
    Target("polly-target", cl::desc("The hardware to target"),
           cl::values(clEnumValN(TARGET_CPU, "cpu", "generate CPU code")
#ifdef GPU_CODEGEN
                          ,
                      clEnumValN(TARGET_GPU, "gpu", "generate GPU code"),
                      clEnumValN(TARGET_HYBRID, "hybrid",
                                 "generate GPU code (preferably) or CPU code")
#endif
                          ),
           cl::init(TARGET_CPU), cl::ZeroOrMore, cl::cat(PollyCategory));

#ifdef GPU_CODEGEN
static cl::opt<GPURuntime> GPURuntimeChoice(
    "polly-gpu-runtime", cl::desc("The GPU Runtime API to target"),
    cl::values(clEnumValN(GPURuntime::CUDA, "libcudart",
                          "use the CUDA Runtime API"),
               clEnumValN(GPURuntime::OpenCL, "libopencl",
                          "use the OpenCL Runtime API")),
    cl::init(GPURuntime::CUDA), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<GPUArch>
    GPUArchChoice("polly-gpu-arch", cl::desc("The GPU Architecture to target"),
                  cl::values(clEnumValN(GPUArch::NVPTX64, "nvptx64",
                                        "target NVIDIA 64-bit architecture"),
                             clEnumValN(GPUArch::SPIR32, "spir32",
                                        "target SPIR 32-bit architecture"),
                             clEnumValN(GPUArch::SPIR64, "spir64",
                                        "target SPIR 64-bit architecture")),
                  cl::init(GPUArch::NVPTX64), cl::ZeroOrMore,
                  cl::cat(PollyCategory));
#endif

VectorizerChoice polly::PollyVectorizerChoice;
static cl::opt<polly::VectorizerChoice, true> Vectorizer(
    "polly-vectorizer", cl::desc("Select the vectorization strategy"),
    cl::values(
        clEnumValN(polly::VECTORIZER_NONE, "none", "No Vectorization"),
        clEnumValN(polly::VECTORIZER_POLLY, "polly",
                   "Polly internal vectorizer"),
        clEnumValN(
            polly::VECTORIZER_STRIPMINE, "stripmine",
            "Strip-mine outer loops for the loop-vectorizer to trigger")),
    cl::location(PollyVectorizerChoice), cl::init(polly::VECTORIZER_NONE),
    cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<bool> ImportJScop(
    "polly-import",
    cl::desc("Import the polyhedral description of the detected Scops"),
    cl::Hidden, cl::init(false), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<bool> FullyIndexedStaticExpansion(
    "polly-enable-mse",
    cl::desc("Fully expand the memory accesses of the detected Scops"),
    cl::Hidden, cl::init(false), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<bool> ExportJScop(
    "polly-export",
    cl::desc("Export the polyhedral description of the detected Scops"),
    cl::Hidden, cl::init(false), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<bool> DeadCodeElim("polly-run-dce",
                                  cl::desc("Run the dead code elimination"),
                                  cl::Hidden, cl::init(false), cl::ZeroOrMore,
                                  cl::cat(PollyCategory));

static cl::opt<bool> PollyViewer(
    "polly-show",
    cl::desc("Highlight the code regions that will be optimized in a "
             "(CFG BBs and LLVM-IR instructions)"),
    cl::init(false), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<bool> PollyOnlyViewer(
    "polly-show-only",
    cl::desc("Highlight the code regions that will be optimized in "
             "a (CFG only BBs)"),
    cl::init(false), cl::cat(PollyCategory));

static cl::opt<bool>
    PollyPrinter("polly-dot", cl::desc("Enable the Polly DOT printer in -O3"),
                 cl::Hidden, cl::value_desc("Run the Polly DOT printer at -O3"),
                 cl::init(false), cl::cat(PollyCategory));

static cl::opt<bool> PollyOnlyPrinter(
    "polly-dot-only",
    cl::desc("Enable the Polly DOT printer in -O3 (no BB content)"), cl::Hidden,
    cl::value_desc("Run the Polly DOT printer at -O3 (no BB content"),
    cl::init(false), cl::cat(PollyCategory));

static cl::opt<bool>
    CFGPrinter("polly-view-cfg",
               cl::desc("Show the Polly CFG right after code generation"),
               cl::Hidden, cl::init(false), cl::cat(PollyCategory));

static cl::opt<bool>
    EnablePolyhedralInfo("polly-enable-polyhedralinfo",
                         cl::desc("Enable polyhedral interface of Polly"),
                         cl::Hidden, cl::init(false), cl::cat(PollyCategory));

static cl::opt<bool>
    EnableForwardOpTree("polly-enable-optree",
                        cl::desc("Enable operand tree forwarding"), cl::Hidden,
                        cl::init(true), cl::cat(PollyCategory));

static cl::opt<bool>
    DumpBefore("polly-dump-before",
               cl::desc("Dump module before Polly transformations into a file "
                        "suffixed with \"-before\""),
               cl::init(false), cl::cat(PollyCategory));

static cl::list<std::string> DumpBeforeFile(
    "polly-dump-before-file",
    cl::desc("Dump module before Polly transformations to the given file"),
    cl::cat(PollyCategory));

static cl::opt<bool>
    DumpAfter("polly-dump-after",
              cl::desc("Dump module after Polly transformations into a file "
                       "suffixed with \"-after\""),
              cl::init(false), cl::cat(PollyCategory));

static cl::list<std::string> DumpAfterFile(
    "polly-dump-after-file",
    cl::desc("Dump module after Polly transformations to the given file"),
    cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<bool>
    EnableDeLICM("polly-enable-delicm",
                 cl::desc("Eliminate scalar loop carried dependences"),
                 cl::Hidden, cl::init(true), cl::cat(PollyCategory));

static cl::opt<bool>
    EnableSimplify("polly-enable-simplify",
                   cl::desc("Simplify SCoP after optimizations"),
                   cl::init(true), cl::cat(PollyCategory));

static cl::opt<bool> EnablePruneUnprofitable(
    "polly-enable-prune-unprofitable",
    cl::desc("Bail out on unprofitable SCoPs before rescheduling"), cl::Hidden,
    cl::init(true), cl::cat(PollyCategory));

namespace polly {
void initializePollyPasses(PassRegistry &Registry) {
  initializeCodeGenerationPass(Registry);

#ifdef GPU_CODEGEN
  initializePPCGCodeGenerationPass(Registry);
  initializeManagedMemoryRewritePassPass(Registry);
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();
#endif
  initializeCodePreparationPass(Registry);
  initializeDeadCodeElimPass(Registry);
  initializeDependenceInfoPass(Registry);
  initializeDependenceInfoWrapperPassPass(Registry);
  initializeJSONExporterPass(Registry);
  initializeJSONImporterPass(Registry);
  initializeMaximalStaticExpanderPass(Registry);
  initializeIslAstInfoWrapperPassPass(Registry);
  initializeIslScheduleOptimizerPass(Registry);
  initializePollyCanonicalizePass(Registry);
  initializePolyhedralInfoPass(Registry);
  initializeScopDetectionWrapperPassPass(Registry);
  initializeScopInlinerPass(Registry);
  initializeScopInfoRegionPassPass(Registry);
  initializeScopInfoWrapperPassPass(Registry);
  initializeRewriteByrefParamsPass(Registry);
  initializeCodegenCleanupPass(Registry);
  initializeFlattenSchedulePass(Registry);
  initializeForwardOpTreePass(Registry);
  initializeDeLICMPass(Registry);
  initializeSimplifyPass(Registry);
  initializeDumpModulePass(Registry);
  initializePruneUnprofitablePass(Registry);
}

/// Register Polly passes such that they form a polyhedral optimizer.
///
/// The individual Polly passes are registered in the pass manager such that
/// they form a full polyhedral optimizer. The flow of the optimizer starts with
/// a set of preparing transformations that canonicalize the LLVM-IR such that
/// the LLVM-IR is easier for us to understand and to optimizes. On the
/// canonicalized LLVM-IR we first run the ScopDetection pass, which detects
/// static control flow regions. Those regions are then translated by the
/// ScopInfo pass into a polyhedral representation. As a next step, a scheduling
/// optimizer is run on the polyhedral representation and finally the optimized
/// polyhedral representation is code generated back to LLVM-IR.
///
/// Besides this core functionality, we optionally schedule passes that provide
/// a graphical view of the scops (Polly[Only]Viewer, Polly[Only]Printer), that
/// allow the export/import of the polyhedral representation
/// (JSCON[Exporter|Importer]) or that show the cfg after code generation.
///
/// For certain parts of the Polly optimizer, several alternatives are provided:
///
/// As scheduling optimizer we support the isl scheduling optimizer
/// (http://freecode.com/projects/isl).
/// It is also possible to run Polly with no optimizer. This mode is mainly
/// provided to analyze the run and compile time changes caused by the
/// scheduling optimizer.
///
/// Polly supports the isl internal code generator.
void registerPollyPasses(llvm::legacy::PassManagerBase &PM) {
  if (DumpBefore)
    PM.add(polly::createDumpModulePass("-before", true));
  for (auto &Filename : DumpBeforeFile)
    PM.add(polly::createDumpModulePass(Filename, false));

  PM.add(polly::createScopDetectionWrapperPassPass());

  if (PollyDetectOnly)
    return;

  if (PollyViewer)
    PM.add(polly::createDOTViewerPass());
  if (PollyOnlyViewer)
    PM.add(polly::createDOTOnlyViewerPass());
  if (PollyPrinter)
    PM.add(polly::createDOTPrinterPass());
  if (PollyOnlyPrinter)
    PM.add(polly::createDOTOnlyPrinterPass());

  PM.add(polly::createScopInfoRegionPassPass());
  if (EnablePolyhedralInfo)
    PM.add(polly::createPolyhedralInfoPass());

  if (EnableSimplify)
    PM.add(polly::createSimplifyPass(0));
  if (EnableForwardOpTree)
    PM.add(polly::createForwardOpTreePass());
  if (EnableDeLICM)
    PM.add(polly::createDeLICMPass());
  if (EnableSimplify)
    PM.add(polly::createSimplifyPass(1));

  if (ImportJScop)
    PM.add(polly::createJSONImporterPass());

  if (DeadCodeElim)
    PM.add(polly::createDeadCodeElimPass());

  if (FullyIndexedStaticExpansion)
    PM.add(polly::createMaximalStaticExpansionPass());

  if (EnablePruneUnprofitable)
    PM.add(polly::createPruneUnprofitablePass());

#ifdef GPU_CODEGEN
  if (Target == TARGET_HYBRID)
    PM.add(
        polly::createPPCGCodeGenerationPass(GPUArchChoice, GPURuntimeChoice));
#endif
  if (Target == TARGET_CPU || Target == TARGET_HYBRID)
    switch (Optimizer) {
    case OPTIMIZER_NONE:
      break; /* Do nothing */

    case OPTIMIZER_ISL:
      PM.add(polly::createIslScheduleOptimizerPass());
      break;
    }

  if (ExportJScop)
    PM.add(polly::createJSONExporterPass());

  if (Target == TARGET_CPU || Target == TARGET_HYBRID)
    switch (CodeGeneration) {
    case CODEGEN_AST:
      PM.add(polly::createIslAstInfoWrapperPassPass());
      break;
    case CODEGEN_FULL:
      PM.add(polly::createCodeGenerationPass());
      break;
    case CODEGEN_NONE:
      break;
    }
#ifdef GPU_CODEGEN
  else {
    PM.add(
        polly::createPPCGCodeGenerationPass(GPUArchChoice, GPURuntimeChoice));
    PM.add(polly::createManagedMemoryRewritePassPass());
  }
#endif

#ifdef GPU_CODEGEN
  if (Target == TARGET_HYBRID)
    PM.add(polly::createManagedMemoryRewritePassPass(GPUArchChoice,
                                                     GPURuntimeChoice));
#endif

  // FIXME: This dummy ModulePass keeps some programs from miscompiling,
  // probably some not correctly preserved analyses. It acts as a barrier to
  // force all analysis results to be recomputed.
  PM.add(createBarrierNoopPass());

  if (DumpAfter)
    PM.add(polly::createDumpModulePass("-after", true));
  for (auto &Filename : DumpAfterFile)
    PM.add(polly::createDumpModulePass(Filename, false));

  if (CFGPrinter)
    PM.add(llvm::createCFGPrinterLegacyPassPass());
}

static bool shouldEnablePolly() {
  if (PollyOnlyPrinter || PollyPrinter || PollyOnlyViewer || PollyViewer)
    PollyTrackFailures = true;

  if (PollyOnlyPrinter || PollyPrinter || PollyOnlyViewer || PollyViewer ||
      ExportJScop || ImportJScop)
    PollyEnabled = true;

  return PollyEnabled;
}

static void
registerPollyEarlyAsPossiblePasses(const llvm::PassManagerBuilder &Builder,
                                   llvm::legacy::PassManagerBase &PM) {
  if (!polly::shouldEnablePolly())
    return;

  if (PassPosition != POSITION_EARLY)
    return;

  registerCanonicalicationPasses(PM);
  polly::registerPollyPasses(PM);
}

static void
registerPollyLoopOptimizerEndPasses(const llvm::PassManagerBuilder &Builder,
                                    llvm::legacy::PassManagerBase &PM) {
  if (!polly::shouldEnablePolly())
    return;

  if (PassPosition != POSITION_AFTER_LOOPOPT)
    return;

  PM.add(polly::createCodePreparationPass());
  polly::registerPollyPasses(PM);
  PM.add(createCodegenCleanupPass());
}

static void
registerPollyScalarOptimizerLatePasses(const llvm::PassManagerBuilder &Builder,
                                       llvm::legacy::PassManagerBase &PM) {
  if (!polly::shouldEnablePolly())
    return;

  if (PassPosition != POSITION_BEFORE_VECTORIZER)
    return;

  PM.add(polly::createCodePreparationPass());
  polly::registerPollyPasses(PM);
  PM.add(createCodegenCleanupPass());
}

static void buildDefaultPollyPipeline(FunctionPassManager &PM,
                                      PassBuilder::OptimizationLevel Level) {
  if (!polly::shouldEnablePolly())
    return;
  PassBuilder PB;
  ScopPassManager SPM;

  // TODO add utility passes for the various command line options, once they're
  // ported
  assert(!DumpBefore && "This option is not implemented");
  assert(DumpBeforeFile.empty() && "This option is not implemented");

  if (PollyDetectOnly)
    return;

  assert(!PollyViewer && "This option is not implemented");
  assert(!PollyOnlyViewer && "This option is not implemented");
  assert(!PollyPrinter && "This option is not implemented");
  assert(!PollyOnlyPrinter && "This option is not implemented");
  assert(!EnablePolyhedralInfo && "This option is not implemented");
  assert(!EnableDeLICM && "This option is not implemented");
  assert(!EnableSimplify && "This option is not implemented");
  if (ImportJScop)
    SPM.addPass(JSONImportPass());
  assert(!DeadCodeElim && "This option is not implemented");
  assert(!EnablePruneUnprofitable && "This option is not implemented");
  if (Target == TARGET_CPU || Target == TARGET_HYBRID)
    switch (Optimizer) {
    case OPTIMIZER_NONE:
      break; /* Do nothing */
    case OPTIMIZER_ISL:
      llvm_unreachable("ISL optimizer is not implemented");
      break;
    }

  assert(!ExportJScop && "This option is not implemented");

  if (Target == TARGET_CPU || Target == TARGET_HYBRID) {
    switch (CodeGeneration) {
    case CODEGEN_FULL:
      SPM.addPass(polly::CodeGenerationPass());
      break;
    case CODEGEN_AST:
    default: // Does it actually make sense to distinguish IslAst codegen?
      break;
    }
  }
#ifdef GPU_CODEGEN
  else
    llvm_unreachable("Hybrid Target with GPU support is not implemented");
#endif

  PM.addPass(CodePreparationPass());
  PM.addPass(createFunctionToScopPassAdaptor(std::move(SPM)));
  PM.addPass(PB.buildFunctionSimplificationPipeline(
      Level, PassBuilder::ThinLTOPhase::None)); // Cleanup

  assert(!DumpAfter && "This option is not implemented");
  assert(DumpAfterFile.empty() && "This option is not implemented");

  if (CFGPrinter)
    PM.addPass(llvm::CFGPrinterPass());
}

/// Register Polly to be available as an optimizer
///
///
/// We can currently run Polly at three different points int the pass manager.
/// a) very early, b) after the canonicalizing loop transformations and c) right
/// before the vectorizer.
///
/// The default is currently a), to register Polly such that it runs as early as
/// possible. This has several implications:
///
///   1) We need to schedule more canonicalization passes
///
///   As nothing is run before Polly, it is necessary to run a set of preparing
///   transformations before Polly to canonicalize the LLVM-IR and to allow
///   Polly to detect and understand the code.
///
///   2) LICM and LoopIdiom pass have not yet been run
///
///   Loop invariant code motion as well as the loop idiom recognition pass make
///   it more difficult for Polly to transform code. LICM may introduce
///   additional data dependences that are hard to eliminate and the loop idiom
///   recognition pass may introduce calls to memset that we currently do not
///   understand. By running Polly early enough (meaning before these passes) we
///   avoid difficulties that may be introduced by these passes.
///
///   3) We get the full -O3 optimization sequence after Polly
///
///   The LLVM-IR that is generated by Polly has been optimized on a high level,
///   but it may be rather inefficient on the lower/scalar level. By scheduling
///   Polly before all other passes, we have the full sequence of -O3
///   optimizations behind us, such that inefficiencies on the low level can
///   be optimized away.
///
/// We are currently evaluating the benefit or running Polly at position b) or
/// c). b) is likely too early as it interacts with the inliner. c) is nice
/// as everything is fully inlined and canonicalized, but we need to be able
/// to handle LICMed code to make it useful.
static llvm::RegisterStandardPasses RegisterPollyOptimizerEarly(
    llvm::PassManagerBuilder::EP_ModuleOptimizerEarly,
    registerPollyEarlyAsPossiblePasses);

static llvm::RegisterStandardPasses
    RegisterPollyOptimizerLoopEnd(llvm::PassManagerBuilder::EP_LoopOptimizerEnd,
                                  registerPollyLoopOptimizerEndPasses);

static llvm::RegisterStandardPasses RegisterPollyOptimizerScalarLate(
    llvm::PassManagerBuilder::EP_VectorizerStart,
    registerPollyScalarOptimizerLatePasses);

static OwningScopAnalysisManagerFunctionProxy
createScopAnalyses(FunctionAnalysisManager &FAM) {
  OwningScopAnalysisManagerFunctionProxy Proxy;
#define SCOP_ANALYSIS(NAME, CREATE_PASS)                                       \
  Proxy.getManager().registerPass([] { return CREATE_PASS; });

#include "PollyPasses.def"

  Proxy.getManager().registerPass(
      [&FAM] { return FunctionAnalysisManagerScopProxy(FAM); });
  return Proxy;
}

static void registerFunctionAnalyses(FunctionAnalysisManager &FAM) {
#define FUNCTION_ANALYSIS(NAME, CREATE_PASS)                                   \
  FAM.registerPass([] { return CREATE_PASS; });

#include "PollyPasses.def"

  FAM.registerPass([&FAM] { return createScopAnalyses(FAM); });
}

static bool
parseFunctionPipeline(StringRef Name, FunctionPassManager &FPM,
                      ArrayRef<PassBuilder::PipelineElement> Pipeline) {
  if (parseAnalysisUtilityPasses<OwningScopAnalysisManagerFunctionProxy>(
          "polly-scop-analyses", Name, FPM))
    return true;

#define FUNCTION_ANALYSIS(NAME, CREATE_PASS)                                   \
  if (parseAnalysisUtilityPasses<                                              \
          std::remove_reference<decltype(CREATE_PASS)>::type>(NAME, Name,      \
                                                              FPM))            \
    return true;

#define FUNCTION_PASS(NAME, CREATE_PASS)                                       \
  if (Name == NAME) {                                                          \
    FPM.addPass(CREATE_PASS);                                                  \
    return true;                                                               \
  }

#include "PollyPasses.def"
  return false;
}

static bool parseScopPass(StringRef Name, ScopPassManager &SPM) {
#define SCOP_ANALYSIS(NAME, CREATE_PASS)                                       \
  if (parseAnalysisUtilityPasses<                                              \
          std::remove_reference<decltype(CREATE_PASS)>::type>(NAME, Name,      \
                                                              SPM))            \
    return true;

#define SCOP_PASS(NAME, CREATE_PASS)                                           \
  if (Name == NAME) {                                                          \
    SPM.addPass(CREATE_PASS);                                                  \
    return true;                                                               \
  }

#include "PollyPasses.def"

  return false;
}

static bool parseScopPipeline(StringRef Name, FunctionPassManager &FPM,
                              ArrayRef<PassBuilder::PipelineElement> Pipeline) {
  if (Name != "scop")
    return false;
  if (!Pipeline.empty()) {
    ScopPassManager SPM;
    for (const auto &E : Pipeline)
      if (!parseScopPass(E.Name, SPM))
        return false;
    FPM.addPass(createFunctionToScopPassAdaptor(std::move(SPM)));
  }
  return true;
}

static bool isScopPassName(StringRef Name) {
#define SCOP_ANALYSIS(NAME, CREATE_PASS)                                       \
  if (Name == "require<" NAME ">")                                             \
    return true;                                                               \
  if (Name == "invalidate<" NAME ">")                                          \
    return true;

#define SCOP_PASS(NAME, CREATE_PASS)                                           \
  if (Name == NAME)                                                            \
    return true;

#include "PollyPasses.def"

  return false;
}

static bool
parseTopLevelPipeline(ModulePassManager &MPM,
                      ArrayRef<PassBuilder::PipelineElement> Pipeline,
                      bool VerifyEachPass, bool DebugLogging) {
  std::vector<PassBuilder::PipelineElement> FullPipeline;
  StringRef FirstName = Pipeline.front().Name;

  if (!isScopPassName(FirstName))
    return false;

  FunctionPassManager FPM(DebugLogging);
  ScopPassManager SPM(DebugLogging);

  for (auto &Element : Pipeline) {
    auto &Name = Element.Name;
    auto &InnerPipeline = Element.InnerPipeline;
    if (!InnerPipeline.empty()) // Scop passes don't have inner pipelines
      return false;
    if (!parseScopPass(Name, SPM))
      return false;
  }

  FPM.addPass(createFunctionToScopPassAdaptor(std::move(SPM)));
  if (VerifyEachPass)
    FPM.addPass(VerifierPass());
  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
  if (VerifyEachPass)
    MPM.addPass(VerifierPass());

  return true;
}

void RegisterPollyPasses(PassBuilder &PB) {
  PB.registerAnalysisRegistrationCallback(registerFunctionAnalyses);
  PB.registerPipelineParsingCallback(parseFunctionPipeline);
  PB.registerPipelineParsingCallback(parseScopPipeline);
  PB.registerParseTopLevelPipelineCallback(parseTopLevelPipeline);

  if (PassPosition == POSITION_BEFORE_VECTORIZER)
    PB.registerVectorizerStartEPCallback(buildDefaultPollyPipeline);
  // FIXME else Error?
}
} // namespace polly

// Plugin Entrypoint:
extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "Polly", LLVM_VERSION_STRING,
          polly::RegisterPollyPasses};
}
