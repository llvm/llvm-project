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
#include "polly/CodeGen/IslAst.h"
#include "polly/CodePreparation.h"
#include "polly/DeLICM.h"
#include "polly/DeadCodeElimination.h"
#include "polly/DependenceInfo.h"
#include "polly/ForwardOpTree.h"
#include "polly/JSONExporter.h"
#include "polly/LinkAllPasses.h"
#include "polly/MaximalStaticExpansion.h"
#include "polly/PruneUnprofitable.h"
#include "polly/ScheduleOptimizer.h"
#include "polly/ScopDetection.h"
#include "polly/ScopGraphPrinter.h"
#include "polly/ScopInfo.h"
#include "polly/Simplify.h"
#include "polly/Support/DumpFunctionPass.h"
#include "polly/Support/DumpModulePass.h"
#include "llvm/Analysis/CFGPrinter.h"
#include "llvm/Config/llvm-config.h" // for LLVM_VERSION_STRING
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Transforms/IPO.h"

namespace cl = llvm::cl;

using llvm::FunctionPassManager;
using llvm::OptimizationLevel;
using llvm::PassBuilder;
using llvm::PassInstrumentationCallbacks;

cl::OptionCategory PollyCategory("Polly Options",
                                 "Configure the polly loop optimizer");

namespace polly {
static cl::opt<bool>
    PollyEnabled("polly",
                 cl::desc("Enable the polly optimizer (with -O1, -O2 or -O3)"),
                 cl::cat(PollyCategory));

static cl::opt<bool> PollyDetectOnly(
    "polly-only-scop-detection",
    cl::desc("Only run scop detection, but no other optimizations"),
    cl::cat(PollyCategory));

enum PassPositionChoice { POSITION_EARLY, POSITION_BEFORE_VECTORIZER };

enum OptimizerChoice { OPTIMIZER_NONE, OPTIMIZER_ISL };

static cl::opt<PassPositionChoice> PassPosition(
    "polly-position", cl::desc("Where to run polly in the pass pipeline"),
    cl::values(clEnumValN(POSITION_EARLY, "early", "Before everything"),
               clEnumValN(POSITION_BEFORE_VECTORIZER, "before-vectorizer",
                          "Right before the vectorizer")),
    cl::Hidden, cl::init(POSITION_BEFORE_VECTORIZER), cl::cat(PollyCategory));

static cl::opt<OptimizerChoice>
    Optimizer("polly-optimizer", cl::desc("Select the scheduling optimizer"),
              cl::values(clEnumValN(OPTIMIZER_NONE, "none", "No optimizer"),
                         clEnumValN(OPTIMIZER_ISL, "isl",
                                    "The isl scheduling optimizer")),
              cl::Hidden, cl::init(OPTIMIZER_ISL), cl::cat(PollyCategory));

enum CodeGenChoice { CODEGEN_FULL, CODEGEN_AST, CODEGEN_NONE };
static cl::opt<CodeGenChoice> CodeGeneration(
    "polly-code-generation", cl::desc("How much code-generation to perform"),
    cl::values(clEnumValN(CODEGEN_FULL, "full", "AST and IR generation"),
               clEnumValN(CODEGEN_AST, "ast", "Only AST generation"),
               clEnumValN(CODEGEN_NONE, "none", "No code generation")),
    cl::Hidden, cl::init(CODEGEN_FULL), cl::cat(PollyCategory));

VectorizerChoice PollyVectorizerChoice;

static cl::opt<VectorizerChoice, true> Vectorizer(
    "polly-vectorizer", cl::desc("Select the vectorization strategy"),
    cl::values(
        clEnumValN(VECTORIZER_NONE, "none", "No Vectorization"),
        clEnumValN(
            VECTORIZER_STRIPMINE, "stripmine",
            "Strip-mine outer loops for the loop-vectorizer to trigger")),
    cl::location(PollyVectorizerChoice), cl::init(VECTORIZER_NONE),
    cl::cat(PollyCategory));

static cl::opt<bool> ImportJScop(
    "polly-import",
    cl::desc("Import the polyhedral description of the detected Scops"),
    cl::Hidden, cl::cat(PollyCategory));

static cl::opt<bool> FullyIndexedStaticExpansion(
    "polly-enable-mse",
    cl::desc("Fully expand the memory accesses of the detected Scops"),
    cl::Hidden, cl::cat(PollyCategory));

static cl::opt<bool> ExportJScop(
    "polly-export",
    cl::desc("Export the polyhedral description of the detected Scops"),
    cl::Hidden, cl::cat(PollyCategory));

static cl::opt<bool> DeadCodeElim("polly-run-dce",
                                  cl::desc("Run the dead code elimination"),
                                  cl::Hidden, cl::cat(PollyCategory));

static cl::opt<bool> PollyViewer(
    "polly-show",
    cl::desc("Highlight the code regions that will be optimized in a "
             "(CFG BBs and LLVM-IR instructions)"),
    cl::cat(PollyCategory));

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
    cl::cat(PollyCategory));

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

namespace {

/// Initialize Polly passes when library is loaded.
///
/// We use the constructor of a statically declared object to initialize the
/// different Polly passes right after the Polly library is loaded. This ensures
/// that the Polly passes are available e.g. in the 'opt' tool.
struct StaticInitializer {
  StaticInitializer() {
    llvm::PassRegistry &Registry = *llvm::PassRegistry::getPassRegistry();
    polly::initializePollyPasses(Registry);
  }
};
static StaticInitializer InitializeEverything;
} // end of anonymous namespace.

void initializePollyPasses(llvm::PassRegistry &Registry) {
  initializeCodeGenerationPass(Registry);

  initializeCodePreparationPass(Registry);
  initializeDeadCodeElimWrapperPassPass(Registry);
  initializeDependenceInfoPass(Registry);
  initializeDependenceInfoPrinterLegacyPassPass(Registry);
  initializeDependenceInfoWrapperPassPass(Registry);
  initializeDependenceInfoPrinterLegacyFunctionPassPass(Registry);
  initializeJSONExporterPass(Registry);
  initializeJSONImporterPass(Registry);
  initializeJSONImporterPrinterLegacyPassPass(Registry);
  initializeMaximalStaticExpanderWrapperPassPass(Registry);
  initializeIslAstInfoWrapperPassPass(Registry);
  initializeIslAstInfoPrinterLegacyPassPass(Registry);
  initializeIslScheduleOptimizerWrapperPassPass(Registry);
  initializeIslScheduleOptimizerPrinterLegacyPassPass(Registry);
  initializePollyCanonicalizePass(Registry);
  initializeScopDetectionWrapperPassPass(Registry);
  initializeScopDetectionPrinterLegacyPassPass(Registry);
  initializeScopInlinerPass(Registry);
  initializeScopInfoRegionPassPass(Registry);
  initializeScopInfoPrinterLegacyRegionPassPass(Registry);
  initializeScopInfoWrapperPassPass(Registry);
  initializeScopInfoPrinterLegacyFunctionPassPass(Registry);
  initializeFlattenSchedulePass(Registry);
  initializeFlattenSchedulePrinterLegacyPassPass(Registry);
  initializeForwardOpTreeWrapperPassPass(Registry);
  initializeForwardOpTreePrinterLegacyPassPass(Registry);
  initializeDeLICMWrapperPassPass(Registry);
  initializeDeLICMPrinterLegacyPassPass(Registry);
  initializeSimplifyWrapperPassPass(Registry);
  initializeSimplifyPrinterLegacyPassPass(Registry);
  initializeDumpModuleWrapperPassPass(Registry);
  initializePruneUnprofitableWrapperPassPass(Registry);
}

static bool shouldEnablePollyForOptimization() { return PollyEnabled; }

static bool shouldEnablePollyForDiagnostic() {
  // FIXME: PollyTrackFailures is user-controlled, should not be set
  // programmatically.
  if (PollyOnlyPrinter || PollyPrinter || PollyOnlyViewer || PollyViewer)
    PollyTrackFailures = true;

  return PollyOnlyPrinter || PollyPrinter || PollyOnlyViewer || PollyViewer ||
         ExportJScop;
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

/// Add the pass sequence required for Polly to the New Pass Manager.
///
/// @param PM           The pass manager itself.
/// @param Level        The optimization level. Used for the cleanup of Polly's
///                     output.
/// @param EnableForOpt Whether to add Polly IR transformations. If False, only
///                     the analysis passes are added, skipping Polly itself.
///                     The IR may still be modified.
static void buildCommonPollyPipeline(FunctionPassManager &PM,
                                     OptimizationLevel Level,
                                     bool EnableForOpt) {
  PassBuilder PB;
  ScopPassManager SPM;

  PM.addPass(CodePreparationPass());

  // TODO add utility passes for the various command line options, once they're
  // ported

  if (PollyDetectOnly) {
    // Don't add more passes other than the ScopPassManager's detection passes.
    PM.addPass(createFunctionToScopPassAdaptor(std::move(SPM)));
    return;
  }

  if (PollyViewer)
    PM.addPass(ScopViewer());
  if (PollyOnlyViewer)
    PM.addPass(ScopOnlyViewer());
  if (PollyPrinter)
    PM.addPass(ScopPrinter());
  if (PollyOnlyPrinter)
    PM.addPass(ScopOnlyPrinter());
  if (EnableSimplify)
    SPM.addPass(SimplifyPass(0));
  if (EnableForwardOpTree)
    SPM.addPass(ForwardOpTreePass());
  if (EnableDeLICM)
    SPM.addPass(DeLICMPass());
  if (EnableSimplify)
    SPM.addPass(SimplifyPass(1));

  if (ImportJScop)
    SPM.addPass(JSONImportPass());

  if (DeadCodeElim)
    SPM.addPass(DeadCodeElimPass());

  if (FullyIndexedStaticExpansion)
    SPM.addPass(MaximalStaticExpansionPass());

  if (EnablePruneUnprofitable)
    SPM.addPass(PruneUnprofitablePass());

  switch (Optimizer) {
  case OPTIMIZER_NONE:
    break; /* Do nothing */
  case OPTIMIZER_ISL:
    SPM.addPass(IslScheduleOptimizerPass());
    break;
  }

  if (ExportJScop)
    SPM.addPass(JSONExportPass());

  if (!EnableForOpt)
    return;

  switch (CodeGeneration) {
  case CODEGEN_AST:
    SPM.addPass(
        llvm::RequireAnalysisPass<IslAstAnalysis, Scop, ScopAnalysisManager,
                                  ScopStandardAnalysisResults &,
                                  SPMUpdater &>());
    break;
  case CODEGEN_FULL:
    SPM.addPass(CodeGenerationPass());
    break;
  case CODEGEN_NONE:
    break;
  }

  PM.addPass(createFunctionToScopPassAdaptor(std::move(SPM)));
  PM.addPass(PB.buildFunctionSimplificationPipeline(
      Level, llvm::ThinOrFullLTOPhase::None)); // Cleanup

  if (CFGPrinter)
    PM.addPass(llvm::CFGPrinterPass());
}

static void buildEarlyPollyPipeline(llvm::ModulePassManager &MPM,
                                    llvm::OptimizationLevel Level) {
  bool EnableForOpt =
      shouldEnablePollyForOptimization() && Level.isOptimizingForSpeed();
  if (!shouldEnablePollyForDiagnostic() && !EnableForOpt)
    return;

  FunctionPassManager FPM = buildCanonicalicationPassesForNPM(MPM, Level);

  if (DumpBefore || !DumpBeforeFile.empty()) {
    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));

    if (DumpBefore)
      MPM.addPass(DumpModulePass("-before", true));
    for (auto &Filename : DumpBeforeFile)
      MPM.addPass(DumpModulePass(Filename, false));

    FPM = FunctionPassManager();
  }

  buildCommonPollyPipeline(FPM, Level, EnableForOpt);
  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));

  if (DumpAfter)
    MPM.addPass(DumpModulePass("-after", true));
  for (auto &Filename : DumpAfterFile)
    MPM.addPass(DumpModulePass(Filename, false));
}

static void buildLatePollyPipeline(FunctionPassManager &PM,
                                   llvm::OptimizationLevel Level) {
  bool EnableForOpt =
      shouldEnablePollyForOptimization() && Level.isOptimizingForSpeed();
  if (!shouldEnablePollyForDiagnostic() && !EnableForOpt)
    return;

  if (DumpBefore)
    PM.addPass(DumpFunctionPass("-before"));
  if (!DumpBeforeFile.empty())
    llvm::report_fatal_error(
        "Option -polly-dump-before-file at -polly-position=late "
        "not supported with NPM",
        false);

  buildCommonPollyPipeline(PM, Level, EnableForOpt);

  if (DumpAfter)
    PM.addPass(DumpFunctionPass("-after"));
  if (!DumpAfterFile.empty())
    llvm::report_fatal_error(
        "Option -polly-dump-after-file at -polly-position=late "
        "not supported with NPM",
        false);
}

static OwningScopAnalysisManagerFunctionProxy
createScopAnalyses(FunctionAnalysisManager &FAM,
                   PassInstrumentationCallbacks *PIC) {
  OwningScopAnalysisManagerFunctionProxy Proxy;
#define SCOP_ANALYSIS(NAME, CREATE_PASS)                                       \
  Proxy.getManager().registerPass([PIC] {                                      \
    (void)PIC;                                                                 \
    return CREATE_PASS;                                                        \
  });
#include "PollyPasses.def"

  Proxy.getManager().registerPass(
      [&FAM] { return FunctionAnalysisManagerScopProxy(FAM); });
  return Proxy;
}

static void registerFunctionAnalyses(FunctionAnalysisManager &FAM,
                                     PassInstrumentationCallbacks *PIC) {

#define FUNCTION_ANALYSIS(NAME, CREATE_PASS)                                   \
  FAM.registerPass([] { return CREATE_PASS; });

#include "PollyPasses.def"

  FAM.registerPass([&FAM, PIC] { return createScopAnalyses(FAM, PIC); });
}

static bool
parseFunctionPipeline(StringRef Name, FunctionPassManager &FPM,
                      ArrayRef<PassBuilder::PipelineElement> Pipeline) {
  if (llvm::parseAnalysisUtilityPasses<OwningScopAnalysisManagerFunctionProxy>(
          "polly-scop-analyses", Name, FPM))
    return true;

#define FUNCTION_ANALYSIS(NAME, CREATE_PASS)                                   \
  if (llvm::parseAnalysisUtilityPasses<                                        \
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

static bool parseScopPass(StringRef Name, ScopPassManager &SPM,
                          PassInstrumentationCallbacks *PIC) {
#define SCOP_ANALYSIS(NAME, CREATE_PASS)                                       \
  if (llvm::parseAnalysisUtilityPasses<                                        \
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
                              PassInstrumentationCallbacks *PIC,
                              ArrayRef<PassBuilder::PipelineElement> Pipeline) {
  if (Name != "scop")
    return false;
  if (!Pipeline.empty()) {
    ScopPassManager SPM;
    for (const auto &E : Pipeline)
      if (!parseScopPass(E.Name, SPM, PIC))
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
parseTopLevelPipeline(llvm::ModulePassManager &MPM,
                      PassInstrumentationCallbacks *PIC,
                      ArrayRef<PassBuilder::PipelineElement> Pipeline) {
  StringRef FirstName = Pipeline.front().Name;

  if (!isScopPassName(FirstName))
    return false;

  FunctionPassManager FPM;
  ScopPassManager SPM;

  for (auto &Element : Pipeline) {
    auto &Name = Element.Name;
    auto &InnerPipeline = Element.InnerPipeline;
    if (!InnerPipeline.empty()) // Scop passes don't have inner pipelines
      return false;
    if (!parseScopPass(Name, SPM, PIC))
      return false;
  }

  FPM.addPass(createFunctionToScopPassAdaptor(std::move(SPM)));
  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));

  return true;
}

/// Register Polly to be available as an optimizer
///
///
/// We can currently run Polly at two different points int the pass manager.
/// a) very early, b) right before the vectorizer.
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
///   2) We get the full -O3 optimization sequence after Polly
///
///   The LLVM-IR that is generated by Polly has been optimized on a high level,
///   but it may be rather inefficient on the lower/scalar level. By scheduling
///   Polly before all other passes, we have the full sequence of -O3
///   optimizations behind us, such that inefficiencies on the low level can
///   be optimized away.
///
/// We are currently evaluating the benefit or running Polly at b). b) is nice
/// as everything is fully inlined and canonicalized, but we need to be able to
/// handle LICMed code to make it useful.
void registerPollyPasses(PassBuilder &PB) {
  PassInstrumentationCallbacks *PIC = PB.getPassInstrumentationCallbacks();
  PB.registerAnalysisRegistrationCallback([PIC](FunctionAnalysisManager &FAM) {
    registerFunctionAnalyses(FAM, PIC);
  });
  PB.registerPipelineParsingCallback(parseFunctionPipeline);
  PB.registerPipelineParsingCallback(
      [PIC](StringRef Name, FunctionPassManager &FPM,
            ArrayRef<PassBuilder::PipelineElement> Pipeline) -> bool {
        return parseScopPipeline(Name, FPM, PIC, Pipeline);
      });
  PB.registerParseTopLevelPipelineCallback(
      [PIC](llvm::ModulePassManager &MPM,
            ArrayRef<PassBuilder::PipelineElement> Pipeline) -> bool {
        return parseTopLevelPipeline(MPM, PIC, Pipeline);
      });

  switch (PassPosition) {
  case POSITION_EARLY:
    PB.registerPipelineStartEPCallback(buildEarlyPollyPipeline);
    break;
  case POSITION_BEFORE_VECTORIZER:
    PB.registerVectorizerStartEPCallback(buildLatePollyPipeline);
    break;
  }
}
} // namespace polly

llvm::PassPluginLibraryInfo getPollyPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "Polly", LLVM_VERSION_STRING,
          polly::registerPollyPasses};
}
