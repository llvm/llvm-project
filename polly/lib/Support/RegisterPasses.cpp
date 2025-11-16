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
#include "polly/MaximalStaticExpansion.h"
#include "polly/Options.h"
#include "polly/Pass/PollyFunctionPass.h"
#include "polly/PruneUnprofitable.h"
#include "polly/ScheduleOptimizer.h"
#include "polly/ScopDetection.h"
#include "polly/ScopGraphPrinter.h"
#include "polly/ScopInfo.h"
#include "polly/ScopInliner.h"
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
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Transforms/IPO.h"

using namespace llvm;
using namespace polly;

namespace cl = llvm::cl;
using namespace polly;

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

static cl::opt<bool>
    PollyPrintDetect("polly-print-detect",
                     cl::desc("Polly - Print static control parts (SCoPs)"),
                     cl::cat(PollyCategory));

static cl::opt<bool>
    PollyPrintScops("polly-print-scops",
                    cl::desc("Print polyhedral description of all regions"),
                    cl::cat(PollyCategory));

static cl::opt<bool> PollyPrintDeps("polly-print-deps",
                                    cl::desc("Polly - Print dependences"),
                                    cl::cat(PollyCategory));

static bool shouldEnablePollyForOptimization() { return PollyEnabled; }

static bool shouldEnablePollyForDiagnostic() {
  // FIXME: PollyTrackFailures is user-controlled, should not be set
  // programmatically.
  if (PollyOnlyPrinter || PollyPrinter || PollyOnlyViewer || PollyViewer)
    PollyTrackFailures = true;

  return PollyOnlyPrinter || PollyPrinter || PollyOnlyViewer || PollyViewer ||
         ExportJScop;
}

/// Parser of parameters for LoopVectorize pass.
static llvm::Expected<PollyPassOptions> parsePollyOptions(StringRef Params,
                                                          bool IsCustom) {
  PassPhase PrevPhase = PassPhase::None;

  bool EnableDefaultOpts = !IsCustom;
  bool EnableEnd2End = !IsCustom;
  std::optional<bool>
      PassEnabled[static_cast<size_t>(PassPhase::PassPhaseLast) + 1];
  PassPhase StopAfter = PassPhase::None;

  // Passes enabled using command-line flags (can be overridden using
  // 'polly<no-pass>')
  if (PollyPrintDetect)
    PassEnabled[static_cast<size_t>(PassPhase::PrintDetect)] = true;
  if (PollyPrintScops)
    PassEnabled[static_cast<size_t>(PassPhase::PrintScopInfo)] = true;
  if (PollyPrintDeps)
    PassEnabled[static_cast<size_t>(PassPhase::PrintDependences)] = true;

  if (PollyViewer)
    PassEnabled[static_cast<size_t>(PassPhase::ViewScops)] = true;
  if (PollyOnlyViewer)
    PassEnabled[static_cast<size_t>(PassPhase::ViewScopsOnly)] = true;
  if (PollyPrinter)
    PassEnabled[static_cast<size_t>(PassPhase::DotScops)] = true;
  if (PollyOnlyPrinter)
    PassEnabled[static_cast<size_t>(PassPhase::DotScopsOnly)] = true;
  if (!EnableSimplify)
    PassEnabled[static_cast<size_t>(PassPhase::Simplify0)] = false;
  if (!EnableForwardOpTree)
    PassEnabled[static_cast<size_t>(PassPhase::Optree)] = false;
  if (!EnableDeLICM)
    PassEnabled[static_cast<size_t>(PassPhase::DeLICM)] = false;
  if (!EnableSimplify)
    PassEnabled[static_cast<size_t>(PassPhase::Simplify1)] = false;
  if (ImportJScop)
    PassEnabled[static_cast<size_t>(PassPhase::ImportJScop)] = true;
  if (DeadCodeElim)
    PassEnabled[static_cast<size_t>(PassPhase::DeadCodeElimination)] = true;
  if (FullyIndexedStaticExpansion)
    PassEnabled[static_cast<size_t>(PassPhase::MaximumStaticExtension)] = true;
  if (!EnablePruneUnprofitable)
    PassEnabled[static_cast<size_t>(PassPhase::PruneUnprofitable)] = false;
  switch (Optimizer) {
  case OPTIMIZER_NONE:
    // explicitly switched off
    PassEnabled[static_cast<size_t>(PassPhase::Optimization)] = false;
    break;
  case OPTIMIZER_ISL:
    // default: enabled
    break;
  }
  if (ExportJScop)
    PassEnabled[static_cast<size_t>(PassPhase::ExportJScop)] = true;
  switch (CodeGeneration) {
  case CODEGEN_AST:
    PassEnabled[static_cast<size_t>(PassPhase::AstGen)] = true;
    PassEnabled[static_cast<size_t>(PassPhase::CodeGen)] = false;
    break;
  case CODEGEN_FULL:
    // default: ast and codegen enabled
    break;
  case CODEGEN_NONE:
    PassEnabled[static_cast<size_t>(PassPhase::AstGen)] = false;
    PassEnabled[static_cast<size_t>(PassPhase::CodeGen)] = false;
    break;
  }

  while (!Params.empty()) {
    StringRef Param;
    std::tie(Param, Params) = Params.split(';');
    auto [ParamName, ParamVal] = Param.split('=');

    if (ParamName == "stopafter") {
      StopAfter = parsePhase(ParamVal);
      if (StopAfter == PassPhase::None)
        return make_error<StringError>(
            formatv("invalid stopafter parameter value '{0}'", ParamVal).str(),
            inconvertibleErrorCode());
      continue;
    }

    if (!ParamVal.empty())
      return make_error<StringError>(
          formatv("parameter '{0}' does not take value", ParamName).str(),
          inconvertibleErrorCode());

    bool Enabled = true;
    if (ParamName.starts_with("no-")) {
      Enabled = false;
      ParamName = ParamName.drop_front(3);
    }

    if (ParamName == "default-opts") {
      EnableDefaultOpts = Enabled;
      continue;
    }

    if (ParamName == "end2end") {
      EnableEnd2End = Enabled;
      continue;
    }

    PassPhase Phase;

    // Shortcut for both simplifys at the same time
    if (ParamName == "simplify") {
      PassEnabled[static_cast<size_t>(PassPhase::Simplify0)] = Enabled;
      PassEnabled[static_cast<size_t>(PassPhase::Simplify1)] = Enabled;
      Phase = PassPhase::Simplify0;
    } else {
      Phase = parsePhase(ParamName);
      if (Phase == PassPhase::None)
        return make_error<StringError>(
            formatv("invalid Polly parameter/phase name '{0}'", ParamName)
                .str(),
            inconvertibleErrorCode());

      if (PrevPhase >= Phase)
        return make_error<StringError>(
            formatv("phases must not be repeated and enumerated in-order: "
                    "'{0}' listed before '{1}'",
                    getPhaseName(PrevPhase), getPhaseName(Phase))
                .str(),
            inconvertibleErrorCode());

      PassEnabled[static_cast<size_t>(Phase)] = Enabled;
    }
    PrevPhase = Phase;
  }

  PollyPassOptions Opts;
  Opts.ViewAll = ViewAll;
  Opts.ViewFilter = ViewFilter;
  Opts.PrintDepsAnalysisLevel = OptAnalysisLevel;

  // Implicitly enable dependent phases first. May be overriden explicitly
  // on/off later.
  for (PassPhase P : llvm::enum_seq_inclusive(PassPhase::PassPhaseFirst,
                                              PassPhase::PassPhaseLast)) {
    bool Enabled = PassEnabled[static_cast<size_t>(P)].value_or(false);
    if (!Enabled)
      continue;

    if (static_cast<size_t>(PassPhase::Detection) < static_cast<size_t>(P))
      Opts.setPhaseEnabled(PassPhase::Detection);

    if (static_cast<size_t>(PassPhase::ScopInfo) < static_cast<size_t>(P))
      Opts.setPhaseEnabled(PassPhase::ScopInfo);

    if (dependsOnDependenceInfo(P))
      Opts.setPhaseEnabled(PassPhase::Dependences);

    if (static_cast<size_t>(PassPhase::AstGen) < static_cast<size_t>(P))
      Opts.setPhaseEnabled(PassPhase::AstGen);
  }

  if (EnableEnd2End)
    Opts.enableEnd2End();

  if (EnableDefaultOpts)
    Opts.enableDefaultOpts();

  for (PassPhase P : llvm::enum_seq_inclusive(PassPhase::PassPhaseFirst,
                                              PassPhase::PassPhaseLast)) {
    std::optional<bool> Enabled = PassEnabled[static_cast<size_t>(P)];

    // Apply only if set explicitly.
    if (Enabled.has_value())
      Opts.setPhaseEnabled(P, *Enabled);
  }

  if (StopAfter != PassPhase::None)
    Opts.disableAfter(StopAfter);

  if (Error CheckResult = Opts.checkConsistency())
    return CheckResult;

  return Opts;
}

static llvm::Expected<PollyPassOptions>
parsePollyDefaultOptions(StringRef Params) {
  return parsePollyOptions(Params, false);
}

static llvm::Expected<PollyPassOptions>
parsePollyCustomOptions(StringRef Params) {
  return parsePollyOptions(Params, true);
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

  ExitOnError Err("Inconsistent Polly configuration: ");
  PollyPassOptions &&Opts =
      Err(parsePollyOptions(StringRef(), /*IsCustom=*/false));
  PM.addPass(PollyFunctionPass(Opts));

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

static llvm::Expected<std::monostate> parseNoOptions(StringRef Params) {
  if (!Params.empty())
    return make_error<StringError>(
        formatv("'{0}' passed to pass that does not take any options", Params)
            .str(),
        inconvertibleErrorCode());

  return std::monostate{};
}

static llvm::Expected<bool>
parseCGPipeline(StringRef Name, llvm::CGSCCPassManager &CGPM,
                PassInstrumentationCallbacks *PIC,
                ArrayRef<PassBuilder::PipelineElement> Pipeline) {
#define CGSCC_PASS(NAME, CREATE_PASS, PARSER)                                  \
  if (PassBuilder::checkParametrizedPassName(Name, NAME)) {                    \
    auto Params = PassBuilder::parsePassParameters(PARSER, Name, NAME);        \
    if (!Params)                                                               \
      return Params.takeError();                                               \
    CGPM.addPass(CREATE_PASS);                                                 \
    return true;                                                               \
  }
#include "PollyPasses.def"

  return false;
}

static llvm::Expected<bool>
parseFunctionPipeline(StringRef Name, FunctionPassManager &FPM,
                      PassInstrumentationCallbacks *PIC,
                      ArrayRef<PassBuilder::PipelineElement> Pipeline) {

#define FUNCTION_PASS(NAME, CREATE_PASS, PARSER)                               \
  if (PassBuilder::checkParametrizedPassName(Name, NAME)) {                    \
    auto ExpectedOpts = PassBuilder::parsePassParameters(PARSER, Name, NAME);  \
    if (!ExpectedOpts)                                                         \
      return ExpectedOpts.takeError();                                         \
    auto &&Opts = *ExpectedOpts;                                               \
    (void)Opts;                                                                \
    FPM.addPass(CREATE_PASS);                                                  \
    return true;                                                               \
  }

#include "PollyPasses.def"
  return false;
}

static llvm::Expected<bool>
parseModulePipeline(StringRef Name, llvm::ModulePassManager &MPM,
                    PassInstrumentationCallbacks *PIC,
                    ArrayRef<PassBuilder::PipelineElement> Pipeline) {
#define MODULE_PASS(NAME, CREATE_PASS, PARSER)                                 \
  if (PassBuilder::checkParametrizedPassName(Name, NAME)) {                    \
    auto ExpectedOpts = PassBuilder::parsePassParameters(PARSER, Name, NAME);  \
    if (!ExpectedOpts)                                                         \
      return ExpectedOpts.takeError();                                         \
    auto &&Opts = *ExpectedOpts;                                               \
    (void)Opts;                                                                \
    MPM.addPass(CREATE_PASS);                                                  \
    return true;                                                               \
  }

#include "PollyPasses.def"

  return false;
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

#define MODULE_PASS(NAME, CREATE_PASS, PARSER)                                 \
  {                                                                            \
    std::remove_reference_t<decltype(*PARSER(StringRef()))> Opts;              \
    (void)Opts;                                                                \
    PIC->addClassToPassName(decltype(CREATE_PASS)::name(), NAME);              \
  }
#define CGSCC_PASS(NAME, CREATE_PASS, PARSER)                                  \
  {                                                                            \
    std::remove_reference_t<decltype(*PARSER(StringRef()))> Opts;              \
    (void)Opts;                                                                \
    PIC->addClassToPassName(decltype(CREATE_PASS)::name(), NAME);              \
  }
#define FUNCTION_PASS(NAME, CREATE_PASS, PARSER)                               \
  {                                                                            \
    std::remove_reference_t<decltype(*PARSER(StringRef()))> Opts;              \
    (void)Opts;                                                                \
    PIC->addClassToPassName(decltype(CREATE_PASS)::name(), NAME);              \
  }
#include "PollyPasses.def"

  PB.registerPipelineParsingCallback(
      [PIC](StringRef Name, FunctionPassManager &FPM,
            ArrayRef<PassBuilder::PipelineElement> Pipeline) -> bool {
        ExitOnError Err("Unable to parse Polly module pass: ");
        return Err(parseFunctionPipeline(Name, FPM, PIC, Pipeline));
      });
  PB.registerPipelineParsingCallback(
      [PIC](StringRef Name, CGSCCPassManager &CGPM,
            ArrayRef<PassBuilder::PipelineElement> Pipeline) -> bool {
        ExitOnError Err("Unable to parse Polly call graph pass: ");
        return Err(parseCGPipeline(Name, CGPM, PIC, Pipeline));
      });
  PB.registerPipelineParsingCallback(
      [PIC](StringRef Name, ModulePassManager &MPM,
            ArrayRef<PassBuilder::PipelineElement> Pipeline) -> bool {
        ExitOnError Err("Unable to parse Polly module pass: ");
        return Err(parseModulePipeline(Name, MPM, PIC, Pipeline));
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
