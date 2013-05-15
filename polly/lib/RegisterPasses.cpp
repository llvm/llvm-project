//===------ RegisterPasses.cpp - Add the Polly Passes to default passes  --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "polly/CodeGen/BlockGenerators.h"
#include "polly/CodeGen/Cloog.h"
#include "polly/CodeGen/CodeGeneration.h"
#include "polly/Dependences.h"
#include "polly/LinkAllPasses.h"
#include "polly/Options.h"
#include "polly/ScopDetection.h"
#include "polly/ScopInfo.h"
#include "polly/TempScopInfo.h"
#include "llvm/Analysis/CFGPrinter.h"
#include "llvm/PassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Vectorize.h"

using namespace llvm;
cl::OptionCategory PollyCategory("Polly Options",
                                 "Configure the polly loop optimizer");

static cl::opt<bool>
PollyEnabled("polly", cl::desc("Enable the polly optimizer (only at -O3)"),
             cl::init(false), cl::ZeroOrMore, cl::cat(PollyCategory));

enum OptimizerChoice {
  OPTIMIZER_NONE,
#ifdef PLUTO_FOUND
  OPTIMIZER_PLUTO,
#endif
#ifdef SCOPLIB_FOUND
  OPTIMIZER_POCC,
#endif
  OPTIMIZER_ISL
};

static cl::opt<OptimizerChoice> Optimizer(
    "polly-optimizer", cl::desc("Select the scheduling optimizer"),
    cl::values(
        clEnumValN(OPTIMIZER_NONE, "none", "No optimizer"),
#ifdef PLUTO_FOUND
        clEnumValN(OPTIMIZER_PLUTO, "pluto", "The Pluto scheduling optimizer"),
#endif
#ifdef SCOPLIB_FOUND
        clEnumValN(OPTIMIZER_POCC, "pocc", "The PoCC scheduling optimizer"),
#endif
        clEnumValN(OPTIMIZER_ISL, "isl", "The isl scheduling optimizer"),
        clEnumValEnd),
    cl::Hidden, cl::init(OPTIMIZER_ISL), cl::ZeroOrMore,
    cl::cat(PollyCategory));

enum CodeGenChoice {
#ifdef CLOOG_FOUND
  CODEGEN_CLOOG,
#endif
  CODEGEN_ISL,
  CODEGEN_NONE
};

#ifdef CLOOG_FOUND
enum CodeGenChoice DefaultCodeGen = CODEGEN_CLOOG;
#else
enum CodeGenChoice DefaultCodeGen = CODEGEN_ISL;
#endif

static cl::opt<CodeGenChoice> CodeGenerator(
    "polly-code-generator", cl::desc("Select the code generator"),
    cl::values(
#ifdef CLOOG_FOUND
        clEnumValN(CODEGEN_CLOOG, "cloog", "CLooG"),
#endif
        clEnumValN(CODEGEN_ISL, "isl", "isl code generator"),
        clEnumValN(CODEGEN_NONE, "none", "no code generation"), clEnumValEnd),
    cl::Hidden, cl::init(DefaultCodeGen), cl::ZeroOrMore,
    cl::cat(PollyCategory));

VectorizerChoice polly::PollyVectorizerChoice;
static cl::opt<polly::VectorizerChoice, true> Vectorizer(
    "polly-vectorizer", cl::desc("Select the vectorization strategy"),
    cl::values(clEnumValN(polly::VECTORIZER_NONE, "none", "No Vectorization"),
               clEnumValN(polly::VECTORIZER_POLLY, "polly",
                          "Polly internal vectorizer"),
               clEnumValN(polly::VECTORIZER_UNROLL_ONLY, "unroll-only",
                          "Only grouped unroll the vectorize candidate loops"),
               clEnumValN(polly::VECTORIZER_BB, "bb",
                          "The Basic Block vectorizer driven by Polly"),
               clEnumValEnd),
    cl::location(PollyVectorizerChoice), cl::init(polly::VECTORIZER_NONE),
    cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<bool>
ImportJScop("polly-import",
            cl::desc("Export the polyhedral description of the detected Scops"),
            cl::Hidden, cl::init(false), cl::ZeroOrMore,
            cl::cat(PollyCategory));

static cl::opt<bool>
ExportJScop("polly-export",
            cl::desc("Export the polyhedral description of the detected Scops"),
            cl::Hidden, cl::init(false), cl::ZeroOrMore,
            cl::cat(PollyCategory));

static cl::opt<bool> DeadCodeElim("polly-run-dce",
                                  cl::desc("Run the dead code elimination"),
                                  cl::Hidden, cl::init(false), cl::ZeroOrMore,
                                  cl::cat(PollyCategory));

static cl::opt<bool>
PollyViewer("polly-show", cl::desc("Enable the Polly DOT viewer in -O3"),
            cl::Hidden, cl::value_desc("Run the Polly DOT viewer at -O3"),
            cl::init(false), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<bool> PollyOnlyViewer(
    "polly-show-only",
    cl::desc("Enable the Polly DOT viewer in -O3 (no BB content)"), cl::Hidden,
    cl::value_desc("Run the Polly DOT viewer at -O3 (no BB content"),
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

namespace {
static void initializePollyPasses(PassRegistry &Registry) {
#ifdef CLOOG_FOUND
  initializeCloogInfoPass(Registry);
  initializeCodeGenerationPass(Registry);
#endif
  initializeIslCodeGenerationPass(Registry);
  initializeCodePreparationPass(Registry);
  initializeDeadCodeElimPass(Registry);
  initializeDependencesPass(Registry);
  initializeIndependentBlocksPass(Registry);
  initializeJSONExporterPass(Registry);
  initializeJSONImporterPass(Registry);
  initializeIslAstInfoPass(Registry);
  initializeIslScheduleOptimizerPass(Registry);
#ifdef SCOPLIB_FOUND
  initializePoccPass(Registry);
#endif
  initializePollyIndVarSimplifyPass(Registry);
  initializeScopDetectionPass(Registry);
  initializeScopInfoPass(Registry);
  initializeTempScopInfoPass(Registry);
}

/// @brief Initialize Polly passes when library is loaded.
///
/// We use the constructor of a statically declared object to initialize the
/// different Polly passes right after the Polly library is loaded. This ensures
/// that the Polly passes are available e.g. in the 'opt' tool.
class StaticInitializer {
public:
  StaticInitializer() {
    PassRegistry &Registry = *PassRegistry::getPassRegistry();
    initializePollyPasses(Registry);
  }
};
static StaticInitializer InitializeEverything;

/// @brief Schedule a set of canonicalization passes to prepare for Polly
///
/// The set of optimization passes was partially taken/copied from the
/// set of default optimization passes in LLVM. It is used to bring the code
/// into a canonical form that simplifies the analysis and optimization passes
/// of Polly. The set of optimization passes scheduled here is probably not yet
/// optimal. TODO: Optimize the set of canonicalization passes.
static void registerCanonicalicationPasses(llvm::PassManagerBase &PM) {
  PM.add(llvm::createPromoteMemoryToRegisterPass());
  PM.add(llvm::createInstructionCombiningPass());
  PM.add(llvm::createCFGSimplificationPass());
  PM.add(llvm::createTailCallEliminationPass());
  PM.add(llvm::createCFGSimplificationPass());
  PM.add(llvm::createReassociatePass());
  PM.add(llvm::createLoopRotatePass());
  PM.add(llvm::createInstructionCombiningPass());

  if (!SCEVCodegen)
    PM.add(polly::createIndVarSimplifyPass());

  PM.add(polly::createCodePreparationPass());
}

/// @brief Register Polly passes such that they form a polyhedral optimizer.
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
/// As scheduling optimizer we support PoCC (http://pocc.sourceforge.net), PLUTO
/// (http://pluto-compiler.sourceforge.net) as well as the isl scheduling
/// optimizer (http://freecode.com/projects/isl). The isl optimizer is the
/// default optimizer.
/// It is also possible to run Polly with no optimizer. This mode is mainly
/// provided to analyze the run and compile time changes caused by the
/// scheduling optimizer.
///
/// Polly supports both CLooG (http://www.cloog.org) as well as the isl internal
/// code generator. For the moment, the CLooG code generator is enabled by
/// default.
static void registerPollyPasses(llvm::PassManagerBase &PM) {
  registerCanonicalicationPasses(PM);

  PM.add(polly::createScopInfoPass());

  if (PollyViewer)
    PM.add(polly::createDOTViewerPass());
  if (PollyOnlyViewer)
    PM.add(polly::createDOTOnlyViewerPass());
  if (PollyPrinter)
    PM.add(polly::createDOTPrinterPass());
  if (PollyOnlyPrinter)
    PM.add(polly::createDOTOnlyPrinterPass());

  if (ImportJScop)
    PM.add(polly::createJSONImporterPass());

  if (DeadCodeElim)
    PM.add(polly::createDeadCodeElimPass());

  switch (Optimizer) {
  case OPTIMIZER_NONE:
    break; /* Do nothing */

#ifdef SCOPLIB_FOUND
  case OPTIMIZER_POCC:
    PM.add(polly::createPoccPass());
    break;
#endif

#ifdef PLUTO_FOUND
  case OPTIMIZER_PLUTO:
    PM.add(polly::createPlutoOptimizerPass());
    break;
#endif

  case OPTIMIZER_ISL:
    PM.add(polly::createIslScheduleOptimizerPass());
    break;
  }

  if (ExportJScop)
    PM.add(polly::createJSONExporterPass());

  switch (CodeGenerator) {
#ifdef CLOOG_FOUND
  case CODEGEN_CLOOG:
    PM.add(polly::createCodeGenerationPass());
    if (PollyVectorizerChoice == VECTORIZER_BB) {
      VectorizeConfig C;
      C.FastDep = true;
      PM.add(createBBVectorizePass(C));
    }
    break;
#endif
  case CODEGEN_ISL:
    PM.add(polly::createIslCodeGenerationPass());
    break;
  case CODEGEN_NONE:
    break;
  }

  if (CFGPrinter)
    PM.add(llvm::createCFGPrinterPass());
}

static void
registerPollyEarlyAsPossiblePasses(const llvm::PassManagerBuilder &Builder,
                                   llvm::PassManagerBase &PM) {

  if (Builder.OptLevel == 0)
    return;

  if (PollyOnlyPrinter || PollyPrinter || PollyOnlyViewer || PollyViewer ||
      ExportJScop || ImportJScop)
    PollyEnabled = true;

  if (!PollyEnabled)
    return;

  // We only run Polly at optimization level '-O3'.
  //
  // This is to ensure that scalar overhead that may be introduced by Polly is
  // properly cleaned up by LLVM later on. We may reinvestigate this decision
  // later on.
  if (Builder.OptLevel != 3) {
    errs() << "Polly should only be run with -O3. Disabling Polly.\n";
    return;
  }

  registerPollyPasses(PM);
}

static void registerPollyOptLevel0Passes(const llvm::PassManagerBuilder &,
                                         llvm::PassManagerBase &PM) {
  registerCanonicalicationPasses(PM);
}

/// @brief Register Polly canonicalization passes at opt level '0'
///
/// At '-O0' we schedule the Polly canonicalization passes. This allows us
/// to easily get the canonicalized IR of a program which can then be used
/// with the Polly passes of the 'opt' optimizer.
static llvm::RegisterStandardPasses
RegisterPollyCanonicalizer(llvm::PassManagerBuilder::EP_EnabledOnOptLevel0,
                           registerPollyOptLevel0Passes);

/// @brief Register Polly to be available as an optimizer
///
/// We currently register Polly such that it runs as early as possible. This has
/// several implications:
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
static llvm::RegisterStandardPasses
RegisterPollyOptimizer(llvm::PassManagerBuilder::EP_EarlyAsPossible,
                       registerPollyEarlyAsPossiblePasses);
} // end of anonymous namespace.
