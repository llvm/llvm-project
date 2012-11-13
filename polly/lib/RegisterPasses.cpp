//===------ RegisterPasses.cpp - Add the Polly Passes to default passes  --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Add the Polly passes to the optimization passes executed at -O3.
//
//===----------------------------------------------------------------------===//
#include "polly/RegisterPasses.h"
#include "polly/LinkAllPasses.h"

#include "polly/CodeGen/Cloog.h"
#include "polly/Dependences.h"
#include "polly/ScopDetection.h"
#include "polly/ScopInfo.h"
#include "polly/TempScopInfo.h"
#include "polly/CodeGen/CodeGeneration.h"

#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/CFGPrinter.h"
#include "llvm/InitializePasses.h"
#include "llvm/PassManager.h"
#include "llvm/PassRegistry.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Vectorize.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Support/CommandLine.h"

#include <string>

using namespace llvm;

static cl::opt<bool>
PollyEnabled("polly", cl::desc("Enable the default passes of Polly in -O3"),
             cl::init(false), cl::ZeroOrMore);

enum OptimizerChoice {
  OPTIMIZER_NONE,
#ifdef SCOPLIB_FOUND
  OPTIMIZER_POCC,
#endif
#ifdef PLUTO_FOUND
  OPTIMIZER_PLUTO,
#endif
  OPTIMIZER_ISL
};

static cl::opt<OptimizerChoice>
Optimizer("polly-optimizer",
          cl::desc("Select the scheduling optimizer"),
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
          cl::Hidden, cl::init(OPTIMIZER_ISL), cl::ZeroOrMore);

enum CodeGenChoice {
#ifdef CLOOG_FOUND
  CODEGEN_CLOOG,
#endif
  CODEGEN_ISL
};

#ifdef CLOOG_FOUND
enum CodeGenChoice DefaultCodeGen = CODEGEN_CLOOG;
#else
enum CodegenChoice DefaultCodeGen = CODEGEN_ISL;
#endif


static cl::opt<CodeGenChoice>
CodeGenerator("polly-code-generator",
              cl::desc("Select the code generator"),
              cl::values(
#ifdef CLOOG_FOUND
                         clEnumValN(CODEGEN_CLOOG, "cloog", "CLooG"),
#endif
                         clEnumValN(CODEGEN_ISL, "isl", "isl code generator"),
                         clEnumValEnd),
          cl::Hidden, cl::init(DefaultCodeGen), cl::ZeroOrMore);

static cl::opt<polly::VectorizerChoice, true>
Vectorizer("polly-vectorizer",
          cl::desc("Select the scheduling optimizer"),
          cl::values(
            clEnumValN(polly::VECTORIZER_NONE, "none", "No Vectorization"),
            clEnumValN(polly::VECTORIZER_POLLY, "polly",
                       "Polly internal vectorizer"),
            clEnumValN(polly::VECTORIZER_UNROLL_ONLY, "unroll-only",
                       "Only grouped unroll the vectorize candidate loops"),
            clEnumValN(polly::VECTORIZER_BB, "bb",
                       "The Basic Block vectorizer driven by Polly"),
            clEnumValEnd),
          cl::Hidden, cl::location(PollyVectorizerChoice),
          cl::init(polly::VECTORIZER_NONE), cl::ZeroOrMore);

static cl::opt<bool>
ImportJScop("polly-import",
            cl::desc("Export the polyhedral description of the detected Scops"),
            cl::Hidden, cl::init(false), cl::ZeroOrMore);
static cl::opt<bool>
ExportJScop("polly-export",
            cl::desc("Export the polyhedral description of the detected Scops"),
            cl::Hidden, cl::init(false), cl::ZeroOrMore);

static cl::opt<bool>
PollyViewer("polly-show",
       cl::desc("Enable the Polly DOT viewer in -O3"), cl::Hidden,
       cl::value_desc("Run the Polly DOT viewer at -O3"),
       cl::init(false), cl::ZeroOrMore);

static cl::opt<bool>
DeadCodeElim("polly-run-dce",
             cl::desc("Run the dead code elimination"),
             cl::Hidden, cl::init(false), cl::ZeroOrMore);

static cl::opt<bool>
PollyOnlyViewer("polly-show-only",
       cl::desc("Enable the Polly DOT viewer in -O3 (no BB content)"),
       cl::Hidden,
       cl::value_desc("Run the Polly DOT viewer at -O3 (no BB content"),
       cl::init(false));
static cl::opt<bool>
PollyPrinter("polly-dot",
       cl::desc("Enable the Polly DOT printer in -O3"), cl::Hidden,
       cl::value_desc("Run the Polly DOT printer at -O3"),
       cl::init(false));
static cl::opt<bool>
PollyOnlyPrinter("polly-dot-only",
       cl::desc("Enable the Polly DOT printer in -O3 (no BB content)"),
       cl::Hidden,
       cl::value_desc("Run the Polly DOT printer at -O3 (no BB content"),
       cl::init(false));

static cl::opt<bool>
CFGPrinter("polly-view-cfg",
       cl::desc("Show the Polly CFG right after code generation"),
       cl::Hidden,
       cl::init(false));

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
  initializeRegionSimplifyPass(Registry);
  initializeScopDetectionPass(Registry);
  initializeScopInfoPass(Registry);
  initializeTempScopInfoPass(Registry);
}

namespace {
// Statically register all Polly passes such that they are available after
// loading Polly.
class StaticInitializer {

public:
    StaticInitializer() {
      PassRegistry &Registry = *PassRegistry::getPassRegistry();
      initializePollyPasses(Registry);
    }
};
} // end of anonymous namespace.

static StaticInitializer InitializeEverything;

static void registerPollyPreoptPasses(llvm::PassManagerBase &PM) {
  // A standard set of optimization passes partially taken/copied from the
  // set of default optimization passes. It is used to bring the code into
  // a canonical form that can than be analyzed by Polly. This set of passes is
  // most probably not yet optimal. TODO: Investigate optimal set of passes.
  PM.add(llvm::createPromoteMemoryToRegisterPass());
  PM.add(llvm::createInstructionCombiningPass());  // Clean up after IPCP & DAE
  PM.add(llvm::createCFGSimplificationPass());     // Clean up after IPCP & DAE
  PM.add(llvm::createTailCallEliminationPass());   // Eliminate tail calls
  PM.add(llvm::createCFGSimplificationPass());     // Merge & remove BBs
  PM.add(llvm::createReassociatePass());           // Reassociate expressions
  PM.add(llvm::createLoopRotatePass());            // Rotate Loop
  PM.add(llvm::createInstructionCombiningPass());
  PM.add(polly::createIndVarSimplifyPass());        // Canonicalize indvars

  PM.add(polly::createCodePreparationPass());
  PM.add(polly::createRegionSimplifyPass());
  // FIXME: The next two passes should not be necessary here. They are currently
  //        because of two problems:
  //
  //        1. The RegionSimplifyPass destroys the canonical form of induction
  //           variables,as it produces PHI nodes with incorrectly ordered
  //           operands. To fix this we run IndVarSimplify.
  //
  //        2. IndVarSimplify does not preserve the region information and
  //           the regioninfo pass does currently not recover simple regions.
  //           As a result we need to run the RegionSimplify pass again to
  //           recover them
  PM.add(polly::createIndVarSimplifyPass());
  PM.add(polly::createRegionSimplifyPass());
}

VectorizerChoice polly::PollyVectorizerChoice;

static void registerPollyPasses(llvm::PassManagerBase &PM) {
  registerPollyPreoptPasses(PM);

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
  }

  if (CFGPrinter)
    PM.add(llvm::createCFGPrinterPass());
}

static
void registerPollyEarlyAsPossiblePasses(const llvm::PassManagerBuilder &Builder,
                                        llvm::PassManagerBase &PM) {

  if (Builder.OptLevel == 0)
    return;

  if (PollyOnlyPrinter || PollyPrinter || PollyOnlyViewer || PollyViewer ||
      ExportJScop || ImportJScop)
    PollyEnabled = true;

  if (!PollyEnabled)
    return;

  // Polly is only enabled at -O3
  if (Builder.OptLevel != 3) {
    errs() << "Polly should only be run with -O3. Disabling Polly.\n";
    return;
  }

  registerPollyPasses(PM);
}

static void registerPollyOptLevel0Passes(const llvm::PassManagerBuilder &,
                                         llvm::PassManagerBase &PM) {
  registerPollyPreoptPasses(PM);
}


// Execute Polly together with a set of preparing passes.
//
// We run Polly that early to run before loop optimizer passes like LICM or
// the LoopIdomPass. Both transform the code in a way that Polly will recognize
// less scops.

static llvm::RegisterStandardPasses
PassRegister(llvm::PassManagerBuilder::EP_EarlyAsPossible,
             registerPollyEarlyAsPossiblePasses);
static llvm::RegisterStandardPasses
PassRegisterPreopt(llvm::PassManagerBuilder::EP_EnabledOnOptLevel0,
                  registerPollyOptLevel0Passes);
