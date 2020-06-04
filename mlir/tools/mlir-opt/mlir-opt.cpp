//===- mlir-opt.cpp - MLIR Optimizer Driver -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for mlir-opt for when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;
using namespace mlir;

namespace mlir {
// Defined in the test directory, no public header.
void registerConvertToTargetEnvPass();
void registerInliner();
void registerMemRefBoundCheck();
void registerPassManagerTestPass();
void registerPatternsTestPass();
void registerPrintOpAvailabilityPass();
void registerSideEffectTestPasses();
void registerSimpleParametricTilingPass();
void registerSymbolTestPasses();
void registerTestAffineDataCopyPass();
void registerTestAllReduceLoweringPass();
void registerTestAffineLoopUnswitchingPass();
void registerTestBufferPlacementPreparationPass();
void registerTestLoopPermutationPass();
void registerTestCallGraphPass();
void registerTestConstantFold();
void registerTestConvertGPUKernelToCubinPass();
void registerTestConvertGPUKernelToHsacoPass();
void registerTestDominancePass();
void registerTestFunc();
void registerTestGpuMemoryPromotionPass();
void registerTestLinalgHoisting();
void registerTestLinalgTransforms();
void registerTestLivenessPass();
void registerTestLoopFusion();
void registerTestLoopMappingPass();
void registerTestLoopUnrollingPass();
void registerTestMatchers();
void registerTestMemRefDependenceCheck();
void registerTestMemRefStrideCalculation();
void registerTestOpaqueLoc();
void registerTestParallelismDetection();
void registerTestGpuParallelLoopMappingPass();
void registerTestSCFUtilsPass();
void registerTestVectorConversions();
void registerVectorizerTestPass();
} // namespace mlir

static cl::opt<std::string>
    inputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));

static cl::opt<bool>
    splitInputFile("split-input-file",
                   cl::desc("Split the input file into pieces and process each "
                            "chunk independently"),
                   cl::init(false));

static cl::opt<bool>
    verifyDiagnostics("verify-diagnostics",
                      cl::desc("Check that emitted diagnostics match "
                               "expected-* lines on the corresponding line"),
                      cl::init(false));

static cl::opt<bool>
    verifyPasses("verify-each",
                 cl::desc("Run the verifier after each transformation pass"),
                 cl::init(true));

static cl::opt<bool> allowUnregisteredDialects(
    "allow-unregistered-dialect",
    cl::desc("Allow operation with no registered dialects"), cl::init(false));

#ifdef MLIR_INCLUDE_TESTS
void registerTestPasses() {
  registerConvertToTargetEnvPass();
  registerInliner();
  registerMemRefBoundCheck();
  registerPassManagerTestPass();
  registerPatternsTestPass();
  registerPrintOpAvailabilityPass();
  registerSideEffectTestPasses();
  registerSimpleParametricTilingPass();
  registerSymbolTestPasses();
  registerTestAffineDataCopyPass();
  registerTestAllReduceLoweringPass();
  registerTestAffineLoopUnswitchingPass();
  registerTestLoopPermutationPass();
  registerTestCallGraphPass();
  registerTestConstantFold();
#if MLIR_CUDA_CONVERSIONS_ENABLED
  registerTestConvertGPUKernelToCubinPass();
#endif
#if MLIR_ROCM_CONVERSIONS_ENABLED
  registerTestConvertGPUKernelToHsacoPass();
#endif
  registerTestBufferPlacementPreparationPass();
  registerTestDominancePass();
  registerTestFunc();
  registerTestGpuMemoryPromotionPass();
  registerTestLinalgHoisting();
  registerTestLinalgTransforms();
  registerTestLivenessPass();
  registerTestLoopFusion();
  registerTestLoopMappingPass();
  registerTestLoopUnrollingPass();
  registerTestMatchers();
  registerTestMemRefDependenceCheck();
  registerTestMemRefStrideCalculation();
  registerTestOpaqueLoc();
  registerTestParallelismDetection();
  registerTestGpuParallelLoopMappingPass();
  registerTestSCFUtilsPass();
  registerTestVectorConversions();
  registerVectorizerTestPass();
}
#endif

static cl::opt<bool>
    showDialects("show-dialects",
                 cl::desc("Print the list of registered dialects"),
                 cl::init(false));

int main(int argc, char **argv) {
  registerAllDialects();
  registerAllPasses();
#ifdef MLIR_INCLUDE_TESTS
  registerTestPasses();
#endif
  InitLLVM y(argc, argv);

  // Register any command line options.
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  PassPipelineCLParser passPipeline("", "Compiler passes to run");

  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv, "MLIR modular optimizer driver\n");

  if(showDialects) {
    llvm::outs() << "Registered Dialects:\n";
    MLIRContext context;
    for(Dialect *dialect : context.getRegisteredDialects()) {
      llvm::outs() << dialect->getNamespace() << "\n";
    }
    return 0;
  }

  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }

  if (failed(MlirOptMain(output->os(), std::move(file), passPipeline,
                         splitInputFile, verifyDiagnostics, verifyPasses,
                         allowUnregisteredDialects))) {
    return 1;
  }
  // Keep the output file if the invocation of MlirOptMain was successful.
  output->keep();
  return 0;
}
