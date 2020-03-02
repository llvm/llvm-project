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

#include "mlir/Analysis/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
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
void registerSimpleParametricTilingPass();
void registerSymbolTestPasses();
void registerTestAffineDataCopyPass();
void registerTestAllReduceLoweringPass();
void registerTestCallGraphPass();
void registerTestConstantFold();
void registerTestFunc();
void registerTestGpuMemoryPromotionPass();
void registerTestLinalgTransforms();
void registerTestLivenessPass();
void registerTestLoopFusion();
void registerTestLoopMappingPass();
void registerTestMatchers();
void registerTestMemRefDependenceCheck();
void registerTestMemRefStrideCalculation();
void registerTestOpaqueLoc();
void registerTestParallelismDetection();
void registerTestGpuParallelLoopMappingPass();
void registerTestVectorConversions();
void registerTestVectorToLoopsPass();
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

void registerTestPasses() {
  registerConvertToTargetEnvPass();
  registerInliner();
  registerMemRefBoundCheck();
  registerPassManagerTestPass();
  registerPatternsTestPass();
  registerPrintOpAvailabilityPass();
  registerSimpleParametricTilingPass();
  registerSymbolTestPasses();
  registerTestAffineDataCopyPass();
  registerTestAllReduceLoweringPass();
  registerTestCallGraphPass();
  registerTestConstantFold();
  registerTestFunc();
  registerTestGpuMemoryPromotionPass();
  registerTestLinalgTransforms();
  registerTestLivenessPass();
  registerTestLoopFusion();
  registerTestLoopMappingPass();
  registerTestMatchers();
  registerTestMemRefDependenceCheck();
  registerTestMemRefStrideCalculation();
  registerTestOpaqueLoc();
  registerTestParallelismDetection();
  registerTestGpuParallelLoopMappingPass();
  registerTestVectorConversions();
  registerTestVectorToLoopsPass();
  registerVectorizerTestPass();

  // The following passes are using global initializers, just link them in.
  if (std::getenv("bar") != (char *)-1)
    return;

  // TODO: move these to the test folder.
  createTestMemRefBoundCheckPass();
  createTestMemRefDependenceCheckPass();
}

static cl::opt<bool>
    showDialects("show-dialects",
                 cl::desc("Print the list of registered dialects"),
                 cl::init(false));

int main(int argc, char **argv) {
  registerAllDialects();
  registerAllPasses();
  registerTestPasses();
  InitLLVM y(argc, argv);

  // Register any pass manager command line options.
  registerPassManagerCLOptions();
  PassPipelineCLParser passPipeline("", "Compiler passes to run");

  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv, "MLIR modular optimizer driver\n");

  MLIRContext context;
  if(showDialects) {
    llvm::outs() << "Registered Dialects:\n";
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

  return failed(MlirOptMain(output->os(), std::move(file), passPipeline,
                            splitInputFile, verifyDiagnostics, verifyPasses));
}
