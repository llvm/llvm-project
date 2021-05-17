//===- mlir-reduce.cpp - The MLIR reducer ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the general framework of the MLIR reducer tool. It
// parses the command line arguments, parses the initial MLIR test case and sets
// up the testing environment. It  outputs the most reduced test case variant
// after executing the reduction passes.
//
//===----------------------------------------------------------------------===//

#include <vector>

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Reducer/OptReductionPass.h"
#include "mlir/Reducer/Passes.h"
#include "mlir/Reducer/Passes/OpReducer.h"
#include "mlir/Reducer/ReductionNode.h"
#include "mlir/Reducer/ReductionTreePass.h"
#include "mlir/Reducer/Tester.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

namespace mlir {
namespace test {
void registerTestDialect(DialectRegistry &);
} // namespace test
} // namespace mlir

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::Required,
                                                llvm::cl::desc("<input file>"));

static llvm::cl::opt<std::string>
    outputFilename("o",
                   llvm::cl::desc("Output filename for the reduced test case"),
                   llvm::cl::init("-"));

// Parse and verify the input MLIR file.
static LogicalResult loadModule(MLIRContext &context, OwningModuleRef &module,
                                StringRef inputFilename) {
  module = parseSourceFile(inputFilename, &context);
  if (!module)
    return failure();

  return success();
}

int main(int argc, char **argv) {

  llvm::InitLLVM y(argc, argv);

  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerAllPasses();
  registerReducerPasses();
  PassPipelineCLParser parser("", "Reduction Passes to Run");

  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "MLIR test case reduction tool.\n");

  std::string errorMessage;

  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output)
    llvm::report_fatal_error(errorMessage);

  mlir::DialectRegistry registry;
  registerAllDialects(registry);
#ifdef MLIR_INCLUDE_TESTS
  mlir::test::registerTestDialect(registry);
#endif
  mlir::MLIRContext context(registry);

  mlir::OwningModuleRef moduleRef;
  if (failed(loadModule(context, moduleRef, inputFilename)))
    llvm::report_fatal_error("Input test case can't be parsed");

  auto errorHandler = [&](const Twine &msg) {
    return emitError(UnknownLoc::get(&context)) << msg;
  };

  // Reduction pass pipeline.
  PassManager pm(&context);
  if (failed(parser.addToPipeline(pm, errorHandler)))
    llvm::report_fatal_error("Failed to add pipeline");

  ModuleOp m = moduleRef.get().clone();

  if (failed(pm.run(m)))
    llvm::report_fatal_error("Error running the reduction pass pipeline");

  m.print(output->os());
  output->keep();

  return 0;
}
