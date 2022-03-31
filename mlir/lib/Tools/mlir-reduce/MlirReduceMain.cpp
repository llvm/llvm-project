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

#include "mlir/Tools/mlir-reduce/MlirReduceMain.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Reducer/Passes.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

// Parse and verify the input MLIR file.
static LogicalResult loadModule(MLIRContext &context,
                                OwningOpRef<ModuleOp> &module,
                                StringRef inputFilename) {
  module = parseSourceFile<ModuleOp>(inputFilename, &context);
  if (!module)
    return failure();

  return success();
}

LogicalResult mlir::mlirReduceMain(int argc, char **argv,
                                   MLIRContext &context) {
  // Override the default '-h' and use the default PrintHelpMessage() which
  // won't print options in categories.
  static llvm::cl::opt<bool> help("h", llvm::cl::desc("Alias for -help"),
                                  llvm::cl::Hidden);

  static llvm::cl::OptionCategory mlirReduceCategory("mlir-reduce options");

  static llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input file>"),
      llvm::cl::cat(mlirReduceCategory));

  static llvm::cl::opt<std::string> outputFilename(
      "o", llvm::cl::desc("Output filename for the reduced test case"),
      llvm::cl::init("-"), llvm::cl::cat(mlirReduceCategory));

  llvm::cl::HideUnrelatedOptions(mlirReduceCategory);

  llvm::InitLLVM y(argc, argv);

  registerReducerPasses();

  PassPipelineCLParser parser("", "Reduction Passes to Run");
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "MLIR test case reduction tool.\n");

  if (help) {
    llvm::cl::PrintHelpMessage();
    return success();
  }

  std::string errorMessage;

  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output)
    return failure();

  OwningOpRef<ModuleOp> moduleRef;
  if (failed(loadModule(context, moduleRef, inputFilename)))
    return failure();

  auto errorHandler = [&](const Twine &msg) {
    return emitError(UnknownLoc::get(&context)) << msg;
  };

  // Reduction pass pipeline.
  PassManager pm(&context);
  if (failed(parser.addToPipeline(pm, errorHandler)))
    return failure();

  OwningOpRef<ModuleOp> m = moduleRef.get().clone();

  if (failed(pm.run(m.get())))
    return failure();

  m->print(output->os());
  output->keep();

  return success();
}
