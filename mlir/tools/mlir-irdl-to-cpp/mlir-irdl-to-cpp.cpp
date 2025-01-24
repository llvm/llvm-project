//===- mlir-irdl-to-cpp.cpp - IRDL to C++ conversion tool -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility that translates an IRDL dialect definition
// into a C++ implementation to be included in MLIR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/IRDL/IR/IRDL.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Target/IRDLToCpp/IRDLToCpp.h"
#include "mlir/Tools/ParseUtilities.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

static LogicalResult translateIRDLToCpp(int argc, char **argv) {
  static llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input file>"),
      llvm::cl::init("-"));

  static llvm::cl::opt<std::string> outputFilename(
      "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
      llvm::cl::init("-"));

  llvm::InitLLVM y(argc, argv);

  llvm::cl::ParseCommandLineOptions(argc, argv, "mlir-irdl-to-cpp");

  std::string errorMessage;
  std::unique_ptr<llvm::MemoryBuffer> input =
      openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  std::unique_ptr<llvm::ToolOutputFile> output =
      openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  auto sourceMgr = std::make_shared<llvm::SourceMgr>();
  sourceMgr->AddNewSourceBuffer(std::move(input), SMLoc());

  DialectRegistry registry;
  registry.insert<irdl::IRDLDialect>();
  MLIRContext ctx(registry);
  ctx.printOpOnDiagnostic(true);

  ParserConfig parseConfig(&ctx);
  OwningOpRef<Operation *> op =
      parseSourceFileForTool(sourceMgr, parseConfig, true);
  if (!op)
    return failure();

  auto moduleOp = llvm::cast<ModuleOp>(*op);

  SourceMgrDiagnosticHandler sourceMgrHandler(*sourceMgr, &ctx);

  for (Operation &op : moduleOp.getOps()) {
    auto dialectOp = llvm::dyn_cast<irdl::DialectOp>(op);
    if (!dialectOp)
      continue;

    // TODO: accept multiple operations in translation to not generate headers
    // multiple times.
    if (failed(irdl::translateIRDLDialectToCpp(dialectOp, output->os())))
      return failure();
  }

  output->keep();
  return success();
}

int main(int argc, char **argv) {
  return failed(translateIRDLToCpp(argc, argv));
}
