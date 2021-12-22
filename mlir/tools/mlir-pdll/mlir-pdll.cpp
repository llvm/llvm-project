//===- mlir-pdll.cpp - MLIR PDLL frontend -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Tools/PDLL/AST/Context.h"
#include "mlir/Tools/PDLL/AST/Nodes.h"
#include "mlir/Tools/PDLL/Parser/Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;
using namespace mlir::pdll;

//===----------------------------------------------------------------------===//
// main
//===----------------------------------------------------------------------===//

/// The desired output type.
enum class OutputType {
  AST,
};

static LogicalResult
processBuffer(raw_ostream &os, std::unique_ptr<llvm::MemoryBuffer> chunkBuffer,
              OutputType outputType, std::vector<std::string> &includeDirs) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.setIncludeDirs(includeDirs);
  sourceMgr.AddNewSourceBuffer(std::move(chunkBuffer), llvm::SMLoc());

  ast::Context astContext;
  FailureOr<ast::Module *> module = parsePDLAST(astContext, sourceMgr);
  if (failed(module))
    return failure();

  switch (outputType) {
  case OutputType::AST:
    (*module)->print(os);
    break;
  }

  return success();
}

int main(int argc, char **argv) {
  llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::init("-"),
      llvm::cl::value_desc("filename"));

  llvm::cl::opt<std::string> outputFilename(
      "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
      llvm::cl::init("-"));

  llvm::cl::list<std::string> includeDirs(
      "I", llvm::cl::desc("Directory of include files"),
      llvm::cl::value_desc("directory"), llvm::cl::Prefix);

  llvm::cl::opt<bool> splitInputFile(
      "split-input-file",
      llvm::cl::desc("Split the input file into pieces and process each "
                     "chunk independently"),
      llvm::cl::init(false));
  llvm::cl::opt<enum OutputType> outputType(
      "x", llvm::cl::init(OutputType::AST),
      llvm::cl::desc("The type of output desired"),
      llvm::cl::values(clEnumValN(OutputType::AST, "ast",
                                  "generate the AST for the input file")));

  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "PDLL Frontend");

  // Set up the input file.
  std::string errorMessage;
  std::unique_ptr<llvm::MemoryBuffer> inputFile =
      openInputFile(inputFilename, &errorMessage);
  if (!inputFile) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  // Set up the output file.
  std::unique_ptr<llvm::ToolOutputFile> outputFile =
      openOutputFile(outputFilename, &errorMessage);
  if (!outputFile) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  // The split-input-file mode is a very specific mode that slices the file
  // up into small pieces and checks each independently.
  auto processFn = [&](std::unique_ptr<llvm::MemoryBuffer> chunkBuffer,
                       raw_ostream &os) {
    return processBuffer(os, std::move(chunkBuffer), outputType, includeDirs);
  };
  if (splitInputFile) {
    if (failed(splitAndProcessBuffer(std::move(inputFile), processFn,
                                     outputFile->os())))
      return 1;
  } else if (failed(processFn(std::move(inputFile), outputFile->os()))) {
    return 1;
  }
  outputFile->keep();
  return 0;
}
