//===- mlir-pdll.cpp - MLIR PDLL frontend -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Tools/PDLL/AST/Context.h"
#include "mlir/Tools/PDLL/AST/Nodes.h"
#include "mlir/Tools/PDLL/CodeGen/CPPGen.h"
#include "mlir/Tools/PDLL/CodeGen/MLIRGen.h"
#include "mlir/Tools/PDLL/ODS/Context.h"
#include "mlir/Tools/PDLL/Parser/Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include <set>

using namespace mlir;
using namespace mlir::pdll;

//===----------------------------------------------------------------------===//
// main
//===----------------------------------------------------------------------===//

/// The desired output type.
enum class OutputType {
  AST,
  MLIR,
  CPP,
};

static LogicalResult
processBuffer(raw_ostream &os, std::unique_ptr<llvm::MemoryBuffer> chunkBuffer,
              OutputType outputType, std::vector<std::string> &includeDirs,
              bool dumpODS, std::set<std::string> *includedFiles) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.setIncludeDirs(includeDirs);
  sourceMgr.AddNewSourceBuffer(std::move(chunkBuffer), SMLoc());

  // If we are dumping ODS information, also enable documentation to ensure the
  // summary and description information is imported as well.
  bool enableDocumentation = dumpODS;

  ods::Context odsContext;
  ast::Context astContext(odsContext);
  FailureOr<ast::Module *> module =
      parsePDLLAST(astContext, sourceMgr, enableDocumentation);
  if (failed(module))
    return failure();

  // Add the files that were included to the set.
  if (includedFiles) {
    for (unsigned i = 1, e = sourceMgr.getNumBuffers(); i < e; ++i) {
      includedFiles->insert(
          sourceMgr.getMemoryBuffer(i + 1)->getBufferIdentifier().str());
    }
  }

  // Print out the ODS information if requested.
  if (dumpODS)
    odsContext.print(llvm::errs());

  // Generate the output.
  if (outputType == OutputType::AST) {
    (*module)->print(os);
    return success();
  }

  MLIRContext mlirContext;
  OwningOpRef<ModuleOp> pdlModule =
      codegenPDLLToMLIR(&mlirContext, astContext, sourceMgr, **module);
  if (!pdlModule)
    return failure();

  if (outputType == OutputType::MLIR) {
    pdlModule->print(os, OpPrintingFlags().enableDebugInfo());
    return success();
  }
  codegenPDLLToCPP(**module, *pdlModule, os);
  return success();
}

/// Create a dependency file for `-d` option.
///
/// This functionality is generally only for the benefit of the build system,
/// and is modeled after the same option in TableGen.
static LogicalResult
createDependencyFile(StringRef outputFilename, StringRef dependencyFile,
                     std::set<std::string> &includedFiles) {
  if (outputFilename == "-") {
    llvm::errs() << "error: the option -d must be used together with -o\n";
    return failure();
  }

  std::string errorMessage;
  std::unique_ptr<llvm::ToolOutputFile> outputFile =
      openOutputFile(dependencyFile, &errorMessage);
  if (!outputFile) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  outputFile->os() << outputFilename << ":";
  for (const auto &includeFile : includedFiles)
    outputFile->os() << ' ' << includeFile;
  outputFile->os() << "\n";
  outputFile->keep();
  return success();
}

int main(int argc, char **argv) {
  // FIXME: This is necessary because we link in TableGen, which defines its
  // options as static variables.. some of which overlap with our options.
  llvm::cl::ResetCommandLineParser();

  llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::init("-"),
      llvm::cl::value_desc("filename"));

  llvm::cl::opt<std::string> outputFilename(
      "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
      llvm::cl::init("-"));

  llvm::cl::list<std::string> includeDirs(
      "I", llvm::cl::desc("Directory of include files"),
      llvm::cl::value_desc("directory"), llvm::cl::Prefix);

  llvm::cl::opt<bool> dumpODS(
      "dump-ods",
      llvm::cl::desc(
          "Print out the parsed ODS information from the input file"),
      llvm::cl::init(false));
  llvm::cl::opt<bool> splitInputFile(
      "split-input-file",
      llvm::cl::desc("Split the input file into pieces and process each "
                     "chunk independently"),
      llvm::cl::init(false));
  llvm::cl::opt<enum OutputType> outputType(
      "x", llvm::cl::init(OutputType::AST),
      llvm::cl::desc("The type of output desired"),
      llvm::cl::values(clEnumValN(OutputType::AST, "ast",
                                  "generate the AST for the input file"),
                       clEnumValN(OutputType::MLIR, "mlir",
                                  "generate the PDL MLIR for the input file"),
                       clEnumValN(OutputType::CPP, "cpp",
                                  "generate a C++ source file containing the "
                                  "patterns for the input file")));
  llvm::cl::opt<std::string> dependencyFilename(
      "d", llvm::cl::desc("Dependency filename"),
      llvm::cl::value_desc("filename"), llvm::cl::init(""));
  llvm::cl::opt<bool> writeIfChanged(
      "write-if-changed",
      llvm::cl::desc("Only write to the output file if it changed"));

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

  // If we are creating a dependency file, we'll also need to track what files
  // get included during processing.
  std::set<std::string> includedFilesStorage;
  std::set<std::string> *includedFiles = nullptr;
  if (!dependencyFilename.empty())
    includedFiles = &includedFilesStorage;

  // The split-input-file mode is a very specific mode that slices the file
  // up into small pieces and checks each independently.
  std::string outputStr;
  llvm::raw_string_ostream outputStrOS(outputStr);
  auto processFn = [&](std::unique_ptr<llvm::MemoryBuffer> chunkBuffer,
                       raw_ostream &os) {
    return processBuffer(os, std::move(chunkBuffer), outputType, includeDirs,
                         dumpODS, includedFiles);
  };
  if (splitInputFile) {
    if (failed(splitAndProcessBuffer(std::move(inputFile), processFn,
                                     outputStrOS)))
      return 1;
  } else if (failed(processFn(std::move(inputFile), outputStrOS))) {
    return 1;
  }

  // Write the output.
  bool shouldWriteOutput = true;
  if (writeIfChanged) {
    // Only update the real output file if there are any differences. This
    // prevents recompilation of all the files depending on it if there aren't
    // any.
    if (auto existingOrErr =
            llvm::MemoryBuffer::getFile(outputFilename, /*IsText=*/true))
      if (std::move(existingOrErr.get())->getBuffer() == outputStrOS.str())
        shouldWriteOutput = false;
  }

  // Populate the output file if necessary.
  if (shouldWriteOutput) {
    std::unique_ptr<llvm::ToolOutputFile> outputFile =
        openOutputFile(outputFilename, &errorMessage);
    if (!outputFile) {
      llvm::errs() << errorMessage << "\n";
      return 1;
    }
    outputFile->os() << outputStrOS.str();
    outputFile->keep();
  }

  // Always write the depfile, even if the main output hasn't changed. If it's
  // missing, Ninja considers the output dirty.
  if (!dependencyFilename.empty()) {
    if (failed(createDependencyFile(outputFilename, dependencyFilename,
                                    includedFilesStorage)))
      return 1;
  }

  return 0;
}
