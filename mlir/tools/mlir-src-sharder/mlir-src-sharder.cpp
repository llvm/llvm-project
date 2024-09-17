//===- mlir-src-sharder.cpp - A tool for sharder generated source files ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

/// Create a dependency file for `-d` option.
///
/// This functionality is generally only for the benefit of the build system,
/// and is modeled after the same option in TableGen.
static LogicalResult createDependencyFile(StringRef outputFilename,
                                          StringRef dependencyFile) {
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

  outputFile->os() << outputFilename << ":\n";
  outputFile->keep();
  return success();
}

int main(int argc, char **argv) {
  // FIXME: This is necessary because we link in TableGen, which defines its
  // options as static variables.. some of which overlap with our options.
  llvm::cl::ResetCommandLineParser();

  llvm::cl::opt<unsigned> opShardIndex(
      "op-shard-index", llvm::cl::desc("The current shard index"));
  llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                           llvm::cl::desc("<input file>"),
                                           llvm::cl::init("-"));
  llvm::cl::opt<std::string> outputFilename(
      "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
      llvm::cl::init("-"));
  llvm::cl::list<std::string> includeDirs(
      "I", llvm::cl::desc("Directory of include files"),
      llvm::cl::value_desc("directory"), llvm::cl::Prefix);
  llvm::cl::opt<std::string> dependencyFilename(
      "d", llvm::cl::desc("Dependency filename"),
      llvm::cl::value_desc("filename"), llvm::cl::init(""));
  llvm::cl::opt<bool> writeIfChanged(
      "write-if-changed",
      llvm::cl::desc("Only write to the output file if it changed"));

  // `ResetCommandLineParser` at the above unregistered the "D" option
  // of `llvm-tblgen`, which caused `TestOps.cpp` to fail due to
  // "Unknnown command line argument '-D...`" when a macros name is
  // present. The following is a workaround to re-register it again.
  llvm::cl::list<std::string> macroNames(
      "D",
      llvm::cl::desc(
          "Name of the macro to be defined -- ignored by mlir-src-sharder"),
      llvm::cl::value_desc("macro name"), llvm::cl::Prefix);

  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv);

  // Open the input file.
  std::string errorMessage;
  std::unique_ptr<llvm::MemoryBuffer> inputFile =
      openInputFile(inputFilename, &errorMessage);
  if (!inputFile) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  // Write the output to a buffer.
  std::string outputStr;
  llvm::raw_string_ostream os(outputStr);
  os << "#define GET_OP_DEFS_" << opShardIndex << "\n"
     << inputFile->getBuffer();

  // Determine whether we need to write the output file.
  bool shouldWriteOutput = true;
  if (writeIfChanged) {
    // Only update the real output file if there are any differences. This
    // prevents recompilation of all the files depending on it if there aren't
    // any.
    if (auto existingOrErr =
            llvm::MemoryBuffer::getFile(outputFilename, /*IsText=*/true))
      if (std::move(existingOrErr.get())->getBuffer() == outputStr)
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
    outputFile->os() << os.str();
    outputFile->keep();
  }

  // Always write the depfile, even if the main output hasn't changed. If it's
  // missing, Ninja considers the output dirty.
  if (!dependencyFilename.empty())
    if (failed(createDependencyFile(outputFilename, dependencyFilename)))
      return 1;

  return 0;
}
