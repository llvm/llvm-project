//===- standalone-translate.cpp ---------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility that translates a file from/to MLIR using one
// of the registered translations.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Translation.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "Standalone/StandaloneDialect.h"

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"));

static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("-"));

static llvm::cl::opt<bool>
    splitInputFile("split-input-file",
                   llvm::cl::desc("Split the input file into pieces and "
                                  "process each chunk independently"),
                   llvm::cl::init(false));

static llvm::cl::opt<bool> verifyDiagnostics(
    "verify-diagnostics",
    llvm::cl::desc("Check that emitted diagnostics match "
                   "expected-* lines on the corresponding line"),
    llvm::cl::init(false));

int main(int argc, char **argv) {
  mlir::registerAllTranslations();

  // TODO: Register standalone translations here.

  llvm::InitLLVM y(argc, argv);

  // Add flags for all the registered translations.
  llvm::cl::opt<const mlir::TranslateFunction *, false, mlir::TranslationParser>
      translationRequested("", llvm::cl::desc("Translation to perform"),
                           llvm::cl::Required);
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR translation driver\n");

  std::string errorMessage;
  auto input = mlir::openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  auto output = mlir::openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  // Processes the memory buffer with a new MLIRContext.
  auto processBuffer = [&](std::unique_ptr<llvm::MemoryBuffer> ownedBuffer,
                           llvm::raw_ostream &os) {
    mlir::MLIRContext context;
    context.allowUnregisteredDialects();
    context.printOpOnDiagnostic(!verifyDiagnostics);
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(ownedBuffer), llvm::SMLoc());

    if (!verifyDiagnostics) {
      mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
      return (*translationRequested)(sourceMgr, os, &context);
    }

    // In the diagnostic verification flow, we ignore whether the translation
    // failed (in most cases, it is expected to fail). Instead, we check if the
    // diagnostics were produced as expected.
    mlir::SourceMgrDiagnosticVerifierHandler sourceMgrHandler(sourceMgr,
                                                              &context);
    (*translationRequested)(sourceMgr, os, &context);
    return sourceMgrHandler.verify();
  };

  if (splitInputFile) {
    if (failed(mlir::splitAndProcessBuffer(std::move(input), processBuffer,
                                           output->os())))
      return 1;
  } else {
    if (failed(processBuffer(std::move(input), output->os())))
      return 1;
  }

  output->keep();
  return 0;
}
