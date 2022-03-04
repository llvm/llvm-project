//===- MlirTranslateMain.cpp - MLIR Translation entry point ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// mlir-translate tool driver
//===----------------------------------------------------------------------===//

LogicalResult
mlir::mlirTranslateMain(int argc, char **argv, llvm::StringRef toolName,
                        const DialectRegistry &extraDialects,
                        llvm::function_ref<void(MLIRContext &)> customization) {

  static llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input file>"),
      llvm::cl::init("-"));

  static llvm::cl::opt<std::string> outputFilename(
      "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
      llvm::cl::init("-"));

  static llvm::cl::opt<bool> allowUnregisteredDialects(
      "allow-unregistered-dialect",
      llvm::cl::desc("Allow operation with no registered dialects"),
      llvm::cl::init(false));

  static llvm::cl::opt<bool> splitInputFile(
      "split-input-file",
      llvm::cl::desc("Split the input file into pieces and "
                     "process each chunk independently"),
      llvm::cl::init(false));

  static llvm::cl::opt<bool> verifyDiagnostics(
      "verify-diagnostics",
      llvm::cl::desc("Check that emitted diagnostics match "
                     "expected-* lines on the corresponding line"),
      llvm::cl::init(false));

  llvm::InitLLVM y(argc, argv);

  // Add flags for all the registered translations.
  llvm::cl::opt<const TranslateFunction *, false, TranslationParser>
      translationRequested("", llvm::cl::desc("Translation to perform"),
                           llvm::cl::Required);
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, toolName);

  std::string errorMessage;
  auto input = openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  // Processes the memory buffer with a new MLIRContext.
  auto processBuffer = [&](std::unique_ptr<llvm::MemoryBuffer> ownedBuffer,
                           raw_ostream &os) {
    MLIRContext context;

    // If the client wanted to register additional dialects, go ahead and add
    // them to our context.
    context.appendDialectRegistry(extraDialects);

    // If a customization callback was provided, apply it to the MLIRContext.
    // This could add dialects to the registry or change context defaults.
    if (customization)
      customization(context);

    // If command line flags were used to customize the context, apply their
    // settings.
    if (allowUnregisteredDialects.getNumOccurrences())
      context.allowUnregisteredDialects(allowUnregisteredDialects);
    context.printOpOnDiagnostic(!verifyDiagnostics);

    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(ownedBuffer), SMLoc());

    if (!verifyDiagnostics) {
      SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
      return (*translationRequested)(sourceMgr, os, &context);
    }

    // In the diagnostic verification flow, we ignore whether the translation
    // failed (in most cases, it is expected to fail). Instead, we check if the
    // diagnostics were produced as expected.
    SourceMgrDiagnosticVerifierHandler sourceMgrHandler(sourceMgr, &context);
    (void)(*translationRequested)(sourceMgr, os, &context);
    return sourceMgrHandler.verify();
  };

  if (splitInputFile) {
    if (failed(splitAndProcessBuffer(std::move(input), processBuffer,
                                     output->os())))
      return failure();
  } else if (failed(processBuffer(std::move(input), output->os()))) {
    return failure();
  }

  output->keep();
  return success();
}
