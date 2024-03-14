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
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/Timing.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Diagnostic Filter
//===----------------------------------------------------------------------===//

namespace {
/// A scoped diagnostic handler that marks non-error diagnostics as handled. As
/// a result, the main diagnostic handler does not print non-error diagnostics.
class ErrorDiagnosticFilter : public ScopedDiagnosticHandler {
public:
  ErrorDiagnosticFilter(MLIRContext *ctx) : ScopedDiagnosticHandler(ctx) {
    setHandler([](Diagnostic &diag) {
      if (diag.getSeverity() != DiagnosticSeverity::Error)
        return success();
      return failure();
    });
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Translate Entry Point
//===----------------------------------------------------------------------===//

LogicalResult mlir::mlirTranslateMain(int argc, char **argv,
                                      llvm::StringRef toolName) {

  static llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input file>"),
      llvm::cl::init("-"));

  static llvm::cl::opt<std::string> outputFilename(
      "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
      llvm::cl::init("-"));

  static llvm::cl::opt<bool> allowUnregisteredDialects(
      "allow-unregistered-dialect",
      llvm::cl::desc("Allow operation with no registered dialects (discouraged: testing only!)"),
      llvm::cl::init(false));

  static llvm::cl::opt<std::string> inputSplitMarker(
      "split-input-file", llvm::cl::ValueOptional,
      llvm::cl::callback([&](const std::string &str) {
        // Implicit value: use default marker if flag was used without value.
        if (str.empty())
          inputSplitMarker.setValue(kDefaultSplitMarker);
      }),
      llvm::cl::desc("Split the input file into chunks using the given or "
                     "default marker and process each chunk independently"),
      llvm::cl::init(""));

  static llvm::cl::opt<bool> verifyDiagnostics(
      "verify-diagnostics",
      llvm::cl::desc("Check that emitted diagnostics match "
                     "expected-* lines on the corresponding line"),
      llvm::cl::init(false));

  static llvm::cl::opt<bool> errorDiagnosticsOnly(
      "error-diagnostics-only",
      llvm::cl::desc("Filter all non-error diagnostics "
                     "(discouraged: testing only!)"),
      llvm::cl::init(false));

  static llvm::cl::opt<std::string> outputSplitMarker(
      "output-split-marker",
      llvm::cl::desc("Split marker to use for merging the ouput"),
      llvm::cl::init(""));

  llvm::InitLLVM y(argc, argv);

  // Add flags for all the registered translations.
  llvm::cl::list<const Translation *, bool, TranslationParser>
      translationsRequested("", llvm::cl::desc("Translations to perform"),
                            llvm::cl::Required);
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  registerTranslationCLOptions();
  registerDefaultTimingManagerCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, toolName);

  // Initialize the timing manager.
  DefaultTimingManager tm;
  applyDefaultTimingManagerCLOptions(tm);
  TimingScope timing = tm.getRootScope();

  std::string errorMessage;
  std::unique_ptr<llvm::MemoryBuffer> input;
  if (auto inputAlignment = translationsRequested[0]->getInputAlignment())
    input = openInputFile(inputFilename, *inputAlignment, &errorMessage);
  else
    input = openInputFile(inputFilename, &errorMessage);
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
    // Temporary buffers for chained translation processing.
    std::string dataIn;
    std::string dataOut;
    LogicalResult result = LogicalResult::success();

    for (size_t i = 0, e = translationsRequested.size(); i < e; ++i) {
      llvm::raw_ostream *stream;
      llvm::raw_string_ostream dataStream(dataOut);

      if (i == e - 1) {
        // Output last translation to output.
        stream = &os;
      } else {
        // Output translation to temporary data buffer.
        stream = &dataStream;
      }

      const Translation *translationRequested = translationsRequested[i];
      TimingScope translationTiming =
          timing.nest(translationRequested->getDescription());

      MLIRContext context;
      context.allowUnregisteredDialects(allowUnregisteredDialects);
      context.printOpOnDiagnostic(!verifyDiagnostics);
      auto sourceMgr = std::make_shared<llvm::SourceMgr>();
      sourceMgr->AddNewSourceBuffer(std::move(ownedBuffer), SMLoc());

      if (verifyDiagnostics) {
        // In the diagnostic verification flow, we ignore whether the
        // translation failed (in most cases, it is expected to fail) and we do
        // not filter non-error diagnostics even if `errorDiagnosticsOnly` is
        // set. Instead, we check if the diagnostics were produced as expected.
        SourceMgrDiagnosticVerifierHandler sourceMgrHandler(*sourceMgr,
                                                            &context);
        (void)(*translationRequested)(sourceMgr, os, &context);
        result = sourceMgrHandler.verify();
      } else if (errorDiagnosticsOnly) {
        SourceMgrDiagnosticHandler sourceMgrHandler(*sourceMgr, &context);
        ErrorDiagnosticFilter diagnosticFilter(&context);
        result = (*translationRequested)(sourceMgr, *stream, &context);
      } else {
        SourceMgrDiagnosticHandler sourceMgrHandler(*sourceMgr, &context);
        result = (*translationRequested)(sourceMgr, *stream, &context);
      }
      if (failed(result))
        return result;

      if (i < e - 1) {
        // If there are further translations, create a new buffer with the
        // output data.
        dataIn = dataOut;
        dataOut.clear();
        ownedBuffer = llvm::MemoryBuffer::getMemBuffer(dataIn);
      }
    }
    return result;
  };

  if (failed(splitAndProcessBuffer(std::move(input), processBuffer,
                                   output->os(), inputSplitMarker,
                                   outputSplitMarker)))
    return failure();

  output->keep();
  return success();
}
