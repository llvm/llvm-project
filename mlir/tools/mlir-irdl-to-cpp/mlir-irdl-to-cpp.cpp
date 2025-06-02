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

static LogicalResult
processBuffer(llvm::raw_ostream &os,
              std::unique_ptr<llvm::MemoryBuffer> ownedBuffer,
              bool verifyDiagnostics, llvm::ThreadPoolInterface *threadPool) {
  // Tell sourceMgr about this buffer, which is what the parser will pick up.
  auto sourceMgr = std::make_shared<llvm::SourceMgr>();
  sourceMgr->AddNewSourceBuffer(std::move(ownedBuffer), SMLoc());

  DialectRegistry registry;
  registry.insert<irdl::IRDLDialect>();
  MLIRContext ctx(registry);

  ctx.printOpOnDiagnostic(!verifyDiagnostics);

  auto runTranslation = [&]() {
    ParserConfig parseConfig(&ctx);
    OwningOpRef<Operation *> op =
        parseSourceFileForTool(sourceMgr, parseConfig, true);
    if (!op)
      return failure();

    auto moduleOp = llvm::cast<ModuleOp>(*op);
    llvm::SmallVector<irdl::DialectOp> dialects{
        moduleOp.getOps<irdl::DialectOp>(),
    };

    return irdl::translateIRDLDialectToCpp(dialects, os);
  };

  if (!verifyDiagnostics) {
    // If no errors are expected, return translation result.
    SourceMgrDiagnosticHandler srcManagerHandler(*sourceMgr, &ctx);
    return runTranslation();
  }

  // If errors are expected, ignore translation result and check for
  // diagnostics.
  SourceMgrDiagnosticVerifierHandler srcManagerHandler(*sourceMgr, &ctx);
  (void)runTranslation();
  return srcManagerHandler.verify();
}

static LogicalResult translateIRDLToCpp(int argc, char **argv) {
  static llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input file>"),
      llvm::cl::init("-"));

  static llvm::cl::opt<std::string> outputFilename(
      "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
      llvm::cl::init("-"));

  static llvm::cl::opt<bool> verifyDiagnostics(
      "verify-diagnostics",
      llvm::cl::desc("Check that emitted diagnostics match "
                     "expected-* lines on the corresponding line"),
      llvm::cl::init(false));

  static llvm::cl::opt<std::string> splitInputFile(
      "split-input-file", llvm::cl::ValueOptional,
      llvm::cl::callback([&](const std::string &str) {
        // Implicit value: use default marker if flag was used without
        // value.
        if (str.empty())
          splitInputFile.setValue(kDefaultSplitMarker);
      }),
      llvm::cl::desc("Split the input file into chunks using the given or "
                     "default marker and process each chunk independently"),
      llvm::cl::init(""));

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

  auto chunkFn = [&](std::unique_ptr<llvm::MemoryBuffer> chunkBuffer,
                     raw_ostream &os) {
    return processBuffer(output->os(), std::move(chunkBuffer),
                         verifyDiagnostics, nullptr);
  };

  auto &splitInputFileDelimiter = splitInputFile.getValue();
  if (splitInputFileDelimiter.size())
    return splitAndProcessBuffer(std::move(input), chunkFn, output->os(),
                                 splitInputFileDelimiter,
                                 splitInputFileDelimiter);

  if (failed(chunkFn(std::move(input), output->os())))
    return failure();

  if (!verifyDiagnostics)
    output->keep();

  return success();
}

int main(int argc, char **argv) {
  return failed(translateIRDLToCpp(argc, argv));
}
