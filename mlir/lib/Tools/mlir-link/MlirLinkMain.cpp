//===- MlirLinkMain.cpp - MLIR Link main ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/mlir-link/MlirLinkMain.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Linker/Linker.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/ParseUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"

using namespace mlir;
using namespace llvm;

OwningOpRef<ModuleOp> makeCompositeModule(MLIRContext &context) {
  OpBuilder builder(&context);
  ModuleOp op = builder.create<ModuleOp>(
      FileLineColLoc::get(&context, "mlir-link", 0, 0));
  OwningOpRef<ModuleOp> composite(op);
  return composite;
}

static LogicalResult linkFile(Linker &linker,
                              std::shared_ptr<SourceMgr> sourceMgr,
                              unsigned flags, bool internalizeLinkedSymbols) {
  // TBD: setup timing

  auto context = linker.getContext();

  // Disable multi-threading when parsing the input file. This removes the
  // unnecessary/costly context synchronization when parsing.
  bool wasThreadingEnabled = context->isMultithreadingEnabled();
  context->disableMultithreading();

  OwningOpRef<Operation *> op =
      parseSourceFileForTool(sourceMgr, context, true /*insertImplicitModule*/);

  if (!op)
    return failure();

  context->enableMultithreading(wasThreadingEnabled);

  // TBD: symbol promotion

  if (internalizeLinkedSymbols) {
    // TBD: internalization
  } else {
    return linker.linkInModule(std::move(op), flags);
  }
}

static LogicalResult linkFile(Linker &linker,
                              std::unique_ptr<MemoryBuffer> buffer,
                              unsigned flags, bool internalizeLinkedSymbols) {
  // TBD: use splitAndProcessBuffer?
  auto sourceMgr = std::make_shared<SourceMgr>();
  sourceMgr->AddNewSourceBuffer(std::move(buffer), SMLoc());

  auto context = linker.getContext();

  // TBD: install debug handler

  // TBD: verify on diagnostics
  SourceMgrDiagnosticHandler sourceMgrHandler(*sourceMgr, context);
  return linkFile(linker, sourceMgr, flags, internalizeLinkedSymbols);
}

static LogicalResult linkFiles(Linker &linker,
                               const cl::list<std::string> &fileNames,
                               unsigned flags, bool internalize) {
  // Filter out flags that don't apply to the first file we load.
  unsigned applicableFlags = flags & Linker::OverrideFromSrc;
  // Similar to some flags, internalization doesn't apply to the first file.
  bool internalizeLinkedSymbols = false;

  std::string errorMessage;
  for (const auto &fileName : fileNames) {
    auto file = openInputFile(fileName, &errorMessage);
    if (!file) {
      llvm::errs() << errorMessage << "\n";
      return failure();
    }

    if (failed(linkFile(linker, std::move(file), applicableFlags,
                        internalizeLinkedSymbols)))
      return failure();

    // Internalization applies to linking of subsequent files.
    internalizeLinkedSymbols = internalize;

    // All linker flags apply to linking of subsequent files.
    applicableFlags = flags;
  }

  return success();
}

LogicalResult mlir::MlirLinkMain(int argc, char **argv,
                                 DialectRegistry &registry) {
  static cl::OptionCategory linkCategory("Link options");

  static cl::list<std::string> inputFilenames(cl::Positional, cl::OneOrMore,
                                              cl::desc("<input mlir files>"),
                                              cl::cat(linkCategory));

  static cl::opt<std::string> outputFilename(
      "o", cl::desc("Override output filename"), cl::init("-"),
      cl::value_desc("filename"), cl::cat(linkCategory));

  static cl::opt<bool> internalize("internalize",
                                   cl::desc("Internalize linked symbols"),
                                   cl::cat(linkCategory));

  static cl::opt<bool> onlyNeeded("only-needed",
                                  cl::desc("Link only needed symbols"),
                                  cl::cat(linkCategory));

  static cl::opt<bool> allowUnregisteredDialects(
      "allow-unregistered-dialect",
      cl::desc("Allow operations coming from an unregistered dialect"),
      cl::init(false));

  static cl::opt<bool> verbose(
      "v", cl::desc("Print information about actions taken"),
      cl::cat(linkCategory));

  static ExitOnError exitOnErr;

  InitLLVM y(argc, argv);
  exitOnErr.setBanner(std::string(argv[0]) + ": ");

  cl::HideUnrelatedOptions({&linkCategory, &getColorCategory()});
  cl::ParseCommandLineOptions(argc, argv, "mlir linker\n");

  MLIRContext context(registry);
  context.allowUnregisteredDialects(allowUnregisteredDialects);

  auto composite = makeCompositeModule(context);
  if (!isa<LinkableModuleOpInterface>(composite->getOperation())) {
    return composite->emitError("expected a LinkableModuleOpInterface");
  }

  Linker linker(cast<LinkableModuleOpInterface>(composite->getOperation()));

  unsigned flags = Linker::Flags::None;
  if (onlyNeeded)
    flags |= Linker::Flags::LinkOnlyNeeded;

  // First add all the regular input files
  if (failed(linkFiles(linker, inputFilenames, flags, internalize)))
    return failure();

  std::string errorMessage;
  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    errs() << errorMessage;
    return failure();
  }

  if (verbose)
    errs() << "Writing linked module to '" << outputFilename << "'\n";

  composite.get()->print(output->os());
  output->keep();
  return success();
}
