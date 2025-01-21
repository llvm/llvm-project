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
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"

using namespace mlir;
using namespace llvm;

OwningOpRef<ModuleOp> makeCompositeModule(MLIRContext *context) {
  OpBuilder builder(context);
  ModuleOp op =
      builder.create<ModuleOp>(FileLineColLoc::get(context, "mlir-link", 0, 0));
  OwningOpRef<ModuleOp> composite(op);
  return composite;
}

static LogicalResult linkFiles(const char *argv0, MLIRContext &context,
                              Linker &linker,
                              const cl::list<std::string> &files,
                              unsigned flags) {
  return failure();
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

  static cl::opt<bool> onlyNeeded("only-needed",
                                  cl::desc("Link only needed symbols"),
                                  cl::cat(linkCategory));

  static cl::opt<bool> verbose(
      "v", cl::desc("Print information about actions taken"),
      cl::cat(linkCategory));

  static ExitOnError exitOnErr;

  InitLLVM y(argc, argv);
  exitOnErr.setBanner(std::string(argv[0]) + ": ");

  cl::HideUnrelatedOptions({&linkCategory, &getColorCategory()});
  cl::ParseCommandLineOptions(argc, argv, "mlir linker\n");

  MLIRContext context(registry);
  auto composite = makeCompositeModule(&context);
  Linker linker(*composite);

  unsigned flags = Linker::Flags::None;
  if (onlyNeeded)
    flags |= Linker::Flags::LinkOnlyNeeded;

  // First add all the regular input files
  if (failed(linkFiles(argv[0], context, linker, inputFilenames, flags)))
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
