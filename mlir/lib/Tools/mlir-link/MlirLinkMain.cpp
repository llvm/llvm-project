//===- MlirLinkMain.cpp - MLIR Link main ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/mlir-link/MlirLinkMain.h"
#include "mlir/IR/Dialect.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/WithColor.h"

using namespace mlir;
using namespace llvm;

LogicalResult mlir::MlirLinkMain(int argc, char **argv,
                                 DialectRegistry &registry) {
  static cl::OptionCategory linkCategory("Link options");

  static cl::list<std::string> inputFilenames(cl::Positional, cl::OneOrMore,
                                              cl::desc("<input mlir files>"),
                                              cl::cat(linkCategory));

  static cl::opt<std::string> outputFilename(
      "o", cl::desc("Override output filename"), cl::init("-"),
      cl::value_desc("filename"), cl::cat(linkCategory));

  static ExitOnError ExitOnErr;

  InitLLVM y(argc, argv);
  ExitOnErr.setBanner(std::string(argv[0]) + ": ");

  cl::HideUnrelatedOptions({&linkCategory, &getColorCategory()});
  cl::ParseCommandLineOptions(argc, argv, "mlir linker\n");

  return success();
}
