//===- mlir-cat.cpp ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <string>
#include <utility>

using namespace mlir;

/// This example parse its input, either from a file or its standard input (in
/// bytecode or textual assembly) and print it back.

int main(int argc, char **argv) {
  // Set up the input file.
  StringRef inputFile;
  if (argc <= 1) {
    llvm::errs() << "Reading from stdin...\n";
    inputFile = "-";
  } else {
    inputFile = argv[1];
  }
  std::string errorMessage;
  auto file = openInputFile(inputFile, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());

  auto output = openOutputFile("-", &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }

  DialectRegistry registry;
  MLIRContext context(registry, MLIRContext::Threading::DISABLED);
  context.allowUnregisteredDialects(true);
  OwningOpRef<Operation *> op = parseSourceFile(sourceMgr, &context);
  if (!op) {
    llvm::errs() << "Failed to parse input file";
    exit(1);
  }
  output->os() << *(op.get()) << "\n";
}
