//===- FunctionDDGPrinter.cpp - Function-scoped DDG DOT printer -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the `-dot-function-ddg` pass, the function-scoped analogue
// of the loop-scoped `-dot-ddg` printer. It builds a DataDependenceGraph for a
// whole function and emits it in DOT format to a file named
// `<prefix>.<function-name>.dot`, reusing the existing DDG DOTGraphTraits so
// loop- and function-scope graphs render identically.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/FunctionDDGPrinter.h"
#include "llvm/Analysis/DDG.h"
#include "llvm/Analysis/DDGPrinter.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::opt<bool>
    FunctionDDGDotOnly("dot-function-ddg-only", cl::Hidden,
                       cl::desc("Print a simplified function-scoped DDG dot "
                                "graph (like -dot-ddg-only at loop scope)."));

static cl::opt<std::string> FunctionDDGDotFilenamePrefix(
    "dot-function-ddg-filename-prefix", cl::init("ddg"), cl::Hidden,
    cl::desc("The prefix used for the function DDG dot file names."));

PreservedAnalyses FunctionDDGDotPrinterPass::run(Function &F,
                                                 FunctionAnalysisManager &AM) {
  auto &DI = AM.getResult<DependenceAnalysis>(F);
  DataDependenceGraph G(F, DI);

  std::string Filename =
      (FunctionDDGDotFilenamePrefix + "." + F.getName() + ".dot").str();
  errs() << "Writing '" << Filename << "'...";

  std::error_code EC;
  raw_fd_ostream File(Filename, EC, sys::fs::OF_Text);
  if (!EC)
    // Only the const DOTGraphTraits specialization is provided, hence the
    // conversion to a const pointer.
    WriteGraph(File, (const DataDependenceGraph *)&G, FunctionDDGDotOnly);
  else
    errs() << "  error opening file for writing!";
  errs() << "\n";

  return PreservedAnalyses::all();
}
