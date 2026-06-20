//===--- llvm-advisor.cpp - LLVM Advisor ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Entry point. Initializes LLVM target infrastructure and dispatches to the
// Client layer. The target registrations are required by the in-process MCA
// analyzer (MCDisassembler + mca::Context) and by ClangAnalyzerUtils.
//
//===----------------------------------------------------------------------===//

#include "Client/CLI/CLIHandler.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"

int main(int argc, char **argv) {
  llvm::InitLLVM X(argc, argv);

  // Register all target backends, MC layers, and disassemblers so that
  // in-process analyzers (MCA, MCDisassembler, ClangCodeGen) can locate
  // any target by its triple at runtime.
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllDisassemblers();

  llvm::advisor::CLIHandler CLI;
  return CLI.run(argc, argv);
}
