//===------ DumpModulePass.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Write a module to a file.
//
//===----------------------------------------------------------------------===//

#include "polly/Support/DumpModulePass.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"

#define DEBUG_TYPE "polly-dump-module"

using namespace llvm;
using namespace polly;

namespace {

static void runDumpModule(llvm::Module &M, StringRef Filename, bool IsSuffix) {
  std::string Dumpfile;
  if (IsSuffix) {
    StringRef ModuleName = M.getName();
    StringRef Stem = sys::path::stem(ModuleName);
    Dumpfile = (Twine(Stem) + Filename + ".ll").str();
  } else {
    Dumpfile = Filename.str();
  }
  LLVM_DEBUG(dbgs() << "Dumping module to " << Dumpfile << '\n');

  std::unique_ptr<ToolOutputFile> Out;
  std::error_code EC;
  Out.reset(new ToolOutputFile(Dumpfile, EC, sys::fs::OF_None));
  if (EC) {
    errs() << EC.message() << '\n';
    return;
  }

  M.print(Out->os(), nullptr);
  Out->keep();
}
} // namespace

llvm::PreservedAnalyses DumpModulePass::run(llvm::Module &M,
                                            llvm::ModuleAnalysisManager &AM) {
  runDumpModule(M, Filename, IsSuffix);
  return PreservedAnalyses::all();
}
