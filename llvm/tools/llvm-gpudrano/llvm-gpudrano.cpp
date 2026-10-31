//===- llvm-gpudrano.cpp - LLVM GPU program analysis utility ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This utility analyzes the given GPU program for uncoaleased accesses and
// block-size invariance property. These are used for performance debugging and
// optimization.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/UncoalescedAnalysisPass.h"
#include "llvm/Transforms/BlockSizeInvarianceAnalysisPass.h"
#include <iostream>
using namespace llvm;

// InputFilename - The filename to read from.
static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input LLVM IR file>"),
              cl::init("-"), cl::value_desc("filename"));

static cl::opt<std::string>
AnalysisName("a", cl::desc("Specify analysis name (one of `uncoaleased-accesses' or `block-size-invariance). \
			Default value: uncoalesced-accesses"),
               cl::value_desc("analysis-name"), cl::init("uncoalesced-accesses"));

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);

  LLVMContext Context;
  cl::ParseCommandLineOptions(argc, argv, "llvm gpudrano\n");

  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseIRFile(InputFilename, Err, Context);

  if (!M.get()) {
    Err.print(argv[0], errs());
    return 1;
  }

  // Run Analysis.
  legacy::PassManager Passes;
  Passes.add(new DominatorTreeWrapperPass());
  if (AnalysisName == "block-size-invariance") {
    std::cout << "Running block-size invariance analysis..." << std::endl;
    Passes.add(new BlockSizeInvarianceAnalysisPass());
  }
  else { // Default case.
    std::cout << "Running uncoalesced accesses analysis..." << std::endl;
    Passes.add(new UncoalescedAnalysisPass());
  }

  Passes.run(*M.get());

  return 0;
}
