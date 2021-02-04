//===-- HelloWorld.cpp - Example Transformations --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/HelloNew/HelloWorld.h"

using namespace llvm;

PreservedAnalyses HelloWorldPass::run(Function &F,
                                      FunctionAnalysisManager &AM) {
  llvm::errs() << "Visiting function " << F.getName() << "\n";

  for (llvm::BasicBlock &BB : F) {
    llvm::errs() << "Visiting bb " << B.getName() << "\n";
    for (llvm::Instruction &II : BB) {
      llvm::errs() << "Visiting II " << II.getName() << "\n";
      llvm::Instruction *I = &II;
      if (llvm::CallInst *CI = llvm::dyn_cast<llvm::CallInst>(I)) {
        llvm::errs() << "Visiting CI " << CI->getName() << "\n";
        if (llvm::DILocation *Loc = I->getDebugLoc()) {
          unsigned Line = Loc->getLine();
          llvm::StringRef File = Loc->getFilename();
          llvm::StringRef Dir = Loc->getDirectory();
          llvm::errs() << Dir << "/" << File << ":" << Line << "\n";
        }
      }
    }
  }
  return PreservedAnalyses::all();
}
