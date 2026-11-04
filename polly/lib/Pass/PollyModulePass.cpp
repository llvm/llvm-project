//===------ PollyModulePass.cpp - Polly module pass  ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "polly/Pass/PollyModulePass.h"
#include "llvm/IR/Module.h"

using namespace llvm;
using namespace polly;

PreservedAnalyses PollyModulePass::run(llvm::Module &M,
                                       llvm::ModuleAnalysisManager &MAM) {
  FunctionAnalysisManager &FAM =
      MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

  bool ModifiedAnyIR = false;
  for (Function &F : M) {
    bool LocalModifiedIR = runPollyPass(F, FAM, Opts);
    ModifiedAnyIR |= LocalModifiedIR;
  }

  // Be conservative about preserved analyses, especially if parallel functions
  // have been outlined.
  return ModifiedAnyIR ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
