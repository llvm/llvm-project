//===------ PollyFunctionPass.cpp - Polly function pass  ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "polly/Pass/PollyFunctionPass.h"

using namespace llvm;
using namespace polly;

PreservedAnalyses PollyFunctionPass::run(llvm::Function &F,
                                         llvm::FunctionAnalysisManager &FAM) {
  bool ModifiedIR = runPollyPass(F, FAM, Opts);

  // Be conservative about preserved analyses.
  // FIXME: May also need to invalidate/update Module/CGSCC passes, but cannot
  // reach them within a FunctionPassManager.
  return ModifiedIR ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
