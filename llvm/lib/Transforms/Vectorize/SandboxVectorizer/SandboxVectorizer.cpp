//===- SandboxVectorizer.cpp - Vectorizer based on Sandbox IR -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/SandboxVectorizer.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/SandboxIR/SandboxIR.h"

using namespace llvm;

#define SV_NAME "sandbox-vectorizer"
#define DEBUG_TYPE SV_NAME

PreservedAnalyses SandboxVectorizerPass::run(Function &F,
                                             FunctionAnalysisManager &AM) {
  TTI = &AM.getResult<TargetIRAnalysis>(F);

  bool Changed = runImpl(F);
  if (!Changed)
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

bool SandboxVectorizerPass::runImpl(Function &F) {
  LLVM_DEBUG(dbgs() << "SBVec: Analyzing " << F.getName() << ".\n");
  sandboxir::Context Ctx(F.getContext());
  // Create SandboxIR for `F`.
  sandboxir::Function &SBF = *Ctx.createFunction(&F);
  // TODO: Initialize SBVec Pass Manager
  (void)SBF;

  return false;
}
