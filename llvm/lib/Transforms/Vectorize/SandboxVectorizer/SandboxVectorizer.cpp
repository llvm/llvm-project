//===- SandboxVectorizer.cpp - Vectorizer based on Sandbox IR -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/SandboxVectorizer.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/SandboxIR/PassManager.h"
#include "llvm/SandboxIR/SandboxIR.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Passes/BottomUpVec.h"

using namespace llvm;

#define SV_NAME "sandbox-vectorizer"
#define DEBUG_TYPE SV_NAME

cl::opt<bool>
    PrintPassPipeline("sbvec-print-pass-pipeline", cl::init(false), cl::Hidden,
                      cl::desc("Prints the pass pipeline and returns."));

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

bool SandboxVectorizerPass::runImpl(Function &LLVMF) {
  // If the target claims to have no vector registers early return.
  if (!TTI->getNumberOfRegisters(TTI->getRegisterClassForType(true))) {
    LLVM_DEBUG(dbgs() << "SBVec: Target has no vector registers, return.\n");
    return false;
  }
  LLVM_DEBUG(dbgs() << "SBVec: Analyzing " << LLVMF.getName() << ".\n");
  // Early return if the attribute NoImplicitFloat is used.
  if (LLVMF.hasFnAttribute(Attribute::NoImplicitFloat)) {
    LLVM_DEBUG(dbgs() << "SBVec: NoImplicitFloat attribute, return.\n");
    return false;
  }

  sandboxir::Context Ctx(LLVMF.getContext());
  // Create SandboxIR for `LLVMF`.
  sandboxir::Function &F = *Ctx.createFunction(&LLVMF);
  // Create the passes and register them with the PassRegistry.
  sandboxir::PassRegistry PR;
  auto &PM = static_cast<sandboxir::FunctionPassManager &>(
      PR.registerPass(std::make_unique<sandboxir::FunctionPassManager>("pm")));
  auto &BottomUpVecPass = static_cast<sandboxir::FunctionPass &>(
      PR.registerPass(std::make_unique<sandboxir::BottomUpVec>()));

  // Create the default pass pipeline.
  PM.addPass(&BottomUpVecPass);

  if (PrintPassPipeline) {
    PM.printPipeline(outs());
    return false;
  }

  // Run the pass pipeline.
  bool Change = PM.runOnFunction(F);
  return Change;
}
