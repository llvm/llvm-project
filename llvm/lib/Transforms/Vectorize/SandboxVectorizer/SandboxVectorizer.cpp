//===- SandboxVectorizer.cpp - Vectorizer based on Sandbox IR -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/SandboxVectorizer.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/SandboxIR/Constant.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/SandboxVectorizerPassBuilder.h"

using namespace llvm;

#define SV_NAME "sandbox-vectorizer"
#define DEBUG_TYPE SV_NAME

static cl::opt<bool>
    PrintPassPipeline("sbvec-print-pass-pipeline", cl::init(false), cl::Hidden,
                      cl::desc("Prints the pass pipeline and returns."));

/// A magic string for the default pass pipeline.
static const char *DefaultPipelineMagicStr = "*";

static cl::opt<std::string> UserDefinedPassPipeline(
    "sbvec-passes", cl::init(DefaultPipelineMagicStr), cl::Hidden,
    cl::desc("Comma-separated list of vectorizer passes. If not set "
             "we run the predefined pipeline."));

SandboxVectorizerPass::SandboxVectorizerPass() : FPM("fpm") {
  if (UserDefinedPassPipeline == DefaultPipelineMagicStr) {
    // TODO: Add region passes to the default pipeline.
    FPM.setPassPipeline(
        "bottom-up-vec<>",
        sandboxir::SandboxVectorizerPassBuilder::createFunctionPass);
  } else {
    // Create the user-defined pipeline.
    FPM.setPassPipeline(
        UserDefinedPassPipeline,
        sandboxir::SandboxVectorizerPassBuilder::createFunctionPass);
  }
}

SandboxVectorizerPass::SandboxVectorizerPass(SandboxVectorizerPass &&) =
    default;

SandboxVectorizerPass::~SandboxVectorizerPass() = default;

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
  if (PrintPassPipeline) {
    FPM.printPipeline(outs());
    return false;
  }

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

  // Create SandboxIR for LLVMF and run BottomUpVec on it.
  sandboxir::Context Ctx(LLVMF.getContext());
  sandboxir::Function &F = *Ctx.createFunction(&LLVMF);
  return FPM.runOnFunction(F);
}
