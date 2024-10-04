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
#include "llvm/SandboxIR/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Passes/BottomUpVec.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Passes/NullPass.h"

using namespace llvm;

#define SV_NAME "sandbox-vectorizer"
#define DEBUG_TYPE SV_NAME

cl::opt<bool>
    PrintPassPipeline("sbvec-print-pass-pipeline", cl::init(false), cl::Hidden,
                      cl::desc("Prints the pass pipeline and returns."));

/// A magic string for the default pass pipeline.
const char *DefaultPipelineMagicStr = "*";

cl::opt<std::string> UserDefinedPassPipeline(
    "sbvec-passes", cl::init(DefaultPipelineMagicStr), cl::Hidden,
    cl::desc("Comma-separated list of vectorizer passes. If not set "
             "we run the predefined pipeline."));

static void registerAllRegionPasses(sandboxir::PassRegistry &PR) {
  PR.registerPass(std::make_unique<sandboxir::NullPass>());
}

static sandboxir::RegionPassManager &
parseAndCreatePassPipeline(sandboxir::PassRegistry &PR, StringRef Pipeline) {
  static constexpr const char EndToken = '\0';
  // Add EndToken to the end to ease parsing.
  std::string PipelineStr = std::string(Pipeline) + EndToken;
  int FlagBeginIdx = 0;
  auto &RPM = static_cast<sandboxir::RegionPassManager &>(
      PR.registerPass(std::make_unique<sandboxir::RegionPassManager>("rpm")));

  for (auto [Idx, C] : enumerate(PipelineStr)) {
    // Keep moving Idx until we find the end of the pass name.
    bool FoundDelim = C == EndToken || C == PR.PassDelimToken;
    if (!FoundDelim)
      continue;
    unsigned Sz = Idx - FlagBeginIdx;
    std::string PassName(&PipelineStr[FlagBeginIdx], Sz);
    FlagBeginIdx = Idx + 1;

    // Get the pass that corresponds to PassName and add it to the pass manager.
    auto *Pass = PR.getPassByName(PassName);
    if (Pass == nullptr) {
      errs() << "Pass '" << PassName << "' not registered!\n";
      exit(1);
    }
    // TODO: Add a type check here. The downcast is correct as long as
    // registerAllRegionPasses only registers regions passes.
    RPM.addPass(static_cast<sandboxir::RegionPass *>(Pass));
  }
  return RPM;
}

SandboxVectorizerPass::SandboxVectorizerPass() {
  registerAllRegionPasses(PR);

  // Create a pipeline to be run on each Region created by BottomUpVec.
  if (UserDefinedPassPipeline == DefaultPipelineMagicStr) {
    // Create the default pass pipeline.
    RPM = &static_cast<sandboxir::RegionPassManager &>(PR.registerPass(
        std::make_unique<sandboxir::FunctionPassManager>("rpm")));
    // TODO: Add passes to the default pipeline.
  } else {
    // Create the user-defined pipeline.
    RPM = &parseAndCreatePassPipeline(PR, UserDefinedPassPipeline);
  }
  BottomUpVecPass = std::make_unique<sandboxir::BottomUpVec>(RPM);
}

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

  if (PrintPassPipeline) {
    RPM->printPipeline(outs());
    return false;
  }

  // Create SandboxIR for LLVMF and run BottomUpVec on it.
  sandboxir::Context Ctx(LLVMF.getContext());
  sandboxir::Function &F = *Ctx.createFunction(&LLVMF);
  return BottomUpVecPass->runOnFunction(F);
}
