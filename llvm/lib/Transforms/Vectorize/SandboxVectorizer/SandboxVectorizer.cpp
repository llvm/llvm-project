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
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Passes/BottomUpVec.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Passes/NullPass.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Passes/PrintInstructionCountPass.h"

using namespace llvm::sandboxir;

namespace llvm {

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

static cl::opt<bool> UseRegionsFromMetadata(
    "sbvec-use-regions-from-metadata", cl::init(false), cl::Hidden,
    cl::desc("Skips bottom-up vectorization, builds regions from metadata "
             "already present in the IR and runs the region pass pipeline."));

static std::unique_ptr<sandboxir::RegionPass> createRegionPass(StringRef Name) {
#define REGION_PASS(NAME, CREATE_PASS)                                         \
  if (Name == NAME)                                                            \
    return std::make_unique<decltype(CREATE_PASS)>(CREATE_PASS);
#include "Passes/PassRegistry.def"
  return nullptr;
}

sandboxir::RegionPassManager createRegionPassManager() {
  sandboxir::RegionPassManager RPM("rpm");
  // Create a pipeline to be run on each Region created by BottomUpVec.
  if (UserDefinedPassPipeline == DefaultPipelineMagicStr) {
    // TODO: Add default passes to RPM.
  } else {
    // Create the user-defined pipeline.
    RPM.setPassPipeline(UserDefinedPassPipeline, createRegionPass);
  }
  return RPM;
}

SandboxVectorizerPass::SandboxVectorizerPass()
    : RPM(createRegionPassManager()), BottomUpVecPass(&RPM) {}

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
    RPM.printPipeline(outs());
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
  if (UseRegionsFromMetadata) {
    SmallVector<std::unique_ptr<sandboxir::Region>> Regions =
        sandboxir::Region::createRegionsFromMD(F);
    for (auto &R : Regions) {
      RPM.runOnRegion(*R);
    }
    return false;
  } else {
    return BottomUpVecPass.runOnFunction(F);
  }
}

} // namespace llvm
