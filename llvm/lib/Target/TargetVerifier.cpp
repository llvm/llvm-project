//===-- TargetVerifier.cpp - LLVM IR Target Verifier ----------------*- C++ -*-===//
////
///// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
///// See https://llvm.org/LICENSE.txt for license information.
///// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
/////
/////===----------------------------------------------------------------------===//
/////
///// This file defines target verifier interfaces that can be used for some
///// validation of input to the system, and for checking that transformations
///// haven't done something bad. In contrast to the Verifier or Lint, the
///// TargetVerifier looks for constructions invalid to a particular target
///// machine.
/////
///// To see what specifically is checked, look at TargetVerifier.cpp or an
///// individual backend's TargetVerifier.
/////
/////===----------------------------------------------------------------------===//

#include "llvm/Target/TargetVerifier.h"
#include "llvm/Target/TargetVerify/AMDGPUTargetVerifier.h"

#include "llvm/InitializePasses.h"
#include "llvm/Analysis/UniformityAnalysis.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Support/Debug.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"

namespace llvm {

bool TargetVerify::run(Function &F) {
  if (TT.isAMDGPU()) {
    AMDGPUTargetVerify TV(Mod);
    TV.run(F);

    dbgs() << TV.MessagesStr.str();
    if (!TV.MessagesStr.str().empty()) {
      TV.IsValid = false;
      return false;
    }
    return true;
  }
  report_fatal_error("Target has no verification method\n");
}

bool TargetVerify::run(Function &F, FunctionAnalysisManager &AM) {
  if (TT.isAMDGPU()) {
    auto *UA = &AM.getResult<UniformityInfoAnalysis>(F);
    auto *DT = &AM.getResult<DominatorTreeAnalysis>(F);
    auto *PDT = &AM.getResult<PostDominatorTreeAnalysis>(F);

    AMDGPUTargetVerify TV(Mod, DT, PDT, UA);
    TV.run(F);

    dbgs() << TV.MessagesStr.str();
    if (!TV.MessagesStr.str().empty()) {
      TV.IsValid = false;
      return false;
    }
    return true;
  }
  report_fatal_error("Target has no verification method\n");
}

PreservedAnalyses TargetVerifierPass::run(Function &F, FunctionAnalysisManager &AM) {
  auto TT = F.getParent()->getTargetTriple();

  if (TT.isAMDGPU()) {
    auto *Mod = F.getParent();

    auto UA = &AM.getResult<UniformityInfoAnalysis>(F);
    auto *DT = &AM.getResult<DominatorTreeAnalysis>(F);
    auto *PDT = &AM.getResult<PostDominatorTreeAnalysis>(F);

    AMDGPUTargetVerify TV(Mod, DT, PDT, UA);
    TV.run(F);

    dbgs() << TV.MessagesStr.str();
    if (!TV.MessagesStr.str().empty()) {
      TV.IsValid = false;
      return PreservedAnalyses::none();
    }
    return PreservedAnalyses::all();
  }
  report_fatal_error("Target has no verification method\n");
}

struct TargetVerifierLegacyPass : public FunctionPass {
  static char ID;

  std::unique_ptr<TargetVerify> TV;
  bool FatalErrors = false;

  TargetVerifierLegacyPass(bool FatalErrors) : FunctionPass(ID),
    FatalErrors(FatalErrors) {
    initializeTargetVerifierLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool doInitialization(Module &M) override {
    TV = std::make_unique<TargetVerify>(&M);
    return false;
  }

  bool runOnFunction(Function &F) override {
    if (!TV->run(F)) {
      errs() << "in function " << F.getName() << '\n';
      if (FatalErrors)
        report_fatal_error("broken function found, compilation aborted!");
      else
        errs() << "broken function found, compilation aborted!\n";
    }
    return false;
  }

  bool doFinalization(Module &M) override {
    bool IsValid = true;
    for (Function &F : M)
      if (F.isDeclaration())
        IsValid &= TV->run(F);

    if (!IsValid)
      if (FatalErrors)
        report_fatal_error("broken module found, compilation aborted!");
      else
        errs() << "broken module found, compilation aborted!\n";
    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }
};
char TargetVerifierLegacyPass::ID = 0;
FunctionPass *createTargetVerifierLegacyPass(bool FatalErrors) {
  return new TargetVerifierLegacyPass(FatalErrors);
}
} // namespace llvm
using namespace llvm;
INITIALIZE_PASS(TargetVerifierLegacyPass, "tgtverifier", "Target Verifier", false, false)
