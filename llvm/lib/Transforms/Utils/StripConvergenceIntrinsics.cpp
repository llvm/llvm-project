//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass strips convergence intrinsics and convergencectrl operand bundles,
// as those are only useful when modifying the CFG during IR passes.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/StripConvergenceIntrinsics.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils.h"

using namespace llvm;

static bool stripConvergenceIntrinsics(Function &F) {
  SmallVector<IntrinsicInst *> ConvergenceIntrinsics;
  bool Changed = false;

  for (BasicBlock &BB : F) {
    for (Instruction &I : make_early_inc_range(BB)) {
      auto *CI = dyn_cast<CallInst>(&I);
      if (!CI)
        continue;

      // Strip a convergencectrl operand bundle if present. Note that
      // convergence intrinsics (e.g. convergence.loop) may use a
      // convergencectrl bundle.
      if (CI->getOperandBundle(LLVMContext::OB_convergencectrl)) {
        auto *NewCall = CallBase::removeOperandBundle(
            CI, LLVMContext::OB_convergencectrl, CI->getIterator());
        NewCall->copyMetadata(*CI);
        CI->replaceAllUsesWith(NewCall);
        CI->eraseFromParent();
        CI = cast<CallInst>(NewCall);
        Changed = true;
      }

      // Collect convergence intrinsics for deferred removal.
      if (auto *II = dyn_cast<IntrinsicInst>(CI))
        if (II->getIntrinsicID() == Intrinsic::experimental_convergence_entry ||
            II->getIntrinsicID() == Intrinsic::experimental_convergence_loop ||
            II->getIntrinsicID() == Intrinsic::experimental_convergence_anchor)
          ConvergenceIntrinsics.push_back(II);
    }
  }

  // Erase all convergence intrinsics now that convergence tokens are no
  // longer in use.
  for (IntrinsicInst *II : ConvergenceIntrinsics)
    II->eraseFromParent();
  Changed |= !ConvergenceIntrinsics.empty();

  return Changed;
}

PreservedAnalyses
StripConvergenceIntrinsicsPass::run(Function &F, FunctionAnalysisManager &) {
  if (!stripConvergenceIntrinsics(F))
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

namespace {
class StripConvergenceIntrinsicsLegacyPass : public FunctionPass {
public:
  static char ID;

  StripConvergenceIntrinsicsLegacyPass() : FunctionPass(ID) {
    initializeStripConvergenceIntrinsicsLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    return stripConvergenceIntrinsics(F);
  }
};
} // namespace

char StripConvergenceIntrinsicsLegacyPass::ID = 0;
INITIALIZE_PASS(StripConvergenceIntrinsicsLegacyPass,
                "strip-convergence-intrinsics",
                "Strip convergence intrinsics and operand bundles", false,
                false)

FunctionPass *llvm::createStripConvergenceIntrinsicsPass() {
  return new StripConvergenceIntrinsicsLegacyPass();
}
