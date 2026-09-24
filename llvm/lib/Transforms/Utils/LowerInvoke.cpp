//===- LowerInvoke.cpp - Eliminate Invoke instructions --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation is designed for use by code generators which do not yet
// support stack unwinding.  This pass converts 'invoke' instructions to 'call'
// instructions, so that any exception-handling 'landingpad' blocks become dead
// code (which can be removed by running the '-simplifycfg' pass afterwards).
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/LowerInvoke.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Instructions.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/Local.h"
using namespace llvm;

#define DEBUG_TYPE "lower-invoke"

STATISTIC(NumInvokes, "Number of invokes replaced");

namespace {
class LowerInvokeLegacyPass : public FunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid
  explicit LowerInvokeLegacyPass() : FunctionPass(ID) {
    initializeLowerInvokeLegacyPassPass(*PassRegistry::getPassRegistry());
  }
  bool runOnFunction(Function &F) override;
};
} // namespace

char LowerInvokeLegacyPass::ID = 0;
INITIALIZE_PASS(LowerInvokeLegacyPass, "lowerinvoke",
                "Lower invoke and unwind, for unwindless code generators",
                false, false)

static bool runImpl(Function &F) {
  bool Changed = false;
  for (BasicBlock &BB : F)
    if (InvokeInst *II = dyn_cast<InvokeInst>(BB.getTerminator())) {
      CallInst *NewCall = createCallMatchingInvoke(II);
      NewCall->takeName(II);
      NewCall->insertBefore(II->getIterator());
      II->replaceAllUsesWith(NewCall);

      // Insert an unconditional branch to the normal destination.
      UncondBrInst::Create(II->getNormalDest(), II->getIterator());

      // Remove any PHI node entries from the exception destination.
      II->getUnwindDest()->removePredecessor(&BB);

      // Remove the invoke instruction now.
      II->eraseFromParent();

      ++NumInvokes;
      Changed = true;
    }
  return Changed;
}

bool LowerInvokeLegacyPass::runOnFunction(Function &F) {
  return runImpl(F);
}

char &llvm::LowerInvokePassID = LowerInvokeLegacyPass::ID;

// Public Interface To the LowerInvoke pass.
FunctionPass *llvm::createLowerInvokePass() {
  return new LowerInvokeLegacyPass();
}

PreservedAnalyses LowerInvokePass::run(Function &F,
                                       FunctionAnalysisManager &AM) {
  bool Changed = runImpl(F);
  if (!Changed)
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}
