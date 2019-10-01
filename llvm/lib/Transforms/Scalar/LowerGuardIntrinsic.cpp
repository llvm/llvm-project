//===- LowerGuardIntrinsic.cpp - Lower the guard intrinsic ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers the llvm.experimental.guard intrinsic to a conditional call
// to @llvm.experimental.deoptimize.  Once this happens, the guard can no longer
// be widened.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/LowerGuardIntrinsic.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/GuardUtils.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/GuardUtils.h"

using namespace llvm;

namespace {
struct LowerGuardIntrinsicLegacyPass : public FunctionPass {
  static char ID;
  LowerGuardIntrinsicLegacyPass() : FunctionPass(ID) {
    initializeLowerGuardIntrinsicLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override;
};
}

static bool lowerGuardIntrinsic(Function &F) {
  // Check if we can cheaply rule out the possibility of not having any work to
  // do.
  auto *GuardDecl = F.getParent()->getFunction(
      Intrinsic::getName(Intrinsic::experimental_guard));
  if (!GuardDecl || GuardDecl->use_empty())
    return false;

  SmallVector<CallInst *, 8> ToLower;
  for (auto &I : instructions(F))
    if (isGuard(&I))
      ToLower.push_back(cast<CallInst>(&I));

  if (ToLower.empty())
    return false;

  auto *DeoptIntrinsic = Intrinsic::getDeclaration(
      F.getParent(), Intrinsic::experimental_deoptimize, {F.getReturnType()});
  DeoptIntrinsic->setCallingConv(GuardDecl->getCallingConv());

  for (auto *CI : ToLower) {
    makeGuardControlFlowExplicit(DeoptIntrinsic, CI);
    CI->eraseFromParent();
  }

  return true;
}

bool LowerGuardIntrinsicLegacyPass::runOnFunction(Function &F) {
  return lowerGuardIntrinsic(F);
}

char LowerGuardIntrinsicLegacyPass::ID = 0;
INITIALIZE_PASS(LowerGuardIntrinsicLegacyPass, "lower-guard-intrinsic",
                "Lower the guard intrinsic to normal control flow", false,
                false)

Pass *llvm::createLowerGuardIntrinsicPass() {
  return new LowerGuardIntrinsicLegacyPass();
}

PreservedAnalyses LowerGuardIntrinsicPass::run(Function &F,
                                               FunctionAnalysisManager &AM) {
  if (lowerGuardIntrinsic(F))
    return PreservedAnalyses::none();

  return PreservedAnalyses::all();
}
