//===- LowerGuardIntrinsic.cpp - Lower the guard intrinsic ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass lowers the llvm.experimental.guard intrinsic to a conditional call
// to @llvm.experimental.deoptimize.  Once this happens, the guard can no longer
// be widened.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

namespace {
struct LowerGuardIntrinsic : public FunctionPass {
  static char ID;
  LowerGuardIntrinsic() : FunctionPass(ID) {
    initializeLowerGuardIntrinsicPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override;
};
}

static void MakeGuardControlFlowExplicit(Function *DeoptIntrinsic,
                                         CallInst *CI) {
  OperandBundleDef DeoptOB(*CI->getOperandBundle(LLVMContext::OB_deopt));
  SmallVector<Value *, 4> Args(std::next(CI->arg_begin()), CI->arg_end());

  auto *CheckBB = CI->getParent();
  auto *DeoptBlockTerm =
      SplitBlockAndInsertIfThen(CI->getArgOperand(0), CI, true);

  auto *CheckBI = cast<BranchInst>(CheckBB->getTerminator());

  // SplitBlockAndInsertIfThen inserts control flow that branches to
  // DeoptBlockTerm if the condition is true.  We want the opposite.
  CheckBI->swapSuccessors();

  CheckBI->getSuccessor(0)->setName("guarded");
  CheckBI->getSuccessor(1)->setName("deopt");

  IRBuilder<> B(DeoptBlockTerm);
  auto *DeoptCall = B.CreateCall(DeoptIntrinsic, Args, {DeoptOB}, "");

  if (DeoptIntrinsic->getReturnType()->isVoidTy()) {
    B.CreateRetVoid();
  } else {
    DeoptCall->setName("deoptcall");
    B.CreateRet(DeoptCall);
  }

  DeoptBlockTerm->eraseFromParent();
}

bool LowerGuardIntrinsic::runOnFunction(Function &F) {
  // Check if we can cheaply rule out the possibility of not having any work to
  // do.
  auto *GuardDecl = F.getParent()->getFunction(
      Intrinsic::getName(Intrinsic::experimental_guard));
  if (!GuardDecl || GuardDecl->use_empty())
    return false;

  SmallVector<CallInst *, 8> ToLower;
  for (auto &I : instructions(F))
    if (auto *CI = dyn_cast<CallInst>(&I))
      if (auto *F = CI->getCalledFunction())
        if (F->getIntrinsicID() == Intrinsic::experimental_guard)
          ToLower.push_back(CI);

  if (ToLower.empty())
    return false;

  auto *DeoptIntrinsic = Intrinsic::getDeclaration(
      F.getParent(), Intrinsic::experimental_deoptimize, {F.getReturnType()});

  for (auto *CI : ToLower) {
    MakeGuardControlFlowExplicit(DeoptIntrinsic, CI);
    CI->eraseFromParent();
  }

  return true;
}

char LowerGuardIntrinsic::ID = 0;
INITIALIZE_PASS(LowerGuardIntrinsic, "lower-guard-intrinsic",
                "Lower the guard intrinsic to normal control flow", false,
                false)

Pass *llvm::createLowerGuardIntrinsicPass() {
  return new LowerGuardIntrinsic();
}
