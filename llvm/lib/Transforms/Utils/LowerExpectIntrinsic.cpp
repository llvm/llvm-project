//===- LowerExpectIntrinsic.cpp - Lower expect intrinsic ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass lowers the 'expect' intrinsic to LLVM metadata.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "lower-expect-intrinsic"
#include "llvm/BasicBlock.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/LLVMContext.h"
#include "llvm/MDBuilder.h"
#include "llvm/Metadata.h"
#include "llvm/Pass.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include <vector>

using namespace llvm;

STATISTIC(IfHandled, "Number of 'expect' intrinsic intructions handled");

static cl::opt<uint32_t>
LikelyBranchWeight("likely-branch-weight", cl::Hidden, cl::init(64),
                   cl::desc("Weight of the branch likely to be taken (default = 64)"));
static cl::opt<uint32_t>
UnlikelyBranchWeight("unlikely-branch-weight", cl::Hidden, cl::init(4),
                   cl::desc("Weight of the branch unlikely to be taken (default = 4)"));

namespace {

  class LowerExpectIntrinsic : public FunctionPass {

    bool HandleSwitchExpect(SwitchInst *SI);

    bool HandleIfExpect(BranchInst *BI);

  public:
    static char ID;
    LowerExpectIntrinsic() : FunctionPass(ID) {
      initializeLowerExpectIntrinsicPass(*PassRegistry::getPassRegistry());
    }

    bool runOnFunction(Function &F);
  };
}


bool LowerExpectIntrinsic::HandleSwitchExpect(SwitchInst *SI) {
  CallInst *CI = dyn_cast<CallInst>(SI->getCondition());
  if (!CI)
    return false;

  Function *Fn = CI->getCalledFunction();
  if (!Fn || Fn->getIntrinsicID() != Intrinsic::expect)
    return false;

  Value *ArgValue = CI->getArgOperand(0);
  ConstantInt *ExpectedValue = dyn_cast<ConstantInt>(CI->getArgOperand(1));
  if (!ExpectedValue)
    return false;

  SwitchInst::CaseIt Case = SI->findCaseValue(ExpectedValue);
  unsigned n = SI->getNumCases(); // +1 for default case.
  std::vector<uint32_t> Weights(n + 1);

  Weights[0] = Case == SI->case_default() ? LikelyBranchWeight
                                          : UnlikelyBranchWeight;
  for (unsigned i = 0; i != n; ++i)
    Weights[i + 1] = i == Case.getCaseIndex() ? LikelyBranchWeight
                                              : UnlikelyBranchWeight;

  SI->setMetadata(LLVMContext::MD_prof,
                  MDBuilder(CI->getContext()).createBranchWeights(Weights));

  SI->setCondition(ArgValue);
  return true;
}


bool LowerExpectIntrinsic::HandleIfExpect(BranchInst *BI) {
  if (BI->isUnconditional())
    return false;

  // Handle non-optimized IR code like:
  //   %expval = call i64 @llvm.expect.i64.i64(i64 %conv1, i64 1)
  //   %tobool = icmp ne i64 %expval, 0
  //   br i1 %tobool, label %if.then, label %if.end

  ICmpInst *CmpI = dyn_cast<ICmpInst>(BI->getCondition());
  if (!CmpI || CmpI->getPredicate() != CmpInst::ICMP_NE)
    return false;

  CallInst *CI = dyn_cast<CallInst>(CmpI->getOperand(0));
  if (!CI)
    return false;

  Function *Fn = CI->getCalledFunction();
  if (!Fn || Fn->getIntrinsicID() != Intrinsic::expect)
    return false;

  Value *ArgValue = CI->getArgOperand(0);
  ConstantInt *ExpectedValue = dyn_cast<ConstantInt>(CI->getArgOperand(1));
  if (!ExpectedValue)
    return false;

  MDBuilder MDB(CI->getContext());
  MDNode *Node;

  // If expect value is equal to 1 it means that we are more likely to take
  // branch 0, in other case more likely is branch 1.
  if (ExpectedValue->isOne())
    Node = MDB.createBranchWeights(LikelyBranchWeight, UnlikelyBranchWeight);
  else
    Node = MDB.createBranchWeights(UnlikelyBranchWeight, LikelyBranchWeight);

  BI->setMetadata(LLVMContext::MD_prof, Node);

  CmpI->setOperand(0, ArgValue);
  return true;
}


bool LowerExpectIntrinsic::runOnFunction(Function &F) {
  for (Function::iterator I = F.begin(), E = F.end(); I != E;) {
    BasicBlock *BB = I++;

    // Create "block_weights" metadata.
    if (BranchInst *BI = dyn_cast<BranchInst>(BB->getTerminator())) {
      if (HandleIfExpect(BI))
        IfHandled++;
    } else if (SwitchInst *SI = dyn_cast<SwitchInst>(BB->getTerminator())) {
      if (HandleSwitchExpect(SI))
        IfHandled++;
    }

    // remove llvm.expect intrinsics.
    for (BasicBlock::iterator BI = BB->begin(), BE = BB->end();
         BI != BE; ) {
      CallInst *CI = dyn_cast<CallInst>(BI++);
      if (!CI)
        continue;

      Function *Fn = CI->getCalledFunction();
      if (Fn && Fn->getIntrinsicID() == Intrinsic::expect) {
        Value *Exp = CI->getArgOperand(0);
        CI->replaceAllUsesWith(Exp);
        CI->eraseFromParent();
      }
    }
  }

  return false;
}


char LowerExpectIntrinsic::ID = 0;
INITIALIZE_PASS(LowerExpectIntrinsic, "lower-expect", "Lower 'expect' "
                "Intrinsics", false, false)

FunctionPass *llvm::createLowerExpectIntrinsicPass() {
  return new LowerExpectIntrinsic();
}
