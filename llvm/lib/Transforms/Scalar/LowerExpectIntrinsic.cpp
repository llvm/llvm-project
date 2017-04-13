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

#include "llvm/Transforms/Scalar/LowerExpectIntrinsic.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Scalar.h"

using namespace llvm;

#define DEBUG_TYPE "lower-expect-intrinsic"

STATISTIC(ExpectIntrinsicsHandled,
          "Number of 'expect' intrinsic instructions handled");

// These default values are chosen to represent an extremely skewed outcome for
// a condition, but they leave some room for interpretation by later passes.
//
// If the documentation for __builtin_expect() was made explicit that it should
// only be used in extreme cases, we could make this ratio higher. As it stands,
// programmers may be using __builtin_expect() / llvm.expect to annotate that a
// branch is likely or unlikely to be taken.
//
// There is a known dependency on this ratio in CodeGenPrepare when transforming
// 'select' instructions. It may be worthwhile to hoist these values to some
// shared space, so they can be used directly by other passes.

static cl::opt<uint32_t> LikelyBranchWeight(
    "likely-branch-weight", cl::Hidden, cl::init(2000),
    cl::desc("Weight of the branch likely to be taken (default = 2000)"));
static cl::opt<uint32_t> UnlikelyBranchWeight(
    "unlikely-branch-weight", cl::Hidden, cl::init(1),
    cl::desc("Weight of the branch unlikely to be taken (default = 1)"));

static bool handleSwitchExpect(SwitchInst &SI) {
  CallInst *CI = dyn_cast<CallInst>(SI.getCondition());
  if (!CI)
    return false;

  Function *Fn = CI->getCalledFunction();
  if (!Fn || Fn->getIntrinsicID() != Intrinsic::expect)
    return false;

  Value *ArgValue = CI->getArgOperand(0);
  ConstantInt *ExpectedValue = dyn_cast<ConstantInt>(CI->getArgOperand(1));
  if (!ExpectedValue)
    return false;

  SwitchInst::CaseHandle Case = *SI.findCaseValue(ExpectedValue);
  unsigned n = SI.getNumCases(); // +1 for default case.
  SmallVector<uint32_t, 16> Weights(n + 1, UnlikelyBranchWeight);

  if (Case == *SI.case_default())
    Weights[0] = LikelyBranchWeight;
  else
    Weights[Case.getCaseIndex() + 1] = LikelyBranchWeight;

  SI.setMetadata(LLVMContext::MD_prof,
                 MDBuilder(CI->getContext()).createBranchWeights(Weights));

  SI.setCondition(ArgValue);
  return true;
}

// Handle both BranchInst and SelectInst.
template <class BrSelInst> static bool handleBrSelExpect(BrSelInst &BSI) {

  // Handle non-optimized IR code like:
  //   %expval = call i64 @llvm.expect.i64(i64 %conv1, i64 1)
  //   %tobool = icmp ne i64 %expval, 0
  //   br i1 %tobool, label %if.then, label %if.end
  //
  // Or the following simpler case:
  //   %expval = call i1 @llvm.expect.i1(i1 %cmp, i1 1)
  //   br i1 %expval, label %if.then, label %if.end

  CallInst *CI;

  ICmpInst *CmpI = dyn_cast<ICmpInst>(BSI.getCondition());
  if (!CmpI) {
    CI = dyn_cast<CallInst>(BSI.getCondition());
  } else {
    if (CmpI->getPredicate() != CmpInst::ICMP_NE)
      return false;
    CI = dyn_cast<CallInst>(CmpI->getOperand(0));
  }

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

  BSI.setMetadata(LLVMContext::MD_prof, Node);

  if (CmpI)
    CmpI->setOperand(0, ArgValue);
  else
    BSI.setCondition(ArgValue);
  return true;
}

static bool handleBranchExpect(BranchInst &BI) {
  if (BI.isUnconditional())
    return false;

  return handleBrSelExpect<BranchInst>(BI);
}

static bool lowerExpectIntrinsic(Function &F) {
  bool Changed = false;

  for (BasicBlock &BB : F) {
    // Create "block_weights" metadata.
    if (BranchInst *BI = dyn_cast<BranchInst>(BB.getTerminator())) {
      if (handleBranchExpect(*BI))
        ExpectIntrinsicsHandled++;
    } else if (SwitchInst *SI = dyn_cast<SwitchInst>(BB.getTerminator())) {
      if (handleSwitchExpect(*SI))
        ExpectIntrinsicsHandled++;
    }

    // Remove llvm.expect intrinsics. Iterate backwards in order
    // to process select instructions before the intrinsic gets
    // removed.
    for (auto BI = BB.rbegin(), BE = BB.rend(); BI != BE;) {
      Instruction *Inst = &*BI++;
      CallInst *CI = dyn_cast<CallInst>(Inst);
      if (!CI) {
        if (SelectInst *SI = dyn_cast<SelectInst>(Inst)) {
          if (handleBrSelExpect(*SI))
            ExpectIntrinsicsHandled++;
        }
        continue;
      }

      Function *Fn = CI->getCalledFunction();
      if (Fn && Fn->getIntrinsicID() == Intrinsic::expect) {
        Value *Exp = CI->getArgOperand(0);
        CI->replaceAllUsesWith(Exp);
        CI->eraseFromParent();
        Changed = true;
      }
    }
  }

  return Changed;
}

PreservedAnalyses LowerExpectIntrinsicPass::run(Function &F,
                                                FunctionAnalysisManager &) {
  if (lowerExpectIntrinsic(F))
    return PreservedAnalyses::none();

  return PreservedAnalyses::all();
}

namespace {
/// \brief Legacy pass for lowering expect intrinsics out of the IR.
///
/// When this pass is run over a function it uses expect intrinsics which feed
/// branches and switches to provide branch weight metadata for those
/// terminators. It then removes the expect intrinsics from the IR so the rest
/// of the optimizer can ignore them.
class LowerExpectIntrinsic : public FunctionPass {
public:
  static char ID;
  LowerExpectIntrinsic() : FunctionPass(ID) {
    initializeLowerExpectIntrinsicPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override { return lowerExpectIntrinsic(F); }
};
}

char LowerExpectIntrinsic::ID = 0;
INITIALIZE_PASS(LowerExpectIntrinsic, "lower-expect",
                "Lower 'expect' Intrinsics", false, false)

FunctionPass *llvm::createLowerExpectIntrinsicPass() {
  return new LowerExpectIntrinsic();
}
