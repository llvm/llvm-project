//===-------- LoopReduceMotion.cpp - Loop Reduce Motion Optimization ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This pass is designed to sink `ReduceCall` operations out of loops to reduce
// the number of instructions within the loop body.
//
// Below is the target pattern to be matched and the resulting pattern
// after the transformation.
//
// before                    | after
// ------                    | ------
// loop:                     | loop:
//   ...                     |   ...
//   vc = vecbin va, vb      |   vc = vecbin va, vb
//   d = reduce_add vc       |   vsum = vadd vsum, vc
//   sum = add sum, d        |   ...
//   ...                     |   ...
// exit:                     | exit:
//   value = sum             |   d = reduce_add sum
//   ...                     |   value = d
//   ...                     |   ...
//   ret                     |   ret
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/LoopReduceMotion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

#define DEBUG_TYPE "loop-reduce-motion"

using namespace llvm;

class LoopReduceMotion : public FunctionPass {
  LoopReduceMotionPass Impl;

public:
  static char ID;

  LoopReduceMotion() : FunctionPass(ID) {}

  StringRef getPassName() const override { return "Loop Reduce Motion Pass"; }

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.setPreservesCFG();
  }
};

char LoopReduceMotion::ID = 0;

PreservedAnalyses LoopReduceMotionPass::run(Function &F,
                                            FunctionAnalysisManager &FAM) {
  LoopInfo &LI = FAM.getResult<LoopAnalysis>(F);
  DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(F);
  bool Changed = false;
  for (Loop *L : LI) {
    Changed |= matchAndTransform(*L, DT, LI);
  }
  if (!Changed)
    return PreservedAnalyses::all();
  return PreservedAnalyses::none();
}

bool LoopReduceMotion::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  auto *TPC = getAnalysisIfAvailable<TargetPassConfig>();
  if (!TPC)
    return false;

  LLVM_DEBUG(dbgs() << "*** " << getPassName() << ": " << F.getName() << "\n");

  DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  bool Changed = false;
  for (Loop *L : LI) {
    Changed |= Impl.matchAndTransform(*L, DT, LI);
  }
  if (!Changed)
    return false;

  return true;
}

bool LoopReduceMotionPass::matchAndTransform(Loop &L, DominatorTree &DT,
                                             LoopInfo &LI) {
  BasicBlock *Header = L.getHeader();
  BasicBlock *Latch = L.getLoopLatch();
  BasicBlock *ExitBlock = L.getExitBlock();
  if (!Header || !Latch || !ExitBlock) {
    LLVM_DEBUG(dbgs() << "LRM: Skipping loop " << Header->getName()
                      << " because it is not a valid loop.\n");
    return false;
  }
  BasicBlock *Preheader = L.getLoopPreheader();
  if (!Preheader) {
    Preheader = InsertPreheaderForLoop(&L, &DT, &LI, nullptr, false);
    if (!Preheader) {
      LLVM_DEBUG(dbgs() << "LRM: Failed to create a preheader for loop "
                        << Header->getName() << ".\n");
      return false;
    }
  }
  for (PHINode &PN : Header->phis()) {
    if (!PN.getType()->isIntegerTy())
      continue;

    RecurrenceDescriptor RecDesc;
    if (!RecurrenceDescriptor::isReductionPHI(&PN, &L, RecDesc))
      continue;

    if (RecDesc.getRecurrenceKind() != RecurKind::Add)
      continue;

    Value *RecurrenceValueFromPHI = PN.getIncomingValueForBlock(Latch);
    Instruction *RecurrenceInst = dyn_cast<Instruction>(RecurrenceValueFromPHI);
    if (!RecurrenceInst || RecurrenceInst->getNumOperands() != 2)
      continue;

    Value *RecurrenceValue = RecurrenceInst->getOperand(0) == &PN
                                 ? RecurrenceInst->getOperand(1)
                                 : RecurrenceInst->getOperand(0);

    CallInst *ReduceCall = dyn_cast<CallInst>(RecurrenceValue);
    if (!ReduceCall)
      continue;
    Function *CalledFunc = ReduceCall->getCalledFunction();

    if (!CalledFunc || !CalledFunc->isIntrinsic() ||
        !(CalledFunc->getIntrinsicID() == Intrinsic::vector_reduce_add))
      continue;

    Value *ReduceOperand = ReduceCall->getArgOperand(0);
    Instruction *VecBin = dyn_cast<Instruction>(ReduceOperand);
    if (!VecBin || (VecBin->getOpcode() != Instruction::Sub &&
                    VecBin->getOpcode() != Instruction::Add))
      continue;
    // pattern match success
    LLVM_DEBUG(dbgs() << "Found pattern to optimize in loop "
                      << Header->getName() << "!\n");

    VectorType *VecTy = cast<VectorType>(VecBin->getType());
    Value *VecZero = ConstantInt::get(VecTy, 0);

    // build new Vector Add to replace Scalar Add
    IRBuilder<> HeaderBuilder(Header, Header->getFirstNonPHIIt());
    PHINode *VecSumPhi = HeaderBuilder.CreatePHI(VecTy, 2, "vec.sum.phi");
    VecSumPhi->addIncoming(VecZero, Preheader);
    IRBuilder<> BodyBuilder(RecurrenceInst);
    Value *NewVecAdd = BodyBuilder.CreateAdd(VecSumPhi, VecBin, "vec.sum.next");
    VecSumPhi->addIncoming(NewVecAdd, Latch);

    // build landingPad for reduce add out of loop
    BasicBlock *ExitingBlock =
        Latch->getTerminator()->getSuccessor(0) == Header ? Latch : Header;
    if (!L.isLoopExiting(ExitingBlock)) {
      ExitingBlock = Header;
    }
    BasicBlock *LandingPad = SplitEdge(ExitingBlock, ExitBlock, &DT, &LI);
    LandingPad->setName("loop.exit.landing");
    IRBuilder<> LandingPadBuilder(LandingPad->getTerminator());
    Value *ScalarTotalSum = LandingPadBuilder.CreateCall(
        ReduceCall->getCalledFunction(), NewVecAdd, "scalar.total.sum");
    Value *PreheaderValue = PN.getIncomingValueForBlock(Preheader);
    Value *LastAdd =
        PreheaderValue
            ? LandingPadBuilder.CreateAdd(PreheaderValue, ScalarTotalSum)
            : ScalarTotalSum;

    // replace use of phi and erase use empty value
    if (!PN.use_empty())
      PN.replaceAllUsesWith(PoisonValue::get(PN.getType()));
    if (PN.use_empty())
      PN.eraseFromParent();

    Instruction *FinalNode = dyn_cast<Instruction>(LastAdd);
    if (!FinalNode)
      return false;
    RecurrenceInst->replaceAllUsesWith(FinalNode);

    if (RecurrenceInst->use_empty())
      RecurrenceInst->eraseFromParent();
    if (ReduceCall->use_empty())
      ReduceCall->eraseFromParent();

    return true;
  }
  return false;
}

FunctionPass *llvm::createLoopReduceMotionPass() {
  return new LoopReduceMotion();
}
