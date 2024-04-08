//===- LowerConditionalStoreIntrinsic.cpp -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/LowerConditionalStoreIntrinsic.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

#define DEBUG_TYPE "lower-cond-store-intrinsic"

// Conditional store intrinsic removal:
// block:
//  llvm.conditional.store.*(Val, Ptr, Condition)
//                         |
//                         V
// block:
//  br %i1 Condition, label cond.store, label block.remaining
// cond.tore:
//  store * Val, Ptr
//  br label block.remaining
// block.remaining:

static bool isCondStoreIntr(Instruction &Instr) {
  CallInst *CI = dyn_cast<CallInst>(&Instr);
  if (!CI)
    return false;

  Function *Fn = CI->getCalledFunction();
  if (Fn && Fn->getIntrinsicID() == Intrinsic::conditional_store)
    return true;
  return false;
}

static void lowerCondStoreIntr(Instruction &Instr, BasicBlock &BB) {
  LLVM_DEBUG(dbgs() << "Found basic with conditional store: " << BB.getName()
                    << "\n");
  auto *Val = Instr.getOperand(0);
  auto *Ptr = Instr.getOperand(1);
  auto *AlignmentVal = dyn_cast<ConstantInt>(Instr.getOperand(2));
  auto Alignment = MaybeAlign(AlignmentVal->getValue().getLimitedValue());
  assert(AlignmentVal && "Invalid intrinsic operands");
  auto *Cond = Instr.getOperand(3);

  Instruction *ThenBlock =
      SplitBlockAndInsertIfThen(Cond, &Instr, /*Unreachable*/ false);

  IRBuilder<> IB(ThenBlock);
  IB.CreateAlignedStore(Val, Ptr, Alignment);

  Instr.eraseFromParent();
}

static bool lowerCondStoreIntrinsicForFunc(Function &F) {
  bool Changed = false;

  for (BasicBlock &BB : F)
    for (Instruction &Instr : llvm::make_early_inc_range(llvm::reverse(BB)))
      if (isCondStoreIntr(Instr)) {
        lowerCondStoreIntr(Instr, BB);
        Changed = true;
      }
  return Changed;
}

PreservedAnalyses
LowerConditionalStoreIntrinsicPass::run(Function &F,
                                        FunctionAnalysisManager &) {
  if (lowerCondStoreIntrinsicForFunc(F))
    return PreservedAnalyses::none();

  return PreservedAnalyses::all();
}

namespace {

class LowerConditionalStoreIntrinsicLegacy : public FunctionPass {
  LowerConditionalStoreIntrinsicPass Impl;

public:
  static char ID;
  LowerConditionalStoreIntrinsicLegacy() : FunctionPass(ID) {
    initializeLowerConditionalStoreIntrinsicLegacyPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    // Don't skip optnone functions; atomics still need to be lowered.
    FunctionAnalysisManager DummyFAM;
    auto PA = Impl.run(F, DummyFAM);
    return !PA.areAllPreserved();
  }
};
} // namespace

char LowerConditionalStoreIntrinsicLegacy::ID = 0;
INITIALIZE_PASS(LowerConditionalStoreIntrinsicLegacy, "lower-conditional-store",
                "Lower conditional store", false, false)

FunctionPass *llvm::createLowerConditionalStoreIntrinsicPass() {
  return new LowerConditionalStoreIntrinsicLegacy();
}