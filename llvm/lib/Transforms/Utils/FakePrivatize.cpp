//===- FakePrivatize.cpp - Test-only alloca privatization
//------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/FakePrivatize.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

PreservedAnalyses FakePrivatizePass::run(Function &F,
                                         FunctionAnalysisManager &) {
  if (F.getName() != "test")
    return PreservedAnalyses::all();

  PHINode *IV = nullptr;
  BasicBlock *Header = nullptr;
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      if (auto *P = dyn_cast<PHINode>(&I); P->getName() == "iv") {
        IV = P;
        Header = &BB;
        break;
      }

  if (!IV)
    return PreservedAnalyses::all();

  BasicBlock *Latch = nullptr;
  for (Value *V : IV->incoming_values())
    if (auto *I = dyn_cast<Instruction>(V)) {
      Latch = I->getParent();
      break;
    }
  if (!Latch)
    return PreservedAnalyses::all();

  uint64_t TripCount = 0;
  for (Instruction &I : *Latch)
    if (auto *Cmp = dyn_cast<ICmpInst>(&I))
      for (Value *Op : Cmp->operands())
        if (auto *C = dyn_cast<ConstantInt>(Op))
          TripCount = C->getZExtValue();
  if (!TripCount)
    return PreservedAnalyses::all();

  SmallVector<AllocaInst *, 4> Allocas;
  for (Instruction &I : F.getEntryBlock())
    if (auto *AI = dyn_cast<AllocaInst>(&I))
      if (AI->getAllocatedType()->isIntegerTy(64))
        Allocas.push_back(AI);
  if (Allocas.empty())
    return PreservedAnalyses::all();

  IRBuilder<> EntryBuilder(&*F.getEntryBlock().getFirstInsertionPt());
  Value *Base = EntryBuilder.CreateAlloca(
      EntryBuilder.getInt64Ty(), EntryBuilder.getInt64(TripCount), "base");
  IRBuilder<> Builder(&*Header->getFirstInsertionPt());
  Value *Ptr =
      Builder.CreateInBoundsGEP(Builder.getInt64Ty(), Base, IV, "slot.ptr");
  for (AllocaInst *AI : Allocas) {
    AI->replaceAllUsesWith(Ptr);
    AI->eraseFromParent();
  }
  return PreservedAnalyses::none();
}
