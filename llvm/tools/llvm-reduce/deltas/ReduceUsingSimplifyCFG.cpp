//===- ReduceUsingSimplifyCFG.h - Specialized Delta Pass ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a function which calls the Generic Delta pass in order
// to call SimplifyCFG on individual basic blocks.
//
//===----------------------------------------------------------------------===//

#include "ReduceUsingSimplifyCFG.h"
#include "Utils.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

void llvm::reduceUsingSimplifyCFGDeltaPass(Oracle &O,
                                           ReducerWorkItem &WorkItem) {
  Module &Program = WorkItem.getModule();
  SmallVector<BasicBlock *, 16> ToSimplify;
  for (auto &F : Program)
    for (auto &BB : F)
      if (!O.shouldKeep())
        ToSimplify.push_back(&BB);
  TargetTransformInfo TTI(Program.getDataLayout());
  for (auto *BB : ToSimplify)
    simplifyCFG(BB, TTI);
}

static void reduceConditionals(Oracle &O, ReducerWorkItem &WorkItem,
                               bool Direction) {
  Module &M = WorkItem.getModule();

  LLVMContext &Ctx = M.getContext();
  ConstantInt *ConstValToSet =
      Direction ? ConstantInt::getTrue(Ctx) : ConstantInt::getFalse(Ctx);

  for (Function &F : M) {
    if (F.isDeclaration())
      continue;

    SmallVector<BasicBlock *, 16> ToSimplify;

    for (auto &BB : F) {
      auto *BR = dyn_cast<BranchInst>(BB.getTerminator());
      if (!BR || !BR->isConditional() || BR->getCondition() == ConstValToSet ||
          O.shouldKeep())
        continue;

      BR->setCondition(ConstValToSet);
      ToSimplify.push_back(&BB);
    }

    if (!ToSimplify.empty()) {
      // TODO: Should probably leave MergeBlockIntoPredecessor for a separate
      // reduction
      simpleSimplifyCFG(F, ToSimplify);
    }
  }
}

void llvm::reduceConditionalsTrueDeltaPass(Oracle &O,
                                           ReducerWorkItem &WorkItem) {
  reduceConditionals(O, WorkItem, true);
}

void llvm::reduceConditionalsFalseDeltaPass(Oracle &O,
                                            ReducerWorkItem &WorkItem) {
  reduceConditionals(O, WorkItem, false);
}
