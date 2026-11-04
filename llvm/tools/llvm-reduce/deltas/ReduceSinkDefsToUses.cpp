//===- ReduceSinkDefsToUses.cpp - Specialized Delta Pass ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Try to move defs to be next to their uses
//
//===----------------------------------------------------------------------===//

#include "ReduceSinkDefsToUses.h"
#include "Utils.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

static bool shouldPreserveUsePosition(const Instruction &I) {
  return isa<AllocaInst>(I) || isa<PHINode>(I) || I.isEHPad();
}

static bool shouldPreserveDefPosition(const Instruction &I) {
  return shouldPreserveUsePosition(I) || I.isTerminator();
}

static void sinkDefsToUsesInFunction(Oracle &O, Function &F) {
  DominatorTree DT(F);

  for (BasicBlock &BB : F) {
    for (Instruction &UseInst : make_early_inc_range(reverse(BB))) {
      if (shouldPreserveUsePosition(UseInst))
        continue;

      for (Value *UseOp : UseInst.operands()) {
        Instruction *DefInst = dyn_cast<Instruction>(UseOp);
        if (!DefInst || shouldPreserveDefPosition(*DefInst))
          continue;

        if (!all_of(DefInst->users(), [&](const User *DefUser) {
              return DefUser == &UseInst ||
                     DT.dominates(&UseInst, cast<Instruction>(DefUser));
            })) {
          continue;
        }

        if (!O.shouldKeep())
          DefInst->moveBeforePreserving(UseInst.getIterator());
      }
    }
  }
}

void llvm::reduceSinkDefsToUsesDeltaPass(Oracle &O, ReducerWorkItem &WorkItem) {
  Module &M = WorkItem.getModule();
  for (Function &F : M) {
    if (!F.isDeclaration())
      sinkDefsToUsesInFunction(O, F);
  }
}
