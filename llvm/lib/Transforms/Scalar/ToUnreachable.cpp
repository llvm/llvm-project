//===- ToUnreachable.cpp - Turn function into unreachable. ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/ToUnreachable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

PreservedAnalyses ToUnreachablePass::run(Function &F,
                                         FunctionAnalysisManager &AM) {
  SmallVector<BasicBlock *> AllBlocks;
  for (BasicBlock &BB : F) {
    AllBlocks.push_back(&BB);
    BB.dropAllReferences();
  }

  for (unsigned I = 1; I < AllBlocks.size(); ++I)
    AllBlocks[I]->eraseFromParent();

  for (Instruction &I : make_early_inc_range(*AllBlocks[0]))
    I.eraseFromParent();

  new UnreachableInst(F.getContext(), AllBlocks[0]);
  return PreservedAnalyses::none();
}
