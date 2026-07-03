//===- Utils.cpp - llvm-reduce utility functions --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains some utility functions supporting llvm-reduce.
//
//===----------------------------------------------------------------------===//

#include "Utils.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalIFunc.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

extern cl::OptionCategory LLVMReduceOptions;

cl::opt<bool> llvm::Verbose("verbose",
                            cl::desc("Print extra debugging information"),
                            cl::init(false), cl::cat(LLVMReduceOptions));

Value *llvm::getDefaultValue(Type *T) {
  if (T->isVoidTy())
    return PoisonValue::get(T);

  if (auto *TET = dyn_cast<TargetExtType>(T)) {
    if (TET->hasProperty(TargetExtType::HasZeroInit))
      return ConstantTargetNone::get(TET);
    return PoisonValue::get(TET);
  }

  return Constant::getNullValue(T);
}

bool llvm::hasAliasUse(Function &F) {
  return any_of(F.users(), [](User *U) {
      return isa<GlobalAlias>(U) || isa<GlobalIFunc>(U);
    });
}

void llvm::simpleSimplifyCFG(Function &F, ArrayRef<BasicBlock *> BBs,
                             bool FoldBlockIntoPredecessor) {

  for (BasicBlock *BB : BBs) {
    ConstantFoldTerminator(BB);
    if (FoldBlockIntoPredecessor)
      MergeBlockIntoPredecessor(BB);
  }

  // Remove unreachable blocks
  //
  // removeUnreachableBlocks can't be used here, it will turn various undefined
  // behavior into unreachables, but llvm-reduce was the thing that generated
  // the undefined behavior, and we don't want it to kill the entire program.
  SmallPtrSet<BasicBlock *, 16> Visited(llvm::from_range,
                                        depth_first(&F.getEntryBlock()));

  SmallVector<BasicBlock *, 16> Unreachable;
  for (BasicBlock &BB : F) {
    if (!Visited.count(&BB))
      Unreachable.push_back(&BB);
  }

  // The dead BB's may be in a dead cycle or otherwise have references to each
  // other.  Because of this, we have to drop all references first, then delete
  // them all at once.
  for (BasicBlock *BB : Unreachable) {
    for (BasicBlock *Successor : successors(&*BB))
      if (Visited.count(Successor))
        Successor->removePredecessor(&*BB, /*KeepOneInputPHIs=*/true);
    BB->dropAllReferences();
  }

  for (BasicBlock *BB : Unreachable)
    BB->eraseFromParent();
}
