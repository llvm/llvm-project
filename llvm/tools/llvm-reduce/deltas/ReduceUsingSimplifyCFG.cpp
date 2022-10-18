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
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

static void reduceUsingSimplifyCFG(Oracle &O, Module &Program) {
  SmallVector<BasicBlock *, 16> ToSimplify;
  for (auto &F : Program)
    for (auto &BB : F)
      if (!O.shouldKeep())
        ToSimplify.push_back(&BB);
  TargetTransformInfo TTI(Program.getDataLayout());
  for (auto *BB : ToSimplify)
    simplifyCFG(BB, TTI);
}

void llvm::reduceUsingSimplifyCFGDeltaPass(TestRunner &Test) {
  runDeltaPass(Test, reduceUsingSimplifyCFG, "Reducing using SimplifyCFG");
}
