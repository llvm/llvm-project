//===-- UnifyFunctionExitNodes.h - Ensure fn's have one return --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass is used to ensure that functions have at most one return and one
// unreachable instruction in them.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_UNIFYFUNCTIONEXITNODES_H
#define LLVM_TRANSFORMS_UTILS_UNIFYFUNCTIONEXITNODES_H

#include "llvm/Pass.h"

namespace llvm {

class BasicBlock;

class UnifyFunctionExitNodes : public FunctionPass {
  bool unifyUnreachableBlocks(Function &F);
  bool unifyReturnBlocks(Function &F);

public:
  static char ID; // Pass identification, replacement for typeid
  UnifyFunctionExitNodes();

  // We can preserve non-critical-edgeness when we unify function exit nodes
  void getAnalysisUsage(AnalysisUsage &AU) const override;

  bool runOnFunction(Function &F) override;
};

Pass *createUnifyFunctionExitNodesPass();

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_UNIFYFUNCTIONEXITNODES_H
