//===-- UnifyFunctionExitNodes.h - Ensure fn's have one return --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass is used to ensure that functions have at most one return and one
// unwind instruction in them.  Additionally, it keeps track of which node is
// the new exit node of the CFG.  If there are no return or unwind instructions
// in the function, the getReturnBlock/getUnwindBlock methods will return a null
// pointer.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_UNIFYFUNCTIONEXITNODES_H
#define LLVM_TRANSFORMS_UTILS_UNIFYFUNCTIONEXITNODES_H

#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"

namespace llvm {

struct UnifyFunctionExitNodes : public FunctionPass {
  BasicBlock *ReturnBlock = nullptr;
  BasicBlock *UnwindBlock = nullptr;
  BasicBlock *UnreachableBlock;

public:
  static char ID; // Pass identification, replacement for typeid
  UnifyFunctionExitNodes() : FunctionPass(ID) {
    initializeUnifyFunctionExitNodesPass(*PassRegistry::getPassRegistry());
  }

  // We can preserve non-critical-edgeness when we unify function exit nodes
  void getAnalysisUsage(AnalysisUsage &AU) const override;

  // getReturn|Unwind|UnreachableBlock - Return the new single (or nonexistent)
  // return, unwind, or unreachable  basic blocks in the CFG.
  //
  BasicBlock *getReturnBlock() const { return ReturnBlock; }
  BasicBlock *getUnwindBlock() const { return UnwindBlock; }
  BasicBlock *getUnreachableBlock() const { return UnreachableBlock; }

  bool runOnFunction(Function &F) override;
};

Pass *createUnifyFunctionExitNodesPass();

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_UNIFYFUNCTIONEXITNODES_H
