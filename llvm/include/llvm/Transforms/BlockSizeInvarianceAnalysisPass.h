//===- BlockSizeInvarianceAnalysisPass.h - Analysis to check block-size invariance of a method -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BLOCK_SIZE_INVARIANCE_ANALYSIS_PASS_H
#define LLVM_BLOCK_SIZE_INVARIANCE_ANALYSIS_PASS_H

#include "llvm/Transforms/BSizeDependenceValue.h"
#include "llvm/Transforms/BSizeGPUState.h"
#include "llvm/Transforms/BlockSizeInvarianceAnalysis.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <list>

namespace llvm {

FunctionPass* createBlockSizeInvarianceAnalysisPass();

struct BlockSizeInvarianceAnalysisPass : public FunctionPass {
  // Set of array accesses that are dependent on block size.
  std::set<const Instruction*> BlockSizeDependentAccesses_;

 public:
  static char ID; // Pass identification, replacement for typeid
  BlockSizeInvarianceAnalysisPass() : FunctionPass(ID) {}

  // Analysis for BlockSizeInvariance Accesses.
  bool runOnFunction(Function &F) override;

  const std::set<const Instruction*>& getBlockSizeDependentAccesses() const {
    return BlockSizeDependentAccesses_;
  }

  // We don't modify the program, so we preserve all analyses.
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.setPreservesAll();
  }
};

}

#endif /* BlockSizeInvarianceAnalysisPass.h */
