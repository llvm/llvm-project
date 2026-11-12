//===- InterproceduralBlockSizeInvarianceAnalysisPass.cpp - Interprocedural analysis to identify block-size invariant GPU kernels -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_INTERPROC_BSI_ANALYSIS_PASS_H
#define LLVM_INTERPROC_BSI_ANALYSIS_PASS_H

#include "llvm/Transforms/BSizeDependenceValue.h"
#include "llvm/Transforms/BlockSizeInvarianceAnalysis.h"

#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <list>

namespace llvm {

struct InterproceduralBlockSizeInvarianceAnalysisPass : public ModulePass {
  std::set<Function *> BlockSizeIndependentMethods_;

 public:
  static char ID; // Pass identification, replacement for typeid
  InterproceduralBlockSizeInvarianceAnalysisPass() : ModulePass(ID) {}

  // Interprocedural analysis for uncoalesced accesses.
  bool runOnModule(Module &M) override;

  bool IsBlockSizeIndependent(Function *F) {
    return (BlockSizeIndependentMethods_.find(F) !=
            BlockSizeIndependentMethods_.end());
  }

  // We don't modify the program, so we preserve all analyses.
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<CallGraphWrapperPass>();
    AU.setPreservesAll();
  }
};

}

#endif /* InterproceduralBlockSizeInvarianceAnalysisPass.h */
