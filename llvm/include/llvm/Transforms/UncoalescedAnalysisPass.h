//===- UncoalescedAnalysisPass.h - Analysis to identify uncoalesced accesses in GPU programs -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This pass does an intra-procedural dataflow analysis to detect potentially
// uncoalesced load/store accesses in GPU programs. It uses an abstract-
// interpretation based analysis to compute the dependence of accessed indices
// on threadID (a unique identifier for threads). If the dependence is
// non-linear or a large linear function, the access is labelled as uncoalesced.
// It assumes that all the initial function arguments are independent of
// threadID.
//===----------------------------------------------------------------------===//

#ifndef LLVM_UNCOALESCED_ANALYSIS_PASS_H
#define LLVM_UNCOALESCED_ANALYSIS_PASS_H

#include "llvm/Transforms/MultiplierValue.h"
#include "llvm/Transforms/UncoalescedAnalysis.h"

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

FunctionPass* createUncoalescedAnalysisPass();

struct UncoalescedAnalysisPass : public FunctionPass {
  std::set<const Instruction*> UncoalescedAccesses_;

 public:
  static char ID; // Pass identification, replacement for typeid
  UncoalescedAnalysisPass() : FunctionPass(ID) {}

  // Analysis for Uncoalesced Accesses.
  bool runOnFunction(Function &F) override;

  const std::set<const Instruction*>& getUncoalescedAccesses() const {
    return UncoalescedAccesses_;
  }

  // We don't modify the program, so we preserve all analyses.
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.setPreservesAll();
  }
};

}

#endif /* UncoalescedAnalysisPass.h */
