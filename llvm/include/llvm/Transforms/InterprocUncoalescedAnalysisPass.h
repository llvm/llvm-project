//===- InterproceduralUncoalescedAnalysisPass.cpp - Interprocedural analysis to identify uncoalesced accesses in GPU programs -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This pass does an inter-procedural dataflow analysis to detect potentially
// uncoalesced load/store accesses in GPU programs. It uses an abstract-
// interpretation based analysis to compute the dependence of accessed indices
// on threadID (a unique identifier for threads). If the dependence is
// non-linear or a large linear function, the access is labelled as uncoalesced.
// 
// It starts with the analysis of the top-most functions in the call-graph and
// then proceeds with the analysis of their callees in a topological order.
// While analyzing a specific callee, it considers the join of the call contexts
// of all its callers.
//===----------------------------------------------------------------------===//

#ifndef LLVM_INTERPROC_UNCOALESCED_ANALYSIS_PASS_H
#define LLVM_INTERPROC_UNCOALESCED_ANALYSIS_PASS_H

#include "llvm/Transforms/MultiplierValue.h"
#include "llvm/Transforms/UncoalescedAnalysis.h"

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

struct InterproceduralUncoalescedAnalysisPass : public ModulePass {
  std::map<const Function*, std::set<const Instruction*>> UncoalescedAccessMap_;

 public:
  static char ID; // Pass identification, replacement for typeid
  InterproceduralUncoalescedAnalysisPass() : ModulePass(ID) {}

  // Interprocedural analysis for uncoalesced accesses.
  bool runOnModule(Module &M) override;

  const std::set<const Instruction*>& getUncoalescedAccesses(const Function* F)
      const {
    return UncoalescedAccessMap_.at(F);
  }

  // We don't modify the program, so we preserve all analyses.
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<CallGraphWrapperPass>();
    AU.setPreservesAll();
  }
};

}

#endif /* InterproceduralUncoalescedAnalysisPass.h */
