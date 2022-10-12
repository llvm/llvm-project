//===- PrintSCC.cpp - Enumerate SCCs in some key graphs -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides passes to print out SCCs in a CFG or a CallGraph.
// Normally, you would not use these passes; instead, you would use the
// scc_iterator directly to enumerate SCCs and process them in some way.  These
// passes serve three purposes:
//
// (1) As a reference for how to use the scc_iterator.
// (2) To print out the SCCs for a CFG or a CallGraph:
//       analyze -print-cfg-sccs            to print the SCCs in each CFG of a module.
//       analyze -print-cfg-sccs -stats     to print the #SCCs and the maximum SCC size.
//       analyze -print-cfg-sccs -debug > /dev/null to watch the algorithm in action.
//
//     and similarly:
//       analyze -print-callgraph-sccs [-stats] [-debug] to print SCCs in the CallGraph
//
// (3) To test the scc_iterator.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SCCIterator.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace {
  struct CFGSCC : public FunctionPass {
    static char ID;  // Pass identification, replacement for typeid
    CFGSCC() : FunctionPass(ID) {}
    bool runOnFunction(Function& func) override;

    void print(raw_ostream &O, const Module* = nullptr) const override { }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesAll();
    }
  };
}

char CFGSCC::ID = 0;
static RegisterPass<CFGSCC>
Y("print-cfg-sccs", "Print SCCs of each function CFG");

bool CFGSCC::runOnFunction(Function &F) {
  unsigned sccNum = 0;
  errs() << "SCCs for Function " << F.getName() << " in PostOrder:";
  for (scc_iterator<Function*> SCCI = scc_begin(&F); !SCCI.isAtEnd(); ++SCCI) {
    const std::vector<BasicBlock *> &nextSCC = *SCCI;
    errs() << "\nSCC #" << ++sccNum << " : ";
    for (BasicBlock *BB : nextSCC) {
      BB->printAsOperand(errs(), false);
      errs() << ", ";
    }
    if (nextSCC.size() == 1 && SCCI.hasCycle())
      errs() << " (Has self-loop).";
  }
  errs() << "\n";

  return true;
}
