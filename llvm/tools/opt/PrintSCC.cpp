//===- PrintSCC.cpp - Enumerate SCCs in some key graphs -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

#include "llvm/Pass.h"
#include "llvm/Module.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Support/CFG.h"
#include "llvm/ADT/SCCIterator.h"
#include <iostream>
using namespace llvm;

namespace {
  struct CFGSCC : public FunctionPass {
    static char ID;  // Pass identification, replacement for typeid
    CFGSCC() : FunctionPass((intptr_t)&ID) {}
    bool runOnFunction(Function& func);

    void print(std::ostream &O, const Module* = 0) const { }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }
  };

  struct CallGraphSCC : public ModulePass {
    static char ID;  // Pass identification, replacement for typeid
    CallGraphSCC() : ModulePass((intptr_t)&ID) {}

    // run - Print out SCCs in the call graph for the specified module.
    bool runOnModule(Module &M);

    void print(std::ostream &O, const Module* = 0) const { }

    // getAnalysisUsage - This pass requires the CallGraph.
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
      AU.addRequired<CallGraph>();
    }
  };

  char CFGSCC::ID = 0;
  RegisterPass<CFGSCC>
  Y("print-cfg-sccs", "Print SCCs of each function CFG");

  char CallGraphSCC::ID = 0;
  RegisterPass<CallGraphSCC>
  Z("print-callgraph-sccs", "Print SCCs of the Call Graph");
}

bool CFGSCC::runOnFunction(Function &F) {
  unsigned sccNum = 0;
  std::cout << "SCCs for Function " << F.getName() << " in PostOrder:";
  for (scc_iterator<Function*> SCCI = scc_begin(&F),
         E = scc_end(&F); SCCI != E; ++SCCI) {
    std::vector<BasicBlock*> &nextSCC = *SCCI;
    std::cout << "\nSCC #" << ++sccNum << " : ";
    for (std::vector<BasicBlock*>::const_iterator I = nextSCC.begin(),
           E = nextSCC.end(); I != E; ++I)
      std::cout << (*I)->getName() << ", ";
    if (nextSCC.size() == 1 && SCCI.hasLoop())
      std::cout << " (Has self-loop).";
  }
  std::cout << "\n";

  return true;
}


// run - Print out SCCs in the call graph for the specified module.
bool CallGraphSCC::runOnModule(Module &M) {
  CallGraphNode* rootNode = getAnalysis<CallGraph>().getRoot();
  unsigned sccNum = 0;
  std::cout << "SCCs for the program in PostOrder:";
  for (scc_iterator<CallGraphNode*> SCCI = scc_begin(rootNode),
         E = scc_end(rootNode); SCCI != E; ++SCCI) {
    const std::vector<CallGraphNode*> &nextSCC = *SCCI;
    std::cout << "\nSCC #" << ++sccNum << " : ";
    for (std::vector<CallGraphNode*>::const_iterator I = nextSCC.begin(),
           E = nextSCC.end(); I != E; ++I)
      std::cout << ((*I)->getFunction() ? (*I)->getFunction()->getName()
                    : std::string("Indirect CallGraph node")) << ", ";
    if (nextSCC.size() == 1 && SCCI.hasLoop())
      std::cout << " (Has self-loop).";
  }
  std::cout << "\n";

  return true;
}
