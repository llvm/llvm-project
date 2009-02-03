//===- GraphPrinters.cpp - DOT printers for various graph types -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines several printers for various different types of graphs used
// by the LLVM infrastructure.  It uses the generic graph interface to convert
// the graph into a .dot graph.  These graphs can then be processed with the
// "dot" tool to convert them to postscript or some other suitable format.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/GraphWriter.h"
#include "llvm/Pass.h"
#include "llvm/Value.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/Dominators.h"
#include <iostream>
#include <fstream>
using namespace llvm;

template<typename GraphType>
static void WriteGraphToFile(std::ostream &O, const std::string &GraphName,
                             const GraphType &GT) {
  std::string Filename = GraphName + ".dot";
  O << "Writing '" << Filename << "'...";
  std::ofstream F(Filename.c_str());

  if (F.good())
    WriteGraph(F, GT);
  else
    O << "  error opening file for writing!";
  O << "\n";
}


//===----------------------------------------------------------------------===//
//                              Call Graph Printer
//===----------------------------------------------------------------------===//

namespace llvm {
  template<>
  struct DOTGraphTraits<CallGraph*> : public DefaultDOTGraphTraits {
    static std::string getGraphName(CallGraph *F) {
      return "Call Graph";
    }

    static std::string getNodeLabel(CallGraphNode *Node, CallGraph *Graph) {
      if (Node->getFunction())
        return ((Value*)Node->getFunction())->getName();
      else
        return "Indirect call node";
    }
  };
}


namespace {
  struct CallGraphPrinter : public ModulePass {
    static char ID; // Pass ID, replacement for typeid
    CallGraphPrinter() : ModulePass((intptr_t)&ID) {}

    virtual bool runOnModule(Module &M) {
      WriteGraphToFile(std::cerr, "callgraph", &getAnalysis<CallGraph>());
      return false;
    }

    void print(std::ostream &OS) const {}
    void print(std::ostream &OS, const llvm::Module*) const {}

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<CallGraph>();
      AU.setPreservesAll();
    }
  };

  char CallGraphPrinter::ID = 0;
  RegisterPass<CallGraphPrinter> P2("dot-callgraph",
                                    "Print Call Graph to 'dot' file");
}

//===----------------------------------------------------------------------===//
//                            DomInfoPrinter Pass
//===----------------------------------------------------------------------===//

namespace {
  class DomInfoPrinter : public FunctionPass {
  public:
    static char ID; // Pass identification, replacement for typeid
    DomInfoPrinter() : FunctionPass((intptr_t)&ID) {}

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
      AU.addRequired<DominatorTree>();
      AU.addRequired<DominanceFrontier>();

    }

    virtual bool runOnFunction(Function &F) {
      DominatorTree &DT = getAnalysis<DominatorTree>();
      DT.dump();
      DominanceFrontier &DF = getAnalysis<DominanceFrontier>();
      DF.dump();
      return false;
    }
  };

  char DomInfoPrinter::ID = 0;
  static RegisterPass<DomInfoPrinter>
  DIP("print-dom-info", "Dominator Info Printer", true, true);
}
