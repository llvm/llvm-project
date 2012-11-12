//===- CFGPrinter.cpp - DOT printer for the control flow graph ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a '-dot-cfg' analysis pass, which emits the
// cfg.<fnname>.dot file for each function in the program, with a graph of the
// CFG for that function.
//
// The other main feature of this file is that it implements the
// Function::viewCFG method, which is useful for debugging passes which operate
// on the CFG.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/CFGPrinter.h"

#include "llvm/Pass.h"
using namespace llvm;

namespace {
  struct CFGViewer : public FunctionPass {
    static char ID; // Pass identifcation, replacement for typeid
    CFGViewer() : FunctionPass(ID) {
      initializeCFGOnlyViewerPass(*PassRegistry::getPassRegistry());
    }

    virtual bool runOnFunction(Function &F) {
      F.viewCFG();
      return false;
    }

    void print(raw_ostream &OS, const Module* = 0) const {}

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }
  };
}

char CFGViewer::ID = 0;
INITIALIZE_PASS(CFGViewer, "view-cfg", "View CFG of function", false, true)

namespace {
  struct CFGOnlyViewer : public FunctionPass {
    static char ID; // Pass identifcation, replacement for typeid
    CFGOnlyViewer() : FunctionPass(ID) {
      initializeCFGOnlyViewerPass(*PassRegistry::getPassRegistry());
    }

    virtual bool runOnFunction(Function &F) {
      F.viewCFGOnly();
      return false;
    }

    void print(raw_ostream &OS, const Module* = 0) const {}

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }
  };
}

char CFGOnlyViewer::ID = 0;
INITIALIZE_PASS(CFGOnlyViewer, "view-cfg-only",
                "View CFG of function (with no function bodies)", false, true)

namespace {
  struct CFGPrinter : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    CFGPrinter() : FunctionPass(ID) {
      initializeCFGPrinterPass(*PassRegistry::getPassRegistry());
    }

    virtual bool runOnFunction(Function &F) {
      std::string Filename = "cfg." + F.getName().str() + ".dot";
      errs() << "Writing '" << Filename << "'...";
      
      std::string ErrorInfo;
      raw_fd_ostream File(Filename.c_str(), ErrorInfo);

      if (ErrorInfo.empty())
        WriteGraph(File, (const Function*)&F);
      else
        errs() << "  error opening file for writing!";
      errs() << "\n";
      return false;
    }

    void print(raw_ostream &OS, const Module* = 0) const {}

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }
  };
}

char CFGPrinter::ID = 0;
INITIALIZE_PASS(CFGPrinter, "dot-cfg", "Print CFG of function to 'dot' file", 
                false, true)

namespace {
  struct CFGOnlyPrinter : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    CFGOnlyPrinter() : FunctionPass(ID) {
      initializeCFGOnlyPrinterPass(*PassRegistry::getPassRegistry());
    }
    
    virtual bool runOnFunction(Function &F) {
      std::string Filename = "cfg." + F.getName().str() + ".dot";
      errs() << "Writing '" << Filename << "'...";

      std::string ErrorInfo;
      raw_fd_ostream File(Filename.c_str(), ErrorInfo);
      
      if (ErrorInfo.empty())
        WriteGraph(File, (const Function*)&F, true);
      else
        errs() << "  error opening file for writing!";
      errs() << "\n";
      return false;
    }
    void print(raw_ostream &OS, const Module* = 0) const {}

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }
  };
}

char CFGOnlyPrinter::ID = 0;
INITIALIZE_PASS(CFGOnlyPrinter, "dot-cfg-only",
   "Print CFG of function to 'dot' file (with no function bodies)",
   false, true)

/// viewCFG - This function is meant for use from the debugger.  You can just
/// say 'call F->viewCFG()' and a ghostview window should pop up from the
/// program, displaying the CFG of the current function.  This depends on there
/// being a 'dot' and 'gv' program in your path.
///
void Function::viewCFG() const {
  ViewGraph(this, "cfg" + getName());
}

/// viewCFGOnly - This function is meant for use from the debugger.  It works
/// just like viewCFG, but it does not include the contents of basic blocks
/// into the nodes, just the label.  If you are only interested in the CFG t
/// his can make the graph smaller.
///
void Function::viewCFGOnly() const {
  ViewGraph(this, "cfg" + getName(), true);
}

FunctionPass *llvm::createCFGPrinterPass () {
  return new CFGPrinter();
}

FunctionPass *llvm::createCFGOnlyPrinterPass () {
  return new CFGOnlyPrinter();
}

