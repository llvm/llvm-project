//===--- VMCore/PrintModulePass.cpp - Module/Function Printer -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// PrintModulePass and PrintFunctionPass implementations.
//
//===----------------------------------------------------------------------===//

#include "llvm/Assembly/PrintModulePass.h"

#include "llvm/Function.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace {

  class PrintModulePass : public ModulePass {
    std::string Banner;
    raw_ostream *Out;       // raw_ostream to print on
    bool DeleteStream;      // Delete the ostream in our dtor?
  public:
    static char ID;
    PrintModulePass() : ModulePass(ID), Out(&dbgs()), 
      DeleteStream(false) {}
    PrintModulePass(const std::string &B, raw_ostream *o, bool DS)
        : ModulePass(ID), Banner(B), Out(o), DeleteStream(DS) {}
    
    ~PrintModulePass() {
      if (DeleteStream) delete Out;
    }
    
    bool runOnModule(Module &M) {
      (*Out) << Banner << M;
      return false;
    }
    
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }
  };
  
  class PrintFunctionPass : public FunctionPass {
    std::string Banner;     // String to print before each function
    raw_ostream *Out;       // raw_ostream to print on
    bool DeleteStream;      // Delete the ostream in our dtor?
  public:
    static char ID;
    PrintFunctionPass() : FunctionPass(ID), Banner(""), Out(&dbgs()), 
                          DeleteStream(false) {}
    PrintFunctionPass(const std::string &B, raw_ostream *o, bool DS)
      : FunctionPass(ID), Banner(B), Out(o), DeleteStream(DS) {}
    
    ~PrintFunctionPass() {
      if (DeleteStream) delete Out;
    }
    
    // runOnFunction - This pass just prints a banner followed by the
    // function as it's processed.
    //
    bool runOnFunction(Function &F) {
      (*Out) << Banner << static_cast<Value&>(F);
      return false;
    }
    
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }
  };
}

char PrintModulePass::ID = 0;
INITIALIZE_PASS(PrintModulePass, "print-module",
                "Print module to stderr", false, false)
char PrintFunctionPass::ID = 0;
INITIALIZE_PASS(PrintFunctionPass, "print-function",
                "Print function to stderr", false, false)

/// createPrintModulePass - Create and return a pass that writes the
/// module to the specified raw_ostream.
ModulePass *llvm::createPrintModulePass(llvm::raw_ostream *OS, 
                                        bool DeleteStream,
                                        const std::string &Banner) {
  return new PrintModulePass(Banner, OS, DeleteStream);
}

/// createPrintFunctionPass - Create and return a pass that prints
/// functions to the specified raw_ostream as they are processed.
FunctionPass *llvm::createPrintFunctionPass(const std::string &Banner,
                                            llvm::raw_ostream *OS, 
                                            bool DeleteStream) {
  return new PrintFunctionPass(Banner, OS, DeleteStream);
}

