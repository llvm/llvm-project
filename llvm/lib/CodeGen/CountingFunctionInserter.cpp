//===- CountingFunctionInserter.cpp - Insert mcount-like function calls ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Insert calls to counter functions, such as mcount, intended to be called
// once per function, at the beginning of each function.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Pass.h"
using namespace llvm;

namespace {
  struct CountingFunctionInserter : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    CountingFunctionInserter() : FunctionPass(ID) {
      initializeCountingFunctionInserterPass(*PassRegistry::getPassRegistry());
    }
    
    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.addPreserved<GlobalsAAWrapperPass>();
    }

    bool runOnFunction(Function &F) override {
      std::string CountingFunctionName =
        F.getFnAttribute("counting-function").getValueAsString();
      if (CountingFunctionName.empty())
        return false;

      Type *VoidTy = Type::getVoidTy(F.getContext());
      Constant *CountingFn =
        F.getParent()->getOrInsertFunction(CountingFunctionName,
                                           VoidTy);
      CallInst::Create(CountingFn, "", &*F.begin()->getFirstInsertionPt());
      return true;
    }
  };
  
  char CountingFunctionInserter::ID = 0;
}

INITIALIZE_PASS(CountingFunctionInserter, "cfinserter", 
                "Inserts calls to mcount-like functions", false, false)

//===----------------------------------------------------------------------===//
//
// CountingFunctionInserter - Give any unnamed non-void instructions "tmp" names.
//
FunctionPass *llvm::createCountingFunctionInserterPass() {
  return new CountingFunctionInserter();
}
