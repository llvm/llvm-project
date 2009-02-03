//===- InstructionNamer.cpp - Give anonymous instructions names -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a little utility pass that gives instructions names, this is mostly
// useful when diffing the effect of an optimization because deleting an
// unnamed instruction can change all other instruction numbering, making the
// diff very noisy.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/Function.h"
#include "llvm/Pass.h"
#include "llvm/Type.h"
using namespace llvm;

namespace {
  struct InstNamer : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    InstNamer() : FunctionPass(&ID) {}
    
    void getAnalysisUsage(AnalysisUsage &Info) const {
      Info.setPreservesAll();
    }

    bool runOnFunction(Function &F) {
      for (Function::arg_iterator AI = F.arg_begin(), AE = F.arg_end();
           AI != AE; ++AI)
        if (!AI->hasName() && AI->getType() != Type::VoidTy)
          AI->setName("tmp");

      for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB) {
        if (!BB->hasName())
          BB->setName("BB");
        
        for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I)
          if (!I->hasName() && I->getType() != Type::VoidTy)
            I->setName("tmp");
      }
      return true;
    }
  };
  
  char InstNamer::ID = 0;
  static RegisterPass<InstNamer> X("instnamer",
                                   "Assign names to anonymous instructions");
}


const PassInfo *const llvm::InstructionNamerID = &X;
//===----------------------------------------------------------------------===//
//
// InstructionNamer - Give any unnamed non-void instructions "tmp" names.
//
FunctionPass *llvm::createInstructionNamerPass() {
  return new InstNamer();
}
