//===-- AllocaHoisting.cpp - Hoist allocas to the entry block --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Hoist the alloca instructions in the non-entry blocks to the entry blocks.
//
//===----------------------------------------------------------------------===//

#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Constants.h"
#include "NVPTXAllocaHoisting.h"

namespace llvm {

bool NVPTXAllocaHoisting::runOnFunction(Function &function) {
  bool               functionModified    = false;
  Function::iterator I                   = function.begin();
  TerminatorInst    *firstTerminatorInst = (I++)->getTerminator();

  for (Function::iterator E = function.end(); I != E; ++I) {
    for (BasicBlock::iterator BI = I->begin(), BE = I->end(); BI != BE;) {
      AllocaInst *allocaInst = dyn_cast<AllocaInst>(BI++);
      if (allocaInst && isa<ConstantInt>(allocaInst->getArraySize())) {
        allocaInst->moveBefore(firstTerminatorInst);
        functionModified = true;
      }
    }
  }

  return functionModified;
}

char NVPTXAllocaHoisting::ID = 1;
RegisterPass<NVPTXAllocaHoisting> X("alloca-hoisting",
                                    "Hoisting alloca instructions in non-entry "
                                    "blocks to the entry block");

FunctionPass *createAllocaHoisting() {
  return new NVPTXAllocaHoisting();
}

} // end namespace llvm
