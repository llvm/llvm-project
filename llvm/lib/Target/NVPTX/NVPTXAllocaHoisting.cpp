//===-- AllocaHoisting.cpp - Hoist allocas to the entry block --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Hoist the alloca instructions in the non-entry blocks to the entry blocks.
//
//===----------------------------------------------------------------------===//

#include "NVPTX.h"
#include "llvm/CodeGen/StackProtector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
using namespace llvm;

static bool hoistAllocas(Function &function) {
  bool functionModified = false;
  Function::iterator I = function.begin();
  Instruction *firstTerminatorInst = (I++)->getTerminator();

  for (Function::iterator E = function.end(); I != E; ++I) {
    for (BasicBlock::iterator BI = I->begin(), BE = I->end(); BI != BE;) {
      AllocaInst *allocaInst = dyn_cast<AllocaInst>(BI++);
      if (allocaInst && isa<ConstantInt>(allocaInst->getArraySize())) {
        allocaInst->moveBefore(firstTerminatorInst->getIterator());
        functionModified = true;
      }
    }
  }

  return functionModified;
}

namespace {
// Hoisting the alloca instructions in the non-entry blocks to the entry
// block.
class NVPTXAllocaHoistingLegacyPass : public FunctionPass {
public:
  static char ID; // Pass ID
  NVPTXAllocaHoistingLegacyPass() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addPreserved<StackProtector>();
  }

  StringRef getPassName() const override {
    return "NVPTX specific alloca hoisting";
  }

  bool runOnFunction(Function &function) override {
    return hoistAllocas(function);
  }
};
} // namespace

char NVPTXAllocaHoistingLegacyPass::ID = 0;

INITIALIZE_PASS(
    NVPTXAllocaHoistingLegacyPass, "alloca-hoisting",
    "Hoisting alloca instructions in non-entry blocks to the entry block",
    false, false)

FunctionPass *llvm::createNVPTXAllocaHoistingLegacyPass() {
  return new NVPTXAllocaHoistingLegacyPass;
}

PreservedAnalyses NVPTXAllocaHoistingPass::run(Function &F,
                                               FunctionAnalysisManager &FAM) {
  if (!hoistAllocas(F))
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  PA.preserve<SSPLayoutAnalysis>();
  return PA;
}
