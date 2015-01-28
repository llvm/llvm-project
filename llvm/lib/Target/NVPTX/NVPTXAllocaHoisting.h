//===-- AllocaHoisting.h - Hosist allocas to the entry block ----*- C++ -*-===//
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

#ifndef LLVM_LIB_TARGET_NVPTX_NVPTXALLOCAHOISTING_H
#define LLVM_LIB_TARGET_NVPTX_NVPTXALLOCAHOISTING_H

#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/CodeGen/StackProtector.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Pass.h"

namespace llvm {

class FunctionPass;
class Function;

// Hoisting the alloca instructions in the non-entry blocks to the entry
// block.
class NVPTXAllocaHoisting : public FunctionPass {
public:
  static char ID; // Pass ID
  NVPTXAllocaHoisting() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DataLayoutPass>();
    AU.addPreserved<MachineFunctionAnalysis>();
    AU.addPreserved<StackProtector>();
  }

  const char *getPassName() const override {
    return "NVPTX specific alloca hoisting";
  }

  bool runOnFunction(Function &function) override;
};

extern FunctionPass *createAllocaHoisting();

} // end namespace llvm

#endif
