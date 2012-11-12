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

#ifndef NVPTX_ALLOCA_HOISTING_H_
#define NVPTX_ALLOCA_HOISTING_H_

#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/Pass.h"
#include "llvm/DataLayout.h"

namespace llvm {

class FunctionPass;
class Function;

// Hoisting the alloca instructions in the non-entry blocks to the entry
// block.
class NVPTXAllocaHoisting : public FunctionPass {
public:
  static char ID; // Pass ID
  NVPTXAllocaHoisting() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired<DataLayout>();
    AU.addPreserved<MachineFunctionAnalysis>();
  }

  virtual const char *getPassName() const {
    return "NVPTX specific alloca hoisting";
  }

  virtual bool runOnFunction(Function &function);
};

extern FunctionPass *createAllocaHoisting();

} // end namespace llvm

#endif // NVPTX_ALLOCA_HOISTING_H_
