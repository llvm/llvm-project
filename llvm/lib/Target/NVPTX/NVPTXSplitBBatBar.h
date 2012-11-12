//===-- llvm/lib/Target/NVPTX/NVPTXSplitBBatBar.h ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the NVIDIA specific declarations
// for splitting basic blocks at barrier instructions.
//
//===----------------------------------------------------------------------===//

#ifndef NVPTX_SPLIT_BB_AT_BAR_H
#define NVPTX_SPLIT_BB_AT_BAR_H

#include "llvm/Pass.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"

namespace llvm {

// actual analysis class, which is a functionpass
struct NVPTXSplitBBatBar : public FunctionPass {
  static char ID;

  NVPTXSplitBBatBar() : FunctionPass(ID) {}
  void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addPreserved<MachineFunctionAnalysis>();
  }
  virtual bool runOnFunction(Function &F);

  virtual const char *getPassName() const {
    return "Split basic blocks at barrier";
  }
};

extern FunctionPass *createSplitBBatBarPass();
}

#endif //NVPTX_SPLIT_BB_AT_BAR_H
