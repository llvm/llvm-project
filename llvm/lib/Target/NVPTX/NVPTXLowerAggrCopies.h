//===-- llvm/lib/Target/NVPTX/NVPTXLowerAggrCopies.h ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the NVIDIA specific lowering of
// aggregate copies
//
//===----------------------------------------------------------------------===//

#ifndef NVPTX_LOWER_AGGR_COPIES_H
#define NVPTX_LOWER_AGGR_COPIES_H

#include "llvm/Pass.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/DataLayout.h"

namespace llvm {

// actual analysis class, which is a functionpass
struct NVPTXLowerAggrCopies : public FunctionPass {
  static char ID;

  NVPTXLowerAggrCopies() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired<DataLayout>();
    AU.addPreserved<MachineFunctionAnalysis>();
  }

  virtual bool runOnFunction(Function &F);

  static const unsigned MaxAggrCopySize = 128;

  virtual const char *getPassName() const {
    return "Lower aggregate copies/intrinsics into loops";
  }
};

extern FunctionPass *createLowerAggrCopies();
}

#endif
