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

#ifndef LLVM_LIB_TARGET_NVPTX_NVPTXLOWERAGGRCOPIES_H
#define LLVM_LIB_TARGET_NVPTX_NVPTXLOWERAGGRCOPIES_H

#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/CodeGen/StackProtector.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Pass.h"

namespace llvm {

// actual analysis class, which is a functionpass
struct NVPTXLowerAggrCopies : public FunctionPass {
  static char ID;

  NVPTXLowerAggrCopies() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DataLayoutPass>();
    AU.addPreserved<MachineFunctionAnalysis>();
    AU.addPreserved<StackProtector>();
  }

  bool runOnFunction(Function &F) override;

  static const unsigned MaxAggrCopySize = 128;

  const char *getPassName() const override {
    return "Lower aggregate copies/intrinsics into loops";
  }
};

extern FunctionPass *createLowerAggrCopies();
}

#endif
