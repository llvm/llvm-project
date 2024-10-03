//===- Reg2Mem.h - Convert registers to allocas -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides the interface for the RegToMem Pass.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_REG2MEM_H
#define LLVM_TRANSFORMS_SCALAR_REG2MEM_H

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

namespace llvm {

class RegToMemPass : public PassInfoMixin<RegToMemPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

class RegToMemWrapperPass : public FunctionPass {
public:
  static char ID;

  RegToMemWrapperPass();

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();

    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();

    AU.addPreserved<LoopInfoWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
  }

  bool runOnFunction(Function &F) override;
};

FunctionPass *createRegToMemWrapperPass();

} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_REG2MEM_H
