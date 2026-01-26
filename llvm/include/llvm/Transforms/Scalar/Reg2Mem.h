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

#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class RegToMemPass : public PassInfoMixin<RegToMemPass> {
public:
  LLVM_ABI PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_REG2MEM_H
