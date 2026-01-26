//===-- AtomicExpand.h - Expand Atomic Instructions -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_ATOMICEXPAND_H
#define LLVM_CODEGEN_ATOMICEXPAND_H

#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class Function;
class TargetMachine;

class AtomicExpandPass : public PassInfoMixin<AtomicExpandPass> {
private:
  const TargetMachine *TM;

public:
  AtomicExpandPass(const TargetMachine &TM) : TM(&TM) {}
  LLVM_ABI PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_CODEGEN_ATOMICEXPAND_H
