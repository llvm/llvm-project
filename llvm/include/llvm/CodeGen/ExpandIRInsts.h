//===- ExpandIRInsts.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_EXPANDIRINSTS_H
#define LLVM_CODEGEN_EXPANDIRINSTS_H

#include "llvm/IR/PassManager.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class TargetMachine;

class ExpandIRInstsPass : public PassInfoMixin<ExpandIRInstsPass> {
private:
  const TargetMachine *TM;
  CodeGenOptLevel OptLevel;

public:
  LLVM_ABI explicit ExpandIRInstsPass(const TargetMachine &TM,
                                      CodeGenOptLevel OptLevel);

  LLVM_ABI PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }
  LLVM_ABI void
  printPipeline(raw_ostream &OS,
                function_ref<StringRef(StringRef)> MapClassName2PassName);
};

} // end namespace llvm

#endif // LLVM_CODEGEN_EXPANDIRINSTS_H
