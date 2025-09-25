//===- ExpandFp.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_EXPANDFP_H
#define LLVM_CODEGEN_EXPANDFP_H

#include "llvm/IR/PassManager.h"
#include "llvm/Support/CodeGen.h"

namespace llvm {

class TargetMachine;

class ExpandFpPass : public PassInfoMixin<ExpandFpPass> {
private:
  const TargetMachine *TM;
  CodeGenOptLevel OptLevel;

public:
  explicit ExpandFpPass(const TargetMachine *TM, CodeGenOptLevel OptLevel);

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }
  void printPipeline(raw_ostream &OS,
                     function_ref<StringRef(StringRef)> MapClassName2PassName);
};

} // end namespace llvm

#endif // LLVM_CODEGEN_EXPANDFP_H
