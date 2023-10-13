//===- MarkColdFunctions.h - ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_MARKCOLDFUNCTIONS_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_MARKCOLDFUNCTIONS_H

#include "llvm/IR/PassManager.h"
#include "llvm/Support/PGOOptions.h"

namespace llvm {

struct MarkColdFunctionsPass : public PassInfoMixin<MarkColdFunctionsPass> {
  MarkColdFunctionsPass(PGOOptions::ColdFuncAttr ColdType)
      : ColdType(ColdType) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

private:
  PGOOptions::ColdFuncAttr ColdType;
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_INSTRUMENTATION_MARKCOLDFUNCTIONS_H
