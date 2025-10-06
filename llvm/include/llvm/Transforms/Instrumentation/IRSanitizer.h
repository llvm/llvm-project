//===----------- IRSanitizer.h - IRSanitizer instrumentation ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the IRSanitizer class.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_IRSANITIZER_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_IRSANITIZER_H

#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"

namespace llvm {

class IRSanitizerPass : public PassInfoMixin<IRSanitizerPass> {
public:
  IRSanitizerPass();
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
  PreservedAnalyses sanitizeFunction(Function &F, FunctionAnalysisManager &FAM);

private:
  Module *M;
  Type *PtrIntTy;
};

} // namespace llvm

#endif
