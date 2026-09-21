//===- ConcurrencySanitizer.h - ConcurrencySanitizer ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_CONCURRENCYSANITIZER_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_CONCURRENCYSANITIZER_H

#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

struct ConcurrencySanitizerPass
    : public RequiredPassInfoMixin<ConcurrencySanitizerPass> {
  LLVM_ABI PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
};

struct ModuleConcurrencySanitizerPass
    : public RequiredPassInfoMixin<ModuleConcurrencySanitizerPass> {
  LLVM_ABI PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_INSTRUMENTATION_CONCURRENCYSANITIZER_H
