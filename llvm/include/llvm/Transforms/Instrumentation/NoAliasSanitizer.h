//===- NoAliasSanitizer.h - NoAlias attribute violation detector ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the NoAliasSanitizer pass, which instruments code to
// detect violations of the noalias attribute semantics at runtime.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_NOALIASSANITIZER_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_NOALIASSANITIZER_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class NoAliasSanitizerPass : public PassInfoMixin<NoAliasSanitizerPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_INSTRUMENTATION_NOALIASSANITIZER_H
