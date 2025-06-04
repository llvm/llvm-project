//===-- KCFI.h - Generic KCFI operand bundle lowering -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass emits generic KCFI indirect call checks for targets that don't
// support lowering KCFI operand bundles in the back-end.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_KCFI_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_KCFI_H

#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
class KCFIPass : public PassInfoMixin<KCFIPass> {
public:
  static bool isRequired() { return true; }
  LLVM_ABI PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};
} // namespace llvm
#endif // LLVM_TRANSFORMS_INSTRUMENTATION_KCFI_H
