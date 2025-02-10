//===- Transforms/Instrumentation/TypeSanitizer.h - TySan Pass -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the type sanitizer pass.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_TYPESANITIZER_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_TYPESANITIZER_H

#include "llvm/IR/PassManager.h"

namespace llvm {
class Function;
class FunctionPass;
class Module;

struct TypeSanitizerPass : public PassInfoMixin<TypeSanitizerPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

} // namespace llvm

#endif /* LLVM_TRANSFORMS_INSTRUMENTATION_TYPESANITIZER_H */
