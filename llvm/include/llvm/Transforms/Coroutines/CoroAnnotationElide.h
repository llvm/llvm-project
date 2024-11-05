//===- CoroAnnotationElide.h - Elide attributed safe coroutine calls ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// This pass transforms all Call or Invoke instructions that are annotated
// "coro_elide_safe" to call the `.noalloc` variant of coroutine instead.
// The frame of the callee coroutine is allocated inside the caller. A pointer
// to the allocated frame will be passed into the `.noalloc` ramp function.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_COROUTINES_COROANNOTATIONELIDE_H
#define LLVM_TRANSFORMS_COROUTINES_COROANNOTATIONELIDE_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class Function;

struct CoroAnnotationElidePass : PassInfoMixin<CoroAnnotationElidePass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
  static bool isRequired() { return false; }
};
} // end namespace llvm

#endif // LLVM_TRANSFORMS_COROUTINES_COROANNOTATIONELIDE_H
