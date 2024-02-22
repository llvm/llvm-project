//===- SoftPointerAuth.h - Software lowering of ptrauth intrins -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_SOFTPOINTERAUTH_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_SOFTPOINTERAUTH_H

#include "llvm/IR/PassManager.h"

namespace llvm {

struct SoftPointerAuthPass : public PassInfoMixin<SoftPointerAuthPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_INSTRUMENTATION_SOFTPOINTERAUTH_H
