//===- ToUnreachable.h - Turn function into unreachable. --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_TOUNREACHABLE_H
#define LLVM_TRANSFORMS_SCALAR_TOUNREACHABLE_H

#include "llvm/IR/PassManager.h"

namespace llvm {
struct ToUnreachablePass : public PassInfoMixin<ToUnreachablePass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};
} // namespace llvm

#endif
