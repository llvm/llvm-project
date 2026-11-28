//===- LoopSplitPass.h - Test driver for LoopSplit --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A command-line driven pass used to exercise the LoopSplit utility from `opt`.
// The split points are provided via the -loop-split-points option as iteration
// offsets relative to the induction start.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_LOOPSPLITPASS_H
#define LLVM_TRANSFORMS_UTILS_LOOPSPLITPASS_H

#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class LoopSplitPass : public OptionalPassInfoMixin<LoopSplitPass> {
public:
  LLVM_ABI PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_LOOPSPLITPASS_H
