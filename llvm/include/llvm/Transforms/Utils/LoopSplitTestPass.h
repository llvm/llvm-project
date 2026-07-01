//===- LoopSplitTestPass.h - Test driver for LoopSplitUtils -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A command-line driven pass used to exercise the LoopSplitUtils utility from
// `opt`. The split points are provided via the -loop-split-points option as
// iteration offsets relative to the induction start.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_LOOPSPLITTESTPASS_H
#define LLVM_TRANSFORMS_UTILS_LOOPSPLITTESTPASS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class LoopSplitTestPass : public PassInfoMixin<LoopSplitTestPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_LOOPSPLITTESTPASS_H
