//===- LoopStripMinePass.h - Tapir loop stripmining -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_TAPIR_LOOPSTRIPMINEPASS_H
#define LLVM_TRANSFORMS_TAPIR_LOOPSTRIPMINEPASS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class Function;

extern cl::opt<bool> EnableTapirLoopStripmine;

/// Loop stripmining pass.  It is a function pass to have access to function and
/// module analyses.
class LoopStripMinePass : public PassInfoMixin<LoopStripMinePass> {
public:
  explicit LoopStripMinePass() {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_TAPIR_LOOPSTRIPMINEPASS_H
