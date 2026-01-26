//===- LoopInterchange.h - Loop interchange pass --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_LOOPINTERCHANGE_H
#define LLVM_TRANSFORMS_SCALAR_LOOPINTERCHANGE_H

#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class LPMUpdater;
class LoopNest;

struct LoopInterchangePass : public PassInfoMixin<LoopInterchangePass> {
  LLVM_ABI PreservedAnalyses run(LoopNest &L, LoopAnalysisManager &AM,
                                 LoopStandardAnalysisResults &AR,
                                 LPMUpdater &U);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_LOOPINTERCHANGE_H
