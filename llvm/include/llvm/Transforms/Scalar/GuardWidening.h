//===- GuardWidening.h - ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Guard widening is an optimization over the @llvm.experimental.guard intrinsic
// that (optimistically) combines multiple guards into one to have fewer checks
// at runtime.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_GUARDWIDENING_H
#define LLVM_TRANSFORMS_SCALAR_GUARDWIDENING_H

#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class LPMUpdater;
class Loop;
class Function;

struct GuardWideningPass : public PassInfoMixin<GuardWideningPass> {
  LLVM_ABI PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  LLVM_ABI PreservedAnalyses run(Loop &L, LoopAnalysisManager &AM,
                                 LoopStandardAnalysisResults &AR,
                                 LPMUpdater &U);
};
}

#endif // LLVM_TRANSFORMS_SCALAR_GUARDWIDENING_H
