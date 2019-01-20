//===- LoopRotation.h - Loop Rotation -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides the interface for the Loop Rotation pass.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_LOOPROTATION_H
#define LLVM_TRANSFORMS_SCALAR_LOOPROTATION_H

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"

namespace llvm {

/// A simple loop rotation transformation.
class LoopRotatePass : public PassInfoMixin<LoopRotatePass> {
public:
  LoopRotatePass(bool EnableHeaderDuplication = true);
  PreservedAnalyses run(Loop &L, LoopAnalysisManager &AM,
                        LoopStandardAnalysisResults &AR, LPMUpdater &U);

private:
  const bool EnableHeaderDuplication;
};
}

#endif // LLVM_TRANSFORMS_SCALAR_LOOPROTATION_H
