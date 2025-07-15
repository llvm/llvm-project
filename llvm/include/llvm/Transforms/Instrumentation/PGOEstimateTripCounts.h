//===- PGOEstimateTripCounts.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_PGOESTIMATETRIPCOUNTS_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_PGOESTIMATETRIPCOUNTS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

struct PGOEstimateTripCountsPass
    : public PassInfoMixin<PGOEstimateTripCountsPass> {
  PGOEstimateTripCountsPass() {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_INSTRUMENTATION_PGOESTIMATETRIPCOUNTS_H
