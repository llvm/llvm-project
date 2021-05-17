//===- polly/ScheduleOptimizer.h - The Schedule Optimizer -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_SCHEDULEOPTIMIZER_H
#define POLLY_SCHEDULEOPTIMIZER_H

#include "polly/ScopPass.h"

namespace llvm {
class Pass;
class PassRegistry;
} // namespace llvm

namespace polly {
llvm::Pass *createIslScheduleOptimizerWrapperPass();

struct IslScheduleOptimizerPass
    : llvm::PassInfoMixin<IslScheduleOptimizerPass> {
  IslScheduleOptimizerPass() {}

  llvm::PreservedAnalyses run(Scop &S, ScopAnalysisManager &SAM,
                              ScopStandardAnalysisResults &SAR, SPMUpdater &U);
};

struct IslScheduleOptimizerPrinterPass
    : llvm::PassInfoMixin<IslScheduleOptimizerPrinterPass> {
  IslScheduleOptimizerPrinterPass(raw_ostream &OS) : OS(OS) {}

  PreservedAnalyses run(Scop &S, ScopAnalysisManager &,
                        ScopStandardAnalysisResults &SAR, SPMUpdater &);

private:
  llvm::raw_ostream &OS;
};

/// Build the desired set of partial tile prefixes.
///
/// We build a set of partial tile prefixes, which are prefixes of the vector
/// loop that have exactly VectorWidth iterations.
///
/// 1. Drop all constraints involving the dimension that represents the
///    vector loop.
/// 2. Constrain the last dimension to get a set, which has exactly VectorWidth
///    iterations.
/// 3. Subtract loop domain from it, project out the vector loop dimension and
///    get a set that contains prefixes, which do not have exactly VectorWidth
///    iterations.
/// 4. Project out the vector loop dimension of the set that was build on the
///    first step and subtract the set built on the previous step to get the
///    desired set of prefixes.
///
/// @param ScheduleRange A range of a map, which describes a prefix schedule
///                      relation.
isl::set getPartialTilePrefixes(isl::set ScheduleRange, int VectorWidth);
} // namespace polly

namespace llvm {
void initializeIslScheduleOptimizerWrapperPassPass(llvm::PassRegistry &);
}

#endif // POLLY_SCHEDULEOPTIMIZER_H
