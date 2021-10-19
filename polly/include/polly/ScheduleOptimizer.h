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
} // namespace polly

namespace llvm {
void initializeIslScheduleOptimizerWrapperPassPass(llvm::PassRegistry &);
}

#endif // POLLY_SCHEDULEOPTIMIZER_H
