//===- PruneUnprofitable.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Mark a SCoP as unfeasible if not deemed profitable to optimize.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_PRUNEUNPROFITABLE_H
#define POLLY_PRUNEUNPROFITABLE_H

#include "polly/ScopPass.h"

namespace llvm {
class Pass;
class PassRegistry;
} // namespace llvm

namespace polly {
llvm::Pass *createPruneUnprofitableWrapperPass();

struct PruneUnprofitablePass : llvm::PassInfoMixin<PruneUnprofitablePass> {
  PruneUnprofitablePass() {}

  llvm::PreservedAnalyses run(Scop &S, ScopAnalysisManager &SAM,
                              ScopStandardAnalysisResults &SAR, SPMUpdater &U);
};
} // namespace polly

namespace llvm {
void initializePruneUnprofitableWrapperPassPass(PassRegistry &);
}

#endif // POLLY_PRUNEUNPROFITABLE_H
