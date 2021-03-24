//===- DeadCodeElimination.h ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Eliminate dead iterations.
//
//===----------------------------------------------------------------------===//


#ifndef POLLY_DEADCODEELIMINATION_H
#define POLLY_DEADCODEELIMINATION_H

#include "polly/ScopPass.h"

namespace llvm {
class PassRegistry;
class Pass;
class raw_ostream;
} // namespace llvm

namespace polly {
llvm::Pass *createDeadCodeElimWrapperPass();

struct DeadCodeElimPass : llvm::PassInfoMixin<DeadCodeElimPass> {
  DeadCodeElimPass() {}

  llvm::PreservedAnalyses run(Scop &S, ScopAnalysisManager &SAM,                              ScopStandardAnalysisResults &SAR, SPMUpdater &U);
};


} // namespace polly

namespace llvm {
void initializeDeadCodeElimWrapperPassPass(llvm::PassRegistry &);
} // namespace llvm

#endif /* POLLY_DEADCODEELIMINATION_H */
