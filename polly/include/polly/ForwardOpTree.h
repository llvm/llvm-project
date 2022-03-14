//===- ForwardOpTree.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Move instructions between statements.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_FORWARDOPTREE_H
#define POLLY_FORWARDOPTREE_H

#include "polly/ScopPass.h"

namespace llvm {
class PassRegistry;
} // namespace llvm

namespace polly {
llvm::Pass *createForwardOpTreeWrapperPass();
llvm::Pass *createForwardOpTreePrinterLegacyPass(llvm::raw_ostream &OS);

struct ForwardOpTreePass : llvm::PassInfoMixin<ForwardOpTreePass> {
  ForwardOpTreePass() {}

  llvm::PreservedAnalyses run(Scop &S, ScopAnalysisManager &SAM,
                              ScopStandardAnalysisResults &SAR, SPMUpdater &U);
};

struct ForwardOpTreePrinterPass
    : llvm::PassInfoMixin<ForwardOpTreePrinterPass> {
  ForwardOpTreePrinterPass(raw_ostream &OS) : OS(OS) {}

  PreservedAnalyses run(Scop &S, ScopAnalysisManager &,
                        ScopStandardAnalysisResults &SAR, SPMUpdater &);

private:
  llvm::raw_ostream &OS;
};

} // namespace polly

namespace llvm {
void initializeForwardOpTreeWrapperPassPass(PassRegistry &);
void initializeForwardOpTreePrinterLegacyPassPass(PassRegistry &);
} // namespace llvm

#endif // POLLY_FORWARDOPTREE_H
