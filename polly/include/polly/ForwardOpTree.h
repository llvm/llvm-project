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

namespace polly {

struct ForwardOpTreePass final : llvm::PassInfoMixin<ForwardOpTreePass> {
  ForwardOpTreePass() {}

  llvm::PreservedAnalyses run(Scop &S, ScopAnalysisManager &SAM,
                              ScopStandardAnalysisResults &SAR, SPMUpdater &U);
};

struct ForwardOpTreePrinterPass final
    : llvm::PassInfoMixin<ForwardOpTreePrinterPass> {
  ForwardOpTreePrinterPass(raw_ostream &OS) : OS(OS) {}

  PreservedAnalyses run(Scop &S, ScopAnalysisManager &,
                        ScopStandardAnalysisResults &SAR, SPMUpdater &);

private:
  llvm::raw_ostream &OS;
};

/// Pass that redirects scalar reads to array elements that are known to contain
/// the same value.
///
/// This reduces the number of scalar accesses and therefore potentially
/// increases the freedom of the scheduler. In the ideal case, all reads of a
/// scalar definition are redirected (We currently do not care about removing
/// the write in this case).  This is also useful for the main DeLICM pass as
/// there are less scalars to be mapped.
bool runForwardOpTree(Scop &S);
} // namespace polly

#endif // POLLY_FORWARDOPTREE_H
