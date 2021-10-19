//===------ DeLICM.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Undo the effect of Loop Invariant Code Motion (LICM) and
// GVN Partial Redundancy Elimination (PRE) on SCoP-level.
//
// Namely, remove register/scalar dependencies by mapping them back to array
// elements.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_DELICM_H
#define POLLY_DELICM_H

#include "polly/ScopPass.h"
#include "isl/isl-noexceptions.h"

namespace llvm {
class PassRegistry;
class Pass;
class raw_ostream;
} // namespace llvm

namespace polly {
/// Create a new DeLICM pass instance.
llvm::Pass *createDeLICMWrapperPass();

struct DeLICMPass : llvm::PassInfoMixin<DeLICMPass> {
  DeLICMPass() {}

  llvm::PreservedAnalyses run(Scop &S, ScopAnalysisManager &SAM,
                              ScopStandardAnalysisResults &SAR, SPMUpdater &U);
};

struct DeLICMPrinterPass : llvm::PassInfoMixin<DeLICMPrinterPass> {
  DeLICMPrinterPass(raw_ostream &OS) : OS(OS) {}

  PreservedAnalyses run(Scop &S, ScopAnalysisManager &,
                        ScopStandardAnalysisResults &SAR, SPMUpdater &);

private:
  llvm::raw_ostream &OS;
};

/// Determine whether two lifetimes are conflicting.
///
/// Used by unittesting.
bool isConflicting(isl::union_set ExistingOccupied,
                   isl::union_set ExistingUnused, isl::union_map ExistingKnown,
                   isl::union_map ExistingWrites,
                   isl::union_set ProposedOccupied,
                   isl::union_set ProposedUnused, isl::union_map ProposedKnown,
                   isl::union_map ProposedWrites,
                   llvm::raw_ostream *OS = nullptr, unsigned Indent = 0);

} // namespace polly

namespace llvm {
void initializeDeLICMWrapperPassPass(llvm::PassRegistry &);
} // namespace llvm

#endif /* POLLY_DELICM_H */
