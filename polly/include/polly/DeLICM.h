//===------ DeLICM.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

#include "polly/Support/GICHelper.h"

namespace llvm {
class PassRegistry;
class Pass;
} // namespace llvm

namespace polly {
/// Create a new DeLICM pass instance.
llvm::Pass *createDeLICMPass();

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
void initializeDeLICMPass(llvm::PassRegistry &);
} // namespace llvm

#endif /* POLLY_DELICM_H */
