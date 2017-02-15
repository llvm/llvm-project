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
} // anonymous namespace

namespace polly {
/// Create a new DeLICM pass instance.
llvm::Pass *createDeLICMPass();

/// Determine whether two lifetimes are conflicting.
///
/// Used by unittesting.
bool isConflicting(IslPtr<isl_union_set> ExistingOccupied,
                   IslPtr<isl_union_set> ExistingUnused,
                   IslPtr<isl_union_set> ExistingWrites,
                   IslPtr<isl_union_set> ProposedOccupied,
                   IslPtr<isl_union_set> ProposedUnused,
                   IslPtr<isl_union_set> ProposedWrites,
                   llvm::raw_ostream *OS = nullptr, unsigned Indent = 0);
} // namespace polly

namespace llvm {
void initializeDeLICMPass(llvm::PassRegistry &);
} // namespace llvm

#endif /* POLLY_DELICM_H */
