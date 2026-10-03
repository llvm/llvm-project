//===- DeadBranchElimination.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Eliminates conditional branches that are unreachable, but that cannot be
// proven unreachable directly because the branch body modifies the values the
// condition depends on (a circular dependency). Uses an optimistic fixed
// point: assume all branch bodies dead, temporarily rewrite the PHI slots
// they feed (journaled and undone in place, no cloning), re-run
// ScalarEvolution, and restore every body whose branch edge cannot be proven
// never-taken, until the assumption set is self-consistent.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_DEADBRANCHELIMINATION_H
#define LLVM_TRANSFORMS_SCALAR_DEADBRANCHELIMINATION_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class Module;

class DeadBranchEliminationPass
    : public PassInfoMixin<DeadBranchEliminationPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_DEADBRANCHELIMINATION_H
