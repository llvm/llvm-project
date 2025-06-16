//=== GuardedLoadHardening.h - Lightweight spectre v1 mitigation *- C++ -*===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
// Lightweight load hardening as a mitigation against Spectre v1.
//===---------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_GUARDEDLOADHARDENING_H
#define LLVM_TRANSFORMS_GUARDEDLOADHARDENING_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class FunctionPass;

class GuardedLoadHardeningPass
    : public PassInfoMixin<GuardedLoadHardeningPass> {
public:
  GuardedLoadHardeningPass() = default;
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
};

FunctionPass *createGuardedLoadHardeningPass();

} // namespace llvm

#endif
