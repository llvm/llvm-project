//===- AIRSimdDivergenceGuard.h - Keep SIMD ops under their branch -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Metal's compiler speculates SIMD-group functions it treats as pure out of
/// the branches that guard them (if-conversion), so they run with every thread
/// of the SIMD-group active. Mark each block that runs such a call under
/// divergent control flow with llvm.sideeffect, which keeps it a branch.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIRSIMDDIVERGENCEGUARD_H
#define LLVM_LIB_TARGET_AIR_AIRSIMDDIVERGENCEGUARD_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class AIRSimdDivergenceGuardPass
    : public RequiredPassInfoMixin<AIRSimdDivergenceGuardPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

class AIRSimdDivergenceGuardLegacy : public ModulePass {
public:
  bool runOnModule(Module &M) override;
  AIRSimdDivergenceGuardLegacy() : ModulePass(ID) {}
  static char ID;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AIR_AIRSIMDDIVERGENCEGUARD_H
