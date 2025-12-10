//===- AMDGPURegPressureGuard.h - Reg Pressure Guard -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Guards transformations by measuring VGPR register pressure using the
/// AMDGPURegPressureEstimator before and after applying a pass. If the
/// pressure increases beyond a configurable threshold, the transformation
/// is reverted to prevent potential register spilling.
///
//===----------------------------------------------------------------------====//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUREGPRESSUREGUARD_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUREGPRESSUREGUARD_H

#include "AMDGPURegPressureEstimator.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class DominatorTree;
class PostDominatorTree;
class UniformityInfoAnalysis;

class Function;

/// Configuration for AMDGPU register pressure guard.
/// Values of 0 use command-line option defaults.
struct AMDGPURegPressureGuardConfig {
  unsigned MaxPercentIncrease = 0; // 0 = use cl::opt default (20)
  unsigned MinBaselineVGPRs = 0;   // 0 = use cl::opt default (96)
};

FunctionPass *createRegPressureBaselineMeasurementPass(
    const AMDGPURegPressureGuardConfig &Config);

FunctionPass *createRegPressureVerificationPass();

FunctionPass *createAMDGPURegPressureGuardLegacyPass(
    FunctionPass *WrappedPass, const AMDGPURegPressureGuardConfig &Config =
                                   AMDGPURegPressureGuardConfig());

template <typename PassT>
class AMDGPURegPressureGuardPass
    : public PassInfoMixin<AMDGPURegPressureGuardPass<PassT>> {
  PassT WrappedPass;
  AMDGPURegPressureGuardConfig Config;

public:
  explicit AMDGPURegPressureGuardPass(
      PassT Pass, const AMDGPURegPressureGuardConfig &Cfg = {})
      : WrappedPass(std::move(Pass)), Config(Cfg) {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  static bool isRequired() { return true; }
};

} // namespace llvm

#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/UniformityAnalysis.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Transforms/Utils/Cloning.h"

namespace llvm {

namespace AMDGPURegPressureGuardHelper {
bool isEnabled();
bool shouldGuardFunction(const AMDGPURegPressureGuardConfig &Config,
                         Function &F, unsigned BaselineVGPRs);
bool shouldRevert(const AMDGPURegPressureGuardConfig &Config,
                  unsigned BaselineVGPRs, unsigned NewVGPRs);
void restoreFunction(Function &F, Function &BackupFunc);
} // namespace AMDGPURegPressureGuardHelper

template <typename PassT>
PreservedAnalyses
AMDGPURegPressureGuardPass<PassT>::run(Function &F,
                                       FunctionAnalysisManager &AM) {
  using namespace AMDGPURegPressureGuardHelper;

  if (!isEnabled())
    return WrappedPass.run(F, AM);

  auto BaselineResult = AM.getResult<AMDGPURegPressureEstimatorAnalysis>(F);
  unsigned BaselineVGPRs = BaselineResult.MaxVGPRs;

  if (!shouldGuardFunction(Config, F, BaselineVGPRs))
    return WrappedPass.run(F, AM);

  ValueToValueMapTy VMap;
  Function *BackupFunc = CloneFunction(&F, VMap);

  PreservedAnalyses PA = WrappedPass.run(F, AM);

  auto NewResult = AM.getResult<AMDGPURegPressureEstimatorAnalysis>(F);
  unsigned NewVGPRs = NewResult.MaxVGPRs;

  bool ShouldRevert = shouldRevert(Config, BaselineVGPRs, NewVGPRs);

  if (ShouldRevert) {
    AM.invalidate(F, PreservedAnalyses::none());
    restoreFunction(F, *BackupFunc);
    BackupFunc->eraseFromParent();
    AM.invalidate(F, PreservedAnalyses::none());
    return PreservedAnalyses::none();
  }

  BackupFunc->eraseFromParent();
  return PA;
}

} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUREGPRESSUREGUARD_H
