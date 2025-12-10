//===- AMDGPURegPressureGuard.cpp - Register Pressure Guarded Pass Wrapper ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a guard mechanism for IR transformations that measures
/// VGPR register pressure before and after applying a pass, reverting the
/// transformation if pressure increases beyond a configurable threshold.
///
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPURegPressureGuard.h"
#include "AMDGPURegPressureEstimator.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/UniformityAnalysis.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <memory>

using namespace llvm;

#define DEBUG_TYPE "amdgpu-reg-pressure-guard"

static cl::opt<bool> EnableRegPressureGuard(
    "amdgpu-enable-reg-pressure-guard",
    cl::desc("Enable AMDGPU register pressure guard to revert transformations "
             "that increase VGPR pressure beyond threshold"),
    cl::init(true));

static cl::opt<unsigned> MaxPercentIncreaseOpt(
    "amdgpu-reg-pressure-max-increase",
    cl::desc("Maximum allowed percentage increase in VGPR pressure"),
    cl::init(20));

static cl::opt<unsigned> MinBaselineVGPRsOpt(
    "amdgpu-reg-pressure-min-baseline",
    cl::desc("Minimum baseline VGPRs required to enable guard"),
    cl::init(96));

STATISTIC(NumFunctionsGuarded, "Number of functions checked by guard");
STATISTIC(NumTransformationsReverted, "Number of transformations reverted");
STATISTIC(NumTransformationsKept, "Number of transformations kept");

namespace llvm {
namespace AMDGPURegPressureGuardHelper {

bool isEnabled() { return EnableRegPressureGuard; }

bool shouldGuardFunction(const AMDGPURegPressureGuardConfig &Config,
                         Function &F, unsigned BaselineVGPRs) {
  unsigned MinBaseline = Config.MinBaselineVGPRs > 0
                             ? Config.MinBaselineVGPRs
                             : MinBaselineVGPRsOpt;

  if (BaselineVGPRs < MinBaseline) {
    LLVM_DEBUG(dbgs() << "AMDGPURegPressureGuard: Skipping " << F.getName()
                      << " - baseline VGPRs (" << BaselineVGPRs
                      << ") below threshold (" << MinBaseline
                      << ")\n");
    return false;
  }
  return true;
}

bool shouldRevert(const AMDGPURegPressureGuardConfig &Config,
                  unsigned BaselineVGPRs, unsigned NewVGPRs) {
  if (NewVGPRs <= BaselineVGPRs) {
    LLVM_DEBUG(dbgs() << "AMDGPURegPressureGuard: Keeping transformation - "
                      << "VGPRs decreased from " << BaselineVGPRs << " to "
                      << NewVGPRs << "\n");
    return false;
  }

  if (BaselineVGPRs == 0)
    return false;

  unsigned PercentIncrease = ((NewVGPRs - BaselineVGPRs) * 100) / BaselineVGPRs;
  unsigned MaxIncrease = Config.MaxPercentIncrease > 0
                             ? Config.MaxPercentIncrease
                             : MaxPercentIncreaseOpt;

  if (PercentIncrease > MaxIncrease) {
    LLVM_DEBUG(dbgs() << "AMDGPURegPressureGuard: Reverting transformation - "
                      << "VGPR increase " << PercentIncrease
                      << "% exceeds limit " << MaxIncrease
                      << "%\n");
    return true;
  }

  LLVM_DEBUG(dbgs() << "AMDGPURegPressureGuard: Keeping transformation - "
                    << "VGPR increase " << PercentIncrease
                    << "% within limit\n");
  return false;
}

void restoreFunction(Function &F, Function &BackupFunc) {
  F.dropAllReferences();

  ValueToValueMapTy VMap;
  auto *DestI = F.arg_begin();
  for (Argument &I : BackupFunc.args()) {
    VMap[&I] = &*DestI++;
  }

  SmallVector<ReturnInst *, 8> Returns;
  CloneFunctionInto(&F, &BackupFunc, VMap,
                    CloneFunctionChangeType::LocalChangesOnly, Returns);
}

} // namespace AMDGPURegPressureGuardHelper
} // namespace llvm

namespace {

struct RegPressureGuardState {
  unsigned BaselineVGPRs = 0;
  Function *BackupFunc = nullptr;
  AMDGPURegPressureGuardConfig Config;
  bool ShouldGuard = false;
};

static DenseMap<Function *, std::unique_ptr<RegPressureGuardState>>
    GuardStateMap;

class RegPressureBaselineMeasurementPass : public FunctionPass {
  AMDGPURegPressureGuardConfig Config;

public:
  static char ID;

  explicit RegPressureBaselineMeasurementPass(
      const AMDGPURegPressureGuardConfig &Cfg = {})
      : FunctionPass(ID), Config(Cfg) {
    initializeRegPressureBaselineMeasurementPassPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    if (!EnableRegPressureGuard)
      return false;

    auto &EstimatorPass =
        getAnalysis<AMDGPURegPressureEstimatorWrapperPass>();
    unsigned BaselineVGPRs = EstimatorPass.getMaxVGPRs();

    bool ShouldGuard = llvm::AMDGPURegPressureGuardHelper::shouldGuardFunction(
        Config, F, BaselineVGPRs);

    if (!ShouldGuard) {
      LLVM_DEBUG(dbgs() << "AMDGPURegPressureGuard: Skipping " << F.getName()
                        << "\n");
      return false;
    }

    ++NumFunctionsGuarded;

    LLVM_DEBUG(dbgs() << "AMDGPURegPressureGuard: Measuring baseline for "
                      << F.getName() << " (baseline: " << BaselineVGPRs
                      << " VGPRs)\n");

    auto State = std::make_unique<RegPressureGuardState>();
    State->BaselineVGPRs = BaselineVGPRs;
    State->Config = Config;
    State->ShouldGuard = true;

    ValueToValueMapTy VMap;
    State->BackupFunc = CloneFunction(&F, VMap);

    GuardStateMap[&F] = std::move(State);

    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AMDGPURegPressureEstimatorWrapperPass>();
    AU.setPreservesAll();
  }

  StringRef getPassName() const override {
    return "AMDGPU Register Pressure Baseline Measurement";
  }
};

class RegPressureVerificationPass : public FunctionPass {
public:
  static char ID;

  RegPressureVerificationPass() : FunctionPass(ID) {
    initializeRegPressureVerificationPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    auto It = GuardStateMap.find(&F);
    if (It == GuardStateMap.end())
      return false;

    auto &State = *It->second;
    if (!State.ShouldGuard) {
      GuardStateMap.erase(It);
      return false;
    }

    auto &EstimatorPass =
        getAnalysis<AMDGPURegPressureEstimatorWrapperPass>();
    unsigned NewVGPRs = EstimatorPass.getMaxVGPRs();

    LLVM_DEBUG(dbgs() << "AMDGPURegPressureGuard: Verifying " << F.getName()
                      << " (baseline: " << State.BaselineVGPRs
                      << ", new: " << NewVGPRs << " VGPRs)\n");

    bool ShouldRevert = llvm::AMDGPURegPressureGuardHelper::shouldRevert(
        State.Config, State.BaselineVGPRs, NewVGPRs);

    if (ShouldRevert) {
      LLVM_DEBUG(dbgs() << "AMDGPURegPressureGuard: Reverting " << F.getName()
                        << "\n");
      llvm::AMDGPURegPressureGuardHelper::restoreFunction(F, *State.BackupFunc);
      ++NumTransformationsReverted;
    } else {
      ++NumTransformationsKept;
    }

    State.BackupFunc->eraseFromParent();
    GuardStateMap.erase(It);

    return ShouldRevert;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AMDGPURegPressureEstimatorWrapperPass>();
  }

  StringRef getPassName() const override {
    return "AMDGPU Register Pressure Verification";
  }
};

} // end anonymous namespace

char RegPressureBaselineMeasurementPass::ID = 0;
char RegPressureVerificationPass::ID = 0;

INITIALIZE_PASS_BEGIN(RegPressureBaselineMeasurementPass,
                      "amdgpu-reg-pressure-baseline",
                      "AMDGPU Register Pressure Baseline Measurement", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(AMDGPURegPressureEstimatorWrapperPass)
INITIALIZE_PASS_END(RegPressureBaselineMeasurementPass,
                    "amdgpu-reg-pressure-baseline",
                    "AMDGPU Register Pressure Baseline Measurement", false,
                    false)

INITIALIZE_PASS_BEGIN(RegPressureVerificationPass,
                      "amdgpu-reg-pressure-verification",
                      "AMDGPU Register Pressure Verification", false, false)
INITIALIZE_PASS_DEPENDENCY(AMDGPURegPressureEstimatorWrapperPass)
INITIALIZE_PASS_END(RegPressureVerificationPass,
                    "amdgpu-reg-pressure-verification",
                    "AMDGPU Register Pressure Verification", false, false)

namespace llvm {

FunctionPass *createRegPressureBaselineMeasurementPass(
    const AMDGPURegPressureGuardConfig &Config) {
  return new RegPressureBaselineMeasurementPass(Config);
}

FunctionPass *createRegPressureVerificationPass() {
  return new RegPressureVerificationPass();
}

} // namespace llvm

class AMDGPURegPressureGuardLegacyPass : public FunctionPass {
public:
  static char ID;

  AMDGPURegPressureGuardLegacyPass() : FunctionPass(ID) {
    initializeAMDGPURegPressureGuardLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &) override {
    llvm_unreachable("Use createRegPressureBaselineMeasurementPass + "
                     "transformation pass + createRegPressureVerificationPass");
    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }
};

char AMDGPURegPressureGuardLegacyPass::ID = 0;

INITIALIZE_PASS(AMDGPURegPressureGuardLegacyPass,
                "amdgpu-reg-pressure-guard-legacy",
                "AMDGPU Register Pressure Guard (Legacy - Deprecated)", false,
                false)

namespace llvm {

FunctionPass *createAMDGPURegPressureGuardLegacyPass(
    FunctionPass *WrappedPass, const AMDGPURegPressureGuardConfig &Config) {
  llvm_unreachable("Use createRegPressureBaselineMeasurementPass + "
                   "transformation pass + createRegPressureVerificationPass");
  return nullptr;
}

} // namespace llvm
