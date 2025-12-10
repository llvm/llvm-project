//===-- AMDGPURegPressureEstimator.h - AMDGPU Reg Pressure -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Estimates VGPR register pressure at IR level for AMDGPU.
///
/// Note: This is a conservative estimate intended for comparing register
/// pressure before and after optimization passes, not for precise register
/// allocation decisions. The estimator may overestimate pressure, especially
/// when there are duplicated extractelement and shufflevector operations,
/// as it does not fully account for optimizations like CSE.
///
//===----------------------------------------------------------------------====//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUREGPRESSUREESTIMATOR_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUREGPRESSUREESTIMATOR_H

#include "llvm/ADT/GenericUniformityInfo.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/SSAContext.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

class Function;
class DominatorTree;
class PostDominatorTree;

struct AMDGPURegPressureEstimatorResult {
  unsigned MaxVGPRs;

  AMDGPURegPressureEstimatorResult() : MaxVGPRs(0) {}
  explicit AMDGPURegPressureEstimatorResult(unsigned VGPRs) : MaxVGPRs(VGPRs) {}

  bool invalidate(Function &, const PreservedAnalyses &PA,
                  FunctionAnalysisManager::Invalidator &) {
    return !(PA.allAnalysesInSetPreserved<CFGAnalyses>() &&
             PA.allAnalysesInSetPreserved<AllAnalysesOn<Function>>());
  }
};

class AMDGPURegPressureEstimatorAnalysis
    : public AnalysisInfoMixin<AMDGPURegPressureEstimatorAnalysis> {
  friend AnalysisInfoMixin<AMDGPURegPressureEstimatorAnalysis>;
  static AnalysisKey Key;

public:
  using Result = AMDGPURegPressureEstimatorResult;

  Result run(Function &F, FunctionAnalysisManager &AM);
};

class AMDGPURegPressureEstimatorPrinterPass
    : public PassInfoMixin<AMDGPURegPressureEstimatorPrinterPass> {
  raw_ostream &OS;

public:
  explicit AMDGPURegPressureEstimatorPrinterPass(raw_ostream &OS) : OS(OS) {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  static bool isRequired() { return true; }
};

class AMDGPURegPressureEstimatorWrapperPass : public FunctionPass {
  unsigned MaxVGPRs = 0;

public:
  static char ID;

  AMDGPURegPressureEstimatorWrapperPass();

  unsigned getMaxVGPRs() const { return MaxVGPRs; }

  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  void print(raw_ostream &OS, const Module *M = nullptr) const override;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUREGPRESSUREESTIMATOR_H