//===- AMDGPUPerfHintAnalysis.h ---- analysis of memory traffic -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Analyzes if a function potentially memory bound and if a kernel
/// kernel may benefit from limiting number of waves to reduce cache thrashing.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUPERFHINTANALYSIS_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUPERFHINTANALYSIS_H

#include "llvm/IR/PassManager.h"
#include "llvm/IR/ValueMap.h"

#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LazyCallGraph.h"

namespace llvm {

class AMDGPUPerfHintAnalysis;
class CallGraphSCC;
class GCNTargetMachine;
class LazyCallGraph;

class AMDGPUPerfHintAnalysis {
public:
  struct FuncInfo {
    unsigned MemInstCost;
    unsigned InstCost;
    unsigned IAMInstCost;      // Indirect access memory instruction count
    unsigned LSMInstCost;      // Large stride memory instruction count
    bool HasDenseGlobalMemAcc; // Set if at least 1 basic block has relatively
                               // high global memory access
    FuncInfo()
        : MemInstCost(0), InstCost(0), IAMInstCost(0), LSMInstCost(0),
          HasDenseGlobalMemAcc(false) {}
  };

  typedef ValueMap<const Function *, FuncInfo> FuncInfoMap;

private:
  FuncInfoMap FIM;

public:
  AMDGPUPerfHintAnalysis() {}

  // OldPM
  bool runOnSCC(const GCNTargetMachine &TM, CallGraphSCC &SCC);

  // NewPM
  bool run(const GCNTargetMachine &TM, LazyCallGraph &CG);

  bool isMemoryBound(const Function *F) const;

  bool needsWaveLimiter(const Function *F) const;
};

struct AMDGPUPerfHintAnalysisPass
    : public PassInfoMixin<AMDGPUPerfHintAnalysisPass> {
  const GCNTargetMachine &TM;
  std::unique_ptr<AMDGPUPerfHintAnalysis> Impl;

  AMDGPUPerfHintAnalysisPass(const GCNTargetMachine &TM)
      : TM(TM), Impl(std::make_unique<AMDGPUPerfHintAnalysis>()) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm
#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUPERFHINTANALYSIS_H
