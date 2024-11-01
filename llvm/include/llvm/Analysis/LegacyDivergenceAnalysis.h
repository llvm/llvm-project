//===- llvm/Analysis/LegacyDivergenceAnalysis.h - KernelDivergence Analysis -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The kernel divergence analysis is an LLVM pass which can be used to find out
// if a branch instruction in a GPU program (kernel) is divergent or not. It can help
// branch optimizations such as jump threading and loop unswitching to make
// better decisions.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_ANALYSIS_LEGACYDIVERGENCEANALYSIS_H
#define LLVM_ANALYSIS_LEGACYDIVERGENCEANALYSIS_H

#include "llvm/ADT/DenseSet.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include <memory>

namespace llvm {
class DivergenceInfo;
class Function;
class Module;
class raw_ostream;
class TargetTransformInfo;
class Use;
class Value;

class LegacyDivergenceAnalysisImpl {
public:
  // Returns true if V is divergent at its definition.
  bool isDivergent(const Value *V) const;

  // Returns true if U is divergent. Uses of a uniform value can be divergent.
  bool isDivergentUse(const Use *U) const;

  // Returns true if V is uniform/non-divergent.
  bool isUniform(const Value *V) const { return !isDivergent(V); }

  // Returns true if U is uniform/non-divergent. Uses of a uniform value can be
  // divergent.
  bool isUniformUse(const Use *U) const { return !isDivergentUse(U); }

  // Keep the analysis results uptodate by removing an erased value.
  void removeValue(const Value *V) { DivergentValues.erase(V); }

  // Print all divergent branches in the function.
  void print(raw_ostream &OS, const Module *) const;

  // Whether analysis should be performed by GPUDivergenceAnalysis.
  bool shouldUseGPUDivergenceAnalysis(const Function &F,
                                      const TargetTransformInfo &TTI,
                                      const LoopInfo &LI);

  void run(Function &F, TargetTransformInfo &TTI, DominatorTree &DT,
           PostDominatorTree &PDT, const LoopInfo &LI);

protected:
  // (optional) handle to new DivergenceAnalysis
  std::unique_ptr<DivergenceInfo> gpuDA;

  // Stores all divergent values.
  DenseSet<const Value *> DivergentValues;

  // Stores divergent uses of possibly uniform values.
  DenseSet<const Use *> DivergentUses;
};

class LegacyDivergenceAnalysis : public FunctionPass,
                                 public LegacyDivergenceAnalysisImpl {
public:
  static char ID;

  LegacyDivergenceAnalysis();
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnFunction(Function &F) override;
};

class LegacyDivergenceAnalysisPass
    : public PassInfoMixin<LegacyDivergenceAnalysisPass>,
      public LegacyDivergenceAnalysisImpl {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

private:
  // (optional) handle to new DivergenceAnalysis
  std::unique_ptr<DivergenceInfo> gpuDA;

  // Stores all divergent values.
  DenseSet<const Value *> DivergentValues;

  // Stores divergent uses of possibly uniform values.
  DenseSet<const Use *> DivergentUses;
};

} // end namespace llvm

#endif // LLVM_ANALYSIS_LEGACYDIVERGENCEANALYSIS_H
