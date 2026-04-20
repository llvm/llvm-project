//===- IncrementalUpdateProfileAnalysis.h -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

#ifndef LLVM_ANALYSIS_INCREMENTALUPDATEPROFILEANALYSIS_H
#define LLVM_ANALYSIS_INCREMENTALUPDATEPROFILEANALYSIS_H

#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/IR/PassManager.h"

namespace llvm {
class IncrementalUpdateProfileAnalysis;

class IncrementalProfDataVerifier {
  BranchProbabilityInfo &BPI;
  BlockFrequencyInfo &BFI;
  Function &F;
  FunctionAnalysisManager &FAM;

  IncrementalProfDataVerifier(Function &F, FunctionAnalysisManager &FAM);
  void verify();
  static std::unique_ptr<IncrementalProfDataVerifier>
  create(Function &F, FunctionAnalysisManager &FAM);
  friend class IncrementalUpdateProfileAnalysis;

public:
  IncrementalProfDataVerifier(const IncrementalProfDataVerifier &) = delete;
  IncrementalProfDataVerifier(IncrementalProfDataVerifier &&) = default;

  BlockFrequencyInfo &bfi() { return BFI; }
  BranchProbabilityInfo &bpi() { return BPI; }

  bool invalidate(Function &, const PreservedAnalyses &PA,
                  FunctionAnalysisManager::Invalidator &) {
    verify();
    // Check whether the analysis has been explicitly invalidated. Otherwise,
    // it's stateless and remains preserved.
    auto PAC = PA.getChecker<IncrementalUpdateProfileAnalysis>();
    return !PAC.preservedWhenStateless();
  }
};

class IncrementalUpdateProfileAnalysis
    : public AnalysisInfoMixin<IncrementalUpdateProfileAnalysis> {

public:
  LLVM_ABI static AnalysisKey Key;
  LLVM_ABI explicit IncrementalUpdateProfileAnalysis() {}

  using Result = std::unique_ptr<IncrementalProfDataVerifier>;

  LLVM_ABI Result run(Function &F, FunctionAnalysisManager &FAM) {
    return IncrementalProfDataVerifier::create(F, FAM);
  }
};
} // namespace llvm

#endif // LLVM_ANALYSIS_INCREMENTALUPDATEPROFILEANALYSIS_H
