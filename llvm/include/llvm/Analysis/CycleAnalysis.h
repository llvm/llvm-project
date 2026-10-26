//===- CycleAnalysis.h - Cycle Info for LLVM IR -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file declares an analysis pass that computes CycleInfo for
/// LLVM IR, specialized from GenericCycleInfo.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_CYCLEANALYSIS_H
#define LLVM_ANALYSIS_CYCLEANALYSIS_H

#include "llvm/IR/CycleInfo.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

/// Enable verification of cycle info.
///
/// The flag enables checks which are expensive and are disabled by default
/// unless the `EXPENSIVE_CHECKS` macro is defined.  The `-verify-cycle-info`
/// flag allows the checks to be enabled selectively without re-compilation.
LLVM_ABI extern bool VerifyCycleInfo;

/// Legacy analysis pass which computes a \ref CycleInfo.
class LLVM_ABI CycleInfoWrapperPass : public FunctionPass {
  Function *F = nullptr;
  CycleInfo CI;

public:
  static char ID;

  CycleInfoWrapperPass();

  CycleInfo &getResult() { return CI; }
  const CycleInfo &getResult() const { return CI; }

  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  void releaseMemory() override;
  void print(raw_ostream &OS, const Module *M = nullptr) const override;
  void verifyAnalysis() const override;
};

/// Analysis pass which computes a \ref CycleInfo.
class CycleAnalysis : public AnalysisInfoMixin<CycleAnalysis> {
  friend AnalysisInfoMixin<CycleAnalysis>;
  LLVM_ABI static AnalysisKey Key;

public:
  /// Provide the result typedef for this analysis pass.
  using Result = CycleInfo;

  using LegacyWrapper = CycleInfoWrapperPass;

  /// Run the analysis pass over a function and produce a dominator tree.
  LLVM_ABI CycleInfo run(Function &F, FunctionAnalysisManager &);

  // TODO: verify analysis?
};

class CycleInfoPrinterPass
    : public RequiredPassInfoMixin<CycleInfoPrinterPass> {
  raw_ostream &OS;

public:
  LLVM_ABI explicit CycleInfoPrinterPass(raw_ostream &OS);
  LLVM_ABI PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

struct CycleInfoVerifierPass
    : public RequiredPassInfoMixin<CycleInfoVerifierPass> {
  LLVM_ABI PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_ANALYSIS_CYCLEANALYSIS_H
