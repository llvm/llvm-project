//===- PGOFlowVerify.h - PGO flow verification ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// InstrProf flow checks. `-verify-pgo-flow` runs after transforming passes;
/// the `verify-pgo-flow` pass is the same walk on demand. Adaptors and pass
/// managers are skipped so a loop nest is not re-walked per adaptor.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_PGOFLOWVERIFY_H
#define LLVM_TRANSFORMS_IPO_PGOFLOWVERIFY_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/IR/IRUnitRef.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
class BasicBlock;
class Function;
class Loop;
class Module;
class PassInstrumentationCallbacks;

/// Walk IR after transforms so InstrProf use-phase flow checks can run.
class PGOFlowVerifier {
public:
  struct BlockFreqInfo {
    unsigned NumUnknownIn = 0;
    unsigned NumUnknownOut = 0;
    uint64_t SumIn = 0;
    uint64_t SumOut = 0;
  };
  using AllBlockFreqInfo = MapVector<const BasicBlock *, BlockFreqInfo>;

  /// True when `-verify-pgo-flow` is set.
  LLVM_ABI static bool isHookEnabled();
  LLVM_ABI void registerCallbacks(PassInstrumentationCallbacks &PIC);
  LLVM_ABI void runAfterPass(StringRef PassID, IRUnitRef IR);

private:
  void invalidateFunctionFrequencyCache(IRUnitRef IR);
  void runAfterPass(const Module *M);
  void runAfterPass(const Function *F);
  void runAfterPass(const LazyCallGraph::SCC *C);
  void runAfterPass(const Loop *L);
  bool hasInstrProfUseSummary(const Module *M) const;

  bool shouldVerifyFunction(const Function *F) const;
  void computeBlockFrequencies(const Function *F);
  void validateBlockFrequencies(const Function *F);
  void validateEntryCountAgainstCallerSum(const Function *F);
  const AllBlockFreqInfo *getCachedBlockFreqInfo(const Function *F) const;

  DenseMap<const Function *, AllBlockFreqInfo> FunctionBlockFreqInfoCache;
};

/// Pipeline pass that runs the same walk as the `-verify-pgo-flow` hook.
class PGOFlowVerifierPass : public RequiredPassInfoMixin<PGOFlowVerifierPass> {
public:
  LLVM_ABI PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
  LLVM_ABI PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);

private:
  PGOFlowVerifier Verifier;
};

} // end namespace llvm
#endif // LLVM_TRANSFORMS_IPO_PGOFLOWVERIFY_H
