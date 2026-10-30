//===-- AIRAliasAnalysis.h - AIR address space alias analysis --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIRALIASANALYSIS_H
#define LLVM_LIB_TARGET_AIR_AIRALIASANALYSIS_H

#include "llvm/Analysis/AliasAnalysis.h"

namespace llvm {

/// Memory in different address spaces does not overlap, except through a
/// generic pointer; constant memory is never written.
class AIRAAResult : public AAResultBase {
public:
  AIRAAResult() = default;
  AIRAAResult(AIRAAResult &&Arg) : AAResultBase(std::move(Arg)) {}

  /// Stateless, so never invalidated.
  bool invalidate(Function &, const PreservedAnalyses &,
                  FunctionAnalysisManager::Invalidator &) {
    return false;
  }

  AliasResult alias(const MemoryLocation &LocA, const MemoryLocation &LocB,
                    AAQueryInfo &AAQI, const Instruction *CtxI = nullptr);

  ModRefInfo getModRefInfoMask(const MemoryLocation &Loc, AAQueryInfo &AAQI,
                               bool IgnoreLocals);
};

class AIRAA : public AnalysisInfoMixin<AIRAA> {
  friend AnalysisInfoMixin<AIRAA>;
  static AnalysisKey Key;

public:
  using Result = AIRAAResult;
  AIRAAResult run(Function &, FunctionAnalysisManager &) {
    return AIRAAResult();
  }
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AIR_AIRALIASANALYSIS_H
