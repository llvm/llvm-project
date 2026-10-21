//===-- AGXAliasAnalysis.h - AGX address space alias analysis --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Device, threadgroup, constant and private memory are disjoint on AGX: two
// accesses in different non-generic address spaces never alias, and nothing
// writes the constant space (kernel arguments and constant data).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AGX_AGXALIASANALYSIS_H
#define LLVM_LIB_TARGET_AGX_AGXALIASANALYSIS_H

#include "llvm/Analysis/AliasAnalysis.h"

namespace llvm {

class AGXAAResult : public AAResultBase {
public:
  AGXAAResult() = default;
  AGXAAResult(AGXAAResult &&Arg) : AAResultBase(std::move(Arg)) {}

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

/// The new pass manager's analysis.
class AGXAA : public AnalysisInfoMixin<AGXAA> {
  friend AnalysisInfoMixin<AGXAA>;
  static AnalysisKey Key;

public:
  using Result = AGXAAResult;
  AGXAAResult run(Function &, AnalysisManager<Function> &) {
    return AGXAAResult();
  }
};

/// The legacy pass manager's wrapper of the result.
class AGXAAWrapperPass : public ImmutablePass {
  std::unique_ptr<AGXAAResult> Result;

public:
  static char ID;
  AGXAAWrapperPass();
  AGXAAResult &getResult() { return *Result; }
  bool doInitialization(Module &) override {
    Result.reset(new AGXAAResult());
    return false;
  }
  bool doFinalization(Module &) override {
    Result.reset();
    return false;
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

/// Adds AGXAAResult to the legacy pass manager's AAResults, ahead of BasicAA.
class AGXExternalAAWrapper : public ExternalAAWrapperPass {
public:
  static char ID;
  AGXExternalAAWrapper()
      : ExternalAAWrapperPass(
            [](Pass &P, Function &, AAResults &AAR) {
              if (auto *W = P.getAnalysisIfAvailable<AGXAAWrapperPass>())
                AAR.addAAResult(W->getResult());
            },
            /*RunEarly=*/true) {}
  StringRef getPassName() const override {
    return "AGX address space alias analysis wrapper";
  }
};

ImmutablePass *createAGXAAWrapperPass();
ImmutablePass *createAGXExternalAAWrapperPass();
void initializeAGXAAWrapperPassPass(PassRegistry &);
void initializeAGXExternalAAWrapperPass(PassRegistry &);

} // namespace llvm

#endif
