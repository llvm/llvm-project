//===- ReplayInlineAdvisor.h - Replay Inline Advisor interface -*- C++ --*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
#ifndef LLVM_ANALYSIS_REPLAYINLINEADVISOR_H
#define LLVM_ANALYSIS_REPLAYINLINEADVISOR_H

#include "llvm/ADT/StringSet.h"
#include "llvm/Analysis/InlineAdvisor.h"
#include "llvm/IR/LLVMContext.h"

namespace llvm {
class BasicBlock;
class CallBase;
class Function;
class Module;
class OptimizationRemarkEmitter;

/// Replay inline advisor that uses optimization remarks from inlining of
/// previous build to guide current inlining. This is useful for inliner tuning.
class ReplayInlineAdvisor : public InlineAdvisor {
public:
  ReplayInlineAdvisor(Module &M, FunctionAnalysisManager &FAM,
                      LLVMContext &Context,
                      std::unique_ptr<InlineAdvisor> OriginalAdvisor,
                      StringRef RemarksFile, bool EmitRemarks);
  std::unique_ptr<InlineAdvice> getAdviceImpl(CallBase &CB) override;
  bool areReplayRemarksLoaded() const { return HasReplayRemarks; }

private:
  StringSet<> InlineSitesFromRemarks;
  std::unique_ptr<InlineAdvisor> OriginalAdvisor;
  bool HasReplayRemarks = false;
  bool EmitRemarks = false;
};
} // namespace llvm
#endif // LLVM_ANALYSIS_REPLAYINLINEADVISOR_H
