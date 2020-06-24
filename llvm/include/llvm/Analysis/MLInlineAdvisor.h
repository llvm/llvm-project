//===- MLInlineAdvisor.h - ML - based InlineAdvisor factories ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_MLINLINEADVISOR_H
#define LLVM_ANALYSIS_MLINLINEADVISOR_H

#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/InlineAdvisor.h"
#include "llvm/Analysis/MLModelRunner.h"
#include "llvm/IR/PassManager.h"

#include <memory>
#include <unordered_map>

namespace llvm {
class Module;
class MLInlineAdvice;

class MLInlineAdvisor : public InlineAdvisor {
public:
  MLInlineAdvisor(Module &M, ModuleAnalysisManager &MAM,
                  std::unique_ptr<MLModelRunner> ModelRunner);

  CallGraph *callGraph() const { return CG.get(); }
  virtual ~MLInlineAdvisor() = default;

  void onPassEntry() override;

  std::unique_ptr<InlineAdvice> getAdvice(CallBase &CB) override;

  int64_t getIRSize(const Function &F) const { return F.getInstructionCount(); }
  void onSuccessfulInlining(const MLInlineAdvice &Advice,
                            bool CalleeWasDeleted);

  bool isForcedToStop() const { return ForceStop; }
  int64_t getLocalCalls(Function &F);
  const MLModelRunner &getModelRunner() const { return *ModelRunner.get(); }

protected:
  virtual std::unique_ptr<MLInlineAdvice>
  getMandatoryAdvice(CallBase &CB, OptimizationRemarkEmitter &ORE);

  virtual std::unique_ptr<MLInlineAdvice>
  getAdviceFromModel(CallBase &CB, OptimizationRemarkEmitter &ORE);

  Module &M;
  std::unique_ptr<MLModelRunner> ModelRunner;

private:
  int64_t getModuleIRSize() const;

  std::unique_ptr<CallGraph> CG;

  int64_t NodeCount = 0;
  int64_t EdgeCount = 0;
  std::map<const Function *, unsigned> FunctionLevels;
  const int32_t InitialIRSize = 0;
  int32_t CurrentIRSize = 0;

  bool ForceStop = false;
};

/// InlineAdvice that tracks changes post inlining. For that reason, it only
/// overrides the "successful inlining" extension points.
class MLInlineAdvice : public InlineAdvice {
public:
  MLInlineAdvice(MLInlineAdvisor *Advisor, CallBase &CB,
                 OptimizationRemarkEmitter &ORE, bool Recommendation)
      : InlineAdvice(Advisor, CB, ORE, Recommendation),
        CallerIRSize(Advisor->isForcedToStop() ? 0
                                               : Advisor->getIRSize(*Caller)),
        CalleeIRSize(Advisor->isForcedToStop() ? 0
                                               : Advisor->getIRSize(*Callee)),
        CallerAndCalleeEdges(Advisor->isForcedToStop()
                                 ? 0
                                 : (Advisor->getLocalCalls(*Caller) +
                                    Advisor->getLocalCalls(*Callee))) {}
  virtual ~MLInlineAdvice() = default;

  void recordInliningImpl() override;
  void recordInliningWithCalleeDeletedImpl() override;
  void recordUnsuccessfulInliningImpl(const InlineResult &Result) override;
  void recordUnattemptedInliningImpl() override;

  Function *getCaller() const { return Caller; }
  Function *getCallee() const { return Callee; }

  const int64_t CallerIRSize;
  const int64_t CalleeIRSize;
  const int64_t CallerAndCalleeEdges;

private:
  void reportContextForRemark(DiagnosticInfoOptimizationBase &OR);

  MLInlineAdvisor *getAdvisor() const {
    return static_cast<MLInlineAdvisor *>(Advisor);
  };
};

} // namespace llvm

#endif // LLVM_ANALYSIS_MLINLINEADVISOR_H