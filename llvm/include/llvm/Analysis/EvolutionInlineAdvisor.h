//===- EvolutionInlineAdvisor.h - LLM+Evolution-based InlineAdvisor factories
//---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_EvolutionInlineAdvisor_H
#define LLVM_ANALYSIS_EvolutionInlineAdvisor_H

#include "llvm/Analysis/FunctionPropertiesAnalysis.h"
#include "llvm/Analysis/InlineAdvisor.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/IR/PassManager.h"

#include <memory>

namespace llvm {

class EvolutionInlineAdvisor : public InlineAdvisor {
public:
  EvolutionInlineAdvisor(Module &M, FunctionAnalysisManager &FAM,
                         InlineContext IC)
      : InlineAdvisor(M, FAM, IC) {}

private:
  std::unique_ptr<InlineAdvice> getAdviceImpl(CallBase &CB) override;
  std::unique_ptr<InlineAdvice> getEvolutionAdviceImpl(CallBase &CB);
};

} // namespace llvm

#endif // LLVM_ANALYSIS_EvolutionInlineAdvisor_H