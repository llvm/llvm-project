//===- EvolutionInlineAdvisor.h - LLM+Evolutionary Algorithm-based InlineAdvisor
//---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_EVOLUTIONINLINEADVISOR_H
#define LLVM_ANALYSIS_EVOLUTIONINLINEADVISOR_H

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
  std::unique_ptr<InlineAdvice> getEvolvableAdvice(CallBase &CB);
};

} // namespace llvm

#endif // LLVM_ANALYSIS_EVOLUTIONINLINEADVISOR_H