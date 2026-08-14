//===- EvolutionInlineAdvisor.cpp - skeleton implementation   ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements EvolutionInlineAdvisor.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/EvolutionInlineAdvisor.h"
#include "llvm/Analysis/InlineAdvisor.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

// EVOLVE-BLOCK-START
// You can include additional LLVM headers here.

std::unique_ptr<InlineAdvice>
EvolutionInlineAdvisor::getEvolvableAdvice(CallBase &CB) {
  // Implementation of inlining strategy. Do not change the function interface.
  bool IsInliningRecommended = false;
  return std::make_unique<InlineAdvice>(
      this, CB,
      FAM.getResult<OptimizationRemarkEmitterAnalysis>(*CB.getCaller()),
      IsInliningRecommended);
}

// EVOLVE-BLOCK-END

std::unique_ptr<InlineAdvice>
EvolutionInlineAdvisor::getAdviceImpl(CallBase &CB) {
  // TODO: refactor the legality check to make them common with
  // MLInlineAdvisor::getAdviceImpl
  auto &Caller = *CB.getCaller();
  auto &Callee = *CB.getCalledFunction();
  auto &ORE = FAM.getResult<OptimizationRemarkEmitterAnalysis>(Caller);

  auto MandatoryKind = InlineAdvisor::getMandatoryKind(CB, FAM, ORE);
  // If this is a "never inline" case, there won't be any changes to internal
  // state we need to track, so we can just return the base InlineAdvice, which
  // will do nothing interesting.
  // Same thing if this is a recursive case.
  if (MandatoryKind == InlineAdvisor::MandatoryInliningKind::Never ||
      &Caller == &Callee)
    return getMandatoryAdvice(CB, false);

  auto IsViable = isInlineViable(Callee);
  if (!IsViable.isSuccess())
    return std::make_unique<InlineAdvice>(this, CB, ORE, false);

  bool Mandatory =
      MandatoryKind == InlineAdvisor::MandatoryInliningKind::Always;

  if (Mandatory)
    return std::make_unique<InlineAdvice>(this, CB, ORE, true);

  return getEvolvableAdvice(CB);
}
