//===- ReplayInlineAdvisor.cpp - Replay InlineAdvisor ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements ReplayInlineAdvisor that replays inline decision based
// on previous inline remarks from optimization remark log.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ReplayInlineAdvisor.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/LineIterator.h"

using namespace llvm;

#define DEBUG_TYPE "inline-replay"

ReplayInlineAdvisor::ReplayInlineAdvisor(FunctionAnalysisManager &FAM,
                                         LLVMContext &Context,
                                         StringRef RemarksFile)
    : InlineAdvisor(FAM), HasReplayRemarks(false) {
  auto BufferOrErr = MemoryBuffer::getFileOrSTDIN(RemarksFile);
  std::error_code EC = BufferOrErr.getError();
  if (EC) {
    Context.emitError("Could not open remarks file: " + EC.message());
    return;
  }

  // Example for inline remarks to parse:
  //   _Z3subii inlined into main [details] at callsite sum:1 @ main:3.1
  // We use the callsite string after `at callsite` to replay inlining.
  line_iterator LineIt(*BufferOrErr.get(), /*SkipBlanks=*/true);
  for (; !LineIt.is_at_eof(); ++LineIt) {
    StringRef Line = *LineIt;
    auto Pair = Line.split(" at callsite ");
    if (Pair.second.empty())
      continue;
    InlineSitesFromRemarks.insert(Pair.second);
  }
  HasReplayRemarks = true;
}

std::unique_ptr<InlineAdvice> ReplayInlineAdvisor::getAdvice(CallBase &CB) {
  assert(HasReplayRemarks);

  Function &Caller = *CB.getCaller();
  auto &ORE = FAM.getResult<OptimizationRemarkEmitterAnalysis>(Caller);

  if (InlineSitesFromRemarks.empty())
    return std::make_unique<InlineAdvice>(this, CB, ORE, false);

  std::string CallSiteLoc = getCallSiteLocation(CB.getDebugLoc());
  bool InlineRecommended = InlineSitesFromRemarks.count(CallSiteLoc) > 0;
  return std::make_unique<InlineAdvice>(this, CB, ORE, InlineRecommended);
}
