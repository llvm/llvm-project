//===- ReplayInlineAdvisor.cpp - Replay InlineAdvisor ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements ReplayInlineAdvisor that replays inline decisions based
// on previous inline remarks from optimization remark log. This is a best
// effort approach useful for testing compiler/source changes while holding
// inlining steady.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ReplayInlineAdvisor.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/LineIterator.h"
#include <memory>

using namespace llvm;

#define DEBUG_TYPE "replay-inline"

ReplayInlineAdvisor::ReplayInlineAdvisor(
    Module &M, FunctionAnalysisManager &FAM, LLVMContext &Context,
    std::unique_ptr<InlineAdvisor> OriginalAdvisor, StringRef RemarksFile,
    ReplayInlineScope Scope, bool EmitRemarks)
    : InlineAdvisor(M, FAM), OriginalAdvisor(std::move(OriginalAdvisor)),
      HasReplayRemarks(false), Scope(Scope), EmitRemarks(EmitRemarks) {

  auto BufferOrErr = MemoryBuffer::getFileOrSTDIN(RemarksFile);
  std::error_code EC = BufferOrErr.getError();
  if (EC) {
    Context.emitError("Could not open remarks file: " + EC.message());
    return;
  }

  // Example for inline remarks to parse:
  //   main:3:1.1: '_Z3subii' inlined into 'main' at callsite sum:1 @
  //   main:3:1.1;
  // We use the callsite string after `at callsite` to replay inlining.
  line_iterator LineIt(*BufferOrErr.get(), /*SkipBlanks=*/true);
  for (; !LineIt.is_at_eof(); ++LineIt) {
    StringRef Line = *LineIt;
    auto Pair = Line.split(" at callsite ");

    auto CalleeCaller = Pair.first.split("' inlined into '");

    StringRef Callee = CalleeCaller.first.rsplit(": '").second;
    StringRef Caller = CalleeCaller.second.rsplit("'").first;

    auto CallSite = Pair.second.split(";").first;

    if (Callee.empty() || Caller.empty() || CallSite.empty()) {
      Context.emitError("Invalid remark format: " + Line);
      return;
    }

    std::string Combined = (Callee + CallSite).str();
    InlineSitesFromRemarks[Combined] = false;
    if (Scope == ReplayInlineScope::Function)
      CallersToReplay.insert(Caller);
  }

  HasReplayRemarks = true;
}

std::unique_ptr<InlineAdvisor> llvm::getReplayInlineAdvisor(
    Module &M, FunctionAnalysisManager &FAM, LLVMContext &Context,
    std::unique_ptr<InlineAdvisor> OriginalAdvisor, StringRef RemarksFile,
    ReplayInlineScope Scope, bool EmitRemarks) {
  auto Advisor = std::make_unique<ReplayInlineAdvisor>(
      M, FAM, Context, std::move(OriginalAdvisor), RemarksFile, Scope,
      EmitRemarks);
  if (!Advisor->areReplayRemarksLoaded())
    Advisor.reset();
  return Advisor;
}

std::unique_ptr<InlineAdvice> ReplayInlineAdvisor::getAdviceImpl(CallBase &CB) {
  assert(HasReplayRemarks);

  Function &Caller = *CB.getCaller();
  auto &ORE = FAM.getResult<OptimizationRemarkEmitterAnalysis>(Caller);

  Optional<InlineCost> InlineRecommended;

  if (Scope == ReplayInlineScope::Module ||
      CallersToReplay.count(CB.getFunction()->getName())) {
    std::string CallSiteLoc = getCallSiteLocation(CB.getDebugLoc());
    StringRef Callee = CB.getCalledFunction()->getName();
    std::string Combined = (Callee + CallSiteLoc).str();

    auto Iter = InlineSitesFromRemarks.find(Combined);
    if (Iter != InlineSitesFromRemarks.end()) {
      InlineSitesFromRemarks[Combined] = true;
      InlineRecommended = llvm::InlineCost::getAlways("previously inlined");
    }
  } else if (Scope == ReplayInlineScope::Function) {
    if (OriginalAdvisor)
      return OriginalAdvisor->getAdvice(CB);
    return {};
  }

  return std::make_unique<DefaultInlineAdvice>(this, CB, InlineRecommended, ORE,
                                               EmitRemarks);
}
