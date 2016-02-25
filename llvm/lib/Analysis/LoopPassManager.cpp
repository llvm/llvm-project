//===- LoopPassManager.cpp - Loop pass management -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LoopPassManager.h"

using namespace llvm;

char LoopAnalysisManagerFunctionProxy::PassID;

LoopAnalysisManagerFunctionProxy::Result
LoopAnalysisManagerFunctionProxy::run(Function &F) {
  // TODO: In FunctionAnalysisManagerModuleProxy we assert that the
  // AnalysisManager is empty, but if we do that here we run afoul of the fact
  // that we still have results for previous functions alive. Should we be
  // clearing those when we finish a function?
  //assert(LAM->empty() && "Loop analyses ran prior to the function proxy!");
  return Result(*LAM);
}

LoopAnalysisManagerFunctionProxy::Result::~Result() {
  // Clear out the analysis manager if we're being destroyed -- it means we
  // didn't even see an invalidate call when we got invalidated.
  LAM->clear();
}

bool LoopAnalysisManagerFunctionProxy::Result::invalidate(
    Function &F, const PreservedAnalyses &PA) {
  // If this proxy isn't marked as preserved, then we can't even invalidate
  // individual loop analyses, there may be an invalid set of Loops in the cache
  // making it impossible to incrementally preserve them. Just clear the entire
  // manager.
  if (!PA.preserved(ID()))
    LAM->clear();

  // Return false to indicate that this result is still a valid proxy.
  return false;
}

char FunctionAnalysisManagerLoopProxy::PassID;
