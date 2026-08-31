//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM Exceptions
//
//===----------------------------------------------------------------------===//
//
// This file provides passes that trigger crashes for testing purposes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_TRIGGERCRASHPASS_H
#define LLVM_TRANSFORMS_UTILS_TRIGGERCRASHPASS_H

#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"

namespace llvm {

class TriggerCrashModulePass
    : public OptionalPassInfoMixin<TriggerCrashModulePass> {
public:
  LLVM_ABI PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

class TriggerCrashFunctionPass
    : public OptionalPassInfoMixin<TriggerCrashFunctionPass> {
public:
  LLVM_ABI PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

class TriggerCrashCGSCCPass
    : public OptionalPassInfoMixin<TriggerCrashCGSCCPass> {
public:
  LLVM_ABI PreservedAnalyses run(LazyCallGraph::SCC &C,
                                 CGSCCAnalysisManager &AM, LazyCallGraph &CG,
                                 CGSCCUpdateResult &UR);
};

class TriggerCrashLoopPass
    : public OptionalPassInfoMixin<TriggerCrashLoopPass> {
public:
  LLVM_ABI PreservedAnalyses run(Loop &L, LoopAnalysisManager &AM,
                                 LoopStandardAnalysisResults &AR,
                                 LPMUpdater &U);
};

LLVM_ABI FunctionPass *createTriggerCrashFunctionPass();

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_TRIGGERCRASHPASS_H
