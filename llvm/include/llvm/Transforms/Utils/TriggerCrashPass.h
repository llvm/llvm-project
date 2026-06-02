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

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class TriggerCrashModulePass
    : public OptionalPassInfoMixin<TriggerCrashModulePass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static StringRef name() { return "TriggerCrashModulePass"; }
};

class TriggerCrashFunctionPass
    : public OptionalPassInfoMixin<TriggerCrashFunctionPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  static StringRef name() { return "TriggerCrashFunctionPass"; }
};

FunctionPass *createTriggerCrashFunctionPass();

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_TRIGGERCRASHPASS_H
