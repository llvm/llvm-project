//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM Exceptions
//
//===----------------------------------------------------------------------===//

#include "llvm/Passes/TriggerCrashPasses.h"

using namespace llvm;

PreservedAnalyses TriggerCrashModulePass::run(Module &,
                                              ModuleAnalysisManager &) {
  abort();
  return PreservedAnalyses::all();
}

PreservedAnalyses TriggerCrashCGSCCPass::run(LazyCallGraph::SCC &,
                                             CGSCCAnalysisManager &,
                                             LazyCallGraph &,
                                             CGSCCUpdateResult &) {
  abort();
  return PreservedAnalyses::all();
}

PreservedAnalyses TriggerCrashFunctionPass::run(Function &,
                                                FunctionAnalysisManager &) {
  abort();
  return PreservedAnalyses::all();
}

PreservedAnalyses TriggerCrashLoopPass::run(Loop &, LoopAnalysisManager &,
                                            LoopStandardAnalysisResults &,
                                            LPMUpdater &) {
  abort();
  return PreservedAnalyses::all();
}

PreservedAnalyses
TriggerCrashMachineFunctionPass::run(MachineFunction &,
                                     MachineFunctionAnalysisManager &) {
  abort();
  return PreservedAnalyses::all();
}
