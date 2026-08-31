//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM Exceptions
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/TriggerCrashMachineFunction.h"

using namespace llvm;

PreservedAnalyses
TriggerCrashMachineFunctionPass::run(MachineFunction &,
                                     MachineFunctionAnalysisManager &) {
  abort();
  return PreservedAnalyses::all();
}
