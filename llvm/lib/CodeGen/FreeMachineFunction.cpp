//===- FreeMachineFunction.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/FreeMachineFunction.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/CodeGen/MachineModuleInfo.h"

using namespace llvm;

PreservedAnalyses
FreeMachineFunctionPass::run(MachineFunction &MF,
                             MachineFunctionAnalysisManager &MFAM) {
  PreservedAnalyses PA = PreservedAnalyses::none();
  PA.abandon<MachineFunctionAnalysis>();
  return PA;
}
