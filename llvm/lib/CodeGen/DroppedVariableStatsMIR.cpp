///===- DroppedVariableStatsMIR.cpp ---------------------------------------===//
///
/// Part of the LLVM Project, under the Apache License v2.0 with LLVM
/// Exceptions. See https://llvm.org/LICENSE.txt for license information.
/// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
///
///===---------------------------------------------------------------------===//
/// \file
/// Dropped Variable Statistics for Debug Information. Reports any number
/// of DBG_VALUEs that get dropped due to an optimization pass.
///
///===---------------------------------------------------------------------===//

#include "llvm/CodeGen/DroppedVariableStatsMIR.h"
#include "llvm/IR/DebugInfoMetadata.h"

using namespace llvm;

void DroppedVariableStatsMIR::runBeforePass(StringRef PassID,
                                            MachineFunction *MF) {
  if (PassID == "Debug Variable Analysis")
    return;
  setup();
  return runOnMachineFunction(MF, true);
}

void DroppedVariableStatsMIR::runAfterPass(StringRef PassID,
                                           MachineFunction *MF) {
  if (PassID == "Debug Variable Analysis")
    return;
  runOnMachineFunction(MF, false);
  calculateDroppedVarStatsOnMachineFunction(MF, PassID, MF->getName().str());
  cleanup();
}

void DroppedVariableStatsMIR::runOnMachineFunction(const MachineFunction *MF,
                                                   bool Before) {
  auto &DebugVariables = DebugVariablesStack.back()[&MF->getFunction()];
  MFunc = MF;
  run(DebugVariables, Before);
}

void DroppedVariableStatsMIR::calculateDroppedVarStatsOnMachineFunction(
    const MachineFunction *MF, StringRef PassID, StringRef FuncOrModName) {
  MFunc = MF;
  const Function *Func = &MF->getFunction();
  DebugVariables &DbgVariables = DebugVariablesStack.back()[Func];
  calculateDroppedStatsAndPrint(DbgVariables, PassID, FuncOrModName,
                                "MachineFunction", Func);
}

void DroppedVariableStatsMIR::visitEveryInstruction(unsigned &DroppedCount,
                                                    VarID Var) {
  unsigned PrevDroppedCount = DroppedCount;
  for (const auto &MBB : *MFunc) {
    for (const auto &MI : MBB) {
      if (!MI.isDebugInstr()) {
        auto *DbgLoc = MI.getDebugLoc().get();
        if (!DbgLoc)
          continue;

        if (updateDroppedCount(DbgLoc, Var, DroppedCount))
          break;
      }
    }
    if (PrevDroppedCount != DroppedCount) {
      PrevDroppedCount = DroppedCount;
      break;
    }
  }
}

void DroppedVariableStatsMIR::visitEveryDebugRecord(DenseSet<VarID> &VarIDSet) {
  for (const auto &MBB : *MFunc) {
    for (const auto &MI : MBB) {
      if (MI.isDebugValueLike()) {
        auto *DbgVar = MI.getDebugVariable();
        if (!DbgVar)
          continue;
        auto DbgLoc = MI.getDebugLoc();
        populateVarIDSet(DbgVar, DbgLoc, VarIDSet);
      }
    }
  }
}
