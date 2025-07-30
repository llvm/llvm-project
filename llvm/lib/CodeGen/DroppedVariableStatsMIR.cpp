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
  auto FuncName = MF->getName();
  MFunc = MF;
  run(DebugVariables, FuncName, Before);
}

void DroppedVariableStatsMIR::calculateDroppedVarStatsOnMachineFunction(
    const MachineFunction *MF, StringRef PassID, StringRef FuncOrModName) {
  MFunc = MF;
  StringRef FuncName = MF->getName();
  const Function *Func = &MF->getFunction();
  DebugVariables &DbgVariables = DebugVariablesStack.back()[Func];
  calculateDroppedStatsAndPrint(DbgVariables, FuncName, PassID, FuncOrModName,
                                "MachineFunction", Func);
}

void DroppedVariableStatsMIR::visitEveryInstruction(
    unsigned &DroppedCount, DenseMap<VarID, DILocation *> &InlinedAtsMap,
    VarID Var) {
  unsigned PrevDroppedCount = DroppedCount;
  const DIScope *DbgValScope = std::get<0>(Var);
  for (const auto &MBB : *MFunc) {
    for (const auto &MI : MBB) {
      if (!MI.isDebugInstr()) {
        auto *DbgLoc = MI.getDebugLoc().get();
        if (!DbgLoc)
          continue;

        auto *Scope = DbgLoc->getScope();
        if (updateDroppedCount(DbgLoc, Scope, DbgValScope, InlinedAtsMap, Var,
                               DroppedCount))
          break;
      }
    }
    if (PrevDroppedCount != DroppedCount) {
      PrevDroppedCount = DroppedCount;
      break;
    }
  }
}

void DroppedVariableStatsMIR::visitEveryDebugRecord(
    DenseSet<VarID> &VarIDSet,
    DenseMap<StringRef, DenseMap<VarID, DILocation *>> &InlinedAtsMap,
    StringRef FuncName, bool Before) {
  for (const auto &MBB : *MFunc) {
    for (const auto &MI : MBB) {
      if (MI.isDebugValueLike()) {
        auto *DbgVar = MI.getDebugVariable();
        if (!DbgVar)
          continue;
        auto DbgLoc = MI.getDebugLoc();
        populateVarIDSetAndInlinedMap(DbgVar, DbgLoc, VarIDSet, InlinedAtsMap,
                                      FuncName, Before);
      }
    }
  }
}
