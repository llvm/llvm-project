///===- DroppedVariableStatsIR.cpp ----------------------------------------===//
///
/// Part of the LLVM Project, under the Apache License v2.0 with LLVM
/// Exceptions. See https://llvm.org/LICENSE.txt for license information.
/// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
///
///===---------------------------------------------------------------------===//
/// \file
/// Dropped Variable Statistics for Debug Information. Reports any number
/// of #dbg_value that get dropped due to an optimization pass.
///
///===---------------------------------------------------------------------===//

#include "llvm/Passes/DroppedVariableStatsIR.h"

using namespace llvm;

void DroppedVariableStatsIR::runOnFunction(const Function *F, bool Before) {
  auto &DebugVariables = DebugVariablesStack.back()[F];
  auto FuncName = F->getName();
  Func = F;
  run(DebugVariables, FuncName, Before);
}

void DroppedVariableStatsIR::calculateDroppedVarStatsOnFunction(
    const Function *F, StringRef PassID, StringRef FuncOrModName,
    StringRef PassLevel) {
  Func = F;
  StringRef FuncName = F->getName();
  DebugVariables &DbgVariables = DebugVariablesStack.back()[F];
  calculateDroppedStatsAndPrint(DbgVariables, FuncName, PassID, FuncOrModName,
                                PassLevel, Func);
}

void DroppedVariableStatsIR::runOnModule(const Module *M, bool Before) {
  for (auto &F : *M)
    runOnFunction(&F, Before);
}

void DroppedVariableStatsIR::calculateDroppedVarStatsOnModule(
    const Module *M, StringRef PassID, StringRef FuncOrModName,
    StringRef PassLevel) {
  for (auto &F : *M) {
    calculateDroppedVarStatsOnFunction(&F, PassID, FuncOrModName, PassLevel);
  }
}

void DroppedVariableStatsIR::registerCallbacks(
    PassInstrumentationCallbacks &PIC) {
  if (!DroppedVariableStatsEnabled)
    return;

  PIC.registerBeforeNonSkippedPassCallback(
      [this](StringRef P, Any IR) { return runBeforePass(IR); });
  PIC.registerAfterPassCallback(
      [this](StringRef P, Any IR, const PreservedAnalyses &PA) {
        return runAfterPass(P, IR);
      });
  PIC.registerAfterPassInvalidatedCallback(
      [this](StringRef P, const PreservedAnalyses &PA) { return cleanup(); });
}

void DroppedVariableStatsIR::visitEveryInstruction(
    unsigned &DroppedCount, DenseMap<VarID, DILocation *> &InlinedAtsMap,
    VarID Var) {
  const DIScope *DbgValScope = std::get<0>(Var);
  for (const auto &I : instructions(Func)) {
    auto *DbgLoc = I.getDebugLoc().get();
    if (!DbgLoc)
      continue;
    if (updateDroppedCount(DbgLoc, DbgLoc->getScope(), DbgValScope,
                           InlinedAtsMap, Var, DroppedCount))
      break;
  }
}

void DroppedVariableStatsIR::visitEveryDebugRecord(
    DenseSet<VarID> &VarIDSet,
    DenseMap<StringRef, DenseMap<VarID, DILocation *>> &InlinedAtsMap,
    StringRef FuncName, bool Before) {
  for (const auto &I : instructions(Func)) {
    for (DbgRecord &DR : I.getDbgRecordRange()) {
      if (auto *Dbg = dyn_cast<DbgVariableRecord>(&DR)) {
        auto *DbgVar = Dbg->getVariable();
        auto DbgLoc = DR.getDebugLoc();
        populateVarIDSetAndInlinedMap(DbgVar, DbgLoc, VarIDSet, InlinedAtsMap,
                                      FuncName, Before);
      }
    }
  }
}
