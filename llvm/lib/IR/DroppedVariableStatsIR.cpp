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

#include "llvm/IR/DroppedVariableStatsIR.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassInstrumentation.h"

using namespace llvm;

template <typename IRUnitT>
const IRUnitT *DroppedVariableStatsIR::unwrapIR(Any IR) {
  const IRUnitT **IRPtr = llvm::any_cast<const IRUnitT *>(&IR);
  return IRPtr ? *IRPtr : nullptr;
}

void DroppedVariableStatsIR::runBeforePass(StringRef P, Any IR) {
  setup();
  if (const auto *M = unwrapIR<Module>(IR))
    return this->runOnModule(P, M, true);
  if (const auto *F = unwrapIR<Function>(IR))
    return this->runOnFunction(P, F, true);
}

void DroppedVariableStatsIR::runAfterPass(StringRef P, Any IR) {
  if (const auto *M = unwrapIR<Module>(IR))
    runAfterPassModule(P, M);
  else if (const auto *F = unwrapIR<Function>(IR))
    runAfterPassFunction(P, F);
  cleanup();
}

void DroppedVariableStatsIR::runAfterPassFunction(StringRef PassID,
                                                  const Function *F) {
  runOnFunction(PassID, F, false);
  calculateDroppedVarStatsOnFunction(F, PassID, F->getName().str(), "Function");
}

void DroppedVariableStatsIR::runAfterPassModule(StringRef PassID,
                                                const Module *M) {
  runOnModule(PassID, M, false);
  calculateDroppedVarStatsOnModule(M, PassID, M->getName().str(), "Module");
}

void DroppedVariableStatsIR::runOnFunction(StringRef PassID, const Function *F,
                                           bool Before) {
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

void DroppedVariableStatsIR::runOnModule(StringRef PassID, const Module *M,
                                         bool Before) {
  for (auto &F : *M) {
    runOnFunction(PassID, &F, Before);
  }
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
      [this](StringRef P, Any IR) { return runBeforePass(P, IR); });
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
