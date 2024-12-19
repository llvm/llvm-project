///===- DroppedVariableStats.cpp ------------------------------------------===//
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

#include "llvm/Passes/DroppedVariableStats.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"

using namespace llvm;

bool DroppedVariableStats::isScopeChildOfOrEqualTo(const DIScope *Scope,
                                                   const DIScope *DbgValScope) {
  while (Scope != nullptr) {
    if (VisitedScope.find(Scope) == VisitedScope.end()) {
      VisitedScope.insert(Scope);
      if (Scope == DbgValScope) {
        VisitedScope.clear();
        return true;
      }
      Scope = Scope->getScope();
    } else {
      VisitedScope.clear();
      return false;
    }
  }
  return false;
}

bool DroppedVariableStats::isInlinedAtChildOfOrEqualTo(
    const DILocation *InlinedAt, const DILocation *DbgValInlinedAt) {
  if (DbgValInlinedAt == InlinedAt)
    return true;
  if (!DbgValInlinedAt)
    return false;
  auto *IA = InlinedAt;
  while (IA) {
    if (IA == DbgValInlinedAt)
      return true;
    IA = IA->getInlinedAt();
  }
  return false;
}

void DroppedVariableStats::calculateDroppedStatsAndPrint(
    DebugVariables &DbgVariables, StringRef FuncName, StringRef PassID,
    StringRef FuncOrModName, StringRef PassLevel, const Function *Func) {
  unsigned DroppedCount = 0;
  DenseSet<VarID> &DebugVariablesBeforeSet = DbgVariables.DebugVariablesBefore;
  DenseSet<VarID> &DebugVariablesAfterSet = DbgVariables.DebugVariablesAfter;
  DenseMap<VarID, DILocation *> &InlinedAtsMap = InlinedAts.back()[FuncName];
  // Find an Instruction that shares the same scope as the dropped #dbg_value or
  // has a scope that is the child of the scope of the #dbg_value, and has an
  // inlinedAt equal to the inlinedAt of the #dbg_value or it's inlinedAt chain
  // contains the inlinedAt of the #dbg_value, if such an Instruction is found,
  // debug information is dropped.
  for (VarID Var : DebugVariablesBeforeSet) {
    if (DebugVariablesAfterSet.contains(Var))
      continue;
    visitEveryInstruction(DroppedCount, InlinedAtsMap, Var);
    removeVarFromAllSets(Var, Func);
  }
  if (DroppedCount > 0) {
    llvm::outs() << PassLevel << ", " << PassID << ", " << DroppedCount << ", "
                 << FuncOrModName << "\n";
    PassDroppedVariables = true;
  } else
    PassDroppedVariables = false;
}

bool DroppedVariableStats::updateDroppedCount(
    DILocation *DbgLoc, const DIScope *Scope, const DIScope *DbgValScope,
    DenseMap<VarID, DILocation *> &InlinedAtsMap, VarID Var,
    unsigned &DroppedCount) {

  // If the Scope is a child of, or equal to the DbgValScope and is inlined at
  // the Var's InlinedAt location, return true to signify that the Var has been
  // dropped.
  if (isScopeChildOfOrEqualTo(Scope, DbgValScope))
    if (isInlinedAtChildOfOrEqualTo(DbgLoc->getInlinedAt(),
                                    InlinedAtsMap[Var])) {
      // Found another instruction in the variable's scope, so there exists a
      // break point at which the variable could be observed. Count it as
      // dropped.
      DroppedCount++;
      return true;
    }
  return false;
}

void DroppedVariableStats::run(DebugVariables &DbgVariables, StringRef FuncName,
                               bool Before) {
  auto &VarIDSet = (Before ? DbgVariables.DebugVariablesBefore
                           : DbgVariables.DebugVariablesAfter);
  auto &InlinedAtsMap = InlinedAts.back();
  if (Before)
    InlinedAtsMap.try_emplace(FuncName, DenseMap<VarID, DILocation *>());
  VarIDSet = DenseSet<VarID>();
  visitEveryDebugRecord(VarIDSet, InlinedAtsMap, FuncName, Before);
}

void DroppedVariableStats::populateVarIDSetAndInlinedMap(
    const DILocalVariable *DbgVar, DebugLoc DbgLoc, DenseSet<VarID> &VarIDSet,
    DenseMap<StringRef, DenseMap<VarID, DILocation *>> &InlinedAtsMap,
    StringRef FuncName, bool Before) {
  VarID Key{DbgVar->getScope(), DbgLoc->getInlinedAtScope(), DbgVar};
  VarIDSet.insert(Key);
  if (Before)
    InlinedAtsMap[FuncName].try_emplace(Key, DbgLoc.getInlinedAt());
}

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
