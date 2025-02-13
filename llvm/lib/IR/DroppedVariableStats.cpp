///===- DroppedVariableStats.cpp ----------------------------------------===//
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

#include "llvm/IR/DroppedVariableStats.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"

using namespace llvm;

DroppedVariableStats::DroppedVariableStats(bool DroppedVarStatsEnabled)
    : DroppedVariableStatsEnabled(DroppedVarStatsEnabled) {
  if (DroppedVarStatsEnabled)
    llvm::outs() << "Pass Level, Pass Name, Num of Dropped Variables, Func or "
                    "Module Name\n";
}

void DroppedVariableStats::setup() {
  DebugVariablesStack.push_back({DenseMap<const Function *, DebugVariables>()});
  InlinedAts.push_back({DenseMap<StringRef, DenseMap<VarID, DILocation *>>()});
}

void DroppedVariableStats::cleanup() {
  assert(!DebugVariablesStack.empty() &&
         "DebugVariablesStack shouldn't be empty!");
  assert(!InlinedAts.empty() && "InlinedAts shouldn't be empty!");
  DebugVariablesStack.pop_back();
  InlinedAts.pop_back();
}

void DroppedVariableStats::calculateDroppedStatsAndPrint(
    DebugVariables &DbgVariables, StringRef FuncName, StringRef PassID,
    StringRef FuncOrModName, StringRef PassLevel, const Function *Func) {
  unsigned DroppedCount = 0;
  DenseSet<VarID> &DebugVariablesBeforeSet = DbgVariables.DebugVariablesBefore;
  DenseSet<VarID> &DebugVariablesAfterSet = DbgVariables.DebugVariablesAfter;
  if (InlinedAts.back().find(FuncName) == InlinedAts.back().end())
    return;
  DenseMap<VarID, DILocation *> &InlinedAtsMap = InlinedAts.back()[FuncName];
  // Find an Instruction that shares the same scope as the dropped #dbg_value
  // or has a scope that is the child of the scope of the #dbg_value, and has
  // an inlinedAt equal to the inlinedAt of the #dbg_value or it's inlinedAt
  // chain contains the inlinedAt of the #dbg_value, if such an Instruction is
  // found, debug information is dropped.
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
  // the Var's InlinedAt location, return true to signify that the Var has
  // been dropped.
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

void DroppedVariableStats::removeVarFromAllSets(VarID Var, const Function *F) {
  // Do not remove Var from the last element, it will be popped from the
  // stack.
  for (auto &DebugVariablesMap : llvm::drop_end(DebugVariablesStack))
    DebugVariablesMap[F].DebugVariablesBefore.erase(Var);
}

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
