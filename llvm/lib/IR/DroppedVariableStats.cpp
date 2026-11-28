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
}

void DroppedVariableStats::cleanup() {
  assert(!DebugVariablesStack.empty() &&
         "DebugVariablesStack shouldn't be empty!");
  DebugVariablesStack.pop_back();
}

void DroppedVariableStats::calculateDroppedStatsAndPrint(
    DebugVariables &DbgVariables, StringRef PassID, StringRef FuncOrModName,
    StringRef PassLevel, const Function *Func) {
  unsigned DroppedCount = 0;
  DenseSet<VarID> &DebugVariablesBeforeSet = DbgVariables.DebugVariablesBefore;
  DenseSet<VarID> &DebugVariablesAfterSet = DbgVariables.DebugVariablesAfter;
  // A variable is considered dropped if an instruction remains at which a
  // debugger could observe it.
  for (VarID Var : DebugVariablesBeforeSet) {
    if (DebugVariablesAfterSet.contains(Var))
      continue;
    visitEveryInstruction(DroppedCount, Var);
    removeVarFromAllSets(Var, Func);
  }
  if (DroppedCount > 0) {
    llvm::outs() << PassLevel << ", " << PassID << ", " << DroppedCount << ", "
                 << FuncOrModName << "\n";
    PassDroppedVariables = true;
  } else
    PassDroppedVariables = false;
}

bool DroppedVariableStats::updateDroppedCount(const DILocation *DbgLoc,
                                              VarID Var,
                                              unsigned &DroppedCount) {
  if (!isVarInScopeAt(DbgLoc, Var))
    return false;
  // Found another instruction in the variable's scope, so there exists a break
  // point at which the variable could be observed. Count it as dropped.
  DroppedCount++;
  return true;
}

void DroppedVariableStats::run(DebugVariables &DbgVariables, bool Before) {
  auto &VarIDSet = (Before ? DbgVariables.DebugVariablesBefore
                           : DbgVariables.DebugVariablesAfter);
  VarIDSet = DenseSet<VarID>();
  visitEveryDebugRecord(VarIDSet);
}

void DroppedVariableStats::populateVarIDSet(const DILocalVariable *DbgVar,
                                            DebugLoc DbgLoc,
                                            DenseSet<VarID> &VarIDSet) {
  VarIDSet.insert(DebugVariableAggregate(
      DebugVariable(DbgVar, std::nullopt, DbgLoc.getInlinedAt())));
}

void DroppedVariableStats::removeVarFromAllSets(VarID Var, const Function *F) {
  // Do not remove Var from the last element, it will be popped from the
  // stack.
  for (auto &DebugVariablesMap : llvm::drop_end(DebugVariablesStack))
    DebugVariablesMap[F].DebugVariablesBefore.erase(Var);
}

bool DroppedVariableStats::isScopeChildOfOrEqualTo(const DIScope *Scope,
                                                   const DIScope *VarScope) {
  while (Scope != nullptr) {
    if (Scope == VarScope)
      return true;
    Scope = Scope->getScope();
  }
  return false;
}

bool DroppedVariableStats::isVarInScopeAt(const DILocation *Loc, VarID Var) {
  // Loc and the call sites in its inlinedAt chain are the frames a debugger
  // stopped at Loc can see. Var can only be in the frame of its own
  // inlined instance.
  for (const DILocation *Frame = Loc; Frame; Frame = Frame->getInlinedAt())
    if (Frame->getInlinedAt() == Var.getInlinedAt())
      return isScopeChildOfOrEqualTo(Frame->getScope(),
                                     Var.getVariable()->getScope());
  return false;
}
