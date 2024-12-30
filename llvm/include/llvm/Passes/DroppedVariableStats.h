///===- DroppedVariableStats.h - Opt Diagnostics -*- C++ -*----------------===//
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

#ifndef LLVM_CODEGEN_DROPPEDVARIABLESTATS_H
#define LLVM_CODEGEN_DROPPEDVARIABLESTATS_H

#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/PassInstrumentation.h"

namespace llvm {

/// A unique key that represents a debug variable.
/// First const DIScope *: Represents the scope of the debug variable.
/// Second const DIScope *: Represents the InlinedAt scope of the debug
/// variable. const DILocalVariable *: It is a pointer to the debug variable
/// itself.
using VarID =
    std::tuple<const DIScope *, const DIScope *, const DILocalVariable *>;

/// A base class to collect and print dropped debug information variable
/// statistics.
class DroppedVariableStats {
public:
  DroppedVariableStats(bool DroppedVarStatsEnabled)
      : DroppedVariableStatsEnabled(DroppedVarStatsEnabled) {
    if (DroppedVarStatsEnabled)
      llvm::outs()
          << "Pass Level, Pass Name, Num of Dropped Variables, Func or "
             "Module Name\n";
  };

  virtual ~DroppedVariableStats() {}

  // We intend this to be unique per-compilation, thus no copies.
  DroppedVariableStats(const DroppedVariableStats &) = delete;
  void operator=(const DroppedVariableStats &) = delete;

  bool getPassDroppedVariables() { return PassDroppedVariables; }

protected:
  void setup() {
    DebugVariablesStack.push_back(
        {DenseMap<const Function *, DebugVariables>()});
    InlinedAts.push_back(
        {DenseMap<StringRef, DenseMap<VarID, DILocation *>>()});
  }

  void cleanup() {
    assert(!DebugVariablesStack.empty() &&
           "DebugVariablesStack shouldn't be empty!");
    assert(!InlinedAts.empty() && "InlinedAts shouldn't be empty!");
    DebugVariablesStack.pop_back();
    InlinedAts.pop_back();
  }

  bool DroppedVariableStatsEnabled = false;
  struct DebugVariables {
    /// DenseSet of VarIDs before an optimization pass has run.
    DenseSet<VarID> DebugVariablesBefore;
    /// DenseSet of VarIDs after an optimization pass has run.
    DenseSet<VarID> DebugVariablesAfter;
  };

protected:
  /// A stack of a DenseMap, that maps DebugVariables for every pass to an
  /// llvm::Function. A stack is used because an optimization pass can call
  /// other passes.
  SmallVector<DenseMap<const Function *, DebugVariables>> DebugVariablesStack;

  /// A DenseSet tracking whether a scope was visited before.
  DenseSet<const DIScope *> VisitedScope;
  /// A stack of DenseMaps, which map the name of an llvm::Function to a
  /// DenseMap of VarIDs and their inlinedAt locations before an optimization
  /// pass has run.
  SmallVector<DenseMap<StringRef, DenseMap<VarID, DILocation *>>> InlinedAts;
  /// Calculate the number of dropped variables in an llvm::Function or
  /// llvm::MachineFunction and print the relevant information to stdout.
  void calculateDroppedStatsAndPrint(DebugVariables &DbgVariables,
                                     StringRef FuncName, StringRef PassID,
                                     StringRef FuncOrModName,
                                     StringRef PassLevel,
                                     const Function *Func) {
    unsigned DroppedCount = 0;
    DenseSet<VarID> &DebugVariablesBeforeSet =
        DbgVariables.DebugVariablesBefore;
    DenseSet<VarID> &DebugVariablesAfterSet = DbgVariables.DebugVariablesAfter;
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
      llvm::outs() << PassLevel << ", " << PassID << ", " << DroppedCount
                   << ", " << FuncOrModName << "\n";
      PassDroppedVariables = true;
    } else
      PassDroppedVariables = false;
  }

  /// Check if a \p Var has been dropped or is a false positive. Also update the
  /// \p DroppedCount if a debug variable is dropped.
  bool updateDroppedCount(DILocation *DbgLoc, const DIScope *Scope,
                          const DIScope *DbgValScope,
                          DenseMap<VarID, DILocation *> &InlinedAtsMap,
                          VarID Var, unsigned &DroppedCount) {
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
  /// Run code to populate relevant data structures over an llvm::Function or
  /// llvm::MachineFunction.
  void run(DebugVariables &DbgVariables, StringRef FuncName, bool Before) {
    auto &VarIDSet = (Before ? DbgVariables.DebugVariablesBefore
                             : DbgVariables.DebugVariablesAfter);
    auto &InlinedAtsMap = InlinedAts.back();
    if (Before)
      InlinedAtsMap.try_emplace(FuncName, DenseMap<VarID, DILocation *>());
    VarIDSet = DenseSet<VarID>();
    visitEveryDebugRecord(VarIDSet, InlinedAtsMap, FuncName, Before);
  }
  /// Populate the VarIDSet and InlinedAtMap with the relevant information
  /// needed for before and after pass analysis to determine dropped variable
  /// status.
  void populateVarIDSetAndInlinedMap(
      const DILocalVariable *DbgVar, DebugLoc DbgLoc, DenseSet<VarID> &VarIDSet,
      DenseMap<StringRef, DenseMap<VarID, DILocation *>> &InlinedAtsMap,
      StringRef FuncName, bool Before) {
    VarID Key{DbgVar->getScope(), DbgLoc->getInlinedAtScope(), DbgVar};
    VarIDSet.insert(Key);
    if (Before)
      InlinedAtsMap[FuncName].try_emplace(Key, DbgLoc.getInlinedAt());
  }
  /// Visit every llvm::Instruction or llvm::MachineInstruction and check if the
  /// debug variable denoted by its ID \p Var may have been dropped by an
  /// optimization pass.
  virtual void
  visitEveryInstruction(unsigned &DroppedCount,
                        DenseMap<VarID, DILocation *> &InlinedAtsMap,
                        VarID Var) = 0;
  /// Visit every debug record in an llvm::Function or llvm::MachineFunction
  /// and call populateVarIDSetAndInlinedMap on it.
  virtual void visitEveryDebugRecord(
      DenseSet<VarID> &VarIDSet,
      DenseMap<StringRef, DenseMap<VarID, DILocation *>> &InlinedAtsMap,
      StringRef FuncName, bool Before) = 0;

private:
  /// Remove a dropped debug variable's VarID from all Sets in the
  /// DroppedVariablesBefore stack.
  void removeVarFromAllSets(VarID Var, const Function *F) {
    // Do not remove Var from the last element, it will be popped from the
    // stack.
    for (auto &DebugVariablesMap : llvm::drop_end(DebugVariablesStack))
      DebugVariablesMap[F].DebugVariablesBefore.erase(Var);
  }
  /// Return true if \p Scope is the same as \p DbgValScope or a child scope of
  /// \p DbgValScope, return false otherwise.
  bool isScopeChildOfOrEqualTo(const DIScope *Scope,
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
  /// Return true if \p InlinedAt is the same as \p DbgValInlinedAt or part of
  /// the InlinedAt chain, return false otherwise.
  bool isInlinedAtChildOfOrEqualTo(const DILocation *InlinedAt,
                                   const DILocation *DbgValInlinedAt) {
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
  bool PassDroppedVariables = false;
};

} // namespace llvm

#endif
