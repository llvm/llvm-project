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

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <tuple>

namespace llvm {

class DIScope;
class DILocalVariable;
class Function;
class DILocation;
class DebugLoc;
class StringRef;

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
  DroppedVariableStats(bool DroppedVarStatsEnabled);

  virtual ~DroppedVariableStats() {}

  // We intend this to be unique per-compilation, thus no copies.
  DroppedVariableStats(const DroppedVariableStats &) = delete;
  void operator=(const DroppedVariableStats &) = delete;

  bool getPassDroppedVariables() { return PassDroppedVariables; }

protected:
  void setup();

  void cleanup();

  bool DroppedVariableStatsEnabled = false;
  struct DebugVariables {
    /// DenseSet of VarIDs before an optimization pass has run.
    DenseSet<VarID> DebugVariablesBefore;
    /// DenseSet of VarIDs after an optimization pass has run.
    DenseSet<VarID> DebugVariablesAfter;
  };

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
                                     StringRef PassLevel, const Function *Func);

  /// Check if a \p Var has been dropped or is a false positive. Also update the
  /// \p DroppedCount if a debug variable is dropped.
  bool updateDroppedCount(DILocation *DbgLoc, const DIScope *Scope,
                          const DIScope *DbgValScope,
                          DenseMap<VarID, DILocation *> &InlinedAtsMap,
                          VarID Var, unsigned &DroppedCount);

  /// Run code to populate relevant data structures over an llvm::Function or
  /// llvm::MachineFunction.
  void run(DebugVariables &DbgVariables, StringRef FuncName, bool Before);

  /// Populate the VarIDSet and InlinedAtMap with the relevant information
  /// needed for before and after pass analysis to determine dropped variable
  /// status.
  void populateVarIDSetAndInlinedMap(
      const DILocalVariable *DbgVar, DebugLoc DbgLoc, DenseSet<VarID> &VarIDSet,
      DenseMap<StringRef, DenseMap<VarID, DILocation *>> &InlinedAtsMap,
      StringRef FuncName, bool Before);

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
  void removeVarFromAllSets(VarID Var, const Function *F);

  /// Return true if \p Scope is the same as \p DbgValScope or a child scope of
  /// \p DbgValScope, return false otherwise.
  bool isScopeChildOfOrEqualTo(const DIScope *Scope,
                               const DIScope *DbgValScope);

  /// Return true if \p InlinedAt is the same as \p DbgValInlinedAt or part of
  /// the InlinedAt chain, return false otherwise.
  bool isInlinedAtChildOfOrEqualTo(const DILocation *InlinedAt,
                                   const DILocation *DbgValInlinedAt);

  bool PassDroppedVariables = false;
};

} // namespace llvm

#endif
