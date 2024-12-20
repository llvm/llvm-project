///===- DroppedVariableStatsMIR.h - Opt Diagnostics -*- C++ -*-------------===//
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

#ifndef LLVM_CODEGEN_DROPPEDVARIABLESTATSMIR_H
#define LLVM_CODEGEN_DROPPEDVARIABLESTATSMIR_H

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Passes/DroppedVariableStats.h"

namespace llvm {

/// A class to collect and print dropped debug information due to MIR
/// optimization passes. After every MIR pass is run, it will print how many
/// #DBG_VALUEs were dropped due to that pass.
class DroppedVariableStatsMIR : public DroppedVariableStats {
public:
  DroppedVariableStatsMIR() : llvm::DroppedVariableStats(false) {}

  void runBeforePass(StringRef PassID, MachineFunction *MF) {
    if (PassID == "Debug Variable Analysis")
      return;
    setup();
    return runOnMachineFunction(MF, true);
  }

  void runAfterPass(StringRef PassID, MachineFunction *MF) {
    if (PassID == "Debug Variable Analysis")
      return;
    runOnMachineFunction(MF, false);
    calculateDroppedVarStatsOnMachineFunction(MF, PassID, MF->getName().str());
    cleanup();
  }

private:
  const MachineFunction *MFunc;
  /// Populate DebugVariablesBefore, DebugVariablesAfter, InlinedAts before or
  /// after a pass has run to facilitate dropped variable calculation for an
  /// llvm::MachineFunction.
  void runOnMachineFunction(const MachineFunction *MF, bool Before);
  /// Iterate over all Instructions in a MachineFunction and report any dropped
  /// debug information.
  void calculateDroppedVarStatsOnMachineFunction(const MachineFunction *MF,
                                                 StringRef PassID,
                                                 StringRef FuncOrModName);
  /// Override base class method to run on an llvm::MachineFunction
  /// specifically.
  virtual void
  visitEveryInstruction(unsigned &DroppedCount,
                        DenseMap<VarID, DILocation *> &InlinedAtsMap,
                        VarID Var) override;
  /// Override base class method to run on DBG_VALUEs specifically.
  virtual void visitEveryDebugRecord(
      DenseSet<VarID> &VarIDSet,
      DenseMap<StringRef, DenseMap<VarID, DILocation *>> &InlinedAtsMap,
      StringRef FuncName, bool Before) override;
};

} // namespace llvm

#endif
