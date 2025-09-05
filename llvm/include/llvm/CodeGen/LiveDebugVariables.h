//===- LiveDebugVariables.h - Tracking debug info variables -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides the interface to the LiveDebugVariables analysis.
//
// The analysis removes DBG_VALUE instructions for virtual registers and tracks
// live user variables in a data structure that can be updated during register
// allocation.
//
// After register allocation new DBG_VALUE instructions are emitted to reflect
// the new locations of user variables.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_LIVEDEBUGVARIABLES_H
#define LLVM_CODEGEN_LIVEDEBUGVARIABLES_H

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

namespace llvm {

template <typename T> class ArrayRef;
class LiveIntervals;
class VirtRegMap;

class LiveDebugVariables {

public:
  class LDVImpl;
  LiveDebugVariables();
  ~LiveDebugVariables();
  LiveDebugVariables(LiveDebugVariables &&);

  void analyze(MachineFunction &MF, LiveIntervals *LIS);
  /// splitRegister - Move any user variables in OldReg to the live ranges in
  /// NewRegs where they are live. Mark the values as unavailable where no new
  /// register is live.
  void splitRegister(Register OldReg, ArrayRef<Register> NewRegs,
                     LiveIntervals &LIS);

  /// emitDebugValues - Emit new DBG_VALUE instructions reflecting the changes
  /// that happened during register allocation.
  /// @param VRM Rename virtual registers according to map.
  void emitDebugValues(VirtRegMap *VRM);

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// dump - Print data structures to dbgs().
  void dump() const;
#endif

  void print(raw_ostream &OS) const;

  void releaseMemory();

  bool invalidate(MachineFunction &MF, const PreservedAnalyses &PA,
                  MachineFunctionAnalysisManager::Invalidator &Inv);

private:
  std::unique_ptr<LDVImpl> PImpl;
};

class LiveDebugVariablesWrapperLegacy : public MachineFunctionPass {
  std::unique_ptr<LiveDebugVariables> Impl;

public:
  static char ID; // Pass identification, replacement for typeid

  LiveDebugVariablesWrapperLegacy();

  bool runOnMachineFunction(MachineFunction &) override;

  LiveDebugVariables &getLDV() { return *Impl; }
  const LiveDebugVariables &getLDV() const { return *Impl; }

  void releaseMemory() override {
    if (Impl)
      Impl->releaseMemory();
  }
  void getAnalysisUsage(AnalysisUsage &) const override;

  MachineFunctionProperties getSetProperties() const override {
    return MachineFunctionProperties().setTracksDebugUserValues();
  }
};

class LiveDebugVariablesAnalysis
    : public AnalysisInfoMixin<LiveDebugVariablesAnalysis> {
  friend AnalysisInfoMixin<LiveDebugVariablesAnalysis>;
  static AnalysisKey Key;

public:
  using Result = LiveDebugVariables;

  MachineFunctionProperties getSetProperties() const {
    return MachineFunctionProperties().setTracksDebugUserValues();
  }

  Result run(MachineFunction &MF, MachineFunctionAnalysisManager &MFAM);
};

class LiveDebugVariablesPrinterPass
    : public PassInfoMixin<LiveDebugVariablesPrinterPass> {
  raw_ostream &OS;

public:
  LiveDebugVariablesPrinterPass(raw_ostream &OS) : OS(OS) {}

  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};
} // end namespace llvm

#endif // LLVM_CODEGEN_LIVEDEBUGVARIABLES_H
