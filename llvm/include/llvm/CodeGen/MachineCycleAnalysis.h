//===- MachineCycleAnalysis.h - Cycle Info for Machine IR -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the MachineCycleInfo class, which is a thin wrapper over
// the Machine IR instance of GenericCycleInfo.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINECYCLEANALYSIS_H
#define LLVM_CODEGEN_MACHINECYCLEANALYSIS_H

#include "llvm/ADT/GenericCycleInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/CodeGen/MachineSSAContext.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

using MachineCycleInfo = GenericCycleInfo<MachineSSAContext>;
using MachineCycle = MachineCycleInfo::CycleT;

/// Legacy analysis pass which computes a \ref MachineCycleInfo.
class LLVM_ABI MachineCycleInfoWrapperPass : public MachineFunctionPass {
  MachineFunction *F = nullptr;
  MachineCycleInfo CI;

public:
  static char ID;

  MachineCycleInfoWrapperPass();

  MachineCycleInfo &getCycleInfo() { return CI; }
  const MachineCycleInfo &getCycleInfo() const { return CI; }

  bool runOnMachineFunction(MachineFunction &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  void releaseMemory() override;
  void print(raw_ostream &OS, const Module *M = nullptr) const override;
};

// TODO: add this function to GenericCycle template after implementing IR
//       version.
LLVM_ABI bool isCycleInvariant(const MachineCycle *Cycle, MachineInstr &I);

class MachineCycleAnalysis : public AnalysisInfoMixin<MachineCycleAnalysis> {
  friend AnalysisInfoMixin<MachineCycleAnalysis>;
  LLVM_ABI static AnalysisKey Key;

public:
  using Result = MachineCycleInfo;

  LLVM_ABI Result run(MachineFunction &MF,
                      MachineFunctionAnalysisManager &MFAM);
};

class MachineCycleInfoPrinterPass
    : public PassInfoMixin<MachineCycleInfoPrinterPass> {
  raw_ostream &OS;

public:
  explicit MachineCycleInfoPrinterPass(raw_ostream &OS) : OS(OS) {}
  LLVM_ABI PreservedAnalyses run(MachineFunction &MF,
                                 MachineFunctionAnalysisManager &MFAM);
  static bool isRequired() { return true; }
};

} // end namespace llvm

#endif // LLVM_CODEGEN_MACHINECYCLEANALYSIS_H
