//===---------------------- AMDGPUNextUseAnalysis.h  ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements Next Use Analysis.
// For each register it goes over all uses and returns the estimated distance of
// the nearest use. This will be used for selecting which registers to spill
// before register allocation.
//
// This is based on ideas from the paper:
// "Register Spilling and Live-Range Splitting for SSA-Form Programs"
// Matthias Braun and Sebastian Hack, CC'09
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUNEXTUSEANALYSIS_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUNEXTUSEANALYSIS_H

#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/IR/PassManager.h"
#include <optional>

namespace llvm {

//==============================================================================
// AMDGPUNextUseAnalysis - Provides next-use distances for live registers or
// sub-registers at a given MachineInstruction suitable for making spilling
// decisions.
//==============================================================================
class AMDGPUNextUseAnalysisImpl;
class AMDGPUNextUseAnalysis {
  friend class AMDGPUNextUseAnalysisLegacyPass;
  friend class AMDGPUNextUseAnalysisPrinterLegacyPass;
  friend class AMDGPUNextUseAnalysisPass;
  friend class AMDGPUNextUseAnalysisPrinterPass;

  std::unique_ptr<AMDGPUNextUseAnalysisImpl> Impl;

  AMDGPUNextUseAnalysis(const MachineFunction *, const MachineLoopInfo *);

public:
  AMDGPUNextUseAnalysis(AMDGPUNextUseAnalysis &&Other);
  ~AMDGPUNextUseAnalysis();

  AMDGPUNextUseAnalysis &operator=(AMDGPUNextUseAnalysis &&Other);

  enum CompatibilityMode { Compute, Graphics };

  CompatibilityMode getCompatibilityMode();
  void setCompatibilityMode(CompatibilityMode);

  /// \Returns the next-use distance for \p LiveReg.
  std::optional<double>
  getNextUseDistance(Register LiveReg, const MachineInstr &CurMI,
                     const SmallVector<const MachineOperand *> &Uses,
                     SmallVector<double> *Distances = nullptr,
                     const MachineOperand **UseOut = nullptr);

  struct UseDistancePair {
    const MachineOperand *Use = nullptr;
    double Dist = 0.0;
    UseDistancePair() = default;
    UseDistancePair(const MachineOperand *Use, double Dist)
        : Use(Use), Dist(Dist) {}
  };

  void getNextUseDistances(const DenseMap<unsigned, LaneBitmask> &LiveRegs,
                           const MachineInstr &MI, UseDistancePair &Furthest,
                           UseDistancePair *FurthestSubreg = nullptr,
                           DenseMap<const MachineOperand *, UseDistancePair>
                               *RelevantUses = nullptr) const;

  void getUses(unsigned Register, LaneBitmask LaneMask, const MachineInstr &MI,
               SmallVector<const MachineOperand *> &Uses);
};

//==============================================================================
// AMDGPUNextUseAnalysisLegacyPass - Legacy and New pass wrapper around
// AMDGPUNextUseAnalysis
//==============================================================================
class AMDGPUNextUseAnalysisLegacyPass : public MachineFunctionPass {

public:
  static char ID;

  AMDGPUNextUseAnalysisLegacyPass();

  AMDGPUNextUseAnalysis &getNextUseAnalysis() { return *NUA; }
  const AMDGPUNextUseAnalysis &getNextUseAnalysis() const { return *NUA; }
  StringRef getPassName() const override;

protected:
  bool runOnMachineFunction(MachineFunction &) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  std::unique_ptr<AMDGPUNextUseAnalysis> NUA;
};

class AMDGPUNextUseAnalysisPass
    : public AnalysisInfoMixin<AMDGPUNextUseAnalysisPass> {
  friend AnalysisInfoMixin<AMDGPUNextUseAnalysisPass>;
  static AnalysisKey Key;

public:
  using Result = AMDGPUNextUseAnalysis;
  Result run(MachineFunction &MF, MachineFunctionAnalysisManager &MFAM);
};

//==============================================================================
// AMDGPUNextUseAnalysisPrinterLegacyPass - Legacy Pass for printing
// AMDGPUNextUseAnalysis results as JSON.
//==============================================================================
class AMDGPUNextUseAnalysisPrinterLegacyPass : public MachineFunctionPass {

public:
  static char ID;

  AMDGPUNextUseAnalysisPrinterLegacyPass();

  StringRef getPassName() const override;

protected:
  bool runOnMachineFunction(MachineFunction &) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

class AMDGPUNextUseAnalysisPrinterPass
    : public PassInfoMixin<AMDGPUNextUseAnalysisPrinterPass> {
  raw_ostream &OS;

public:
  explicit AMDGPUNextUseAnalysisPrinterPass(raw_ostream &OS) : OS(OS) {}
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
  static bool isRequired() { return true; }
};

} // namespace llvm
#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUNEXTUSEANALYSIS_H
