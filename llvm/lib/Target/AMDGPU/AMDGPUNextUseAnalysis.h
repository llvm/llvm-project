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
#include <optional>

namespace llvm {

//------------------------------------------------------------------------------
//
//------------------------------------------------------------------------------
class AMDGPUNextUseAnalysisImpl;
class AMDGPUNextUseAnalysis {
  friend class AMDGPUNextUseAnalysisPass;

  std::unique_ptr<AMDGPUNextUseAnalysisImpl> Impl;

  void initialize(const MachineFunction *, const MachineLoopInfo *,
                  const MachineDominatorTree *);

public:
  enum CompatibilityMode { Compute, Graphics };

  CompatibilityMode getCompatibilityMode();
  void setCompatibilityMode(CompatibilityMode);

  /// \Returns the next-use distance for \p DefReg.
  std::optional<double>
  getNextUseDistance(Register LiveReg, const MachineInstr &CurMI,
                     const SmallVector<const MachineOperand *> &Uses,
                     SmallVector<double> *Distances = nullptr,
                     const MachineOperand **UseOut = nullptr);

  void getUses(unsigned Register, LaneBitmask LaneMask, const MachineInstr &MI,
               SmallVector<const MachineOperand *> &Uses);

  void printFurthestDistancesAsJson(raw_ostream &OS, const LiveIntervals *LIS);
};

//------------------------------------------------------------------------------
//
//------------------------------------------------------------------------------
class AMDGPUNextUseAnalysisPass : public MachineFunctionPass {

public:
  static char ID;

  AMDGPUNextUseAnalysisPass() : MachineFunctionPass(ID) {}

  AMDGPUNextUseAnalysis &getAMDGPUNextUseAnalysis() { return *NUA; }
  const AMDGPUNextUseAnalysis &getAMDGPUNextUseAnalysis() const { return *NUA; }
  StringRef getPassName() const override { return "Next Use Analysis"; }

protected:
  bool runOnMachineFunction(MachineFunction &) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  std::unique_ptr<AMDGPUNextUseAnalysis> NUA;
};

} // namespace llvm
#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUNEXTUSEANALYSIS_H
