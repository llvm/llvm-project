//===- AMDGPURegPressAnalysis.h --- analysis of BB reg pressure -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Analyzes the register pressure for each basic block. Provides simple
/// interface to determine SGPR and VGPR max pressure
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUREGPRESSANALYSIS_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUREGPRESSANALYSIS_H

#include "GCNRegPressure.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Pass.h"

namespace llvm {

struct AMDGPURegPressAnalysis : public MachineFunctionPass {
  static char ID;

public:
  AMDGPURegPressAnalysis() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervals>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  GCNRegPressure getPressure(const MachineBasicBlock *MBB) const;
  GCNRegPressure getMaxPressure() const;

private:
  using MBBPressureMap = std::map<const MachineBasicBlock *, GCNRegPressure>;

  MBBPressureMap BlockPressure;
  GCNRegPressure MaxPressure;
};
} // namespace llvm
#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUREGPRESSANALYSIS_H
