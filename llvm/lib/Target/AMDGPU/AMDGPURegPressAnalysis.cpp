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

#include "AMDGPURegPressAnalysis.h"
#include "AMDGPU.h"
#include "llvm/CodeGen/LiveIntervals.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-reg-press"

char llvm::AMDGPURegPressAnalysis::ID = 0;
char &llvm::AMDGPURegPressAnalysisID = AMDGPURegPressAnalysis::ID;

INITIALIZE_PASS(AMDGPURegPressAnalysis, DEBUG_TYPE,
                "Analysis of per BB register pressure", true, true)

GCNRegPressure AMDGPURegPressAnalysis::getMaxPressure() const {
  return MaxPressure;
}

GCNRegPressure
AMDGPURegPressAnalysis::getPressure(const MachineBasicBlock *MBB) const {
  assert(BlockPressure.count(MBB) && "Could not find MBB in map");
  return BlockPressure.at(MBB);
}

bool AMDGPURegPressAnalysis::runOnMachineFunction(MachineFunction &MF) {

  // Reset trackers, or else they will carry across functions in a module.
  BlockPressure.clear();
  MaxPressure.clear();

  if (skipFunction(MF.getFunction()))
    return false;

  LiveIntervalsWrapperPass &LIS = getAnalysis<LiveIntervalsWrapperPass>();

  GCNUpwardRPTracker RPT(LIS.getLIS());

  for (auto &MBB : MF) {
    GCNRegPressure BBMaxPressure;

    if (!MBB.empty()) {
      RPT.reset(MBB.instr_back());
      for (auto &MI : reverse(MBB))
        RPT.recede(MI);

      BBMaxPressure = RPT.getMaxPressureAndReset();
    }

    BlockPressure[&MBB] = BBMaxPressure;
    MaxPressure = max(BBMaxPressure, MaxPressure);
  }

  return false;
}
