//===- AMDGPUStaticSimulator.h - Static performance simulator ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Defines aggregate metrics reported by the AMDGPU static simulator pass.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUSTATICSIMULATOR_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUSTATICSIMULATOR_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

class MachineBasicBlock;
class MachineFunction;

namespace AMDGPU {

/// Aggregate instruction and stall metrics for one machine basic block.
struct StaticSimulatorBlockMetrics {
  unsigned NumInstructions = 0;
  unsigned NumVALU = 0;
  unsigned NumSALU = 0;
  unsigned NumTRANS = 0;
  unsigned NumWMMA = 0;
  unsigned NumVOPD = 0;
  unsigned NumPacked = 0;
  unsigned NumDSRead = 0;
  unsigned NumDSWrite = 0;
  unsigned NumVMEM = 0;
  unsigned NumSMEM = 0;
  unsigned NumTDM = 0;
  unsigned NumBranch = 0;
  unsigned NumBarrier = 0;
  unsigned NumNop = 0;
  unsigned NumDelayAlu = 0;
  unsigned NumMSBSet = 0;
  unsigned NumSpill = 0;
  unsigned NumReload = 0;
  unsigned NumSGPRToVGPR = 0;
  unsigned NumVGPRToSGPR = 0;
  unsigned NumWaitcnt = 0;

  unsigned NumBytes = 0;
  unsigned TotalCycles = 0;
  unsigned TotalStallCycles = 0;

  // Non-MSB effective stalls are assigned to one dominant category.
  unsigned StallFunctionalUnit = 0;
  unsigned StallCoExec = 0;
  unsigned StallDelayAlu = 0;
  unsigned StallMemFIFO = 0;
  unsigned StallWaitCnt = 0;
  unsigned StallLongLatVALU = 0;
  unsigned StallLOLVALUTRANS = 0;
  unsigned StallVaSSRC = 0;
  unsigned StallVaVdst = 0;
  unsigned StallOther = 0;

  // Unmasked exposed MSB sets add one direct stall cycle. If the WMMA window
  // remains active after that cycle, it is added again as a coexecution miss.
  unsigned NumMSBSetExposed = 0;
  // Masked MSB sets are informational and do not add stall cycles.
  unsigned NumMSBSetMasked = 0;

  // These fields partition StallCoExec cycles by instruction class.
  unsigned CoExecMissVALU = 0;
  unsigned CoExecMissTRANS = 0;
  unsigned CoExecMissMemory = 0;
  unsigned CoExecMissOther = 0;

  unsigned WMMAOccupancyCycles = 0;
  unsigned WMMACoExecUsed = 0;
  unsigned WMMACoExecBlocked = 0;
  unsigned ISlotTotal = 0;
  unsigned ISlotUsedByVALU = 0;
  unsigned ISlotWastedOnNonVALU = 0;

  void add(const StaticSimulatorBlockMetrics &Other);

  /// Print nonzero stall categories and MSB exposure counts to \p OS.
  void printStallBreakdown(raw_ostream &OS) const;
};

/// Per block metrics and their function total.
/// PerBlock keys are nonowning and their blocks must outlive this report.
struct StaticSimulatorReport {
  DenseMap<const MachineBasicBlock *, StaticSimulatorBlockMetrics> PerBlock;
  StaticSimulatorBlockMetrics Total;

  /// Print an assembly comment style function summary to \p OS.
  void print(raw_ostream &OS, const MachineFunction &MF) const;
};

} // namespace AMDGPU
} // namespace llvm

#endif
