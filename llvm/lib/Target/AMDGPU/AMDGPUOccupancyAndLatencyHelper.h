//===- AMDGPUOccupancyAndLatencyHelper.h - Helpers for occupancy ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Helper functions for occupancy and latency used by the AMDGPU hot
/// block rematerialization pass to decide whether gaining occupancy is worth
/// more than the cost of re-executing rematerialized instructions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUOCCUPANCYANDLATENCYHELPER_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUOCCUPANCYANDLATENCYHELPER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/MC/MCInstrItineraries.h"

namespace llvm {

class MachineInstr;
class MachineFunction;
class GCNSubtarget;
class MachineLoopInfo;
class SIInstrInfo;

struct SchedScore {
  unsigned Occupancy = 0;
  bool SgprSpill = false;
  // Latency that could be hidden by overlapping with other work.
  unsigned LatencyHide = 0;
  // Memory latency that could not be hidden.
  unsigned MemLatency = 0;
  // Vector and scalar ALU can run in parallel when occupancy > 1.
  unsigned MixAlu = 0;
  unsigned Alu = 0;
  unsigned Lds = 0;

  SchedScore() = default;

  void sum(const SchedScore &S, unsigned LoopDepth = 0);
  // Returns true when reaching TargetOccupancy hides enough extra memory
  // latency to justify the pass's transformations.
  bool isMemBound(unsigned TargetOccupancy, unsigned ExtraOcc = 1) const;
  // Extra memory latency that ExtraOcc additional waves would hide.
  unsigned latencyGain(unsigned TargetOccupancy, unsigned ExtraOcc) const;
};

struct AMDGPULatencyTracker {
  AMDGPULatencyTracker(const GCNSubtarget &ST);

  const SIInstrInfo *SIII;
  const InstrItineraryData *ItineraryData;
  // Maps a memory-latency instruction's destination register to its remaining
  // cycle count.
  DenseMap<unsigned, int> LatencyMIs;
  SchedScore Score;

  enum class AluStatus {
    Nothing,
    Vector,
    Scalar,
  } PrevStatus = AluStatus::Nothing;

  void scan(const MachineInstr &MI);
};

SchedScore collectLatency(MachineFunction &MF, const GCNSubtarget &ST,
                          const MachineLoopInfo *MLI = nullptr);

} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUOCCUPANCYANDLATENCYHELPER_H
