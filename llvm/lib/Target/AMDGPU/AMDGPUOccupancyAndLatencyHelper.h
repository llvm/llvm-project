//==- AMDGPUOccupancyAndLatencyHelper.cpp - Helpers for occupancy + latency ==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Helper functions for occupancy and latency.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUOCCUPANCYANDLATENCYHELPER_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUOCCUPANCYANDLATENCYHELPER_H

namespace llvm {

class MachineFunction;
class GCNSubtarget;
class MachineLoopInfo;

struct SchedScore {
  // Score for this Sched result.
  unsigned Occupancy = 0;
  bool SgprSpill = false;
  unsigned LatencyHide = 0; // Only latency hide will split 2 load into 2 pass?
  unsigned MemLatency = 0;  // Only save mem latency.
                            // We want mem latency small and hide big. Compare
                            // memLatency - hide * Occ, smaller is better.
  unsigned MixAlu = 0;      // VAlu and SAlu can running parallel if Occ > 1.
  unsigned Alu = 0; // avoid sequence of s_alu inst count less then occupancy.
  unsigned Lds = 0; // Todo: count lds.
  SchedScore() {}

  // Other info which can help compare schedule result.
  float computeScore() const;
  float computeScore2() const;

  void sum(const SchedScore &S, unsigned LoopDepth = 0);
  bool isBetter(const SchedScore &S) const;
  bool isMemBound(unsigned TargetOccupancy, unsigned ExtraOcc = 1) const;
  // More latency can be hiden with ExtraOcc.
  unsigned latencyGain(unsigned TargetOccupancy, unsigned ExtraOcc) const;
};

SchedScore collectLatency(llvm::MachineFunction &MF,
                          const llvm::GCNSubtarget &ST,
                          const llvm::MachineLoopInfo *MLI = nullptr);

}
#endif
