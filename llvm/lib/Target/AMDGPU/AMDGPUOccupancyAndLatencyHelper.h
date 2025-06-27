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

#include "llvm/ADT/DenseMap.h"
#include "llvm/MC/MCInstrItineraries.h"

namespace llvm {

class MachineInstr;
class MachineFunction;
class GCNSubtarget;
class MachineLoopInfo;
class SIInstrInfo;

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

  void sum(const SchedScore &S, unsigned LoopDepth = 0);
  bool isMemBound(unsigned TargetOccupancy, unsigned ExtraOcc = 1) const;
  // More latency can be hiden with ExtraOcc.
  unsigned latencyGain(unsigned TargetOccupancy, unsigned ExtraOcc) const;
};

struct AMDGPULatencyTracker {
  AMDGPULatencyTracker(const llvm::GCNSubtarget &ST);
  const llvm::SIInstrInfo *SIII;
  const llvm::InstrItineraryData *ItinerayData;
  // Latency MI dst reg to cycle map.
  llvm::DenseMap<unsigned, int> LatencyMIs;
  SchedScore Score;
  // Low latency MI not wait.
  unsigned HideLatency = 0;
  unsigned MemLatency = 0;
  // For simple, only consider mixture as one valu one salu.
  // Not group now.
  unsigned PrevSAlu = 0;
  unsigned PrevVAlu = 0;
  enum class AluStatus {
    Nothing,
    Vector,
    Scalar,
  } PrevStatus = AluStatus::Nothing;
  void scan(const llvm::MachineInstr &MI);
};

SchedScore collectLatency(llvm::MachineFunction &MF,
                          const llvm::GCNSubtarget &ST,
                          const llvm::MachineLoopInfo *MLI = nullptr);

} // namespace llvm
#endif
