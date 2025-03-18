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

#include "llvm/ADT/DenseMap.h"
#include "llvm/MC/MCInstrItineraries.h"

namespace llvm {

class MachineFunction;
class GCNSubtarget;
class MachineInstr;
class SIInstrInfo;
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

  void sum(const SchedScore &s, unsigned loopDepth = 0);
  bool isBetter(const SchedScore &s) const;
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
  SchedScore score;
  // Low latency MI not wait.
  unsigned hideLatency = 0;
  unsigned memLatency = 0;
  // For simple, only consider mixture as one valu one salu.
  // Not group now.
  unsigned prevSAlu = 0;
  unsigned prevVAlu = 0;
  enum class AluStatus {
    Nothing,
    Vector,
    Scalar,
  } prevStatus = AluStatus::Nothing;
  void scan(const llvm::MachineInstr &MI);
};

SchedScore CollectLatency(llvm::MachineFunction &MF,
                          const llvm::GCNSubtarget &ST,
                          const llvm::MachineLoopInfo *MLI = nullptr);
} // namespace llvm
