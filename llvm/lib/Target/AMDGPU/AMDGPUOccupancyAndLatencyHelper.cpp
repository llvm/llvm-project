//===-- AMDGPUOccupancyAndLatencyHelper - Helper functions for occupancy and latency --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===--------------------------------------------------------------------------------===//
//
/// \file
/// \brief Helper functions for occupancy and latency.
//
//===--------------------------------------------------------------------------------===//

#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "AMDGPUSubtarget.h"
#include "GCNSubtarget.h"
#include "AMDGPUOccupancyAndLatencyHelper.h"

#include "llvm/CodeGen/MachineLoopInfo.h"

namespace llvm {

// Other info which can help compare schedule result.
float SchedScore::computeScore() const {
  // Occupancy 1 cannot mix alu.
  unsigned MixHidenAlu = Alu - MixAlu;
  if (Occupancy == 1)
    MixHidenAlu = 0;
  return ((float)MemLatency - (float)MixHidenAlu) / (float)Occupancy -
         LatencyHide;
}
float SchedScore::computeScore2() const {
  float cycles = 0;
  cycles = (MixAlu * Occupancy + MemLatency);
  cycles /= Occupancy;
  return cycles;
}

void SchedScore::sum(const SchedScore &s, unsigned loopDepth) {
  unsigned loopCount = loopDepth > 0 ? std::pow(3, loopDepth) : 1;
  LatencyHide += loopCount * s.LatencyHide;
  MemLatency += loopCount * s.MemLatency;
  MixAlu += loopCount * s.MixAlu;
  Alu += loopCount * s.Alu;
  Lds += loopCount * s.Lds;
  SgprSpill |= s.SgprSpill;
}
bool SchedScore::isBetter(const SchedScore &s) const {
  float score = computeScore();
  float newScore = s.computeScore();
  bool spillBetter = !SgprSpill && s.SgprSpill;
  return spillBetter ? true : newScore >= score;
}
// Does more occupancy give more perf.
bool SchedScore::isMemBound(unsigned TargetOccupancy, unsigned ExtraOcc) const {
  unsigned gain = latencyGain(TargetOccupancy, ExtraOcc);
  // 10% is good enough.
  if ((10*gain) >= Alu)
    return true;
  else
    return false;
}

unsigned SchedScore::latencyGain(unsigned TgtOcc, unsigned ExtraOcc) const {
  unsigned latency = MemLatency;
  return (latency / (TgtOcc))- (latency / (TgtOcc + ExtraOcc));
}

// AMDGPULatencyTracker
AMDGPULatencyTracker::AMDGPULatencyTracker(const GCNSubtarget &ST)
    : SIII(ST.getInstrInfo()), ItinerayData(ST.getInstrItineraryData()) {}

void AMDGPULatencyTracker::scan(const MachineInstr &MI) {
  if (MI.isDebugInstr()) return;
  int latency = SIII->getInstrLatency(ItinerayData, MI);
  // If inside latency hide.
  if (!LatencyMIs.empty()) {
    bool bWaitCnt = false;
    for (auto &MO : MI.operands()) {
      if (MO.isReg()) {
        unsigned reg = MO.getReg();
        auto it = LatencyMIs.find(reg);
        if (it != LatencyMIs.end()) {
          bWaitCnt = true;
          // If MI use mem result, update latency to mem latency.
          int cycle = it->second;
          if (cycle > latency)
            latency = cycle;
        }
      }
    }
    // Update latency for each mem latency inst.
    for (auto it = LatencyMIs.begin(); it != LatencyMIs.end();) {
      auto prev = it;
      auto l = (it++);
      int cycle = l->second;
      if (cycle <= latency) {
        // Only left cycles.
        // Remove the reg.
        LatencyMIs.erase(prev);
        if (bWaitCnt && cycle == latency) {
          score.MemLatency += cycle;
          // Only count memLatency once, the rest is hide.
          bWaitCnt = false;
        } else {
          // Hide cycle or count mem latency?
          score.LatencyHide += cycle;
        }
      } else {
        l->second -= latency;
        // Hide latency.
        score.LatencyHide += latency;
      }
    }

  } else {
    // TODO: check branch/lds?
    // TODO: check prevVAlu?
    auto getAluStatus = [](const MachineInstr &MI,
                           const llvm::SIInstrInfo *SIII) {
      AluStatus status = AluStatus::Nothing;
      if (SIII->isVALU(MI.getOpcode())) {
        status = AluStatus::Vector;
      } else if (SIII->isSALU(MI.getOpcode())) {
        status = AluStatus::Scalar;
      }
      return status;
    };
    AluStatus status = getAluStatus(MI, SIII);

    switch (prevStatus) {
    case AluStatus::Nothing: {
      score.Alu += latency;
      score.MixAlu += latency;
      prevStatus = status;
    } break;
    case AluStatus::Vector:
    case AluStatus::Scalar: {
      score.Alu += latency;
      // Ignore mix alu.
      if (prevStatus != status) {
        prevStatus = AluStatus::Nothing;
      } else {
        score.MixAlu += latency;
      }
    } break;
    }
  }
  // Update latency inst.
  if (SIII->isHighLatencyInstruction(MI) && MI.mayLoad()) {
    unsigned reg = MI.getOperand(0).getReg();
    // TODO: get correct latency.
    // SIII->getInstrLatency(ItinerayData, MI);
    constexpr unsigned kHighLetency = 180;
    LatencyMIs[reg] = kHighLetency;
  } else if (SIII->isLowLatencyInstruction(MI) && MI.mayLoad()) {
    unsigned reg = MI.getOperand(0).getReg();
    // TODO: get correct latency.
    // SIII->getInstrLatency(ItinerayData, MI);
    constexpr unsigned kLowLetency = 35;
    LatencyMIs[reg] = kLowLetency;
  }
}

SchedScore CollectLatency(MachineFunction &MF, const llvm::GCNSubtarget &ST,
                          const llvm::MachineLoopInfo *MLI) {
  SchedScore totalScore;
  for (auto &MFI : MF) {
    MachineBasicBlock &MBB = MFI;
    MachineBasicBlock::iterator Next;
    AMDGPULatencyTracker latencyTracker(ST);
    for (auto &MI : MBB) {
      latencyTracker.scan(MI);
    }
    unsigned loopDepth = 0;
    if (MLI) {
      loopDepth = MLI->getLoopDepth(&MBB);
    }
    totalScore.sum(latencyTracker.score, loopDepth);
  }
  return totalScore;
}

} // namespace llvm


