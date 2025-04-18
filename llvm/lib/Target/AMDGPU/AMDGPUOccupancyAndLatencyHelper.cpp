//==- AMDGPUOccupancyAndLatencyHelper.cpp - Helpers for occupancy + latency ==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==------------------------------------------------------------------------==//
//
/// \file
/// \brief Helper functions for occupancy and latency.
//
//==------------------------------------------------------------------------==//

#include "AMDGPUOccupancyAndLatencyHelper.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineLoopInfo.h"

#include <cmath>

namespace llvm {

void SchedScore::sum(const SchedScore &S, unsigned LoopDepth) {
  unsigned LoopCount = LoopDepth > 0 ? std::pow(3, LoopDepth) : 1;
  LatencyHide += LoopCount * S.LatencyHide;
  MemLatency += LoopCount * S.MemLatency;
  MixAlu += LoopCount * S.MixAlu;
  Alu += LoopCount * S.Alu;
  Lds += LoopCount * S.Lds;
  SgprSpill |= S.SgprSpill;
}
// Does more occupancy give more perf.
bool SchedScore::isMemBound(unsigned TargetOccupancy, unsigned ExtraOcc) const {
  unsigned Gain = latencyGain(TargetOccupancy, ExtraOcc);
  // 10% is good enough.
  if ((10 * Gain) >= Alu)
    return true;
  return false;
}

unsigned SchedScore::latencyGain(unsigned TgtOcc, unsigned ExtraOcc) const {
  unsigned Latency = MemLatency;
  return (Latency / (TgtOcc)) - (Latency / (TgtOcc + ExtraOcc));
}

// AMDGPULatencyTracker
AMDGPULatencyTracker::AMDGPULatencyTracker(const GCNSubtarget &ST)
    : SIII(ST.getInstrInfo()), ItinerayData(ST.getInstrItineraryData()) {}

void AMDGPULatencyTracker::scan(const MachineInstr &MI) {
  if (MI.isDebugInstr())
    return;
  int Latency = SIII->getInstrLatency(ItinerayData, MI);
  // If inside latency hide.
  if (!LatencyMIs.empty()) {
    bool IsWaitCnt = false;
    for (auto &MO : MI.operands()) {
      if (MO.isReg()) {
        Register Reg = MO.getReg();
        auto It = LatencyMIs.find(Reg);
        if (It != LatencyMIs.end()) {
          IsWaitCnt = true;
          // If MI use mem result, update latency to mem latency.
          int Cycle = It->second;
          if (Cycle > Latency)
            Latency = Cycle;
        }
      }
    }
    // Update latency for each mem latency inst.
    for (auto It = LatencyMIs.begin(); It != LatencyMIs.end();) {
      auto Prev = It;
      auto L = (It++);
      int Cycle = L->second;
      if (Cycle <= Latency) {
        // Only left cycles.
        // Remove the reg.
        LatencyMIs.erase(Prev);
        if (IsWaitCnt && Cycle == Latency) {
          Score.MemLatency += Cycle;
          // Only count memLatency once, the rest is hide.
          IsWaitCnt = false;
        } else {
          // Hide cycle or count mem latency?
          Score.LatencyHide += Cycle;
        }
      } else {
        L->second -= Latency;
        // Hide latency.
        Score.LatencyHide += Latency;
      }
    }

  } else {
    // TODO: check branch/lds?
    // TODO: check prevVAlu?
    auto GetAluStatus = [](const MachineInstr &MI,
                           const llvm::SIInstrInfo *SIII) {
      AluStatus Status = AluStatus::Nothing;
      if (SIII->isVALU(MI.getOpcode())) {
        Status = AluStatus::Vector;
      } else if (SIII->isSALU(MI.getOpcode())) {
        Status = AluStatus::Scalar;
      }
      return Status;
    };
    AluStatus Status = GetAluStatus(MI, SIII);

    switch (PrevStatus) {
    case AluStatus::Nothing: {
      Score.Alu += Latency;
      Score.MixAlu += Latency;
      PrevStatus = Status;
    } break;
    case AluStatus::Vector:
    case AluStatus::Scalar: {
      Score.Alu += Latency;
      // Ignore mix alu.
      if (PrevStatus != Status) {
        PrevStatus = AluStatus::Nothing;
      } else {
        Score.MixAlu += Latency;
      }
    } break;
    }
  }
  // Update latency inst.
  if (SIII->isHighLatencyDef(MI.getOpcode()) && MI.mayLoad()) {
    Register Reg = MI.getOperand(0).getReg();
    // TODO: get correct latency.
    // SIII->getInstrLatency(ItinerayData, MI);
    constexpr unsigned kHighLetency = 180;
    LatencyMIs[Reg] = kHighLetency;
  } else if (SIII->isLowLatencyInstruction(MI) && MI.mayLoad()) {
    Register Reg = MI.getOperand(0).getReg();
    // TODO: get correct latency.
    // SIII->getInstrLatency(ItinerayData, MI);
    constexpr unsigned kLowLetency = 35;
    LatencyMIs[Reg] = kLowLetency;
  }
}


SchedScore collectLatency(MachineFunction &MF, const llvm::GCNSubtarget &ST,
                          const llvm::MachineLoopInfo *MLI) {
  SchedScore TotalScore;
  for (auto &MFI : MF) {
    MachineBasicBlock &MBB = MFI;
    MachineBasicBlock::iterator Next;
    AMDGPULatencyTracker LatencyTracker(ST);
    for (auto &MI : MBB) {
      LatencyTracker.scan(MI);
    }
    unsigned LoopDepth = 0;
    if (MLI) {
      LoopDepth = MLI->getLoopDepth(&MBB);
    }
    TotalScore.sum(LatencyTracker.Score, LoopDepth);
  }
  return TotalScore;
}

} // namespace llvm


