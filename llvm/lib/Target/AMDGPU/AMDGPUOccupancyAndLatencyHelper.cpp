//===- AMDGPUOccupancyAndLatencyHelper.cpp - Helpers for occupancy --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Helper functions for occupancy and latency.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUOccupancyAndLatencyHelper.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"

#include "llvm/ADT/SmallVector.h"

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineLoopInfo.h"

#include <cmath>

using namespace llvm;

namespace llvm {

void SchedScore::sum(const SchedScore &S, unsigned LoopDepth) {
  // Weight a block's contribution by its loop nesting: instructions inside
  // loops execute far more often, so their latency matters more.
  unsigned LoopCount = LoopDepth > 0 ? std::pow(3, LoopDepth) : 1;
  LatencyHide += LoopCount * S.LatencyHide;
  MemLatency += LoopCount * S.MemLatency;
  MixAlu += LoopCount * S.MixAlu;
  Alu += LoopCount * S.Alu;
  Lds += LoopCount * S.Lds;
  SgprSpill |= S.SgprSpill;
}

bool SchedScore::isMemBound(unsigned TargetOccupancy, unsigned ExtraOcc) const {
  unsigned Gain = latencyGain(TargetOccupancy, ExtraOcc);
  // If the extra hidden latency is at least 10% of ALU work, occupancy wins.
  return (10 * Gain) >= Alu;
}

unsigned SchedScore::latencyGain(unsigned TgtOcc, unsigned ExtraOcc) const {
  unsigned Latency = MemLatency;
  return (Latency / TgtOcc) - (Latency / (TgtOcc + ExtraOcc));
}

AMDGPULatencyTracker::AMDGPULatencyTracker(const GCNSubtarget &ST)
    : SIII(ST.getInstrInfo()), ItineraryData(ST.getInstrItineraryData()) {}

void AMDGPULatencyTracker::scan(const MachineInstr &MI) {
  if (MI.isDebugInstr())
    return;
  int Latency = SIII->getInstrLatency(ItineraryData, MI);

  if (!LatencyMIs.empty()) {
    bool IsWaitCnt = false;
    for (const MachineOperand &MO : MI.operands()) {
      if (!MO.isReg())
        continue;
      Register Reg = MO.getReg();
      auto It = LatencyMIs.find(Reg);
      if (It != LatencyMIs.end()) {
        IsWaitCnt = true;
        // This instruction consumes a pending memory result; the effective
        // latency is at least the remaining cycles of that load.
        int Cycle = It->second;
        if (Cycle > Latency)
          Latency = Cycle;
      }
    }
    // Collect the registers whose pending latency fully retires this cycle;
    // erase them after iterating so we do not mutate the map while walking it.
    SmallVector<unsigned, 8> RetiredRegs;
    for (auto &Entry : LatencyMIs) {
      int Cycle = Entry.second;
      if (Cycle <= Latency) {
        RetiredRegs.push_back(Entry.first);
        if (IsWaitCnt && Cycle == Latency) {
          Score.MemLatency += Cycle;
          // Count the exposed memory latency only once; the rest is hidden.
          IsWaitCnt = false;
        } else {
          Score.LatencyHide += Cycle;
        }
      } else {
        Entry.second -= Latency;
        Score.LatencyHide += Latency;
      }
    }
    for (unsigned Reg : RetiredRegs)
      LatencyMIs.erase(Reg);
  } else {
    auto GetAluStatus = [](const MachineInstr &MI, const SIInstrInfo *SIII) {
      if (SIII->isVALU(MI.getOpcode(), false))
        return AluStatus::Vector;
      if (SIII->isSALU(MI.getOpcode()))
        return AluStatus::Scalar;
      return AluStatus::Nothing;
    };
    AluStatus Status = GetAluStatus(MI, SIII);

    switch (PrevStatus) {
    case AluStatus::Nothing:
      Score.Alu += Latency;
      Score.MixAlu += Latency;
      PrevStatus = Status;
      break;
    case AluStatus::Vector:
    case AluStatus::Scalar:
      Score.Alu += Latency;
      if (PrevStatus != Status)
        PrevStatus = AluStatus::Nothing;
      else
        Score.MixAlu += Latency;
      break;
    }
  }

  if (SIII->isHighLatencyDef(MI.getOpcode()) && MI.mayLoad()) {
    Register Reg = MI.getOperand(0).getReg();
    constexpr unsigned kHighLatency = 180;
    LatencyMIs[Reg] = kHighLatency;
  } else if (SIII->isLowLatencyInstruction(MI) && MI.mayLoad()) {
    Register Reg = MI.getOperand(0).getReg();
    constexpr unsigned kLowLatency = 35;
    LatencyMIs[Reg] = kLowLatency;
  }
}

SchedScore collectLatency(MachineFunction &MF, const GCNSubtarget &ST,
                          const MachineLoopInfo *MLI) {
  SchedScore TotalScore;
  for (MachineBasicBlock &MBB : MF) {
    AMDGPULatencyTracker LatencyTracker(ST);
    for (MachineInstr &MI : MBB)
      LatencyTracker.scan(MI);
    unsigned LoopDepth = MLI ? MLI->getLoopDepth(&MBB) : 0;
    TotalScore.sum(LatencyTracker.Score, LoopDepth);
  }
  return TotalScore;
}

} // namespace llvm
