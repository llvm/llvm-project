//===--- AMDGPUWMMASchedule.cpp - AMDGPU WMMA Schedule Adjustment ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file contains a DAG scheduling mutation to add additional
///       edges between ds_load instructions and wmma instructions that
///       occur a certain amount away from the actual wmma consumer of
///       said ds_load. This forces the ds_load to properly prefetch
///       and prevent early bunching of ds_loads that then lead to long
///       stalls.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUWMMASchedule.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include "llvm/Support/Debug.h"
#include <optional>
#define DEBUG_TYPE "amdgpu-wmma-sched"

using namespace llvm;

namespace {

// A ds_load plus the program order positions (among WMMAs) of its
// earliest and latest WMMA consumer.
struct LoadInfo {
  SUnit *SU;
  unsigned MinPos = UINT_MAX;  // earliest consumer (UINT_MAX means none in this region)
  unsigned MaxPos = 0;         // latest consumer
};

class WMMASchedule : public ScheduleDAGMutation {
private:
  const GCNSubtarget &ST;
  const SIRegisterInfo &TRI;
  const MachineRegisterInfo &MRI;

public:
  WMMASchedule(MachineFunction *MF)
      : ST(MF->getSubtarget<GCNSubtarget>()), TRI(*ST.getRegisterInfo()),
        MRI(MF->getRegInfo()) {}
  void apply(ScheduleDAGInstrs *DAG) override;
};

void WMMASchedule::apply(ScheduleDAGInstrs *DAG) {
  if (!ST.hasGFX1250Insts())
    return;
  const TargetSchedModel *SM = DAG->getSchedModel();
  const SIInstrInfo *TII = ST.getInstrInfo();

  // Gather WMMAs (numbered in program order) and ds_loads.
  MapVector<SUnit *, unsigned> Wmmas; // WMMA SUnit and its program position relative to one another
  SmallVector<LoadInfo> Loads;
  std::optional<unsigned> LoadLatency = 64; 
  std::optional<unsigned> WmmaLatency;
  std::optional<double> LDSBandwidth;

  for (SUnit &SU : DAG->SUnits) {
    MachineInstr *MI = SU.getInstr();
    if (!MI)
      continue;

    // Gather WMMAs
    if (TII->isMFMAorWMMA(*MI)) {
      if (!WmmaLatency)
        WmmaLatency = SM->computeInstrLatency(MI);
      Wmmas.insert({&SU, Wmmas.size()});
      continue;
    }

    // Gather DS_LOADs
    if (TII->isDS(*MI) && MI->mayLoad()) {
      if (!LoadLatency) 
        LoadLatency = SM->computeInstrLatency(MI);
      if (!LDSBandwidth)
        LDSBandwidth = std::ceil(SM->computeReciprocalThroughput(MI));
      Loads.push_back({&SU});
    }
  }

  // The following means the DAG Mutation cannot do anything useful.
  if (!LoadLatency || !LDSBandwidth || !WmmaLatency || Wmmas.empty())
    return;

  // Order the WMMAs.
  auto [PrevSU, _] = *Wmmas.begin();
  for (auto *It = std::next(Wmmas.begin()); It != Wmmas.end(); ++It) {
    auto [SU, _] = *It;
    bool Success = DAG->addEdge(SU, SDep(PrevSU, SDep::Artificial));
    LLVM_DEBUG(dbgs() << "wmma SU" << SU->NodeNum << " <- after wmma SU"
                      << PrevSU->NodeNum << (Success ? "\n" : " FAIL (cycle)\n"));
    PrevSU = SU;
  }

  // For each load, find the earliest and latest consuming WMMA positions.
  // Additionally correct the latency of the ds_load -> earliest WMMA consumer 
  // data edge.
  for (LoadInfo& LI : Loads) {
    for (const SDep &D : LI.SU->Succs) {
      if (D.getKind() != SDep::Data)
        continue;
      auto *It = Wmmas.find(D.getSUnit());
      // Check to see if successor is a WMMA
      if (It == Wmmas.end())
        continue;
      LI.MinPos = std::min(LI.MinPos, It->second);
      LI.MaxPos = std::max(LI.MaxPos, It->second);
    }
    if (LI.MinPos == UINT_MAX) 
      continue;
    SUnit *EarliestConsumer = Wmmas.begin()[LI.MinPos].first;
    // Correct latency of edges between ds_load and earliest WMMA consumer
    for (SDep &S : LI.SU->Succs)
      if (S.getSUnit() == EarliestConsumer && S.getKind() == SDep::Data)
        S.setLatency(*LoadLatency);
    for (SDep &P : EarliestConsumer->Preds)
      if (P.getSUnit() == LI.SU && P.getKind() == SDep::Data)
        P.setLatency(*LoadLatency);
    EarliestConsumer->setDepthDirty();
    LI.SU->setHeightDirty();
  }

  // Order the Loads
  llvm::stable_sort(Loads, [](const LoadInfo &A, const LoadInfo &B) {
    return A.MinPos < B.MinPos;
  });
  SUnit* Prev = nullptr; 
  for (LoadInfo &LI : Loads) {
    if (LI.MinPos == UINT_MAX) 
      continue;
    if (Prev) {
      SDep D(Prev, SDep::Artificial);
      D.setLatency(*LDSBandwidth);
      DAG->addEdge(LI.SU, D);
    }
    Prev = LI.SU;
  }

  // MaxPos in order
  std::vector<bool> Present(Wmmas.size(), false);
  for (const LoadInfo &LI : Loads)
    // Check if there's a consumer for this load in the region
    if (LI.MinPos != UINT_MAX)
      Present[LI.MaxPos] = true;
  
  // Latest position that's <= position Pos at which some load's register frees
  // A load's register becomes free at its MaxPos, which is its last WMMA consumer.
  std::vector<std::optional<unsigned>> DeadBy(Wmmas.size(), std::nullopt);
  std::optional<unsigned> Latest;
  for (unsigned Pos = 0; Pos < Wmmas.size(); ++Pos) {
    if (Present[Pos])
      Latest = Pos;
    DeadBy[Pos] = Latest;
  }

  // Create the edges that constrain where the ds_loads can be placed
  // minimum distance of a load is LI.MinPos - Dist
  // maximum distance of a load is DeadBy[LI.MinPos - Dist]
  unsigned Dist = std::max(1u, static_cast<unsigned>(std::ceil(static_cast<double>(*LoadLatency) / *WmmaLatency)));
  for (const LoadInfo &LI : Loads) {
    if (LI.MinPos == UINT_MAX || LI.MinPos < Dist)
      continue;
    
    // Minimum distance edge
    // Load scheduled before this point (heuristically)
    long MinDist = LI.MinPos - Dist;

    // Maximum distance edge
    // Load scheduled after this point
    bool MaxSuccess = true;
    std::optional<unsigned> MaxDist = DeadBy[MinDist];
    if (MaxDist && *MaxDist < MinDist)
      MaxSuccess = DAG->addEdge(LI.SU, SDep(Wmmas.begin()[*MaxDist].first, SDep::Artificial));

    LLVM_DEBUG(dbgs() << "load SU" << LI.SU->NodeNum << " Win=[" << (MaxDist ? *MaxDist : -1) << ","
                      << MinDist << "] MinPos=" << LI.MinPos << " MaxPos=" << LI.MaxPos
                      << (MaxSuccess ? "\n" : " (edge REJECTED: cycle)\n"));
  }

}

} // end namespace

std::unique_ptr<ScheduleDAGMutation>
llvm::createAMDGPUWMMAScheduleDAGMutation(MachineFunction *MF) {
  return std::make_unique<WMMASchedule>(MF);
}
