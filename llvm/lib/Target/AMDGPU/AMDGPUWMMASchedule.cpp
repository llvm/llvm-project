//===--- AMDGPUWMMASchedule.cpp - AMDGPU WMMA Schedule Adjustment ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file contains a DAG scheduling mutation that shapes how gfx1250
///       ds_load (LDS) prefetches are placed relative to the WMMA instructions
///       that consume them, to prevent the pre-RA scheduler from bunching all
///       the loads at the head of the block (which forces the WMMAs behind
///       long s_wait_dscnt stalls and inflates register pressure).
///
///       It does the following:
///       - Order the WMMAs (WMMA -> WMMA edges added).
///       - Order the ds_loads (ds_load -> ds_load edges added
///         with latency attached to prevent them overhwelming
///         the LDS bus and becoming memory bound).
///       - Add WMMA -> ds_load edges to stop loads from being bunched at
///         the start of the block
///       - Build a live range histogram of the A/B operand fragments under
///         an as late as possible schedule, recording the minimum VGPRs
///         needed for such a schedule (so the WMMA -> ds_load edges can
///         be placed earlier if the minimum VGPR budget can afford it).
///
//===----------------------------------------------------------------------===//

#include "AMDGPUWMMASchedule.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include <optional>
#define DEBUG_TYPE "amdgpu-wmma-sched"

using namespace llvm;

namespace {

// A single ds_load and their order among the WMMAs.
struct LoadInfo {
  SUnit *SU;
  unsigned MinPos =
      UINT_MAX;        // earliest WMMA consumer (UINT_MAX means none in region)
  unsigned MaxPos = 0; // latest WMMA consumer
  long LatestCycle = 0; // as late as possible cycle
};

// A fragment: the wide vreg several ds_loads build (for example - a vreg_512
// from four DS_READ_B128). This is the unit for VGPR pressure - the DS_READ
// subloads share one register, so counting per ds_load instead of by fragments
// would multiply the pressure.
struct FragInfo {
  unsigned VGPRs = 0;
  unsigned MinPos = UINT_MAX;
  unsigned MaxPos = 0;
  long LatestCycle = LONG_MAX; // earliest subload's as late as possible cycle
  SmallVector<SUnit *, 4> Subloads;
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
  MapVector<SUnit *, unsigned> Wmmas; // Ordered WMMA SUnits
  SmallVector<LoadInfo> Loads;
  std::optional<unsigned> LoadLatency;
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
    DAG->addEdge(SU, SDep(PrevSU, SDep::Artificial));
    PrevSU = SU;
  }

  // For each load, find earliest and latest consuming WMMA positions, and
  // correct the ds_load -> earliest consumer data edge latency (Both the
  // Succs and Preds SDep is updated)
  for (LoadInfo &LI : Loads) {
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

  // Chain consecutive loads with an LDS bandwidth latency
  SUnit *Prev = nullptr;
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

  // Determing each load's as late as possible cycle - this means the
  // latest cycle that still meets the load latency, then pushed earlier
  // if the ds_load -> ds_load edges requires spacing (ds_loads cannot be
  // too close to each other or it could overwhelm the LDS bus and lead to
  // the program being memory bound).
  for (LoadInfo &LI : Loads)
    if (LI.MinPos != UINT_MAX)
      LI.LatestCycle = (long)LI.MinPos * (*WmmaLatency) - (long)(*LoadLatency);
  long PrevLatest = LONG_MAX;
  for (int I = (int)Loads.size() - 1; I >= 0; --I) {
    LoadInfo &LI = Loads[I];
    if (LI.MinPos == UINT_MAX)
      continue;
    long Spaced = PrevLatest - (long)(*LDSBandwidth);
    if (Spaced < LI.LatestCycle)
      LI.LatestCycle = Spaced;
    PrevLatest = LI.LatestCycle;
  }

  // Group subloads into fragments and build the live range histogram
  // with a schedule as late as possible. Each fragment is live from
  // its earliest subload to its last WMMA consumer. The peak of the
  // histogram is the minimum VGPRs needed.
  MapVector<Register, FragInfo> Frags;
  for (LoadInfo &LI : Loads) {
    if (LI.MinPos == UINT_MAX)
      continue;
    Register R = LI.SU->getInstr()->getOperand(0).getReg();
    FragInfo &F = Frags[R];
    if (F.Subloads.empty() && R.isVirtual())
      F.VGPRs = TRI.getRegSizeInBits(*MRI.getRegClass(R)) / 32;
    F.MinPos = std::min(F.MinPos, LI.MinPos);
    F.MaxPos = std::max(F.MaxPos, LI.MaxPos);
    F.LatestCycle = std::min(F.LatestCycle, LI.LatestCycle);
    F.Subloads.push_back(LI.SU);
  }

  std::vector<unsigned> Hist(Wmmas.size(), 0);
  for (auto &KV : Frags) {
    FragInfo &F = KV.second;
    long Pos = F.LatestCycle / (long)(*WmmaLatency);
    unsigned StartPos = Pos < 0 ? 0 : (unsigned)Pos;
    for (unsigned P = StartPos; P <= F.MaxPos && P < Wmmas.size(); ++P)
      Hist[P] += F.VGPRs;
  }

  unsigned Budget = 0;
  for (unsigned P = 0; P < Wmmas.size(); ++P)
    Budget = std::max(Budget, Hist[P]);

  // For each fragment (in order), find the earliest position it
  // can be placed so the live set never exceeds the budget, then
  // add a WMMAS[Earliest] -> ds_load edge - this is what leads to
  // the debunching.
  for (auto &KV : Frags) {
    FragInfo &F = KV.second;
    long Pos = F.LatestCycle / (long)(*WmmaLatency);
    unsigned LateStartPos = Pos < 0 ? 0 : (unsigned)Pos;
    unsigned Earliest = LateStartPos;
    for (int P = (int)LateStartPos - 1; P >= 0; --P) {
      if (Hist[(unsigned)P] + F.VGPRs <= Budget)
        Earliest = (unsigned)P;
      else
        break;
    }
    // Update the histogram so later fragments don't schedule earlier and
    // exceed the budget.
    for (unsigned P = Earliest; P < LateStartPos; ++P)
      Hist[P] += F.VGPRs;
    // No need to add an edge if the load can be scheduled at the beginning.
    if (Earliest == 0)
      continue;
    for (SUnit *L : F.Subloads)
      DAG->addEdge(L, SDep(Wmmas.begin()[Earliest].first, SDep::Artificial));
  }
}

} // end namespace

std::unique_ptr<ScheduleDAGMutation>
llvm::createAMDGPUWMMAScheduleDAGMutation(MachineFunction *MF) {
  return std::make_unique<WMMASchedule>(MF);
}
