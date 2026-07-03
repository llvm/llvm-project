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
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include <optional>
#define DEBUG_TYPE "amdgpu-wmma-sched"

using namespace llvm;

namespace {

// Disables the whole mutation (used to capture a no-mutation baseline schedule
// for the before/after visualization).
static cl::opt<bool> DisableWMMASchedule(
    "amdgpu-wmma-sched-disable", cl::init(false),
    cl::desc("Disable the AMDGPU WMMA ds_load scheduling mutation"));

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
  if (!ST.hasGFX1250Insts() || DisableWMMASchedule)
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

  LLVM_DEBUG(
      dbgs()
      << "\n========================================================\n"
         "AMDGPUWMMASchedule: ds_load scheduling mutation\n"
         "Shapes where LDS (ds_load) prefetches sit relative to the WMMAs\n"
         "that consume them, so the pre-RA scheduler doesn't bunch every\n"
         "load at the top of the block (which stalls the WMMAs behind\n"
         "s_wait_dscnt and inflates register pressure).\n"
         "WMMAs are numbered W[0..N-1] in program order.\n"
         "========================================================\n"
      << "config: " << Wmmas.size() << " WMMAs, " << Loads.size()
      << " ds_loads; loadlat=" << *LoadLatency << " wmmalat=" << *WmmaLatency
      << " ldsbw=" << (unsigned)*LDSBandwidth << "\n");

  // Order the WMMAs.
  LLVM_DEBUG(
      dbgs()
      << "\n--- [1] WMMA ordering ------------------------------------\n"
         "Chain each WMMA to the next in program order with an artificial\n"
         "edge, so load placement can be reasoned about relative to fixed\n" 
         "W[] positions.\n");
  auto [PrevSU, PrevPos] = *Wmmas.begin();
  for (auto *It = std::next(Wmmas.begin()); It != Wmmas.end(); ++It) {
    auto [SU, Pos] = *It;
    DAG->addEdge(SU, SDep(PrevSU, SDep::Artificial));
    LLVM_DEBUG(dbgs() << "[1] WMMA W[" << PrevPos << "] SU" << PrevSU->NodeNum
                      << " -> W[" << Pos << "] SU" << SU->NodeNum << "\n");
    PrevSU = SU;
    PrevPos = Pos;
  }

  // For each load, find earliest and latest consuming WMMA positions, and
  // correct the ds_load -> earliest consumer data edge latency (Both the
  // Succs and Preds SDep is updated)
  LLVM_DEBUG(
      dbgs()
      << "\n--- [2] ds_load -> earliest consuming WMMA latency -------\n"
         "For each ds_load, find which WMMAs consume it (MinPos = earliest,\n"
         "MaxPos = latest). Update the data edge latency of the earliest\n"
         "consumer to be the real LDS load latency, so the scheduler keeps\n"
         "the load issued far enough ahead of the WMMAs that need it.\n");
  for (LoadInfo &LI : Loads) {
    SmallVector<SUnit *, 8> Consumers; // WMMA consumers (program order)
    for (const SDep &D : LI.SU->Succs) {
      if (D.getKind() != SDep::Data)
        continue;
      auto *It = Wmmas.find(D.getSUnit());
      // Check to see if successor is a WMMA
      if (It == Wmmas.end())
        continue;
      LI.MinPos = std::min(LI.MinPos, It->second);
      LI.MaxPos = std::max(LI.MaxPos, It->second);
      Consumers.push_back(D.getSUnit());
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
    LLVM_DEBUG({
      dbgs() << "[2] ds_load SU" << LI.SU->NodeNum << ": MinPos=" << LI.MinPos
             << " MaxPos=" << LI.MaxPos << "; consumers W[" << LI.MinPos << ".."
             << LI.MaxPos << "] (";
      for (unsigned I = 0; I < Consumers.size(); ++I)
        dbgs() << (I ? ", " : "") << "SU" << Consumers[I]->NodeNum;
      dbgs() << "); set latency " << *LoadLatency << " on edge -> W["
             << LI.MinPos << "]\n";
    });
  }

  // Order the Loads
  llvm::stable_sort(Loads, [](const LoadInfo &A, const LoadInfo &B) {
    return A.MinPos < B.MinPos;
  });

  // Chain consecutive loads with an LDS bandwidth latency
  LLVM_DEBUG(
      dbgs()
      << "\n--- [3+4] ds_load -> ds_load spacing ---------------------\n"
         "Sort loads by their earliest consumer, then chain consecutive\n"
         "loads in that order with a small latency so they don't all issue\n" 
         "back to back and saturate the LDS bus (which would make the kernel\n" 
         "memory bound).\n");
  SUnit *Prev = nullptr;
  for (LoadInfo &LI : Loads) {
    if (LI.MinPos == UINT_MAX)
      continue;
    if (Prev) {
      SDep D(Prev, SDep::Artificial);
      D.setLatency(*LDSBandwidth);
      DAG->addEdge(LI.SU, D);
      LLVM_DEBUG(dbgs() << "[3+4] ds_load SU" << Prev->NodeNum << " -> SU"
                        << LI.SU->NodeNum << " (spacing latency "
                        << (unsigned)*LDSBandwidth << ")\n");
    }
    Prev = LI.SU;
  }

  // Determing each load's as late as possible cycle - this means the
  // latest cycle that still meets the load latency, then pushed earlier
  // if the ds_load -> ds_load edges requires spacing (ds_loads cannot be
  // too close to each other or it could overwhelm the LDS bus and lead to
  // the program being memory bound).
  LLVM_DEBUG(
      dbgs()
      << "\n--- [lat]/[space] as late as possible cycle --------------\n"
         "[lat]: latest cycle each load could issue and still feed its\n"
         "earliest consumer in time (MinPos*wmmalat - loadlat).\n"
         "[space]: loop through the ordered loads from last to first and\n"
         "pull any that are too close to the next one earlier in order to\n" 
         "honor the ds_load -> ds_load spacing.\n");
  for (LoadInfo &LI : Loads)
    if (LI.MinPos != UINT_MAX) {
      LI.LatestCycle = (long)LI.MinPos * (*WmmaLatency) - (long)(*LoadLatency);
      LLVM_DEBUG(dbgs() << "[lat] ds_load SU" << LI.SU->NodeNum
                        << ": LatestCycle=" << LI.LatestCycle << " (W["
                        << LI.MinPos << "]*" << *WmmaLatency << " - "
                        << *LoadLatency << ")\n");
    }
  long PrevLatest = LONG_MAX;
  for (int I = (int)Loads.size() - 1; I >= 0; --I) {
    LoadInfo &LI = Loads[I];
    if (LI.MinPos == UINT_MAX)
      continue;
    long Spaced = PrevLatest - (long)(*LDSBandwidth);
    if (Spaced < LI.LatestCycle) {
      LLVM_DEBUG(dbgs() << "[space] ds_load SU" << LI.SU->NodeNum
                        << ": LatestCycle " << LI.LatestCycle << " -> " << Spaced
                        << " (spaced " << (unsigned)*LDSBandwidth
                        << " before next load's " << PrevLatest << ")\n");
      LI.LatestCycle = Spaced;
    }
    PrevLatest = LI.LatestCycle;
  }

  // Group subloads into fragments and build the live range histogram
  // with a schedule as late as possible. Each fragment is live from
  // its earliest subload to its last WMMA consumer. The peak of the
  // histogram is the minimum VGPRs needed.
  LLVM_DEBUG(
      dbgs()
      << "\n--- [hist] live VGPR histogram (as late as possible schedule) ---\n"
         "Group subloads that build one wide vreg into a 'fragment' (a\n"
         "unit of VGPR pressure), then accumulate each fragment's VGPR\n"
         "usage across the WMMA positions it is live over, under the\n"
         "as late as possible schedule above.\n");
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
    LLVM_DEBUG({
      dbgs() << "[hist] frag (";
      for (unsigned I = 0; I < F.Subloads.size(); ++I)
        dbgs() << (I ? ", " : "") << "SU" << F.Subloads[I]->NodeNum;
      dbgs() << ") (vgprs=" << F.VGPRs << ", LatestCycle=" << F.LatestCycle
             << ") live over W[" << StartPos << ".." << F.MaxPos << "]\n";
    });
  }

  unsigned Budget = 0;
  for (unsigned P = 0; P < Wmmas.size(); ++P)
    Budget = std::max(Budget, Hist[P]);

  LLVM_DEBUG({
    dbgs() << "\n--- [5] histogram BEFORE debunch (sets the budget) -------\n"
              "Per WMMA position VGPR totals from the as late as possible\n" 
              "schedule. The peak becomes the VGPR budget: the debunch may\n"
              "pull loads earlier as long as no position exceeds it.\n";
    dbgs() << "[5] live-VGPR histogram BEFORE slack (min VGPRs / budget = "
           << Budget << "):\n";
    for (unsigned P = 0; P < Wmmas.size(); ++P)
      if (Hist[P])
        dbgs() << "    W[" << P << "] = " << Hist[P] << "\n";
  });

  // For each fragment (in order), find the earliest position it
  // can be placed so the live set never exceeds the budget, then
  // add a WMMAS[Earliest] -> ds_load edge - this is what leads to
  // the debunching.
  LLVM_DEBUG(
      dbgs()
      << "\n--- [6] debunch: pull loads earlier into budget slack ----\n"
         "For each fragment, scan earlier W[] positions while the budget\n"
         "still has room. The earliest such position gets an artificial\n"
         "WMMA -> ds_load edge that stops the scheduler from bunching that load\n"
         "any earlier. 'unconstrained' = it already fits at W[0], so no edge\n"
         "is needed; the histogram is updated cumulatively so later\n"
         "fragments only use the slack that's left over.\n");
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
    LLVM_DEBUG({
      dbgs() << "[6] frag (";
      for (unsigned I = 0; I < F.Subloads.size(); ++I)
        dbgs() << (I ? ", " : "") << "SU" << F.Subloads[I]->NodeNum;
      dbgs() << ") (vgprs=" << F.VGPRs << ", consumers W[" << F.MinPos << ".."
             << F.MaxPos << "]) earliest=W[" << Earliest << "]"
             << (Earliest ? " edge added\n" : " unconstrained\n");
    });
    // No need to add an edge if the load can be scheduled at the beginning.
    if (Earliest == 0)
      continue;
    for (SUnit *L : F.Subloads)
      DAG->addEdge(L, SDep(Wmmas.begin()[Earliest].first, SDep::Artificial));
  }

  LLVM_DEBUG({
    unsigned Peak = 0;
    for (unsigned P = 0; P < Wmmas.size(); ++P)
      Peak = std::max(Peak, Hist[P]);
    dbgs() << "\n--- [6] histogram AFTER debunch -------------------------\n"
              "Same usage after the debunch edges. Loads now sit as\n"
              "early as the budget allows; the peak should still be within\n"
              "the budget from [5].\n";
    dbgs() << "[6] live-VGPR histogram AFTER slack (peak = " << Peak << "):\n";
    for (unsigned P = 0; P < Wmmas.size(); ++P)
      if (Hist[P])
        dbgs() << "    W[" << P << "] = " << Hist[P] << "\n";
  });
}

} // end namespace

std::unique_ptr<ScheduleDAGMutation>
llvm::createAMDGPUWMMAScheduleDAGMutation(MachineFunction *MF) {
  return std::make_unique<WMMASchedule>(MF);
}
