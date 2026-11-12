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
///       It estimates the latest useful issue point and live range of each
///       fragment, then adds WMMA -> ds_load edges to stop loads from being
///       bunched at the start of the block without increasing the estimated
///       VGPR register pressure. It also corrects the latency of the existing
///       data edge from each ds_load to its earliest WMMA consumer so the
///       default and coexec schedulers use the same latency.
///
//===----------------------------------------------------------------------===//

#include "AMDGPUWMMASchedule.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include "llvm/Support/Debug.h"
#include <optional>
#define DEBUG_TYPE "amdgpu-wmma-sched"

using namespace llvm;

namespace {

static constexpr StringLiteral WMMAScheduleAttr = "amdgpu-wmma-schedule";

// A single ds_load and its consumers among the WMMAs.
struct LoadInfo {
  SUnit *SU;
  unsigned MinPos =
      UINT_MAX;        // earliest WMMA consumer (UINT_MAX means none in region)
  unsigned MaxPos = 0; // latest WMMA consumer
  long LatestCycle = 0; // as late as possible cycle

  explicit LoadInfo(SUnit *SU) : SU(SU) {}
};

// A fragment: the wide vreg several ds_loads build (for example - a vreg_512
// from four DS_READ_B128). This is the unit for VGPR pressure - the DS_READ
// subloads share one register, so counting per ds_load instead of by fragments
// would multiply the pressure.
struct FragInfo {
  unsigned VGPRs = 0;
  unsigned MaxPos = 0;
  long LatestCycle = LONG_MAX; // earliest subload's as late as possible cycle
  SmallVector<SUnit *, 4> Subloads;
};

class WMMASchedule : public ScheduleDAGMutation {
private:
  const GCNSubtarget &ST;
  const SIRegisterInfo &TRI;
  const MachineRegisterInfo &MRI;
  bool Enabled;

public:
  WMMASchedule(MachineFunction *MF)
      : ST(MF->getSubtarget<GCNSubtarget>()), TRI(*ST.getRegisterInfo()),
        MRI(MF->getRegInfo()),
        Enabled(MF->getFunction().hasFnAttribute(WMMAScheduleAttr)) {}
  void apply(ScheduleDAGInstrs *DAG) override;
};

void WMMASchedule::apply(ScheduleDAGInstrs *DAG) {
  if (!Enabled || !ST.hasGFX1250Insts())
    return;
  const TargetSchedModel *SM = DAG->getSchedModel();
  if (!SM->hasInstrSchedModel())
    return;
  const SIInstrInfo *TII = ST.getInstrInfo();

  // Gather WMMAs (numbered in program order) and ds_loads.
  SmallVector<SUnit *> Wmmas; // Ordered WMMA SUnits
  SmallVector<LoadInfo> Loads;
  std::optional<unsigned> WmmaLatency;

  for (SUnit &SU : DAG->SUnits) {
    MachineInstr *MI = SU.getInstr();
    if (!MI)
      continue;

    // Gather WMMAs
    if (TII->isMFMAorWMMA(*MI)) {
      if (!WmmaLatency)
        WmmaLatency = SM->computeInstrLatency(MI);
      Wmmas.push_back(&SU);
      continue;
    }

    // Gather DS_LOADs
    if (TII->isDS(*MI) && MI->mayLoad())
      Loads.emplace_back(&SU);
  }

  // The following means the DAG Mutation cannot do anything useful.
  if (Loads.empty() || Wmmas.empty())
    return;

  LLVM_DEBUG(dbgs() << "AMDGPUWMMASchedule: " << Wmmas.size() << " WMMAs, "
                    << Loads.size() << " ds_loads, WMMA latency "
                    << *WmmaLatency << "\n");

  // For each load, find the earliest and latest consuming WMMA positions,
  // correct the latency to the earliest consumer, and estimate the latest
  // cycle at which the load can be issued.
  for (LoadInfo &LI : Loads) {
    for (const SDep &D : LI.SU->Succs) {
      if (D.getKind() != SDep::Data)
        continue;
      auto It = llvm::find(Wmmas, D.getSUnit());
      if (It == Wmmas.end())
        continue;
      unsigned Pos = static_cast<unsigned>(It - Wmmas.begin());
      LI.MinPos = std::min(LI.MinPos, Pos);
      LI.MaxPos = std::max(LI.MaxPos, Pos);
    }
    if (LI.MinPos == UINT_MAX)
      continue;
    SUnit *EarliestConsumer = Wmmas[LI.MinPos];
    const unsigned LoadLatency = SM->computeInstrLatency(LI.SU->getInstr());
    for (SDep &S : LI.SU->Succs)
      if (S.getSUnit() == EarliestConsumer && S.getKind() == SDep::Data)
        S.setLatency(LoadLatency);
    for (SDep &P : EarliestConsumer->Preds)
      if (P.getSUnit() == LI.SU && P.getKind() == SDep::Data)
        P.setLatency(LoadLatency);
    EarliestConsumer->setDepthDirty();
    LI.SU->setHeightDirty();
    LI.LatestCycle = static_cast<long>(LI.MinPos) * (*WmmaLatency) -
                     static_cast<long>(LoadLatency);
    LLVM_DEBUG(dbgs() << "ds_load SU" << LI.SU->NodeNum << ": consumers W["
                      << LI.MinPos << ".." << LI.MaxPos
                      << "], latency=" << LoadLatency
                      << ", LatestCycle=" << LI.LatestCycle << "\n");
  }

  // Loads without a MinPos have no WMMA consumer in this scheduling region.
  llvm::erase_if(Loads,
                 [](const LoadInfo &LI) { return LI.MinPos == UINT_MAX; });

  // Group subloads into fragments and build the live range histogram
  // with a schedule as late as possible. Each fragment is live from
  // its earliest subload to its last WMMA consumer. The peak of the
  // histogram is the minimum VGPRs needed.
  MapVector<Register, FragInfo> Frags;
  for (LoadInfo &LI : Loads) {
    Register R = LI.SU->getInstr()->getOperand(0).getReg();
    FragInfo &F = Frags[R];
    if (F.Subloads.empty() && R.isVirtual())
      F.VGPRs = TRI.getRegClassWeight(MRI.getRegClass(R)).RegWeight;
    F.MaxPos = std::max(F.MaxPos, LI.MaxPos);
    F.LatestCycle = std::min(F.LatestCycle, LI.LatestCycle);
    F.Subloads.push_back(LI.SU);
  }

  std::vector<unsigned> Hist(Wmmas.size(), 0);
  for (auto &[_, F] : Frags) {
    long Pos = F.LatestCycle / static_cast<long>(*WmmaLatency);
    unsigned StartPos = Pos < 0 ? 0 : static_cast<unsigned>(Pos);
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

  unsigned Budget = *llvm::max_element(Hist);

  LLVM_DEBUG(dbgs() << "[hist] live-VGPR budget = " << Budget << "\n");

  // For each fragment (in order), find the earliest WMMA at which it can be
  // live without exceeding the budget, then add a
  // Wmmas[EarliestLivePos - 1] -> ds_load edge to prevent its subloads from
  // being scheduled before that boundary.
  for (auto &[_, F] : Frags) {
    long Pos = F.LatestCycle / static_cast<long>(*WmmaLatency);
    unsigned LateStartPos = Pos < 0 ? 0 : static_cast<unsigned>(Pos);
    unsigned EarliestLivePos = LateStartPos;
    for (int P = static_cast<int>(LateStartPos) - 1; P >= 0; --P) {
      const unsigned Candidate = static_cast<unsigned>(P);
      if (Hist[Candidate] + F.VGPRs > Budget)
        break;
      EarliestLivePos = Candidate;
    }
    LLVM_DEBUG({
      dbgs() << "[anchor] frag (";
      for (unsigned I = 0; I < F.Subloads.size(); ++I)
        dbgs() << (I ? ", " : "") << "SU" << F.Subloads[I]->NodeNum;
      dbgs() << ") (vgprs=" << F.VGPRs << ", last consumer W[" << F.MaxPos
             << "]) EarliestLivePos=W[" << EarliestLivePos << "]";
      if (EarliestLivePos)
        dbgs() << " anchor=W[" << EarliestLivePos - 1 << "]\n";
      else
        dbgs() << " unconstrained\n";
    });
    SUnit *Anchor = EarliestLivePos == 0 ? nullptr : Wmmas[EarliestLivePos - 1];
    if (Anchor) {
      bool AllEdgesLegal = llvm::all_of(
          F.Subloads, [&](SUnit *L) { return DAG->canAddEdge(L, Anchor); });
      if (!AllEdgesLegal) {
        LLVM_DEBUG(
            dbgs() << "[anchor] skipped: an edge would create a cycle\n");
        continue;
      }
    }

    // Commit the fragment's use of the available histogram slack only after
    // every proposed edge has been validated.
    for (unsigned P = EarliestLivePos; P < LateStartPos; ++P)
      Hist[P] += F.VGPRs;

    // No need to add an edge if the load can be scheduled at the beginning.
    if (!Anchor)
      continue;
    // Hist[EarliestLivePos] models the fragment as live at
    // Wmmas[EarliestLivePos], so the edge must come from the
    // preceding WMMA.
    for (SUnit *L : F.Subloads) {
      bool Added = DAG->addEdge(L, SDep(Anchor, SDep::Artificial));
      assert(Added && "prevalidated WMMA scheduling edge became illegal");
    }
  }

  LLVM_DEBUG(dbgs() << "[hist] live-VGPR peak after debunch = "
                    << *llvm::max_element(Hist) << "\n");
}

} // end namespace

std::unique_ptr<ScheduleDAGMutation>
llvm::createAMDGPUWMMAScheduleDAGMutation(MachineFunction *MF) {
  return std::make_unique<WMMASchedule>(MF);
}
