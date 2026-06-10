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
  LLVM_DEBUG(dbgs() << "WMMASchedule running, " << DAG->SUnits.size()
                    << "SUnits\n");

  SmallVector<SUnit *> Loads;
  MapVector<SUnit *, unsigned> Wmmas;

  std::optional<unsigned> LoadLatency = std::nullopt;
  std::optional<unsigned> WmmaLatency = std::nullopt;

  // Gather all WMMAs and DS_LOADs
  for (auto &SU : DAG->SUnits) {
    MachineInstr *MI = SU.getInstr();
    if (!MI)
      continue;

    // Gather WMMAs
    if (TII->isMFMAorWMMA(*MI)) {
      if (WmmaLatency == std::nullopt)
        WmmaLatency = SM->computeInstrLatency(MI);
      Wmmas.insert({&SU, Wmmas.size()});
      continue;
    }

    // Gather DS_LOADs
    if (TII->isDS(*MI) && MI->mayLoad()) {
      if (LoadLatency == std::nullopt)
        LoadLatency = SM->computeInstrLatency(MI);
      Loads.push_back(&SU);
    }
  }

  // Calculate how many WMMAs away from consuming WMMA a load must be
  // before it will certainly ready for consumer
  unsigned Dist;
  if (LoadLatency && WmmaLatency) {
    Dist = std::ceil(static_cast<double>(*LoadLatency) / *WmmaLatency);
    if (Dist < 1)
      Dist = 1;
  } else {
    return; // Either missing Wmmas or Loads. No point continuing.
  }
  LLVM_DEBUG(dbgs() << "Dist " << Dist << "\n");

  // For every load, determine earliest WMMA reliant on it,
  // and add an anchor.
  for (SUnit *L : Loads) {
    SUnit *Earliest = nullptr;
    unsigned EarliestPos = UINT_MAX;
    for (const SDep &D : L->Succs) {
      if (D.getKind() != SDep::Data)
        continue;
      SUnit *S = D.getSUnit();
      auto *It = Wmmas.find(S);
      if (It == Wmmas.end())
        continue;
      if (It->second < EarliestPos) {
        EarliestPos = It->second;
        Earliest = S;
      }
    };
    LLVM_DEBUG(dbgs() << "load SU" << L->NodeNum << " -> earliest WMMA SU"
                      << (Earliest ? (int)Earliest->NodeNum : -1) << " (pos "
                      << EarliestPos << ")\n");
    if (Earliest && EarliestPos >= Dist) {
      LLVM_DEBUG(dbgs() << "Window Created!\n");
      SUnit *Anchor = Wmmas.begin()[EarliestPos - Dist].first;
      bool Ok = DAG->addEdge(L, SDep(Anchor, SDep::Artificial));
      LLVM_DEBUG(dbgs() << "  leash SU" << L->NodeNum << " after WMMA SU"
                        << Anchor->NodeNum
                        << (Ok ? "\n" : " (REJECTED: cycle)\n"));
    };
  };
}

} // end namespace

std::unique_ptr<ScheduleDAGMutation>
llvm::createAMDGPUWMMAScheduleDAGMutation(MachineFunction *MF) {
  return std::make_unique<WMMASchedule>(MF);
}
