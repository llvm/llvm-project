//===--- AMDGPUHazardLatency.cpp - AMDGPU Hazard Latency Adjustment -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file contains a DAG scheduling mutation to adjust the
///       latency of data edges between instructions which use registers
///       potentially subject to additional hazard waits not accounted
///       for in the normal scheduling model.
///       While the scheduling model is typically still accurate in these
///       scenarios, adjusting latency of relevant edges can improve wait
///       merging and reduce pipeline impact of any required waits.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUHazardLatency.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"

using namespace llvm;

namespace {

class HazardLatency : public ScheduleDAGMutation {
private:
  const GCNSubtarget &ST;
  const SIRegisterInfo &TRI;
  const MachineRegisterInfo &MRI;

public:
  HazardLatency(MachineFunction *MF)
      : ST(MF->getSubtarget<GCNSubtarget>()), TRI(*ST.getRegisterInfo()),
        MRI(MF->getRegInfo()) {}
  void apply(ScheduleDAGInstrs *DAG) override;
};

void HazardLatency::apply(ScheduleDAGInstrs *DAG) {
  constexpr unsigned MaskLatencyBoost = 3;

  // Hazard only manifests in Wave64
  if (!ST.hasVALUMaskWriteHazard() || !ST.isWave64())
    return;

  for (SUnit &SU : DAG->SUnits) {
    const MachineInstr *MI = SU.getInstr();
    if (!SIInstrInfo::isVALU(*MI))
      continue;
    if (MI->getOpcode() == AMDGPU::V_READLANE_B32 ||
        MI->getOpcode() == AMDGPU::V_READFIRSTLANE_B32)
      continue;
    for (SDep &SuccDep : SU.Succs) {
      if (SuccDep.isCtrl())
        continue;
      // Boost latency on VALU writes to SGPRs used by VALUs.
      // Reduce risk of premature VALU pipeline stall on associated reads.
      MachineInstr *DestMI = SuccDep.getSUnit()->getInstr();
      if (!SIInstrInfo::isVALU(*DestMI))
        continue;
      Register Reg = SuccDep.getReg();
      if (!TRI.isSGPRReg(MRI, Reg))
        continue;
      SuccDep.setLatency(SuccDep.getLatency() * MaskLatencyBoost);
    }
  }
}

} // end namespace

std::unique_ptr<ScheduleDAGMutation>
llvm::createAMDGPUHazardLatencyDAGMutation(MachineFunction *MF) {
  return std::make_unique<HazardLatency>(MF);
}
