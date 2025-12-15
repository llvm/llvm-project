//===--- AMDGPUBarrierLatency.cpp - AMDGPU Barrier Latency ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file contains a DAG scheduling mutation to add latency to
///       barrier edges between ATOMIC_FENCE instructions and preceding
///       memory accesses potentially affected by the fence.
///       This encourages the scheduling of more instructions before
///       ATOMIC_FENCE instructions.  ATOMIC_FENCE instructions may
///       introduce wait counting or indicate an impending S_BARRIER
///       wait.  Having more instructions in-flight across these
///       constructs improves latency hiding.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUBarrierLatency.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"

using namespace llvm;

namespace {

class BarrierLatency : public ScheduleDAGMutation {
private:
  SmallSet<SyncScope::ID, 4> IgnoredScopes;

public:
  BarrierLatency(MachineFunction *MF) {
    LLVMContext &Context = MF->getFunction().getContext();
    IgnoredScopes.insert(SyncScope::SingleThread);
    IgnoredScopes.insert(Context.getOrInsertSyncScopeID("wavefront"));
    IgnoredScopes.insert(Context.getOrInsertSyncScopeID("wavefront-one-as"));
    IgnoredScopes.insert(Context.getOrInsertSyncScopeID("singlethread-one-as"));
  }
  void apply(ScheduleDAGInstrs *DAG) override;
};

void BarrierLatency::apply(ScheduleDAGInstrs *DAG) {
  constexpr unsigned SyntheticLatency = 2000;
  for (SUnit &SU : DAG->SUnits) {
    const MachineInstr *MI = SU.getInstr();
    if (MI->getOpcode() != AMDGPU::ATOMIC_FENCE)
      continue;

    // Update latency on barrier edges of ATOMIC_FENCE.
    // Ignore scopes not expected to have any latency.
    SyncScope::ID SSID = static_cast<SyncScope::ID>(MI->getOperand(1).getImm());
    if (IgnoredScopes.contains(SSID))
      continue;

    for (SDep &PredDep : SU.Preds) {
      if (!PredDep.isBarrier())
        continue;
      SUnit *PredSU = PredDep.getSUnit();
      MachineInstr *MI = PredSU->getInstr();
      // Only consider memory loads
      if (!MI->mayLoad() || MI->mayStore())
        continue;
      SDep ForwardD = PredDep;
      ForwardD.setSUnit(&SU);
      for (SDep &SuccDep : PredSU->Succs) {
        if (SuccDep == ForwardD) {
          SuccDep.setLatency(SuccDep.getLatency() + SyntheticLatency);
          break;
        }
      }
      PredDep.setLatency(PredDep.getLatency() + SyntheticLatency);
      PredSU->setDepthDirty();
      SU.setDepthDirty();
    }
  }
}

} // end namespace

std::unique_ptr<ScheduleDAGMutation>
llvm::createAMDGPUBarrierLatencyDAGMutation(MachineFunction *MF) {
  return std::make_unique<BarrierLatency>(MF);
}
