//===--- AMDGPUBarrierLatency.cpp - AMDGPU Barrier Latency ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file contains a DAG scheduling mutation to add latency to
///       barrier edges between ATOMIC_FENCE instructions and preceeding
///       memory accesses potentially affected by the fence.
///       This is beneficial when a fence would cause wait count insertion,
///       as more instructions will be scheduled before the fence hiding
///       memory latency.
///       It also reduces the risk of a fence causing a premature wait
///       on all active memory operations.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUBarrierLatency.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"

using namespace llvm;

namespace {

class BarrierLatency : public ScheduleDAGMutation {
public:
  BarrierLatency() = default;
  void apply(ScheduleDAGInstrs *DAG) override;
};

static bool isMemLoad(const MachineInstr *MI) {
  auto isLoad = [](const MachineInstr *MI) {
    return (SIInstrInfo::isDS(*MI) || SIInstrInfo::isVMEM(*MI) ||
            SIInstrInfo::isSMRD(*MI)) &&
           MI->mayLoad();
  };

  if (MI->isBundle()) {
    auto I = std::next(MI->getIterator());
    return I != MI->getParent()->instr_end() && I->isInsideBundle() &&
           isLoad(&*I);
  }

  return isLoad(MI);
}

void BarrierLatency::apply(ScheduleDAGInstrs *DAG) {
  const unsigned SyntheticLatency = 2000;
  for (SUnit &SU : DAG->SUnits) {
    const MachineInstr *MI = SU.getInstr();
    if (MI->getOpcode() != AMDGPU::ATOMIC_FENCE)
      continue;

    // Update latency on barrier edges of ATOMIC_FENCE.
    // We don't consider the scope of the fence or type of instruction
    // involved in the barrier edge.
    for (SDep &PredDep : SU.Preds) {
      if (!PredDep.isBarrier())
        continue;
      SUnit *PredSU = PredDep.getSUnit();
      if (!isMemLoad(PredSU->getInstr()))
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
llvm::createAMDGPUBarrierLatencyDAGMutation() {
  return std::make_unique<BarrierLatency>();
}
