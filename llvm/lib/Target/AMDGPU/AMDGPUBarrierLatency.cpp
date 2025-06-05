//===--- AMDGPUBarrierLatency.cpp - AMDGPU Barrier Latency ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file contains a DAG scheduling mutation to add data dependency
///       edges between ATOMIC_FENCE instructions and preceeding memory
///       accesses that might be affected by the fence.
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

static bool isMemRead(const MachineInstr *MI) {
  return (SIInstrInfo::isDS(*MI) || SIInstrInfo::isVMEM(*MI) ||
          SIInstrInfo::isSMRD(*MI)) &&
         MI->mayLoad();
}

static const MachineInstr *getReadInstr(const MachineInstr *MI) {
  if (MI->isBundle()) {
    auto I = std::next(MI->getIterator());
    if (I != MI->getParent()->instr_end() && I->isInsideBundle() &&
        isMemRead(&*I))
      return &*I;
  } else if (isMemRead(MI)) {
    return MI;
  }

  return nullptr;
}

void BarrierLatency::apply(ScheduleDAGInstrs *DAG) {
  const unsigned SyntheticLatency = 2000;
  const unsigned MaxTracked = 32;
  SmallVector<std::pair<SUnit *, const MachineInstr *>, MaxTracked> ReadOps;
  unsigned NextIdx = 0;

  for (SUnit &SU : DAG->SUnits) {
    auto *MI = SU.getInstr();
    auto *ReadMI = getReadInstr(MI);

    // Record read operations.
    // If SU represents a bundle, then ReadMI is the first instruction in the
    // bundle.
    if (ReadMI) {
      if (ReadOps.size() < MaxTracked) {
        ReadOps.emplace_back(&SU, ReadMI);
      } else {
        ReadOps[NextIdx] = std::pair(&SU, ReadMI);
        NextIdx = (NextIdx + 1) % MaxTracked;
      }
      continue;
    }

    // Create new edges on ATOMIC_FENCE for recorded reads.
    // We don't consider the scope of the fence so it is possible there will
    // be no impact of this fence on the recorded operations.
    if (MI->getOpcode() == AMDGPU::ATOMIC_FENCE) {
      for (auto &DSOp : ReadOps) {
        Register DstReg = DSOp.second->getOperand(0).getReg();
        SDep Edge = SDep(DSOp.first, SDep::Data, DstReg);
        Edge.setLatency(SyntheticLatency);
        DAG->addEdge(&SU, Edge);
      }
      // Clear tracked operations
      ReadOps.clear();
      NextIdx = 0;
      continue;
    }
  }
}

} // end namespace

std::unique_ptr<ScheduleDAGMutation>
llvm::createAMDGPUBarrierLatencyDAGMutation() {
  return std::make_unique<BarrierLatency>();
}
