//===--- AMDGPUBarrierLatency.cpp - AMDGPU Barrier Latency ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file contains a DAG scheduling mutation to add latency to:
///       1. Barrier edges between ATOMIC_FENCE instructions and preceding
///          memory accesses potentially affected by the fence.
///          This encourages the scheduling of more instructions before
///          ATOMIC_FENCE instructions.  ATOMIC_FENCE instructions may
///          introduce wait counting or indicate an impending S_BARRIER
///          wait.  Having more instructions in-flight across these
///          constructs improves latency hiding.
///       2. Barrier edges from S_BARRIER_SIGNAL to S_BARRIER_WAIT.
///          This encourages independent work to be scheduled between
///          signal and wait, hiding barrier synchronization latency.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUBarrierLatency.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

static cl::opt<unsigned> BarrierSignalWaitLatencyOpt(
    "amdgpu-barrier-signal-wait-latency",
    cl::desc("Synthetic latency between S_BARRIER_SIGNAL and S_BARRIER_WAIT "
             "to encourage scheduling independent work between them"),
    cl::init(16), cl::Hidden);

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

    const GCNSubtarget &ST = MF->getSubtarget<GCNSubtarget>();
    if (!ST.requiresWaitOnWorkgroupReleaseFence()) {
      // Prior to GFX10 workgroup scope does not normally require waitcnts
      IgnoredScopes.insert(Context.getOrInsertSyncScopeID("workgroup"));
    }
  }
  void apply(ScheduleDAGInstrs *DAG) override;
};

void addLatencyToEdge(SDep &PredDep, SUnit &SU, unsigned Latency) {
  SUnit *PredSU = PredDep.getSUnit();
  SDep ForwardD = PredDep;
  ForwardD.setSUnit(&SU);
  for (SDep &SuccDep : PredSU->Succs) {
    if (SuccDep == ForwardD) {
      SuccDep.setLatency(SuccDep.getLatency() + Latency);
      break;
    }
  }
  PredDep.setLatency(PredDep.getLatency() + Latency);
  PredSU->setDepthDirty();
  SU.setDepthDirty();
}

void setLatencyForEdge(SDep &PredDep, SUnit &SU, unsigned Latency) {
  SUnit *PredSU = PredDep.getSUnit();
  SDep ForwardD = PredDep;
  ForwardD.setSUnit(&SU);
  for (SDep &SuccDep : PredSU->Succs) {
    if (SuccDep == ForwardD) {
      SuccDep.setLatency(Latency);
      break;
    }
  }
  PredDep.setLatency(Latency);
  PredSU->setDepthDirty();
  SU.setDepthDirty();
}

void BarrierLatency::apply(ScheduleDAGInstrs *DAG) {
  const SIInstrInfo *TII = static_cast<const SIInstrInfo *>(DAG->TII);
  constexpr unsigned FenceLatency = 2000;
  const unsigned BarrierSignalWaitLatency = BarrierSignalWaitLatencyOpt;
  SmallVector<SUnit *, 8> RegionTDM;
  SmallVector<SUnit *, 8> RegionAsync;
  const TargetSchedModel *SchedModel = DAG->getSchedModel();

  for (SUnit &SU : DAG->SUnits) {
    const MachineInstr *MI = SU.getInstr();
    unsigned Op = MI->getOpcode();

    if (Op == AMDGPU::ATOMIC_FENCE) {
      // Update latency on barrier edges of ATOMIC_FENCE.
      // Ignore scopes not expected to have any latency.
      SyncScope::ID SSID =
          static_cast<SyncScope::ID>(MI->getOperand(1).getImm());
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

        addLatencyToEdge(PredDep, SU,
                         SchedModel ? SchedModel->computeInstrLatency(MI, false)
                                    : FenceLatency);
      }
    } else if (Op == AMDGPU::S_BARRIER_WAIT) {
      for (SDep &PredDep : SU.Preds) {
        SUnit *PredSU = PredDep.getSUnit();
        const MachineInstr *PredMI = PredSU->getInstr();
        if (TII->isBarrierStart(PredMI->getOpcode())) {
          addLatencyToEdge(PredDep, SU, BarrierSignalWaitLatency);
        }
      }
    } else if (TII->isLDSDMA(*MI)) {
      if (MI->getDesc().TSFlags & SIInstrFlags::TENSOR_CNT)
        RegionTDM.push_back(&SU);
      else if (MI->getDesc().TSFlags & SIInstrFlags::ASYNC_CNT)
        RegionAsync.push_back(&SU);
    } else if (Op == AMDGPU::S_WAIT_TENSORCNT ||
               Op == AMDGPU::S_WAIT_ASYNCCNT) {
      auto needWaitFor = [&](SmallVectorImpl<SUnit *> &RegionLDSDMA, SUnit *SU,
                             int64_t Count) {
        if (RegionLDSDMA.size() <= static_cast<uint64_t>(Count)) {
          return false;
        }

        int64_t Counter = 0;
        auto I = RegionLDSDMA.rbegin(), E = RegionLDSDMA.rend();
        for (; I != E; I++) {
          if (Counter >= Count)
            return true;

          if (SU->NodeNum == (*I)->NodeNum)
            return false;

          ++Counter;
        }
        llvm_unreachable("Malformed RegionLDSDMA");
      };

      int64_t WaitVal = MI->getOperand(0).getImm();
      for (SDep &PredDep : SU.Preds) {
        if (PredDep.getKind() != SDep::Kind::Data)
          continue;

        Register DepReg = PredDep.getReg();
        Register LDSDMACnt = AMDGPU::TENSORcnt;
        uint64_t LDSDMAFlags = SIInstrFlags::TENSOR_CNT;
        if (Op == AMDGPU::S_WAIT_ASYNCCNT) {
          LDSDMACnt = AMDGPU::ASYNCcnt;
          LDSDMAFlags = SIInstrFlags::ASYNC_CNT;
        }

        if (DepReg != LDSDMACnt)
          continue;

        SUnit *PredSU = PredDep.getSUnit();

        // The data dep can be carried by a non-LDSDMA SU
        // (e.g. an intervening COPY or pseudo). Such predecessors are not
        // tracked, so needWaitFor cannot reason about them.
        if (!(PredSU->getInstr()->getDesc().TSFlags & LDSDMAFlags))
          continue;

        if (!needWaitFor(Op == AMDGPU::S_WAIT_ASYNCCNT ? RegionAsync
                                                       : RegionTDM,
                         PredSU, WaitVal)) {
          setLatencyForEdge(PredDep, SU, 1);
        }
      }
    }
  }
}

} // end namespace

std::unique_ptr<ScheduleDAGMutation>
llvm::createAMDGPUBarrierLatencyDAGMutation(MachineFunction *MF) {
  return std::make_unique<BarrierLatency>(MF);
}
