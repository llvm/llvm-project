//===- RISCVMachineScheduler.cpp - MI Scheduler for RISC-V ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RISCVMachineScheduler.h"
#include "MCTargetDesc/RISCVBaseInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/Support/Debug.h"
#include <limits>

using namespace llvm;

#define DEBUG_TYPE "riscv-prera-sched-strategy"

static cl::opt<bool> EnableVTypeSchedHeuristic(
    "riscv-enable-vtype-sched-heuristic", cl::init(false), cl::Hidden,
    cl::desc("Enable scheduling RVV instructions based on vtype heuristic "
             "(pick instruction with compatible vtype first)"));

static VTypeInfo getVTypeInfo(GenericSchedulerBase::SchedCandidate &Cand) {
  MachineInstr *CandMI = Cand.SU->getInstr();
  const MCInstrDesc &Desc = CandMI->getDesc();
  if (RISCVII::hasSEWOp(Desc.TSFlags)) {
    unsigned CurVSEW = CandMI->getOperand(RISCVII::getSEWOpNum(Desc)).getImm();
    RISCVVType::VLMUL CurVLMUL = RISCVII::getLMul(Desc.TSFlags);
    // FIXME: We should consider vl and policy here.
    return {CurVLMUL, CurVSEW};
  }
  return {RISCVVType::LMUL_RESERVED, std::numeric_limits<unsigned>::max()};
}

static bool isValidVTypeInfo(VTypeInfo Info) {
  return Info.first != RISCVVType::LMUL_RESERVED &&
         Info.second != std::numeric_limits<unsigned>::max();
}

bool RISCVPreRAMachineSchedStrategy::tryVType(
    VTypeInfo TryVType, VTypeInfo CandVtype,
    GenericSchedulerBase::SchedCandidate &TryCand,
    GenericSchedulerBase::SchedCandidate &Cand,
    GenericSchedulerBase::CandReason Reason) const {
  // Do not compare the vtype changes between top and bottom
  // boundary.
  if (Cand.AtTop != TryCand.AtTop)
    return false;

  // Try Cand first.
  // We prefer the top node as it is straightforward from the perspective of
  // vtype dataflow.
  if (isValidVTypeInfo(CandVtype) && isValidVTypeInfo(TopVType) && Cand.AtTop &&
      CandVtype == TopVType) {
    return true;
  }

  if (isValidVTypeInfo(CandVtype) && isValidVTypeInfo(BottomVType) &&
      !Cand.AtTop && CandVtype == BottomVType) {
    return true;
  }

  // Then try TryCand.
  if (isValidVTypeInfo(TryVType) && isValidVTypeInfo(TopVType) &&
      TryCand.AtTop && TryVType == TopVType) {
    TryCand.Reason = Reason;
    return true;
  }

  if (isValidVTypeInfo(TryVType) && isValidVTypeInfo(BottomVType) &&
      !TryCand.AtTop && TryVType == BottomVType) {
    TryCand.Reason = Reason;
    return true;
  }

  return false;
}

bool RISCVPreRAMachineSchedStrategy::tryCandidate(SchedCandidate &Cand,
                                                  SchedCandidate &TryCand,
                                                  SchedBoundary *Zone) const {
  //-------------------------------------------------------------------------//
  // Below is copied from `GenericScheduler::tryCandidate`.
  // FIXME: Is there a way to not replicate this?
  //-------------------------------------------------------------------------//
  // Initialize the candidate if needed.
  if (!Cand.isValid()) {
    TryCand.Reason = FirstValid;
    return true;
  }

  // Bias PhysReg Defs and copies to their uses and defined respectively.
  if (tryGreater(biasPhysReg(TryCand.SU, TryCand.AtTop),
                 biasPhysReg(Cand.SU, Cand.AtTop), TryCand, Cand, PhysReg))
    return TryCand.Reason != NoCand;

  // Avoid exceeding the target's limit.
  if (DAG->isTrackingPressure() &&
      tryPressure(TryCand.RPDelta.Excess, Cand.RPDelta.Excess, TryCand, Cand,
                  RegExcess, TRI, DAG->MF))
    return TryCand.Reason != NoCand;

  // Avoid increasing the max critical pressure in the scheduled region.
  if (DAG->isTrackingPressure() &&
      tryPressure(TryCand.RPDelta.CriticalMax, Cand.RPDelta.CriticalMax,
                  TryCand, Cand, RegCritical, TRI, DAG->MF))
    return TryCand.Reason != NoCand;

  // We only compare a subset of features when comparing nodes between
  // Top and Bottom boundary. Some properties are simply incomparable, in many
  // other instances we should only override the other boundary if something
  // is a clear good pick on one boundary. Skip heuristics that are more
  // "tie-breaking" in nature.
  bool SameBoundary = Zone != nullptr;
  if (SameBoundary) {
    // For loops that are acyclic path limited, aggressively schedule for
    // latency. Within an single cycle, whenever CurrMOps > 0, allow normal
    // heuristics to take precedence.
    if (Rem.IsAcyclicLatencyLimited && !Zone->getCurrMOps() &&
        tryLatency(TryCand, Cand, *Zone))
      return TryCand.Reason != NoCand;

    // Prioritize instructions that read unbuffered resources by stall cycles.
    if (tryLess(Zone->getLatencyStallCycles(TryCand.SU),
                Zone->getLatencyStallCycles(Cand.SU), TryCand, Cand, Stall))
      return TryCand.Reason != NoCand;
  }

  // Keep clustered nodes together to encourage downstream peephole
  // optimizations which may reduce resource requirements.
  //
  // This is a best effort to set things up for a post-RA pass. Optimizations
  // like generating loads of multiple registers should ideally be done within
  // the scheduler pass by combining the loads during DAG postprocessing.
  unsigned CandZoneCluster = Cand.AtTop ? TopClusterID : BotClusterID;
  unsigned TryCandZoneCluster = TryCand.AtTop ? TopClusterID : BotClusterID;
  bool CandIsClusterSucc =
      isTheSameCluster(CandZoneCluster, Cand.SU->ParentClusterIdx);
  bool TryCandIsClusterSucc =
      isTheSameCluster(TryCandZoneCluster, TryCand.SU->ParentClusterIdx);

  if (tryGreater(TryCandIsClusterSucc, CandIsClusterSucc, TryCand, Cand,
                 Cluster))
    return TryCand.Reason != NoCand;

  if (SameBoundary) {
    // Weak edges are for clustering and other constraints.
    if (tryLess(getWeakLeft(TryCand.SU, TryCand.AtTop),
                getWeakLeft(Cand.SU, Cand.AtTop), TryCand, Cand, Weak))
      return TryCand.Reason != NoCand;
  }

  // Avoid increasing the max pressure of the entire region.
  if (DAG->isTrackingPressure() &&
      tryPressure(TryCand.RPDelta.CurrentMax, Cand.RPDelta.CurrentMax, TryCand,
                  Cand, RegMax, TRI, DAG->MF))
    return TryCand.Reason != NoCand;

  if (SameBoundary) {
    // Avoid critical resource consumption and balance the schedule.
    TryCand.initResourceDelta(DAG, SchedModel);
    if (tryLess(TryCand.ResDelta.CritResources, Cand.ResDelta.CritResources,
                TryCand, Cand, ResourceReduce))
      return TryCand.Reason != NoCand;
    if (tryGreater(TryCand.ResDelta.DemandedResources,
                   Cand.ResDelta.DemandedResources, TryCand, Cand,
                   ResourceDemand))
      return TryCand.Reason != NoCand;

    // Avoid serializing long latency dependence chains.
    // For acyclic path limited loops, latency was already checked above.
    if (!RegionPolicy.DisableLatencyHeuristic && TryCand.Policy.ReduceLatency &&
        !Rem.IsAcyclicLatencyLimited && tryLatency(TryCand, Cand, *Zone))
      return TryCand.Reason != NoCand;

    // Fall through to original instruction order.
    if ((Zone->isTop() && TryCand.SU->NodeNum < Cand.SU->NodeNum) ||
        (!Zone->isTop() && TryCand.SU->NodeNum > Cand.SU->NodeNum)) {
      TryCand.Reason = NodeOrder;
      return true;
    }
  }

  //-------------------------------------------------------------------------//
  // Below is RISC-V specific scheduling heuristics.
  //-------------------------------------------------------------------------//

  // TODO: We should not use `CandReason::Cluster` here, but is there a
  // mechanism to extend this enum?
  if (EnableVTypeSchedHeuristic &&
      tryVType(getVTypeInfo(TryCand), getVTypeInfo(Cand), TryCand, Cand,
               Cluster))
    return TryCand.Reason != NoCand;

  return false;
}

void RISCVPreRAMachineSchedStrategy::enterMBB(MachineBasicBlock *MBB) {
  TopVType = {RISCVVType::LMUL_RESERVED, std::numeric_limits<unsigned>::max()};
  BottomVType = {RISCVVType::LMUL_RESERVED,
                 std::numeric_limits<unsigned>::max()};
}

void RISCVPreRAMachineSchedStrategy::leaveMBB() {
  TopVType = {RISCVVType::LMUL_RESERVED, std::numeric_limits<unsigned>::max()};
  BottomVType = {RISCVVType::LMUL_RESERVED,
                 std::numeric_limits<unsigned>::max()};
}

void RISCVPreRAMachineSchedStrategy::schedNode(SUnit *SU, bool IsTopNode) {
  GenericScheduler::schedNode(SU, IsTopNode);
  MachineInstr *MI = SU->getInstr();
  const MCInstrDesc &Desc = MI->getDesc();
  if (RISCVII::hasSEWOp(Desc.TSFlags)) {
    unsigned VSEW = MI->getOperand(RISCVII::getSEWOpNum(Desc)).getImm();
    RISCVVType::VLMUL VLMUL = RISCVII::getLMul(Desc.TSFlags);
    if (IsTopNode)
      TopVType = {VLMUL, VSEW};
    else
      BottomVType = {VLMUL, VSEW};
    LLVM_DEBUG({
      dbgs() << "Previous scheduled Unit: \n";
      dbgs() << "  IsTop: " << IsTopNode << "\n";
      dbgs() << "  SU(" << SU->NodeNum << ") - ";
      SU->getInstr()->dump();
      dbgs() << " VSEW : " << (1 << VSEW) << "\n";
      auto LMUL = RISCVVType::decodeVLMUL(VLMUL);
      dbgs() << " VLMUL: m" << (LMUL.second ? "f" : "") << LMUL.first << "\n";
    });
  }
}
