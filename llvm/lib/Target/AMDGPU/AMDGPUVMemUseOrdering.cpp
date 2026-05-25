//===-- AMDGPUVMemUseOrdering.cpp - AMDGPU VMEM Use Ordering --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file DAG mutation that prevents the post-RA scheduler from hoisting
///       consumers of VMEM-pending registers ahead of in-region VMEM loads.
///
///       A register is "VMEM-pending" if its value was last written by a pure
///       VMEM load that has not yet been waited on.  Two sources are covered:
///
///       (A) Cross-block: a physreg live-in whose last writer on some
///           predecessor path is a VMEM load (backward CFG walk,
///           depth-bounded).
///
///       (B) Same-block pre-region: an instruction earlier in the same basic
///           block (before the current scheduling region) whose last write to
///           a physreg was a VMEM load.  This covers multi-region blocks where
///           an earlier region issued a VMEM load and the result is still
///           in-flight when this region is scheduled.
///
///       In both cases the producer is outside the current scheduling region
///       and invisible to the DAG, so the scheduler treats the consumer as
///       free and may hoist it before in-region loads, forcing SIInsertWaitcnts
///       to emit an early wait that serialises those loads.
///
///       The fix adds Artificial Order edges from each in-region VMEM-load
///       SUnit to every consumer of a VMEM-pending register so the scheduler
///       keeps consumers after the loads.  A single later wait then covers
///       both the pending counter and the in-region cluster.
///       This applies to VALU consumers and VMEM consumers (e.g. a cascaded
///       BVH whose node pointer comes from a prior BVH's output).
///
///       Requires RemoveKillFlags=true on the owning DAG: reordering via
///       artificial edges invalidates kill flags, and GCNPostScheduleDAGMILive
///       already sets this.
///
///       Pre-RA schedulers: the mutation is intentionally registered only in
///       createPostMachineScheduler.  It relies on physical register live-ins
///       and physreg def operands, which only exist post-RA.  The iterative
///       pre-RA schedulers (gcn-iterative-*) operate on virtual registers and
///       have no vmcnt/pending-register concept, so registering this mutation
///       there would be both incorrect and unnecessary.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUVMemUseOrdering.h"
#include "AMDGPUWaitcntUtils.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "amdgpu-vmem-use-ordering"

// TODO: The right depth bound is not well-established.  8 was chosen
// conservatively; profiling on real shader workloads may show a lower value
// (e.g. 4) is sufficient and reduces compile-time for deep CFGs.
static cl::opt<unsigned> VMemUseOrderingMaxDepth(
    "amdgpu-vmem-use-ordering-max-depth",
    cl::desc("Maximum predecessor-block depth searched when classifying "
             "VMEM-pending live-ins"),
    cl::init(8), cl::Hidden);

namespace {

/// Post-RA DAG mutation that prevents the scheduler from hoisting consumers
/// of VMEM-pending registers ahead of in-region VMEM loads.  It identifies
/// pending registers from two sources (cross-block live-ins and same-block
/// pre-region defs), then adds Artificial order edges from each in-region
/// VMEM-load SUnit to every consumer of a pending unit.
class VMemUseOrdering : public ScheduleDAGMutation {
  const GCNSubtarget &ST;

public:
  VMemUseOrdering(MachineFunction *MF) : ST(MF->getSubtarget<GCNSubtarget>()) {}
  void apply(ScheduleDAGInstrs *DAG) override;
};

// True iff MI is a non-store, side-effect-free VMEM load completing on a
// VMEM counter alone.  Excludes spill reloads (PseudoSourceValue MMOs).
// BUNDLE: every member must satisfy the same criteria.
static bool isPureVMemLoad(const MachineInstr &MI, const GCNSubtarget &ST) {
  if (!MI.mayLoad() || MI.mayStore() || MI.hasUnmodeledSideEffects())
    return false;
  for (const MachineMemOperand *MMO : MI.memoperands())
    if (MMO->getPseudoValue())
      return false;
  if (MI.getOpcode() != TargetOpcode::BUNDLE)
    return SIInstrInfo::isVmemCounterLoad(MI, ST);
  bool SawMember = false;
  for (auto It = std::next(MI.getIterator()), E = MI.getParent()->instr_end();
       It != E && It->isBundledWithPred(); ++It) {
    if (It->isMetaInstruction() || It->isDebugInstr())
      continue;
    if (!It->mayLoad() || It->mayStore() || It->hasUnmodeledSideEffects())
      return false;
    if (!SIInstrInfo::isVmemCounterLoad(*It, ST))
      return false;
    SawMember = true;
  }
  return SawMember;
}

// Tracks which VMEM counter classes have been waited to zero during a block
// scan.  Pre-GFX12 has one unified counter (LOAD_CNT); GFX12+ splits into
// LOAD_CNT, SAMPLE_CNT, and BVH_CNT.
struct VmemClearedState {
  bool LoadCnt = false;
  bool SampleCnt = false;
  bool BvhCnt = false;

  void update(const MachineInstr &MI, const GCNSubtarget &ST) {
    if (MI.getNumOperands() == 0 || !MI.getOperand(0).isImm())
      return;
    uint64_t Imm = MI.getOperand(0).getImm();
    if (!ST.hasExtendedWaitCounts()) {
      if (MI.getOpcode() == AMDGPU::S_WAITCNT) {
        AMDGPU::Waitcnt W =
            AMDGPU::decodeWaitcnt(AMDGPU::getIsaVersion(ST.getCPU()), Imm);
        if (W.get(AMDGPU::LOAD_CNT) == 0)
          LoadCnt = true;
      }
      return;
    }
    switch (MI.getOpcode()) {
    case AMDGPU::S_WAIT_LOADCNT:
      if (Imm == 0)
        LoadCnt = true;
      break;
    case AMDGPU::S_WAIT_SAMPLECNT:
      if (Imm == 0)
        SampleCnt = true;
      break;
    case AMDGPU::S_WAIT_BVHCNT:
      if (Imm == 0)
        BvhCnt = true;
      break;
    case AMDGPU::S_WAIT_LOADCNT_DSCNT: {
      AMDGPU::Waitcnt W =
          AMDGPU::decodeLoadcntDscnt(AMDGPU::getIsaVersion(ST.getCPU()), Imm);
      if (W.get(AMDGPU::LOAD_CNT) == 0)
        LoadCnt = true;
      break;
    }
    default:
      break;
    }
  }

  bool clears(AMDGPU::InstCounterType CT) const {
    switch (CT) {
    case AMDGPU::LOAD_CNT:
      return LoadCnt;
    case AMDGPU::SAMPLE_CNT:
      return SampleCnt;
    case AMDGPU::BVH_CNT:
      return BvhCnt;
    default:
      return false;
    }
  }

  // True if all VMEM counter classes relevant to ST have been zeroed.
  bool allCleared(const GCNSubtarget &ST) const {
    if (!ST.hasExtendedWaitCounts())
      return LoadCnt;
    return LoadCnt && SampleCnt && BvhCnt;
  }
};

// Walk the predecessor CFG backward from CurMBB, classifying all register
// units in Candidates.  Units whose last reaching definition on any predecessor
// path is an unwaited VMEM load are added to PendingUnits.  The shared Visited
// set prevents re-entering blocks across all units, so each predecessor block
// is scanned at most once regardless of how many live-ins are being classified.
static void
gatherVMemPendingUnits(const MachineBasicBlock *CurMBB,
                       const SmallDenseSet<MCRegUnit, 16> &Candidates,
                       const TargetRegisterInfo &TRI, const GCNSubtarget &ST,
                       SmallDenseSet<MCRegUnit, 16> &PendingUnits,
                       SmallPtrSetImpl<const MachineBasicBlock *> &Visited,
                       unsigned Depth) {
  if (Depth > VMemUseOrderingMaxDepth)
    return;

  for (const MachineBasicBlock *Pred : CurMBB->predecessors()) {
    if (!Visited.insert(Pred).second)
      continue;

    SmallDenseSet<MCRegUnit, 16> Unresolved = Candidates;
    VmemClearedState Cleared;

    for (auto It = Pred->rbegin(), E = Pred->rend();
         It != E && !Unresolved.empty(); ++It) {
      const MachineInstr &MI = *It;
      if (MI.isMetaInstruction() || MI.isDebugInstr())
        continue;

      // Check defs before updating Cleared: Cleared must only reflect
      // instructions that appear after the def in program order.
      // A non-VMEM def resolves the unit as not-pending: a unit is only
      // pending when its last writer is positively identified as a pure
      // VMEM load.
      bool IsVMem = isPureVMemLoad(MI, ST);
      AMDGPU::InstCounterType CT =
          IsVMem ? SIInstrInfo::getVmemLoadCounter(MI, ST) : AMDGPU::LOAD_CNT;
      SmallVector<MCRegUnit, 4> JustResolved;
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isReg() || !MO.isDef() || !MO.getReg().isPhysical())
          continue;
        for (MCRegUnit DefUnit : TRI.regunits(MO.getReg().asMCReg())) {
          if (!Unresolved.count(DefUnit))
            continue;
          JustResolved.push_back(DefUnit);
          if (IsVMem && !Cleared.clears(CT))
            PendingUnits.insert(DefUnit);
        }
      }
      for (MCRegUnit U : JustResolved)
        Unresolved.erase(U);

      Cleared.update(MI, ST);
      if (Cleared.allCleared(ST))
        break;
    }

    if (!Unresolved.empty() && !Cleared.allCleared(ST))
      gatherVMemPendingUnits(Pred, Unresolved, TRI, ST, PendingUnits, Visited,
                             Depth + 1);
  }
}

void VMemUseOrdering::apply(ScheduleDAGInstrs *DAG) {
  // EntrySU/ExitSU are not in DAG->SUnits, so every element has a real MI.
  if (DAG->SUnits.empty())
    return;
  MachineBasicBlock *MBB = DAG->begin()->getParent();
  const TargetRegisterInfo *TRI = DAG->TRI;

  // Build a unified set of VMEM-pending register units from two sources.
  // MCRegUnit throughout gives uniform sub-register aliasing for the
  // pending classification and the consumer check below.
  SmallDenseSet<MCRegUnit, 16> PendingUnits;

  // Source A: cross-block VMEM-pending live-ins.
  // Classify all live-in reg-units in one CFG walk; each predecessor block is
  // scanned at most once regardless of the number of live-ins.
  if (!MBB->pred_empty()) {
    SmallDenseSet<MCRegUnit, 16> Candidates;
    for (const auto &LI : MBB->liveins())
      for (MCRegUnit U : TRI->regunits(LI.PhysReg))
        Candidates.insert(U);
    if (!Candidates.empty()) {
      SmallPtrSet<const MachineBasicBlock *, 16> Visited;
      Visited.insert(MBB);
      gatherVMemPendingUnits(MBB, Candidates, *TRI, ST, PendingUnits, Visited,
                             0);
    }
  }

  // Source B: same-block pre-region VMEM-pending registers.
  // Map from register unit to the counter type of its last VMEM writer.
  // A unit absent from the map is not VMEM-pending (last writer was non-VMEM,
  // or the unit was cleared by a wait).
  //
  // Skip leading debug instructions so the guard reflects real content;
  // remaining non-debug meta (KILL, IMPLICIT_DEF, CFI, ...) is filtered below.
  auto PreRegionBegin =
      skipDebugInstructionsForward(MBB->begin(), DAG->begin());
  if (PreRegionBegin != DAG->begin()) {
    SmallDenseMap<MCRegUnit, AMDGPU::InstCounterType, 32> UnitLastVMemCT;
    for (const MachineInstr &MI : make_range(PreRegionBegin, DAG->begin())) {
      if (MI.isMetaInstruction())
        continue;
      VmemClearedState Cleared;
      Cleared.update(MI, ST);
      if (Cleared.LoadCnt || Cleared.SampleCnt || Cleared.BvhCnt) {
        SmallVector<MCRegUnit, 8> ToErase;
        for (const auto &[Unit, CT] : UnitLastVMemCT)
          if (Cleared.clears(CT))
            ToErase.push_back(Unit);
        for (MCRegUnit Unit : ToErase)
          UnitLastVMemCT.erase(Unit);
        continue;
      }
      bool IsVMem = isPureVMemLoad(MI, ST);
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isReg() || !MO.isDef() || !MO.getReg().isPhysical())
          continue;
        for (MCRegUnit Unit : TRI->regunits(MO.getReg().asMCReg())) {
          if (IsVMem)
            UnitLastVMemCT[Unit] = SIInstrInfo::getVmemLoadCounter(MI, ST);
          else
            UnitLastVMemCT.erase(Unit);
        }
      }
    }
    for (const auto &[Unit, CT] : UnitLastVMemCT)
      PendingUnits.insert(Unit);
  }

  if (PendingUnits.empty())
    return;

  LLVM_DEBUG({
    dbgs() << "VMemUseOrdering: " << PendingUnits.size()
           << " pending unit(s) in " << MBB->getFullName() << "\n";
    for (MCRegUnit U : PendingUnits)
      dbgs() << "  pending: " << printRegUnit(U, TRI) << "\n";
  });

  // Collect all in-region pure VMEM-load SUnits.  Cascaded loads (e.g. a BVH
  // whose node pointer comes from a prior BVH result) are included; the
  // self-edge check and DAG->addEdge()'s cycle rejection handle them below.
  SmallVector<SUnit *, 8> VMemLoadSUs;
  for (SUnit &SU : DAG->SUnits) {
    const MachineInstr *MI = SU.getInstr();
    if (MI && isPureVMemLoad(*MI, ST))
      VMemLoadSUs.push_back(&SU);
  }
  if (VMemLoadSUs.empty())
    return;

  // Prune to "leaf" loads: if load A's transitive successor cone contains
  // another in-region VMEM load B, then the edge B → consumer already implies
  // A → consumer (through A → ... → B → consumer), so A → consumer is
  // redundant.  Keeping only leaves reduces added edges and avoids pulling
  // unrelated in-region loads onto the consumer's critical path.
  if (VMemLoadSUs.size() > 1) {
    SmallPtrSet<SUnit *, 8> VMemSet(VMemLoadSUs.begin(), VMemLoadSUs.end());
    SmallPtrSet<SUnit *, 8> HasVMemSucc;
    SmallVector<SUnit *, 16> Stack;
    SmallPtrSet<SUnit *, 32> Visited;
    for (SUnit *L : VMemLoadSUs) {
      Stack.clear();
      Visited.clear();
      for (const SDep &D : L->Succs)
        Stack.push_back(D.getSUnit());
      while (!Stack.empty()) {
        SUnit *S = Stack.pop_back_val();
        if (!Visited.insert(S).second)
          continue;
        if (VMemSet.count(S)) {
          HasVMemSucc.insert(L);
          break;
        }
        for (const SDep &D : S->Succs)
          Stack.push_back(D.getSUnit());
      }
    }
    if (!HasVMemSucc.empty()) {
      SmallVector<SUnit *, 8> Leaves;
      for (SUnit *L : VMemLoadSUs)
        if (!HasVMemSucc.count(L))
          Leaves.push_back(L);
      VMemLoadSUs = std::move(Leaves);
    }
  }

  LLVM_DEBUG({
    dbgs() << "VMemUseOrdering: " << VMemLoadSUs.size()
           << " leaf VMEM-load SUnit(s)\n";
    for (SUnit *L : VMemLoadSUs)
      dbgs() << "  leaf: SU(" << L->NodeNum << ") " << *L->getInstr();
  });

  // For each consumer of a VMEM-pending register, add Artificial order edges
  // from every leaf in-region VMEM-load SUnit to it.
  // addEdge() silently drops edges that would create a cycle; WAR/WAW
  // dependencies already in the DAG make some of these edges redundant.
  for (SUnit &SU : DAG->SUnits) {
    MachineInstr *MI = SU.getInstr();
    if (!MI || MI->isMetaInstruction())
      continue;

    bool ReadsPending = false;
    for (const MachineOperand &MO : MI->uses()) {
      if (!MO.isReg() || !MO.getReg().isPhysical())
        continue;
      for (MCRegUnit Unit : TRI->regunits(MO.getReg().asMCReg())) {
        if (PendingUnits.count(Unit)) {
          ReadsPending = true;
          break;
        }
      }
      if (ReadsPending)
        break;
    }
    if (!ReadsPending)
      continue;

    unsigned EdgesAdded = 0;
    for (SUnit *L : VMemLoadSUs) {
      if (L == &SU)
        continue;
      if (DAG->addEdge(&SU, SDep(L, SDep::Artificial)))
        ++EdgesAdded;
    }
    LLVM_DEBUG(if (EdgesAdded) dbgs()
               << "VMemUseOrdering: added " << EdgesAdded << " edge(s) to SU("
               << SU.NodeNum << ") " << *MI);
  }
}

} // end namespace

std::unique_ptr<ScheduleDAGMutation>
llvm::createAMDGPUVMemUseOrderingDAGMutation(MachineFunction *MF) {
  return std::make_unique<VMemUseOrdering>(MF);
}
