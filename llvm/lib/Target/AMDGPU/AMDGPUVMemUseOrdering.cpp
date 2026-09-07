//===-- AMDGPUVMemUseOrdering.cpp - AMDGPU VMEM Use Ordering --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file Post-RA DAG mutation that keeps consumers of VMEM-pending registers
///       from being hoisted ahead of the region's own VMEM loads.
///
///       A register is VMEM-pending at region entry when its closest reaching
///       definition is a pure VMEM load.  Hoisting such a consumer to the top
///       of the region forces SIInsertWaitcnts to emit an early wait that
///       serialises the loads; adding Artificial order edges from each
///       in-region VMEM load to the consumer instead lets one later partial
///       wait cover them all.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUVMemUseOrdering.h"
#include "AMDGPUWaitcntUtils.h"
#include "GCNSubtarget.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdint>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "amdgpu-vmem-use-ordering"

// Maximum predecessor-block depth the backward classification walk descends.
// Bounds compile-time on deep CFGs; the value is a conservative, untuned guess.
static cl::opt<unsigned> VMemUseOrderingMaxDepth(
    "amdgpu-vmem-use-ordering-max-depth",
    cl::desc("Maximum predecessor-block depth searched when classifying "
             "VMEM-pending registers"),
    cl::init(8), cl::Hidden);

// A counter class already holding many outstanding loads at region entry is
// counter-bound; adding edges there collapses SIInsertWaitcnts's partial-wait
// staircase into a full drain, so leave such classes alone (0 disables).
static cl::opt<unsigned> VMemUseOrderingMaxPendingDepth(
    "amdgpu-vmem-use-ordering-max-pending-depth",
    cl::desc("Skip adding VMEM-load ordering edges for a counter class once "
             "more than this many register units are already VMEM-pending at "
             "region entry on that class (0 disables the gate)"),
    cl::init(8), cl::Hidden);

namespace {

// Only VGPR/AGPR units can ever be VMEM-pending, since a pure VMEM load only
// writes vector registers.
static bool isVectorPhysReg(MCRegister Reg, const TargetRegisterInfo &TRI) {
  const TargetRegisterClass *RC = TRI.getPhysRegBaseClass(Reg);
  return RC &&
         (SIRegisterInfo::isVGPRClass(RC) || SIRegisterInfo::isAGPRClass(RC));
}

class VMemUseOrdering : public ScheduleDAGMutation {
  // In-region pure VMEM loads paired with the counter class each completes on.
  using VMemLoadList =
      SmallVector<std::pair<SUnit *, AMDGPU::InstCounterType>, 8>;

  const GCNSubtarget &ST;

  // Scratch reused across regions: grown lazily, reset per region.
  BitVector VisitedMBBs; // [MBB number]  block already seen by classify()
  BitVector DfsBV;       // [SUnit NodeNum] node seen by leaf-pruning DFS
  BitVector CandsBV;     // [reg unit]    candidate use awaiting classify
  BitVector DefinedBV;   // [reg unit]    defined somewhere in the region
  // [reg unit] bitmask of VMEM counter classes the unit is pending on (0 = not
  // pending, one bit = unambiguous, several = matches any VMEM counter).
  SmallVector<uint16_t> PendingMask;
  unsigned PendingDepth[AMDGPU::NUM_INST_CNTS] = {}; // #units pending per class
  bool AnyPending = false;                           // any unit pending at all

  // MCRegUnit is enum class : unsigned; BitVector needs a plain unsigned index.
  static unsigned ruIdx(MCRegUnit U) { return static_cast<unsigned>(U); }

  void growSU(unsigned N) {
    if (DfsBV.size() < N)
      DfsBV.resize(N);
  }
  void growRU(unsigned N) {
    if (CandsBV.size() < N)
      CandsBV.resize(N);
    if (DefinedBV.size() < N)
      DefinedBV.resize(N);
    if (PendingMask.size() < N)
      PendingMask.resize(N);
  }

  /// Mark reg unit \p U as VMEM-pending on counter class \p Cls: the first
  /// class seen sticks; a later different class makes the unit ambiguous (set
  /// in every VMEM class so it matches any counter); the same class is a no-op.
  /// PendingDepth (per-class unit count) is maintained incrementally.
  void markPending(unsigned U, AMDGPU::InstCounterType Cls) {
    uint16_t Bit = uint16_t(1) << Cls;
    uint16_t &M = PendingMask[U];
    if (M == Bit)
      return;
    AnyPending = true;
    if (M == 0) { // first reaching VMEM def: its class sticks
      M = Bit;
      ++PendingDepth[Cls];
      return;
    }
    // Conflicting class: promote to ambiguous (pending on every VMEM class).
    for (AMDGPU::InstCounterType C :
         {AMDGPU::LOAD_CNT, AMDGPU::SAMPLE_CNT, AMDGPU::BVH_CNT})
      if (!(M & (uint16_t(1) << C))) {
        M |= uint16_t(1) << C;
        ++PendingDepth[C];
      }
  }

  /// Counter class completed by a pure VMEM load. getVmemLoadCounter treats a
  /// BUNDLE header as LOAD_CNT, so for a clause use its first VMEM-load member.
  AMDGPU::InstCounterType loadCounter(const MachineInstr &MI) const {
    if (!MI.isBundle())
      return AMDGPU::getVmemLoadCounter(MI, ST);
    for (auto It = std::next(MI.getIterator()), E = MI.getParent()->instr_end();
         It != E && It->isBundledWithPred(); ++It) {
      if (It->isMetaInstruction())
        continue;
      if (AMDGPU::isVmemCounterLoad(*It, ST))
        return AMDGPU::getVmemLoadCounter(*It, ST);
    }
    return AMDGPU::LOAD_CNT;
  }

  /// Backward walk from \p Stop in \p MBB, then predecessors (depth-bounded,
  /// each block visited once via VisitedMBBs).  Resolves each candidate at its
  /// closest reaching reference: a pure VMEM-load def marks it pending, any
  /// other def or a use drops it.  CandsBV/CandsCount are backtracked on return
  /// so sibling paths see the original set.
  void classify(const MachineBasicBlock *MBB,
                MachineBasicBlock::const_iterator Stop, unsigned &CandsCount,
                const TargetRegisterInfo &TRI, unsigned Depth) {
    SmallVector<MCRegUnit, 16> Resolved; // restored on return (backtracking)
    for (auto It = MachineBasicBlock::const_reverse_iterator(Stop),
              E = MBB->rend();
         It != E && CandsCount > 0; ++It) {
      const MachineInstr &MI = *It;
      if (MI.isMetaInstruction())
        continue;
      // Classification is only needed once this instruction actually resolves
      // a candidate def, so defer it (most scanned instructions resolve none).
      int IsVMem = -1; // -1 unknown, 0 no, 1 yes
      AMDGPU::InstCounterType Cls = AMDGPU::LOAD_CNT;
      // Defs first: a pure VMEM-load def leaves the unit pending in its class.
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isReg() || !MO.isDef() || !MO.getReg().isPhysical())
          continue;
        for (MCRegUnit U : TRI.regunits(MO.getReg().asMCReg())) {
          if (!CandsBV.test(ruIdx(U)))
            continue;
          if (IsVMem < 0) {
            IsVMem = AMDGPU::isPureVMemLoad(MI, ST) ? 1 : 0;
            if (IsVMem)
              Cls = loadCounter(MI);
          }
          CandsBV.reset(ruIdx(U));
          --CandsCount;
          Resolved.push_back(U);
          if (IsVMem)
            markPending(ruIdx(U), Cls);
        }
      }
      // A use reached before any def means the value was consumed (waited)
      // pre-region.  Defs run first, so a read-modify-write is classified by
      // its def.
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isReg() || MO.isDef() || !MO.getReg().isPhysical())
          continue;
        for (MCRegUnit U : TRI.regunits(MO.getReg().asMCReg())) {
          if (!CandsBV.test(ruIdx(U)))
            continue;
          CandsBV.reset(ruIdx(U));
          --CandsCount;
          Resolved.push_back(U);
        }
      }
    }
    if (CandsCount > 0 && Depth < VMemUseOrderingMaxDepth) {
      for (const MachineBasicBlock *Pred : MBB->predecessors()) {
        unsigned PredNum = static_cast<unsigned>(Pred->getNumber());
        if (!VisitedMBBs.test(PredNum)) {
          VisitedMBBs.set(PredNum);
          classify(Pred, Pred->end(), CandsCount, TRI, Depth + 1);
        }
      }
    }
    // Backtrack so sibling predecessors and the caller see the original set.
    for (MCRegUnit U : Resolved) {
      CandsBV.set(ruIdx(U));
      ++CandsCount;
    }
  }

  /// Collects in-region pure VMEM loads with the class each completes on.
  /// \returns false if there are none (nothing to sink consumers past).
  bool collectRegionLoads(ScheduleDAGInstrs *DAG,
                          VMemLoadList &VMemLoads) const {
    for (SUnit &SU : DAG->SUnits) {
      const MachineInstr *MI = SU.getInstr();
      if (MI && AMDGPU::isPureVMemLoad(*MI, ST))
        VMemLoads.push_back({&SU, loadCounter(*MI)});
    }
    return !VMemLoads.empty();
  }

  /// Drops a load whose successor cone reaches another in-region load of the
  /// same class: the survivor's edges already cover it.  Classes are read from
  /// the original list so a pruned load never stands in for another.
  void pruneCoveredLoads(ScheduleDAGInstrs *DAG, VMemLoadList &VMemLoads) {
    if (VMemLoads.size() <= 1)
      return;
    growSU(DAG->SUnits.size());
    auto reachesLoadOfClass = [&](unsigned NodeNum,
                                  AMDGPU::InstCounterType Cls) {
      for (auto &[L, C] : VMemLoads)
        if (L->NodeNum == NodeNum)
          return C == Cls;
      return false;
    };
    SmallVector<SUnit *, 16> Stack;
    llvm::erase_if(VMemLoads,
                   [&](const std::pair<SUnit *, AMDGPU::InstCounterType> &LC) {
                     DfsBV.reset();
                     Stack.clear();
                     for (const SDep &D : LC.first->Succs)
                       if (D.getSUnit()->NodeNum < DfsBV.size())
                         Stack.push_back(D.getSUnit());
                     while (!Stack.empty()) {
                       SUnit *S = Stack.pop_back_val();
                       if (DfsBV.test(S->NodeNum))
                         continue;
                       DfsBV.set(S->NodeNum);
                       if (reachesLoadOfClass(S->NodeNum, LC.second))
                         return true;
                       for (const SDep &D : S->Succs)
                         if (D.getSUnit()->NodeNum < DfsBV.size())
                           Stack.push_back(D.getSUnit());
                     }
                     return false;
                   });
  }

  /// Seeds candidate reg units from in-region vector uses, minus units also
  /// defined in-region (those already carry a Data dep on the in-region
  /// producer).
  /// \returns the number of candidate reg units.
  unsigned collectCandidateUnits(ScheduleDAGInstrs *DAG,
                                 const TargetRegisterInfo &TRI) {
    growRU(TRI.getNumRegUnits());
    CandsBV.reset();
    DefinedBV.reset();
    for (SUnit &SU : DAG->SUnits) {
      const MachineInstr *MI = SU.getInstr();
      if (!MI || MI->isMetaInstruction())
        continue;
      for (const MachineOperand &MO : MI->operands()) {
        if (!MO.isReg() || !MO.getReg().isPhysical())
          continue;
        MCRegister Reg = MO.getReg().asMCReg();
        if (MO.isDef())
          for (MCRegUnit U : TRI.regunits(Reg))
            DefinedBV.set(ruIdx(U));
        else if (isVectorPhysReg(Reg, TRI))
          for (MCRegUnit U : TRI.regunits(Reg))
            CandsBV.set(ruIdx(U));
      }
    }
    CandsBV.reset(DefinedBV); // CandsBV &= ~DefinedBV
    return CandsBV.count();
  }

  /// Resets per-region pending state and marks \p MBB as the classify() origin.
  void resetPendingState(const MachineBasicBlock *MBB) {
    std::fill(PendingMask.begin(), PendingMask.end(), uint16_t(0));
    std::fill(std::begin(PendingDepth), std::end(PendingDepth), 0u);
    AnyPending = false;
    unsigned NumBlocks = MBB->getParent()->getNumBlockIDs();
    if (VisitedMBBs.size() < NumBlocks)
      VisitedMBBs.resize(NumBlocks);
    VisitedMBBs.reset();
    VisitedMBBs.set(static_cast<unsigned>(MBB->getNumber()));
  }

  /// Drops loads whose counter class is already deep at region entry (see
  /// VMemUseOrderingMaxPendingDepth); they can never contribute an edge.
  void applyPendingDepthGate(VMemLoadList &VMemLoads) const {
    if (!VMemUseOrderingMaxPendingDepth)
      return;
    llvm::erase_if(
        VMemLoads, [&](const std::pair<SUnit *, AMDGPU::InstCounterType> &LC) {
          return PendingDepth[LC.second] > VMemUseOrderingMaxPendingDepth;
        });
  }

  /// \returns the counter classes \p MI waits on via the pending units it reads
  /// (0 if none).  An ambiguous unit selects every class.
  unsigned pendingClassesRead(const MachineInstr &MI,
                              const TargetRegisterInfo &TRI) const {
    const unsigned AllMask = (1u << AMDGPU::LOAD_CNT) |
                             (1u << AMDGPU::SAMPLE_CNT) |
                             (1u << AMDGPU::BVH_CNT);
    unsigned ClassMask = 0;
    for (const MachineOperand &MO : MI.uses()) {
      if (!MO.isReg() || !MO.getReg().isPhysical())
        continue;
      for (MCRegUnit U : TRI.regunits(MO.getReg().asMCReg()))
        ClassMask |= PendingMask[ruIdx(U)];
      if (ClassMask == AllMask) // cannot gain more classes; stop scanning
        break;
    }
    return ClassMask;
  }

  /// For each SUnit reading a pending unit, adds Artificial order edges from
  /// the matching in-region loads.  addEdge() drops self/cycle edges; redundant
  /// ones are harmless (Artificial latency is 0).
  void addOrderingEdges(ScheduleDAGInstrs *DAG, const TargetRegisterInfo &TRI,
                        const VMemLoadList &VMemLoads) const {
    // Bucket by class so each consumer visits only the classes it waits on.
    SmallVector<SUnit *, 8> LoadsByClass[AMDGPU::NUM_INST_CNTS];
    for (auto &[L, C] : VMemLoads)
      LoadsByClass[C].push_back(L);

    for (SUnit &SU : DAG->SUnits) {
      MachineInstr *MI = SU.getInstr();
      if (!MI || MI->isMetaInstruction())
        continue;
      unsigned ClassMask = pendingClassesRead(*MI, TRI);
      if (!ClassMask)
        continue;
      [[maybe_unused]] unsigned Added = 0;
      for (unsigned C = 0; C != AMDGPU::NUM_INST_CNTS; ++C) {
        if (!(ClassMask & (1u << C)))
          continue;
        for (SUnit *L : LoadsByClass[C])
          if (L != &SU && DAG->addEdge(&SU, SDep(L, SDep::Artificial)))
            ++Added;
      }
      LLVM_DEBUG(if (Added) dbgs()
                 << "VMemUseOrdering: added " << Added << " edge(s) to SU("
                 << SU.NodeNum << ") " << *MI);
    }
  }

public:
  VMemUseOrdering(MachineFunction *MF) : ST(MF->getSubtarget<GCNSubtarget>()) {}

  void apply(ScheduleDAGInstrs *DAG) override {
    // EntrySU/ExitSU are not in DAG->SUnits, so every element has a real MI.
    if (DAG->SUnits.empty())
      return;
    const TargetRegisterInfo &TRI = *DAG->TRI;

    VMemLoadList VMemLoads;
    if (!collectRegionLoads(DAG, VMemLoads))
      return;

    pruneCoveredLoads(DAG, VMemLoads);

    unsigned CandsCount = collectCandidateUnits(DAG, TRI);
    if (CandsCount == 0)
      return;

    const MachineBasicBlock *MBB = DAG->begin()->getParent();
    resetPendingState(MBB);
    classify(MBB, DAG->begin(), CandsCount, TRI, 0);
    if (!AnyPending)
      return;

    LLVM_DEBUG({
      dbgs() << "VMemUseOrdering: pending units by class in "
             << MBB->getFullName() << "\n";
      for (unsigned C = 0; C != AMDGPU::NUM_INST_CNTS; ++C)
        if (PendingDepth[C])
          dbgs() << "  class " << C << ": " << PendingDepth[C]
                 << (VMemUseOrderingMaxPendingDepth &&
                             PendingDepth[C] > VMemUseOrderingMaxPendingDepth
                         ? " (saturated)"
                         : "")
                 << "\n";
    });

    applyPendingDepthGate(VMemLoads);
    if (VMemLoads.empty())
      return;

    addOrderingEdges(DAG, TRI, VMemLoads);
  }
};

} // end namespace

std::unique_ptr<ScheduleDAGMutation>
llvm::createAMDGPUVMemUseOrderingDAGMutation(MachineFunction *MF) {
  return std::make_unique<VMemUseOrdering>(MF);
}
