//===--- AMDGPUIGroupLP.cpp - AMDGPU IGroupLP  ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file This file defines a set of schedule DAG mutations that can be used to
// override default scheduler behavior to enforce specific scheduling patterns.
// They should be used in cases where runtime performance considerations such as
// inter-wavefront interactions, mean that compile-time heuristics cannot
// predict the optimal instruction ordering, or in kernels where optimum
// instruction scheduling is important enough to warrant manual intervention.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUIGroupLP.h"
#include "AMDGPUTargetMachine.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/TargetOpcodes.h"

using namespace llvm;

#define DEBUG_TYPE "machine-scheduler"

namespace {

static cl::opt<bool>
    EnableIGroupLP("amdgpu-igrouplp",
                   cl::desc("Enable construction of Instruction Groups and "
                            "their ordering for scheduling"),
                   cl::init(false));

static cl::opt<Optional<unsigned>>
    VMEMGroupMaxSize("amdgpu-igrouplp-vmem-group-size", cl::init(None),
                     cl::Hidden,
                     cl::desc("The maximum number of instructions to include "
                              "in VMEM group."));

static cl::opt<Optional<unsigned>>
    MFMAGroupMaxSize("amdgpu-igrouplp-mfma-group-size", cl::init(None),
                     cl::Hidden,
                     cl::desc("The maximum number of instructions to include "
                              "in MFMA group."));

static cl::opt<Optional<unsigned>>
    LDRGroupMaxSize("amdgpu-igrouplp-ldr-group-size", cl::init(None),
                    cl::Hidden,
                    cl::desc("The maximum number of instructions to include "
                             "in lds/gds read group."));

static cl::opt<Optional<unsigned>>
    LDWGroupMaxSize("amdgpu-igrouplp-ldw-group-size", cl::init(None),
                    cl::Hidden,
                    cl::desc("The maximum number of instructions to include "
                             "in lds/gds write group."));

typedef function_ref<bool(const MachineInstr &, const SIInstrInfo *)>
    CanAddMIFn;

// Classify instructions into groups to enable fine tuned control over the
// scheduler. These groups may be more specific than current SchedModel
// instruction classes.
class SchedGroup {
private:
  // Function that returns true if a non-bundle MI may be inserted into this
  // group.
  const CanAddMIFn canAddMI;

  // Maximum number of SUnits that can be added to this group.
  Optional<unsigned> MaxSize;

  // Collection of SUnits that are classified as members of this group.
  SmallVector<SUnit *, 32> Collection;

  ScheduleDAGInstrs *DAG;

  void tryAddEdge(SUnit *A, SUnit *B) {
    if (A != B && DAG->canAddEdge(B, A)) {
      DAG->addEdge(B, SDep(A, SDep::Artificial));
      LLVM_DEBUG(dbgs() << "Adding edge...\n"
                        << "from: SU(" << A->NodeNum << ") " << *A->getInstr()
                        << "to: SU(" << B->NodeNum << ") " << *B->getInstr());
    }
  }

public:
  // Add DAG dependencies from all SUnits in this SchedGroup and this SU. If
  // MakePred is true, SU will be a predecessor of the SUnits in this
  // SchedGroup, otherwise SU will be a successor.
  void link(SUnit &SU, bool MakePred = false) {
    for (auto A : Collection) {
      SUnit *B = &SU;
      if (MakePred)
        std::swap(A, B);

      tryAddEdge(A, B);
    }
  }

  // Add DAG dependencies from all SUnits in this SchedGroup and this SU. Use
  // the predicate to determine whether SU should be a predecessor (P = true)
  // or a successor (P = false) of this SchedGroup.
  void link(SUnit &SU, function_ref<bool(const SUnit *A, const SUnit *B)> P) {
    for (auto A : Collection) {
      SUnit *B = &SU;
      if (P(A, B))
        std::swap(A, B);

      tryAddEdge(A, B);
    }
  }

  // Add DAG dependencies such that SUnits in this group shall be ordered
  // before SUnits in OtherGroup.
  void link(SchedGroup &OtherGroup) {
    for (auto B : OtherGroup.Collection)
      link(*B);
  }

  // Returns true if no more instructions may be added to this group.
  bool isFull() { return MaxSize && Collection.size() >= *MaxSize; }

  // Returns true if SU can be added to this SchedGroup.
  bool canAddSU(SUnit &SU, const SIInstrInfo *TII) {
    if (isFull())
      return false;

    MachineInstr &MI = *SU.getInstr();
    if (MI.getOpcode() != TargetOpcode::BUNDLE)
      return canAddMI(MI, TII);

    // Special case for bundled MIs.
    const MachineBasicBlock *MBB = MI.getParent();
    MachineBasicBlock::instr_iterator B = MI.getIterator(), E = ++B;
    while (E != MBB->end() && E->isBundledWithPred())
      ++E;

    // Return true if all of the bundled MIs can be added to this group.
    return std::all_of(
        B, E, [this, TII](MachineInstr &MI) { return canAddMI(MI, TII); });
  }

  void add(SUnit &SU) { Collection.push_back(&SU); }

  SchedGroup(CanAddMIFn canAddMI, Optional<unsigned> MaxSize,
             ScheduleDAGInstrs *DAG)
      : canAddMI(canAddMI), MaxSize(MaxSize), DAG(DAG) {}
};

bool isMFMASGMember(const MachineInstr &MI, const SIInstrInfo *TII) {
  return TII->isMFMA(MI);
}

bool isVALUSGMember(const MachineInstr &MI, const SIInstrInfo *TII) {
  return TII->isVALU(MI) && !TII->isMFMA(MI);
}

bool isSALUSGMember(const MachineInstr &MI, const SIInstrInfo *TII) {
  return TII->isSALU(MI);
}

bool isVMEMSGMember(const MachineInstr &MI, const SIInstrInfo *TII) {
  return TII->isVMEM(MI) || (TII->isFLAT(MI) && !TII->isDS(MI));
}

bool isVMEMReadSGMember(const MachineInstr &MI, const SIInstrInfo *TII) {
  return MI.mayLoad() &&
         (TII->isVMEM(MI) || (TII->isFLAT(MI) && !TII->isDS(MI)));
}

bool isVMEMWriteSGMember(const MachineInstr &MI, const SIInstrInfo *TII) {
  return MI.mayStore() &&
         (TII->isVMEM(MI) || (TII->isFLAT(MI) && !TII->isDS(MI)));
}

bool isDSWriteSGMember(const MachineInstr &MI, const SIInstrInfo *TII) {
  return MI.mayStore() && TII->isDS(MI);
}

bool isDSReadSGMember(const MachineInstr &MI, const SIInstrInfo *TII) {
  return MI.mayLoad() && TII->isDS(MI);
}

class IGroupLPDAGMutation : public ScheduleDAGMutation {
public:
  const SIInstrInfo *TII;
  ScheduleDAGMI *DAG;

  IGroupLPDAGMutation() = default;
  void apply(ScheduleDAGInstrs *DAGInstrs) override;
};

// DAG mutation that coordinates with the SCHED_BARRIER instruction and
// corresponding builtin. The mutation adds edges from specific instruction
// classes determined by the SCHED_BARRIER mask so that they cannot be
// scheduled around the SCHED_BARRIER.
class SchedBarrierDAGMutation : public ScheduleDAGMutation {
private:
  const SIInstrInfo *TII;

  ScheduleDAGMI *DAG;

  // Components of the mask that determines which instructions may not be
  // scheduled across the SCHED_BARRIER.
  enum class SchedBarrierMasks {
    NONE = 0u,
    ALU = 1u << 0,
    VALU = 1u << 1,
    SALU = 1u << 2,
    MFMA = 1u << 3,
    VMEM = 1u << 4,
    VMEM_READ = 1u << 5,
    VMEM_WRITE = 1u << 6,
    DS = 1u << 7,
    DS_READ = 1u << 8,
    DS_WRITE = 1u << 9,
    LLVM_MARK_AS_BITMASK_ENUM(/* LargestFlag = */ DS_WRITE)
  };

  // Cache SchedGroups of each type if we have multiple SCHED_BARRIERs in a
  // region.
  //
  std::unique_ptr<SchedGroup> MFMASchedGroup = nullptr;
  std::unique_ptr<SchedGroup> VALUSchedGroup = nullptr;
  std::unique_ptr<SchedGroup> SALUSchedGroup = nullptr;
  std::unique_ptr<SchedGroup> VMEMReadSchedGroup = nullptr;
  std::unique_ptr<SchedGroup> VMEMWriteSchedGroup = nullptr;
  std::unique_ptr<SchedGroup> DSWriteSchedGroup = nullptr;
  std::unique_ptr<SchedGroup> DSReadSchedGroup = nullptr;

  // Use a SCHED_BARRIER's mask to identify instruction SchedGroups that should
  // not be reordered accross the SCHED_BARRIER.
  void getSchedGroupsFromMask(int32_t Mask,
                              SmallVectorImpl<SchedGroup *> &SchedGroups);

  // Add DAG edges that enforce SCHED_BARRIER ordering.
  void addSchedBarrierEdges(SUnit &SU);

  // Classify instructions and add them to the SchedGroup.
  void initSchedGroup(SchedGroup *SG);

  // Remove all existing edges from a SCHED_BARRIER.
  void resetSchedBarrierEdges(SUnit &SU);

public:
  void apply(ScheduleDAGInstrs *DAGInstrs) override;

  SchedBarrierDAGMutation() = default;
};

void IGroupLPDAGMutation::apply(ScheduleDAGInstrs *DAGInstrs) {
  const GCNSubtarget &ST = DAGInstrs->MF.getSubtarget<GCNSubtarget>();
  TII = ST.getInstrInfo();
  DAG = static_cast<ScheduleDAGMI *>(DAGInstrs);
  const TargetSchedModel *TSchedModel = DAGInstrs->getSchedModel();
  if (!TSchedModel || DAG->SUnits.empty())
    return;

  LLVM_DEBUG(dbgs() << "Applying IGroupLPDAGMutation...\n");

  // The order of InstructionGroups in this vector defines the
  // order in which edges will be added. In other words, given the
  // present ordering, we will try to make each VMEMRead instruction
  // a predecessor of each DSRead instruction, and so on.
  SmallVector<SchedGroup, 4> PipelineOrderGroups = {
      SchedGroup(isVMEMSGMember, VMEMGroupMaxSize, DAG),
      SchedGroup(isDSReadSGMember, LDRGroupMaxSize, DAG),
      SchedGroup(isMFMASGMember, MFMAGroupMaxSize, DAG),
      SchedGroup(isDSWriteSGMember, LDWGroupMaxSize, DAG)};

  for (SUnit &SU : DAG->SUnits) {
    LLVM_DEBUG(dbgs() << "Checking Node"; DAG->dumpNode(SU));
    for (auto &SG : PipelineOrderGroups)
      if (SG.canAddSU(SU, TII))
        SG.add(SU);
  }

  for (unsigned i = 0; i < PipelineOrderGroups.size() - 1; i++) {
    auto &GroupA = PipelineOrderGroups[i];
    for (unsigned j = i + 1; j < PipelineOrderGroups.size(); j++) {
      auto &GroupB = PipelineOrderGroups[j];
      GroupA.link(GroupB);
    }
  }
}

void SchedBarrierDAGMutation::apply(ScheduleDAGInstrs *DAGInstrs) {
  const TargetSchedModel *TSchedModel = DAGInstrs->getSchedModel();
  if (!TSchedModel || DAGInstrs->SUnits.empty())
    return;

  LLVM_DEBUG(dbgs() << "Applying SchedBarrierDAGMutation...\n");

  const GCNSubtarget &ST = DAGInstrs->MF.getSubtarget<GCNSubtarget>();
  TII = ST.getInstrInfo();
  DAG = static_cast<ScheduleDAGMI *>(DAGInstrs);
  for (auto &SU : DAG->SUnits)
    if (SU.getInstr()->getOpcode() == AMDGPU::SCHED_BARRIER)
      addSchedBarrierEdges(SU);
}

void SchedBarrierDAGMutation::addSchedBarrierEdges(SUnit &SchedBarrier) {
  MachineInstr &MI = *SchedBarrier.getInstr();
  assert(MI.getOpcode() == AMDGPU::SCHED_BARRIER);
  // Remove all existing edges from the SCHED_BARRIER that were added due to the
  // instruction having side effects.
  resetSchedBarrierEdges(SchedBarrier);
  SmallVector<SchedGroup *, 4> SchedGroups;
  int32_t Mask = MI.getOperand(0).getImm();
  getSchedGroupsFromMask(Mask, SchedGroups);
  for (auto SG : SchedGroups)
    SG->link(
        SchedBarrier, (function_ref<bool(const SUnit *A, const SUnit *B)>)[](
                          const SUnit *A, const SUnit *B) {
          return A->NodeNum > B->NodeNum;
        });
}

void SchedBarrierDAGMutation::getSchedGroupsFromMask(
    int32_t Mask, SmallVectorImpl<SchedGroup *> &SchedGroups) {
  SchedBarrierMasks SBMask = (SchedBarrierMasks)Mask;
  // See IntrinsicsAMDGPU.td for an explanation of these masks and their
  // mappings.
  //
  if ((SBMask & SchedBarrierMasks::VALU) == SchedBarrierMasks::NONE &&
      (SBMask & SchedBarrierMasks::ALU) == SchedBarrierMasks::NONE) {
    if (!VALUSchedGroup) {
      VALUSchedGroup = std::make_unique<SchedGroup>(isVALUSGMember, None, DAG);
      initSchedGroup(VALUSchedGroup.get());
    }

    SchedGroups.push_back(VALUSchedGroup.get());
  }

  if ((SBMask & SchedBarrierMasks::SALU) == SchedBarrierMasks::NONE &&
      (SBMask & SchedBarrierMasks::ALU) == SchedBarrierMasks::NONE) {
    if (!SALUSchedGroup) {
      SALUSchedGroup = std::make_unique<SchedGroup>(isSALUSGMember, None, DAG);
      initSchedGroup(SALUSchedGroup.get());
    }

    SchedGroups.push_back(SALUSchedGroup.get());
  }

  if ((SBMask & SchedBarrierMasks::MFMA) == SchedBarrierMasks::NONE &&
      (SBMask & SchedBarrierMasks::ALU) == SchedBarrierMasks::NONE) {
    if (!MFMASchedGroup) {
      MFMASchedGroup = std::make_unique<SchedGroup>(isMFMASGMember, None, DAG);
      initSchedGroup(MFMASchedGroup.get());
    }

    SchedGroups.push_back(MFMASchedGroup.get());
  }

  if ((SBMask & SchedBarrierMasks::VMEM_READ) == SchedBarrierMasks::NONE &&
      (SBMask & SchedBarrierMasks::VMEM) == SchedBarrierMasks::NONE) {
    if (!VMEMReadSchedGroup) {
      VMEMReadSchedGroup =
          std::make_unique<SchedGroup>(isVMEMReadSGMember, None, DAG);
      initSchedGroup(VMEMReadSchedGroup.get());
    }

    SchedGroups.push_back(VMEMReadSchedGroup.get());
  }

  if ((SBMask & SchedBarrierMasks::VMEM_WRITE) == SchedBarrierMasks::NONE &&
      (SBMask & SchedBarrierMasks::VMEM) == SchedBarrierMasks::NONE) {
    if (!VMEMWriteSchedGroup) {
      VMEMWriteSchedGroup =
          std::make_unique<SchedGroup>(isVMEMWriteSGMember, None, DAG);
      initSchedGroup(VMEMWriteSchedGroup.get());
    }

    SchedGroups.push_back(VMEMWriteSchedGroup.get());
  }

  if ((SBMask & SchedBarrierMasks::DS_READ) == SchedBarrierMasks::NONE &&
      (SBMask & SchedBarrierMasks::DS) == SchedBarrierMasks::NONE) {
    if (!DSReadSchedGroup) {
      DSReadSchedGroup =
          std::make_unique<SchedGroup>(isDSReadSGMember, None, DAG);
      initSchedGroup(DSReadSchedGroup.get());
    }

    SchedGroups.push_back(DSReadSchedGroup.get());
  }

  if ((SBMask & SchedBarrierMasks::DS_WRITE) == SchedBarrierMasks::NONE &&
      (SBMask & SchedBarrierMasks::DS) == SchedBarrierMasks::NONE) {
    if (!DSWriteSchedGroup) {
      DSWriteSchedGroup =
          std::make_unique<SchedGroup>(isDSWriteSGMember, None, DAG);
      initSchedGroup(DSWriteSchedGroup.get());
    }

    SchedGroups.push_back(DSWriteSchedGroup.get());
  }
}

void SchedBarrierDAGMutation::initSchedGroup(SchedGroup *SG) {
  assert(SG);
  for (auto &SU : DAG->SUnits)
    if (SG->canAddSU(SU, TII))
      SG->add(SU);
}

void SchedBarrierDAGMutation::resetSchedBarrierEdges(SUnit &SU) {
  assert(SU.getInstr()->getOpcode() == AMDGPU::SCHED_BARRIER);
  for (auto &P : SU.Preds)
    SU.removePred(P);

  for (auto &S : SU.Succs) {
    for (auto &SP : S.getSUnit()->Preds) {
      if (SP.getSUnit() == &SU) {
        S.getSUnit()->removePred(SP);
      }
    }
  }
}

} // namespace

namespace llvm {

std::unique_ptr<ScheduleDAGMutation> createIGroupLPDAGMutation() {
  return EnableIGroupLP ? std::make_unique<IGroupLPDAGMutation>() : nullptr;
}

std::unique_ptr<ScheduleDAGMutation> createSchedBarrierDAGMutation() {
  return std::make_unique<SchedBarrierDAGMutation>();
}

} // end namespace llvm
