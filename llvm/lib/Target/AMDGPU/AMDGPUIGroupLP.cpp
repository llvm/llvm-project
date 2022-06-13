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
#include "llvm/ADT/DenseMap.h"
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

// Components of the mask that determines which instruction types may be may be
// classified into a SchedGroup.
enum class SchedGroupMask {
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
  ALL = ALU | VALU | SALU | MFMA | VMEM | VMEM_READ | VMEM_WRITE | DS |
        DS_READ | DS_WRITE,
  LLVM_MARK_AS_BITMASK_ENUM(/* LargestFlag = */ ALL)
};

// Classify instructions into groups to enable fine tuned control over the
// scheduler. These groups may be more specific than current SchedModel
// instruction classes.
class SchedGroup {
private:
  // Mask that defines which instruction types can be classified into this
  // SchedGroup. The instruction types correspond to the mask from SCHED_BARRIER
  // and SCHED_GROUP_BARRIER.
  SchedGroupMask SGMask;

  // Maximum number of SUnits that can be added to this group.
  Optional<unsigned> MaxSize;

  // SchedGroups will only synchronize with other SchedGroups that have the same
  // SyncID.
  int SyncID = 0;

  // Collection of SUnits that are classified as members of this group.
  SmallVector<SUnit *, 32> Collection;

  ScheduleDAGInstrs *DAG;

  const SIInstrInfo *TII;

  // Try to add and edge from SU A to SU B.
  bool tryAddEdge(SUnit *A, SUnit *B);

  // Use SGMask to determine whether we can classify MI as a member of this
  // SchedGroup object.
  bool canAddMI(const MachineInstr &MI) const;

  // Returns true if SU can be added to this SchedGroup.
  bool canAddSU(SUnit &SU) const;

  // Returns true if no more instructions may be added to this group.
  bool isFull() const;

  // Add SU to the SchedGroup.
  void add(SUnit &SU) {
    LLVM_DEBUG(dbgs() << "For SchedGroup with mask "
                      << format_hex((int)SGMask, 10, true) << " adding "
                      << *SU.getInstr());
    Collection.push_back(&SU);
  }

public:
  // Add DAG dependencies from all SUnits in this SchedGroup and this SU. If
  // MakePred is true, SU will be a predecessor of the SUnits in this
  // SchedGroup, otherwise SU will be a successor.
  void link(SUnit &SU, bool MakePred = false);

  // Add DAG dependencies from all SUnits in this SchedGroup and this SU. Use
  // the predicate to determine whether SU should be a predecessor (P = true)
  // or a successor (P = false) of this SchedGroup.
  void link(SUnit &SU, function_ref<bool(const SUnit *A, const SUnit *B)> P);

  // Add DAG dependencies such that SUnits in this group shall be ordered
  // before SUnits in OtherGroup.
  void link(SchedGroup &OtherGroup);

  // Returns true if no more instructions may be added to this group.
  bool isFull() { return MaxSize && Collection.size() >= *MaxSize; }

  // Identify and add all relevant SUs from the DAG to this SchedGroup.
  void initSchedGroup();

  // Add instructions to the SchedGroup bottom up starting from RIter.
  // ConflictedInstrs is a set of instructions that should not be added to the
  // SchedGroup even when the other conditions for adding it are satisfied.
  // RIter will be added to the SchedGroup as well, and dependencies will be
  // added so that RIter will always be scheduled at the end of the group.
  void initSchedGroup(std::vector<SUnit>::reverse_iterator RIter,
                      DenseSet<SUnit *> &ConflictedInstrs);

  int getSyncID() { return SyncID; }

  SchedGroup(SchedGroupMask SGMask, Optional<unsigned> MaxSize,
             ScheduleDAGInstrs *DAG, const SIInstrInfo *TII)
      : SGMask(SGMask), MaxSize(MaxSize), DAG(DAG), TII(TII) {}

  SchedGroup(SchedGroupMask SGMask, Optional<unsigned> MaxSize, int SyncID,
             ScheduleDAGInstrs *DAG, const SIInstrInfo *TII)
      : SGMask(SGMask), MaxSize(MaxSize), SyncID(SyncID), DAG(DAG), TII(TII) {}
};

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
class SchedBarrierDAGMutation : public ScheduleDAGMutation {
private:
  const SIInstrInfo *TII;

  ScheduleDAGMI *DAG;

  // Organize lists of SchedGroups by their SyncID. SchedGroups /
  // SCHED_GROUP_BARRIERs with different SyncIDs will have no edges added
  // between then.
  DenseMap<int, SmallVector<SchedGroup, 4>> SyncedSchedGroupsMap;

  // Used to track instructions that are already to added to a different
  // SchedGroup with the same SyncID.
  DenseMap<int, DenseSet<SUnit *>> SyncedInstrsMap;

  // Add DAG edges that enforce SCHED_BARRIER ordering.
  void addSchedBarrierEdges(SUnit &SU);

  // Use a SCHED_BARRIER's mask to identify instruction SchedGroups that should
  // not be reordered accross the SCHED_BARRIER. This is used for the base
  // SCHED_BARRIER, and not SCHED_GROUP_BARRIER. The difference is that
  // SCHED_BARRIER will always block all instructions that can be classified
  // into a particular SchedClass, whereas SCHED_GROUP_BARRIER has a fixed size
  // and may only synchronize with some SchedGroups. Returns the inverse of
  // Mask. SCHED_BARRIER's mask describes which instruction types should be
  // allowed to be scheduled across it. Invert the mask to get the
  // SchedGroupMask of instructions that should be barred.
  SchedGroupMask invertSchedBarrierMask(SchedGroupMask Mask) const;

  // Create SchedGroups for a SCHED_GROUP_BARRIER.
  void initSchedGroupBarrier(std::vector<SUnit>::reverse_iterator RIter);

  // Add DAG edges that try to enforce ordering defined by SCHED_GROUP_BARRIER
  // instructions.
  void addSchedGroupBarrierEdges();

public:
  void apply(ScheduleDAGInstrs *DAGInstrs) override;

  SchedBarrierDAGMutation() = default;
};

bool SchedGroup::tryAddEdge(SUnit *A, SUnit *B) {
  if (A != B && DAG->canAddEdge(B, A)) {
    DAG->addEdge(B, SDep(A, SDep::Artificial));
    LLVM_DEBUG(dbgs() << "Adding edge...\n"
                      << "from: SU(" << A->NodeNum << ") " << *A->getInstr()
                      << "to: SU(" << B->NodeNum << ") " << *B->getInstr());
    return true;
  }
  return false;
}

bool SchedGroup::canAddMI(const MachineInstr &MI) const {
  bool Result = false;
  if (MI.isMetaInstruction() || MI.getOpcode() == AMDGPU::SCHED_BARRIER ||
      MI.getOpcode() == AMDGPU::SCHED_GROUP_BARRIER)
    Result = false;

  else if (((SGMask & SchedGroupMask::ALU) != SchedGroupMask::NONE) &&
           (TII->isVALU(MI) || TII->isMFMA(MI) || TII->isSALU(MI)))
    Result = true;

  else if (((SGMask & SchedGroupMask::VALU) != SchedGroupMask::NONE) &&
           TII->isVALU(MI) && !TII->isMFMA(MI))
    Result = true;

  else if (((SGMask & SchedGroupMask::SALU) != SchedGroupMask::NONE) &&
           TII->isSALU(MI))
    Result = true;

  else if (((SGMask & SchedGroupMask::MFMA) != SchedGroupMask::NONE) &&
           TII->isMFMA(MI))
    Result = true;

  else if (((SGMask & SchedGroupMask::VMEM) != SchedGroupMask::NONE) &&
           (TII->isVMEM(MI) || (TII->isFLAT(MI) && !TII->isDS(MI))))
    Result = true;

  else if (((SGMask & SchedGroupMask::VMEM_READ) != SchedGroupMask::NONE) &&
           MI.mayLoad() &&
           (TII->isVMEM(MI) || (TII->isFLAT(MI) && !TII->isDS(MI))))
    Result = true;

  else if (((SGMask & SchedGroupMask::VMEM_WRITE) != SchedGroupMask::NONE) &&
           MI.mayStore() &&
           (TII->isVMEM(MI) || (TII->isFLAT(MI) && !TII->isDS(MI))))
    Result = true;

  else if (((SGMask & SchedGroupMask::DS) != SchedGroupMask::NONE) &&
           TII->isDS(MI))
    Result = true;

  else if (((SGMask & SchedGroupMask::DS_READ) != SchedGroupMask::NONE) &&
           MI.mayLoad() && TII->isDS(MI))
    Result = true;

  else if (((SGMask & SchedGroupMask::DS_WRITE) != SchedGroupMask::NONE) &&
           MI.mayStore() && TII->isDS(MI))
    Result = true;

  LLVM_DEBUG(
      dbgs() << "For SchedGroup with mask " << format_hex((int)SGMask, 10, true)
             << (Result ? " could classify " : " unable to classify ") << MI);

  return Result;
}

void SchedGroup::link(SUnit &SU, bool MakePred) {
  for (auto A : Collection) {
    SUnit *B = &SU;
    if (MakePred)
      std::swap(A, B);

    tryAddEdge(A, B);
  }
}

void SchedGroup::link(SUnit &SU,
                      function_ref<bool(const SUnit *A, const SUnit *B)> P) {
  for (auto A : Collection) {
    SUnit *B = &SU;
    if (P(A, B))
      std::swap(A, B);

    tryAddEdge(A, B);
  }
}

void SchedGroup::link(SchedGroup &OtherGroup) {
  for (auto B : OtherGroup.Collection)
    link(*B);
}

bool SchedGroup::isFull() const {
  return MaxSize && Collection.size() >= *MaxSize;
}

bool SchedGroup::canAddSU(SUnit &SU) const {
  MachineInstr &MI = *SU.getInstr();
  if (MI.getOpcode() != TargetOpcode::BUNDLE)
    return canAddMI(MI);

  // Special case for bundled MIs.
  const MachineBasicBlock *MBB = MI.getParent();
  MachineBasicBlock::instr_iterator B = MI.getIterator(), E = ++B;
  while (E != MBB->end() && E->isBundledWithPred())
    ++E;

  // Return true if all of the bundled MIs can be added to this group.
  return std::all_of(B, E, [this](MachineInstr &MI) { return canAddMI(MI); });
}

void SchedGroup::initSchedGroup() {
  for (auto &SU : DAG->SUnits) {
    if (isFull())
      break;

    if (canAddSU(SU))
      add(SU);
  }
}

static bool canFitIntoPipeline(SUnit &SU, ScheduleDAGInstrs *DAG,
                               DenseSet<SUnit *> &ConflictedInstrs) {
  return std::all_of(
      ConflictedInstrs.begin(), ConflictedInstrs.end(),
      [DAG, &SU](SUnit *SuccSU) { return DAG->canAddEdge(SuccSU, &SU); });
}

void SchedGroup::initSchedGroup(std::vector<SUnit>::reverse_iterator RIter,
                                DenseSet<SUnit *> &ConflictedInstrs) {
  SUnit &InitSU = *RIter;
  for (auto E = DAG->SUnits.rend(); RIter != E; ++RIter) {
    auto &SU = *RIter;
    if (isFull())
      break;

    if (canAddSU(SU) && !ConflictedInstrs.count(&SU) &&
        canFitIntoPipeline(SU, DAG, ConflictedInstrs)) {
      add(SU);
      ConflictedInstrs.insert(&SU);
      tryAddEdge(&SU, &InitSU);
    }
  }

  add(InitSU);
  assert(MaxSize);
  (*MaxSize)++;
}

// Create a pipeline from the SchedGroups in PipelineOrderGroups such that we
// try to enforce the relative ordering of instructions in each group.
static void makePipeline(SmallVectorImpl<SchedGroup> &PipelineOrderGroups) {
  auto I = PipelineOrderGroups.begin();
  auto E = PipelineOrderGroups.end();
  for (; I != E; ++I) {
    auto &GroupA = *I;
    for (auto J = std::next(I); J != E; ++J) {
      auto &GroupB = *J;
      GroupA.link(GroupB);
    }
  }
}

// Same as makePipeline but with reverse ordering.
static void
makeReversePipeline(SmallVectorImpl<SchedGroup> &PipelineOrderGroups) {
  auto I = PipelineOrderGroups.rbegin();
  auto E = PipelineOrderGroups.rend();
  for (; I != E; ++I) {
    auto &GroupA = *I;
    for (auto J = std::next(I); J != E; ++J) {
      auto &GroupB = *J;
      GroupA.link(GroupB);
    }
  }
}

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
      SchedGroup(SchedGroupMask::VMEM, VMEMGroupMaxSize, DAG, TII),
      SchedGroup(SchedGroupMask::DS_READ, LDRGroupMaxSize, DAG, TII),
      SchedGroup(SchedGroupMask::MFMA, MFMAGroupMaxSize, DAG, TII),
      SchedGroup(SchedGroupMask::DS_WRITE, LDWGroupMaxSize, DAG, TII)};

  for (auto &SG : PipelineOrderGroups)
    SG.initSchedGroup();

  makePipeline(PipelineOrderGroups);
}

// Remove all existing edges from a SCHED_BARRIER or SCHED_GROUP_BARRIER.
static void resetEdges(SUnit &SU, ScheduleDAGInstrs *DAG) {
  assert(SU.getInstr()->getOpcode() == AMDGPU::SCHED_BARRIER ||
         SU.getInstr()->getOpcode() == AMDGPU::SCHED_GROUP_BARRIER);

  while (!SU.Preds.empty())
    for (auto &P : SU.Preds)
      SU.removePred(P);

  while (!SU.Succs.empty())
    for (auto &S : SU.Succs)
      for (auto &SP : S.getSUnit()->Preds)
        if (SP.getSUnit() == &SU)
          S.getSUnit()->removePred(SP);
}

void SchedBarrierDAGMutation::apply(ScheduleDAGInstrs *DAGInstrs) {
  const TargetSchedModel *TSchedModel = DAGInstrs->getSchedModel();
  if (!TSchedModel || DAGInstrs->SUnits.empty())
    return;

  LLVM_DEBUG(dbgs() << "Applying SchedBarrierDAGMutation...\n");
  const GCNSubtarget &ST = DAGInstrs->MF.getSubtarget<GCNSubtarget>();
  TII = ST.getInstrInfo();
  DAG = static_cast<ScheduleDAGMI *>(DAGInstrs);
  SyncedInstrsMap.clear();
  SyncedSchedGroupsMap.clear();
  for (auto R = DAG->SUnits.rbegin(), E = DAG->SUnits.rend(); R != E; ++R) {
    if (R->getInstr()->getOpcode() == AMDGPU::SCHED_BARRIER)
      addSchedBarrierEdges(*R);

    else if (R->getInstr()->getOpcode() == AMDGPU::SCHED_GROUP_BARRIER)
      initSchedGroupBarrier(R);
  }

  // SCHED_GROUP_BARRIER edges can only be added after we have found and
  // initialized all of the SCHED_GROUP_BARRIER SchedGroups.
  addSchedGroupBarrierEdges();
}

void SchedBarrierDAGMutation::addSchedBarrierEdges(SUnit &SchedBarrier) {
  MachineInstr &MI = *SchedBarrier.getInstr();
  assert(MI.getOpcode() == AMDGPU::SCHED_BARRIER);
  // Remove all existing edges from the SCHED_BARRIER that were added due to the
  // instruction having side effects.
  resetEdges(SchedBarrier, DAG);
  auto InvertedMask =
      invertSchedBarrierMask((SchedGroupMask)MI.getOperand(0).getImm());
  SchedGroup SG(InvertedMask, None, DAG, TII);
  SG.initSchedGroup();
  // Preserve original instruction ordering relative to the SCHED_BARRIER.
  SG.link(
      SchedBarrier,
      (function_ref<bool(const SUnit *A, const SUnit *B)>)[](
          const SUnit *A, const SUnit *B) { return A->NodeNum > B->NodeNum; });
}

SchedGroupMask
SchedBarrierDAGMutation::invertSchedBarrierMask(SchedGroupMask Mask) const {
  // Invert mask and erase bits for types of instructions that are implied to be
  // allowed past the SCHED_BARRIER.
  SchedGroupMask InvertedMask = ~Mask;

  // ALU implies VALU, SALU, MFMA.
  if ((InvertedMask & SchedGroupMask::ALU) == SchedGroupMask::NONE)
    InvertedMask &=
        ~SchedGroupMask::VALU & ~SchedGroupMask::SALU & ~SchedGroupMask::MFMA;
  // VALU, SALU, MFMA implies ALU.
  else if ((InvertedMask & SchedGroupMask::VALU) == SchedGroupMask::NONE ||
           (InvertedMask & SchedGroupMask::SALU) == SchedGroupMask::NONE ||
           (InvertedMask & SchedGroupMask::MFMA) == SchedGroupMask::NONE)
    InvertedMask &= ~SchedGroupMask::ALU;

  // VMEM implies VMEM_READ, VMEM_WRITE.
  if ((InvertedMask & SchedGroupMask::VMEM) == SchedGroupMask::NONE)
    InvertedMask &= ~SchedGroupMask::VMEM_READ & ~SchedGroupMask::VMEM_WRITE;
  // VMEM_READ, VMEM_WRITE implies VMEM.
  else if ((InvertedMask & SchedGroupMask::VMEM_READ) == SchedGroupMask::NONE ||
           (InvertedMask & SchedGroupMask::VMEM_WRITE) == SchedGroupMask::NONE)
    InvertedMask &= ~SchedGroupMask::VMEM;

  // DS implies DS_READ, DS_WRITE.
  if ((InvertedMask & SchedGroupMask::DS) == SchedGroupMask::NONE)
    InvertedMask &= ~SchedGroupMask::DS_READ & ~SchedGroupMask::DS_WRITE;
  // DS_READ, DS_WRITE implies DS.
  else if ((InvertedMask & SchedGroupMask::DS_READ) == SchedGroupMask::NONE ||
           (InvertedMask & SchedGroupMask::DS_WRITE) == SchedGroupMask::NONE)
    InvertedMask &= ~SchedGroupMask::DS;

  return InvertedMask;
}

void SchedBarrierDAGMutation::initSchedGroupBarrier(
    std::vector<SUnit>::reverse_iterator RIter) {
  // Remove all existing edges from the SCHED_GROUP_BARRIER that were added due
  // to the instruction having side effects.
  resetEdges(*RIter, DAG);
  MachineInstr &SGB = *RIter->getInstr();
  assert(SGB.getOpcode() == AMDGPU::SCHED_GROUP_BARRIER);
  int32_t SGMask = SGB.getOperand(0).getImm();
  int32_t Size = SGB.getOperand(1).getImm();
  int32_t SyncID = SGB.getOperand(2).getImm();
  // Create a new SchedGroup and add it to a list that is mapped to the SyncID.
  // SchedGroups only enforce ordering between SchedGroups with the same SyncID.
  auto &SG = SyncedSchedGroupsMap[SyncID].emplace_back((SchedGroupMask)SGMask,
                                                       Size, SyncID, DAG, TII);

  // SyncedInstrsMap is used here is used to avoid adding the same SUs in
  // multiple SchedGroups that have the same SyncID. This only matters for
  // SCHED_GROUP_BARRIER and not SCHED_BARRIER.
  SG.initSchedGroup(RIter, SyncedInstrsMap[SG.getSyncID()]);
}

void SchedBarrierDAGMutation::addSchedGroupBarrierEdges() {
  // Since we traversed the DAG in reverse order when initializing
  // SCHED_GROUP_BARRIERs we need to reverse the order in the vector to maintain
  // user intentions and program order.
  for (auto &SchedGroups : SyncedSchedGroupsMap)
    makeReversePipeline(SchedGroups.second);
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
