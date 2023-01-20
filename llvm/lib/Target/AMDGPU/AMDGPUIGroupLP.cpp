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

#define DEBUG_TYPE "igrouplp"

namespace {

static cl::opt<bool> EnableExactSolver(
    "amdgpu-igrouplp-exact-solver", cl::Hidden,
    cl::desc("Whether to use the exponential time solver to fit "
             "the instructions to the pipeline as closely as "
             "possible."),
    cl::init(false));

static cl::opt<unsigned> CutoffForExact(
    "amdgpu-igrouplp-exact-solver-cutoff", cl::init(0), cl::Hidden,
    cl::desc("The maximum number of scheduling group conflicts "
             "which we attempt to solve with the exponential time "
             "exact solver. Problem sizes greater than this will"
             "be solved by the less accurate greedy algorithm. Selecting "
             "solver by size is superseded by manually selecting "
             "the solver (e.g. by amdgpu-igrouplp-exact-solver"));

static cl::opt<uint64_t> MaxBranchesExplored(
    "amdgpu-igrouplp-exact-solver-max-branches", cl::init(0), cl::Hidden,
    cl::desc("The amount of branches that we are willing to explore with"
             "the exact algorithm before giving up."));

static cl::opt<bool> UseCostHeur(
    "amdgpu-igrouplp-exact-solver-cost-heur", cl::init(true), cl::Hidden,
    cl::desc("Whether to use the cost heuristic to make choices as we "
             "traverse the search space using the exact solver. Defaulted "
             "to on, and if turned off, we will use the node order -- "
             "attempting to put the later nodes in the later sched groups. "
             "Experimentally, results are mixed, so this should be set on a "
             "case-by-case basis."));

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

typedef DenseMap<SUnit *, SmallVector<int, 4>> SUnitsToCandidateSGsMap;

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
  std::optional<unsigned> MaxSize;

  // SchedGroups will only synchronize with other SchedGroups that have the same
  // SyncID.
  int SyncID = 0;

  // SGID is used to map instructions to candidate SchedGroups
  unsigned SGID;

  // Count of the number of created SchedGroups, used to initialize SGID.
  static unsigned NumSchedGroups;

  ScheduleDAGInstrs *DAG;

  const SIInstrInfo *TII;

  // Try to add and edge from SU A to SU B.
  bool tryAddEdge(SUnit *A, SUnit *B);

  // Use SGMask to determine whether we can classify MI as a member of this
  // SchedGroup object.
  bool canAddMI(const MachineInstr &MI) const;

public:
  // Collection of SUnits that are classified as members of this group.
  SmallVector<SUnit *, 32> Collection;

  // Returns true if SU can be added to this SchedGroup.
  bool canAddSU(SUnit &SU) const;

  // Add DAG dependencies from all SUnits in this SchedGroup and this SU. If
  // MakePred is true, SU will be a predecessor of the SUnits in this
  // SchedGroup, otherwise SU will be a successor.
  void link(SUnit &SU, bool MakePred = false);

  // Add DAG dependencies and track which edges are added, and the count of
  // missed edges
  int link(SUnit &SU, bool MakePred,
           std::vector<std::pair<SUnit *, SUnit *>> &AddedEdges);

  // Add DAG dependencies from all SUnits in this SchedGroup and this SU.
  // Use the predicate to determine whether SU should be a predecessor (P =
  // true) or a successor (P = false) of this SchedGroup.
  void link(SUnit &SU, function_ref<bool(const SUnit *A, const SUnit *B)> P);

  // Add DAG dependencies such that SUnits in this group shall be ordered
  // before SUnits in OtherGroup.
  void link(SchedGroup &OtherGroup);

  // Returns true if no more instructions may be added to this group.
  bool isFull() const { return MaxSize && Collection.size() >= *MaxSize; }

  // Add SU to the SchedGroup.
  void add(SUnit &SU) {
    LLVM_DEBUG(dbgs() << "For SchedGroup with mask "
                      << format_hex((int)SGMask, 10, true) << " adding "
                      << *SU.getInstr());
    Collection.push_back(&SU);
  }

  // Remove last element in the SchedGroup
  void pop() { Collection.pop_back(); }

  // Identify and add all relevant SUs from the DAG to this SchedGroup.
  void initSchedGroup();

  // Add instructions to the SchedGroup bottom up starting from RIter.
  // PipelineInstrs is a set of instructions that should not be added to the
  // SchedGroup even when the other conditions for adding it are satisfied.
  // RIter will be added to the SchedGroup as well, and dependencies will be
  // added so that RIter will always be scheduled at the end of the group.
  void initSchedGroup(std::vector<SUnit>::reverse_iterator RIter,
                      SUnitsToCandidateSGsMap &SyncedInstrs);

  void initSchedGroup(SUnitsToCandidateSGsMap &SyncedInstrs);

  int getSyncID() { return SyncID; }

  int getSGID() { return SGID; }

  SchedGroupMask getMask() { return SGMask; }

  SchedGroup(SchedGroupMask SGMask, std::optional<unsigned> MaxSize,
             ScheduleDAGInstrs *DAG, const SIInstrInfo *TII)
      : SGMask(SGMask), MaxSize(MaxSize), DAG(DAG), TII(TII) {
    SGID = NumSchedGroups++;
  }

  SchedGroup(SchedGroupMask SGMask, std::optional<unsigned> MaxSize, int SyncID,
             ScheduleDAGInstrs *DAG, const SIInstrInfo *TII)
      : SGMask(SGMask), MaxSize(MaxSize), SyncID(SyncID), DAG(DAG), TII(TII) {
    SGID = NumSchedGroups++;
  }
};

// Remove all existing edges from a SCHED_BARRIER or SCHED_GROUP_BARRIER.
static void resetEdges(SUnit &SU, ScheduleDAGInstrs *DAG) {
  assert(SU.getInstr()->getOpcode() == AMDGPU::SCHED_BARRIER ||
         SU.getInstr()->getOpcode() == AMDGPU::SCHED_GROUP_BARRIER ||
         SU.getInstr()->getOpcode() == AMDGPU::IGLP_OPT);

  while (!SU.Preds.empty())
    for (auto &P : SU.Preds)
      SU.removePred(P);

  while (!SU.Succs.empty())
    for (auto &S : SU.Succs)
      for (auto &SP : S.getSUnit()->Preds)
        if (SP.getSUnit() == &SU)
          S.getSUnit()->removePred(SP);
}

typedef std::pair<SUnit *, SmallVector<int, 4>> SUToCandSGsPair;
typedef SmallVector<SUToCandSGsPair, 4> SUsToCandSGsVec;

// The PipelineSolver is used to assign SUnits to SchedGroups in a pipeline
// in non-trivial cases. For example, if the requested pipeline is
// {VMEM_READ, VALU, MFMA, VMEM_READ} and we encounter a VMEM_READ instruction
// in the DAG, then we will have an instruction that can not be trivially
// assigned to a SchedGroup. The PipelineSolver class implements two algorithms
// to find a good solution to the pipeline -- a greedy algorithm and an exact
// algorithm. The exact algorithm has an exponential time complexity and should
// only be used for small sized problems or medium sized problems where an exact
// solution is highly desired.
class PipelineSolver {
  ScheduleDAGMI *DAG;

  // Instructions that can be assigned to multiple SchedGroups
  DenseMap<int, SUnitsToCandidateSGsMap> SyncedInstrs;
  SmallVector<SUsToCandSGsVec, 4> PipelineInstrs;
  DenseMap<int, SmallVector<SchedGroup, 4>> SyncedSchedGroups;
  // The current working pipeline
  SmallVector<SmallVector<SchedGroup, 4>, 4> CurrPipeline;
  // The pipeline that has the best solution found so far
  SmallVector<SmallVector<SchedGroup, 4>, 4> BestPipeline;

  // Whether or not we actually have any SyncedInstrs to try to solve.
  bool NeedsSolver = false;

  // Compute an estimate of the size of search tree -- the true size is
  // the product of each conflictedInst.Matches.size() across all SyncPipelines
  unsigned computeProblemSize();

  // The cost penalty of not assigning a SU to a SchedGroup
  int MissPenalty = 0;

  // Costs in terms of the number of edges we are unable to add
  int BestCost = -1;
  int CurrCost = 0;

  // Index pointing to the conflicting instruction that is currently being
  // fitted
  int CurrConflInstNo = 0;
  // Index to the pipeline that is currently being fitted
  int CurrSyncGroupIdx = 0;
  // The first non trivial pipeline
  int BeginSyncGroupIdx = 0;

  // How many branches we have explored
  uint64_t BranchesExplored = 0;

  // Update indices to fit next conflicting instruction
  void advancePosition();
  // Recede indices to attempt to find better fit for previous conflicting
  // instruction
  void retreatPosition();

  // The exponential time algorithm which finds the provably best fit
  bool solveExact();
  // The polynomial time algorithm which attempts to find a good fit
  bool solveGreedy();
  // Whether or not the current solution is optimal
  bool checkOptimal();
  // Populate the ready list, prioiritizing fewest missed edges first
  void populateReadyList(SUToCandSGsPair &CurrSU,
                         SmallVectorImpl<std::pair<int, int>> &ReadyList,
                         SmallVectorImpl<SchedGroup> &SyncPipeline);
  // Add edges corresponding to the SchedGroups as assigned by solver
  void makePipeline();
  // Add the edges from the SU to the other SchedGroups in pipeline, and
  // return the number of edges missed.
  int addEdges(SmallVectorImpl<SchedGroup> &SyncPipeline, SUnit *SU, int SGID,
               std::vector<std::pair<SUnit *, SUnit *>> &AddedEdges);
  // Remove the edges passed via AddedEdges
  void removeEdges(const std::vector<std::pair<SUnit *, SUnit *>> &AddedEdges);
  // Convert the passed in maps to arrays for bidirectional iterators
  void convertSyncMapsToArrays();

  void reset();

public:
  // Invoke the solver to map instructions to instruction groups. Heuristic &&
  // command-line-option determines to use exact or greedy algorithm.
  void solve();

  PipelineSolver(DenseMap<int, SmallVector<SchedGroup, 4>> &SyncedSchedGroups,
                 DenseMap<int, SUnitsToCandidateSGsMap> &SyncedInstrs,
                 ScheduleDAGMI *DAG)
      : DAG(DAG), SyncedInstrs(SyncedInstrs),
        SyncedSchedGroups(SyncedSchedGroups) {

    for (auto &PipelineInstrs : SyncedInstrs) {
      if (PipelineInstrs.second.size() > 0) {
        NeedsSolver = true;
        break;
      }
    }

    if (!NeedsSolver)
      return;

    convertSyncMapsToArrays();

    CurrPipeline = BestPipeline;

    while (static_cast<size_t>(BeginSyncGroupIdx) < PipelineInstrs.size() &&
           PipelineInstrs[BeginSyncGroupIdx].size() == 0)
      ++BeginSyncGroupIdx;

    if (static_cast<size_t>(BeginSyncGroupIdx) >= PipelineInstrs.size())
      return;
  }
};

void PipelineSolver::reset() {

  for (auto &SyncPipeline : CurrPipeline) {
    for (auto &SG : SyncPipeline) {
      SmallVector<SUnit *, 32> TempCollection = SG.Collection;
      SG.Collection.clear();
      auto SchedBarr = llvm::find_if(TempCollection, [](SUnit *SU) {
        return SU->getInstr()->getOpcode() == AMDGPU::SCHED_GROUP_BARRIER;
      });
      if (SchedBarr != TempCollection.end())
        SG.Collection.push_back(*SchedBarr);
    }
  }

  CurrSyncGroupIdx = BeginSyncGroupIdx;
  CurrConflInstNo = 0;
  CurrCost = 0;
}

void PipelineSolver::convertSyncMapsToArrays() {
  for (auto &SyncPipe : SyncedSchedGroups) {
    BestPipeline.insert(BestPipeline.begin(), SyncPipe.second);
  }

  int PipelineIDx = SyncedInstrs.size() - 1;
  PipelineInstrs.resize(SyncedInstrs.size());
  for (auto &SyncInstrMap : SyncedInstrs) {
    for (auto &SUsToCandSGs : SyncInstrMap.second) {
      if (PipelineInstrs[PipelineIDx].size() == 0) {
        PipelineInstrs[PipelineIDx].push_back(
            std::pair(SUsToCandSGs.first, SUsToCandSGs.second));
        continue;
      }
      auto SortPosition = PipelineInstrs[PipelineIDx].begin();
      // Insert them in sorted order -- this allows for good parsing order in
      // the greedy algorithm
      while (SortPosition != PipelineInstrs[PipelineIDx].end() &&
             SUsToCandSGs.first->NodeNum > SortPosition->first->NodeNum)
        ++SortPosition;
      PipelineInstrs[PipelineIDx].insert(
          SortPosition, std::pair(SUsToCandSGs.first, SUsToCandSGs.second));
    }
    --PipelineIDx;
  }
}

void PipelineSolver::makePipeline() {
  // Preserve the order of barrier for subsequent SchedGroupBarrier mutations
  for (auto &SyncPipeline : BestPipeline) {
    for (auto &SG : SyncPipeline) {
      SUnit *SGBarr = nullptr;
      for (auto &SU : SG.Collection) {
        if (SU->getInstr()->getOpcode() == AMDGPU::SCHED_GROUP_BARRIER)
          SGBarr = SU;
      }
      // Command line requested IGroupLP doesn't have SGBarr
      if (!SGBarr)
        continue;
      resetEdges(*SGBarr, DAG);
      SG.link(*SGBarr, false);
    }
  }

  for (auto &SyncPipeline : BestPipeline) {
    auto I = SyncPipeline.rbegin();
    auto E = SyncPipeline.rend();
    for (; I != E; ++I) {
      auto &GroupA = *I;
      for (auto J = std::next(I); J != E; ++J) {
        auto &GroupB = *J;
        GroupA.link(GroupB);
      }
    }
  }
}

int PipelineSolver::addEdges(
    SmallVectorImpl<SchedGroup> &SyncPipeline, SUnit *SU, int SGID,
    std::vector<std::pair<SUnit *, SUnit *>> &AddedEdges) {
  int AddedCost = 0;
  bool MakePred = false;

  // The groups in the pipeline are in reverse order. Thus,
  // by traversing them from last to first, we are traversing
  // them in the order as they were introduced in the code. After we
  // pass the group the SU is being assigned to, it should be
  // linked as a predecessor of the subsequent SchedGroups
  auto GroupNo = (int)SyncPipeline.size() - 1;
  for (; GroupNo >= 0; GroupNo--) {
    if (SyncPipeline[GroupNo].getSGID() == SGID) {
      MakePred = true;
      continue;
    }
    auto Group = &SyncPipeline[GroupNo];
    AddedCost += Group->link(*SU, MakePred, AddedEdges);
    assert(AddedCost >= 0);
  }

  return AddedCost;
}

void PipelineSolver::removeEdges(
    const std::vector<std::pair<SUnit *, SUnit *>> &EdgesToRemove) {
  // Only remove the edges that we have added when testing
  // the fit.
  for (auto &PredSuccPair : EdgesToRemove) {
    SUnit *Pred = PredSuccPair.first;
    SUnit *Succ = PredSuccPair.second;

    auto Match = llvm::find_if(
        Succ->Preds, [&Pred](SDep &P) { return P.getSUnit() == Pred; });
    if (Match != Succ->Preds.end()) {
      assert(Match->isArtificial());
      Succ->removePred(*Match);
    }
  }
}

void PipelineSolver::advancePosition() {
  ++CurrConflInstNo;

  if (static_cast<size_t>(CurrConflInstNo) >=
      PipelineInstrs[CurrSyncGroupIdx].size()) {
    CurrConflInstNo = 0;
    ++CurrSyncGroupIdx;
    // Advance to next non-trivial pipeline
    while (static_cast<size_t>(CurrSyncGroupIdx) < PipelineInstrs.size() &&
           PipelineInstrs[CurrSyncGroupIdx].size() == 0)
      ++CurrSyncGroupIdx;
  }
}

void PipelineSolver::retreatPosition() {
  assert(CurrConflInstNo >= 0);
  assert(CurrSyncGroupIdx >= 0);

  if (CurrConflInstNo > 0) {
    --CurrConflInstNo;
    return;
  }

  if (CurrConflInstNo == 0) {
    // If we return to the starting position, we have explored
    // the entire tree
    if (CurrSyncGroupIdx == BeginSyncGroupIdx)
      return;

    --CurrSyncGroupIdx;
    // Go to previous non-trivial pipeline
    while (PipelineInstrs[CurrSyncGroupIdx].size() == 0)
      --CurrSyncGroupIdx;

    CurrConflInstNo = PipelineInstrs[CurrSyncGroupIdx].size() - 1;
  }
}

bool PipelineSolver::checkOptimal() {
  if (static_cast<size_t>(CurrSyncGroupIdx) == PipelineInstrs.size()) {
    if (BestCost == -1 || CurrCost < BestCost) {
      BestPipeline = CurrPipeline;
      BestCost = CurrCost;
      LLVM_DEBUG(dbgs() << "Found Fit with cost " << BestCost << "\n");
    }
    assert(BestCost >= 0);
  }

  bool DoneExploring = false;
  if (MaxBranchesExplored > 0 && BranchesExplored >= MaxBranchesExplored)
    DoneExploring = true;

  return (DoneExploring || BestCost == 0);
}

void PipelineSolver::populateReadyList(
    SUToCandSGsPair &CurrSU, SmallVectorImpl<std::pair<int, int>> &ReadyList,
    SmallVectorImpl<SchedGroup> &SyncPipeline) {
  assert(CurrSU.second.size() >= 1);
  auto I = CurrSU.second.rbegin();
  auto E = CurrSU.second.rend();
  for (; I != E; ++I) {
    std::vector<std::pair<SUnit *, SUnit *>> AddedEdges;
    int CandSGID = *I;
    SchedGroup *Match;
    for (auto &SG : SyncPipeline) {
      if (SG.getSGID() == CandSGID)
        Match = &SG;
    }

    if (UseCostHeur) {
      if (Match->isFull()) {
        ReadyList.push_back(std::pair(*I, MissPenalty));
        continue;
      }

      int TempCost = addEdges(SyncPipeline, CurrSU.first, CandSGID, AddedEdges);
      ReadyList.push_back(std::pair(*I, TempCost));
      removeEdges(AddedEdges);
    } else
      ReadyList.push_back(std::pair(*I, -1));
  }

  if (UseCostHeur) {
    std::sort(ReadyList.begin(), ReadyList.end(),
              [](std::pair<int, int> A, std::pair<int, int> B) {
                return A.second < B.second;
              });
  }

  assert(ReadyList.size() == CurrSU.second.size());
}

bool PipelineSolver::solveExact() {
  if (checkOptimal())
    return true;

  if (static_cast<size_t>(CurrSyncGroupIdx) == PipelineInstrs.size())
    return false;

  assert(static_cast<size_t>(CurrSyncGroupIdx) < PipelineInstrs.size());
  assert(static_cast<size_t>(CurrConflInstNo) <
         PipelineInstrs[CurrSyncGroupIdx].size());
  SUToCandSGsPair CurrSU = PipelineInstrs[CurrSyncGroupIdx][CurrConflInstNo];
  LLVM_DEBUG(dbgs() << "Fitting SU(" << CurrSU.first->NodeNum
                    << ") in Pipeline # " << CurrSyncGroupIdx << "\n");

  // SchedGroup -> Cost pairs
  SmallVector<std::pair<int, int>, 4> ReadyList;
  // Prioritize the candidate sched groups in terms of lowest cost first
  populateReadyList(CurrSU, ReadyList, CurrPipeline[CurrSyncGroupIdx]);

  auto I = ReadyList.begin();
  auto E = ReadyList.end();
  for (; I != E; ++I) {
    // If we are trying SGs in least cost order, and the current SG is cost
    // infeasible, then all subsequent SGs will also be cost infeasible, so we
    // can prune.
    if (BestCost != -1 && (CurrCost + I->second > BestCost))
      return false;

    int CandSGID = I->first;
    int AddedCost = 0;
    std::vector<std::pair<SUnit *, SUnit *>> AddedEdges;
    auto &SyncPipeline = CurrPipeline[CurrSyncGroupIdx];
    SchedGroup *Match;
    for (auto &SG : SyncPipeline) {
      if (SG.getSGID() == CandSGID)
        Match = &SG;
    }

    if (Match->isFull())
      continue;

    LLVM_DEBUG(dbgs() << "Assigning to SchedGroup with Mask "
                      << (int)Match->getMask() << "and ID " << CandSGID
                      << "\n");
    Match->add(*CurrSU.first);
    AddedCost = addEdges(SyncPipeline, CurrSU.first, CandSGID, AddedEdges);
    LLVM_DEBUG(dbgs() << "Cost of Assignment: " << AddedCost << "\n");
    CurrCost += AddedCost;
    advancePosition();
    ++BranchesExplored;
    bool FinishedExploring = false;
    // If the Cost after adding edges is greater than a known solution,
    // backtrack
    if (CurrCost < BestCost || BestCost == -1) {
      if (solveExact()) {
        FinishedExploring = BestCost != 0;
        if (!FinishedExploring)
          return true;
      }
    }

    retreatPosition();
    CurrCost -= AddedCost;
    removeEdges(AddedEdges);
    Match->pop();
    CurrPipeline[CurrSyncGroupIdx] = SyncPipeline;
    if (FinishedExploring)
      return true;
  }

  // Try the pipeline where the current instruction is omitted
  // Potentially if we omit a problematic instruction from the pipeline,
  // all the other instructions can nicely fit.
  CurrCost += MissPenalty;
  advancePosition();

  LLVM_DEBUG(dbgs() << "NOT Assigned (" << CurrSU.first->NodeNum << ")\n");

  bool FinishedExploring = false;
  if (CurrCost < BestCost || BestCost == -1) {
    if (solveExact()) {
      bool FinishedExploring = BestCost != 0;
      if (!FinishedExploring)
        return true;
    }
  }

  retreatPosition();
  CurrCost -= MissPenalty;
  return FinishedExploring;
}

bool PipelineSolver::solveGreedy() {
  BestCost = 0;
  std::vector<std::pair<SUnit *, SUnit *>> AddedEdges;

  while (static_cast<size_t>(CurrSyncGroupIdx) < PipelineInstrs.size()) {
    SUToCandSGsPair CurrSU = PipelineInstrs[CurrSyncGroupIdx][CurrConflInstNo];
    int BestNodeCost = -1;
    int TempCost;
    SchedGroup *BestGroup = nullptr;
    int BestGroupID = -1;
    auto &SyncPipeline = CurrPipeline[CurrSyncGroupIdx];
    LLVM_DEBUG(dbgs() << "Fitting SU(" << CurrSU.first->NodeNum
                      << ") in Pipeline # " << CurrSyncGroupIdx << "\n");

    // Since we have added the potential SchedGroups from bottom up, but
    // traversed the DAG from top down, parse over the groups from last to
    // first. If we fail to do this for the greedy algorithm, the solution will
    // likely not be good in more complex cases.
    auto I = CurrSU.second.rbegin();
    auto E = CurrSU.second.rend();
    for (; I != E; ++I) {
      std::vector<std::pair<SUnit *, SUnit *>> AddedEdges;
      int CandSGID = *I;
      SchedGroup *Match;
      for (auto &SG : SyncPipeline) {
        if (SG.getSGID() == CandSGID)
          Match = &SG;
      }

      LLVM_DEBUG(dbgs() << "Trying SGID # " << CandSGID << " with Mask "
                        << (int)Match->getMask() << "\n");

      if (Match->isFull()) {
        LLVM_DEBUG(dbgs() << "SGID # " << CandSGID << " is full\n");
        continue;
      }
      TempCost = addEdges(SyncPipeline, CurrSU.first, CandSGID, AddedEdges);
      LLVM_DEBUG(dbgs() << "Cost of Group " << TempCost << "\n");
      if (TempCost < BestNodeCost || BestNodeCost == -1) {
        BestGroup = Match;
        BestNodeCost = TempCost;
        BestGroupID = CandSGID;
      }
      removeEdges(AddedEdges);
      if (BestNodeCost == 0)
        break;
    }

    if (BestGroupID != -1) {
      BestGroup->add(*CurrSU.first);
      addEdges(SyncPipeline, CurrSU.first, BestGroupID, AddedEdges);
      LLVM_DEBUG(dbgs() << "Best Group has ID: " << BestGroupID << " and Mask"
                        << (int)BestGroup->getMask() << "\n");
      BestCost += TempCost;
    } else
      BestCost += MissPenalty;

    CurrPipeline[CurrSyncGroupIdx] = SyncPipeline;
    advancePosition();
  }
  BestPipeline = CurrPipeline;
  removeEdges(AddedEdges);
  return false;
}

unsigned PipelineSolver::computeProblemSize() {
  unsigned ProblemSize = 0;
  for (auto &PipeConflicts : PipelineInstrs) {
    ProblemSize += PipeConflicts.size();
  }

  return ProblemSize;
}

void PipelineSolver::solve() {
  if (!NeedsSolver)
    return;

  unsigned ProblemSize = computeProblemSize();
  assert(ProblemSize > 0);

  bool BelowCutoff = (CutoffForExact > 0) && ProblemSize <= CutoffForExact;
  MissPenalty = (ProblemSize / 2) + 1;

  LLVM_DEBUG(DAG->dump());
  if (EnableExactSolver || BelowCutoff) {
    LLVM_DEBUG(dbgs() << "Starting Greedy pipeline solver\n");
    solveGreedy();
    reset();
    LLVM_DEBUG(dbgs() << "Greedy produced best cost of " << BestCost << "\n");
    if (BestCost > 0) {
      LLVM_DEBUG(dbgs() << "Starting EXACT pipeline solver\n");
      solveExact();
      LLVM_DEBUG(dbgs() << "Exact produced best cost of " << BestCost << "\n");
    }
  } else { // Use the Greedy Algorithm by default
    LLVM_DEBUG(dbgs() << "Starting GREEDY pipeline solver\n");
    solveGreedy();
  }

  makePipeline();
}

enum IGLPStrategyID : int { MFMASmallGemmOptID = 0 };

// Implement a IGLP scheduling strategy.
class IGLPStrategy {
protected:
  ScheduleDAGInstrs *DAG;

  const SIInstrInfo *TII;

public:
  // Add SchedGroups to \p Pipeline to implement this Strategy.
  virtual void applyIGLPStrategy(
      DenseMap<int, SUnitsToCandidateSGsMap> &SyncedInstrs,
      DenseMap<int, SmallVector<SchedGroup, 4>> &SyncedSchedGroups) = 0;

  // Returns true if this strategy should be applied to a ScheduleDAG.
  virtual bool shouldApplyStrategy(ScheduleDAGInstrs *DAG) = 0;

  IGLPStrategy(ScheduleDAGInstrs *DAG, const SIInstrInfo *TII)
      : DAG(DAG), TII(TII) {}

  virtual ~IGLPStrategy() = default;
};

class MFMASmallGemmOpt final : public IGLPStrategy {
public:
  void applyIGLPStrategy(
      DenseMap<int, SUnitsToCandidateSGsMap> &SyncedInstrs,
      DenseMap<int, SmallVector<SchedGroup, 4>> &SyncedSchedGroups) override;

  bool shouldApplyStrategy(ScheduleDAGInstrs *DAG) override { return true; }

  MFMASmallGemmOpt(ScheduleDAGInstrs *DAG, const SIInstrInfo *TII)
      : IGLPStrategy(DAG, TII) {}
};

void MFMASmallGemmOpt::applyIGLPStrategy(
    DenseMap<int, SUnitsToCandidateSGsMap> &SyncedInstrs,
    DenseMap<int, SmallVector<SchedGroup, 4>> &SyncedSchedGroups) {
  // Count the number of MFMA instructions.
  unsigned MFMACount = 0;
  for (const MachineInstr &I : *DAG)
    if (TII->isMFMAorWMMA(I))
      ++MFMACount;

  const unsigned PipelineSyncID = 0;
  SchedGroup *SG = nullptr;
  for (unsigned I = 0; I < MFMACount * 3; ++I) {
    SG = &SyncedSchedGroups[PipelineSyncID].emplace_back(
        SchedGroupMask::DS, 2, PipelineSyncID, DAG, TII);
    SG->initSchedGroup(SyncedInstrs[SG->getSyncID()]);

    SG = &SyncedSchedGroups[PipelineSyncID].emplace_back(
        SchedGroupMask::MFMA, 1, PipelineSyncID, DAG, TII);
    SG->initSchedGroup(SyncedInstrs[SG->getSyncID()]);
  }
}

static std::unique_ptr<IGLPStrategy>
createIGLPStrategy(IGLPStrategyID ID, ScheduleDAGInstrs *DAG,
                   const SIInstrInfo *TII) {
  switch (ID) {
  case MFMASmallGemmOptID:
    return std::make_unique<MFMASmallGemmOpt>(DAG, TII);
  }

  llvm_unreachable("Unknown IGLPStrategyID");
}

class IGroupLPDAGMutation : public ScheduleDAGMutation {
private:
  const SIInstrInfo *TII;

  ScheduleDAGMI *DAG;

  // Organize lists of SchedGroups by their SyncID. SchedGroups /
  // SCHED_GROUP_BARRIERs with different SyncIDs will have no edges added
  // between then.
  DenseMap<int, SmallVector<SchedGroup, 4>> SyncedSchedGroups;

  // Used to track instructions that can be mapped to multiple sched groups
  DenseMap<int, SUnitsToCandidateSGsMap> SyncedInstrs;

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
  void initSchedGroupBarrierPipelineStage(
      std::vector<SUnit>::reverse_iterator RIter);

  void initIGLPOpt(SUnit &SU);

public:
  void apply(ScheduleDAGInstrs *DAGInstrs) override;

  IGroupLPDAGMutation() = default;
};

unsigned SchedGroup::NumSchedGroups = 0;

bool SchedGroup::tryAddEdge(SUnit *A, SUnit *B) {
  if (A != B && DAG->canAddEdge(B, A)) {
    DAG->addEdge(B, SDep(A, SDep::Artificial));
    return true;
  }
  return false;
}

bool SchedGroup::canAddMI(const MachineInstr &MI) const {
  bool Result = false;
  if (MI.isMetaInstruction())
    Result = false;

  else if (((SGMask & SchedGroupMask::ALU) != SchedGroupMask::NONE) &&
           (TII->isVALU(MI) || TII->isMFMAorWMMA(MI) || TII->isSALU(MI)))
    Result = true;

  else if (((SGMask & SchedGroupMask::VALU) != SchedGroupMask::NONE) &&
           TII->isVALU(MI) && !TII->isMFMAorWMMA(MI))
    Result = true;

  else if (((SGMask & SchedGroupMask::SALU) != SchedGroupMask::NONE) &&
           TII->isSALU(MI))
    Result = true;

  else if (((SGMask & SchedGroupMask::MFMA) != SchedGroupMask::NONE) &&
           TII->isMFMAorWMMA(MI))
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

int SchedGroup::link(SUnit &SU, bool MakePred,
                     std::vector<std::pair<SUnit *, SUnit *>> &AddedEdges) {
  int MissedEdges = 0;
  for (auto *A : Collection) {
    SUnit *B = &SU;
    if (A == B || A->getInstr()->getOpcode() == AMDGPU::SCHED_GROUP_BARRIER)
      continue;
    if (MakePred)
      std::swap(A, B);

    if (DAG->IsReachable(B, A))
      continue;
    // tryAddEdge returns false if there is a dependency that makes adding
    // the A->B edge impossible, otherwise it returns true;
    bool Added = tryAddEdge(A, B);
    if (Added)
      AddedEdges.push_back(std::pair(A, B));
    else
      ++MissedEdges;
  }

  return MissedEdges;
}

void SchedGroup::link(SUnit &SU, bool MakePred) {
  for (auto *A : Collection) {
    SUnit *B = &SU;
    if (A->getInstr()->getOpcode() == AMDGPU::SCHED_GROUP_BARRIER)
      continue;
    if (MakePred)
      std::swap(A, B);

    tryAddEdge(A, B);
  }
}

void SchedGroup::link(SUnit &SU,
                      function_ref<bool(const SUnit *A, const SUnit *B)> P) {
  for (auto *A : Collection) {
    SUnit *B = &SU;
    if (P(A, B))
      std::swap(A, B);

    tryAddEdge(A, B);
  }
}

void SchedGroup::link(SchedGroup &OtherGroup) {
  for (auto *B : OtherGroup.Collection)
    link(*B);
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

void SchedGroup::initSchedGroup(std::vector<SUnit>::reverse_iterator RIter,
                                SUnitsToCandidateSGsMap &SyncedInstrs) {
  SUnit &InitSU = *RIter;
  for (auto E = DAG->SUnits.rend(); RIter != E; ++RIter) {
    auto &SU = *RIter;
    if (isFull())
      break;

    if (canAddSU(SU))
      SyncedInstrs[&SU].push_back(SGID);
  }

  add(InitSU);
  assert(MaxSize);
  (*MaxSize)++;
}

void SchedGroup::initSchedGroup(SUnitsToCandidateSGsMap &SyncedInstrs) {
  auto I = DAG->SUnits.rbegin();
  auto E = DAG->SUnits.rend();
  for (; I != E; ++I) {
    auto &SU = *I;
    if (isFull())
      break;

    if (canAddSU(SU))
      SyncedInstrs[&SU].push_back(SGID);
  }
}

void IGroupLPDAGMutation::apply(ScheduleDAGInstrs *DAGInstrs) {
  const TargetSchedModel *TSchedModel = DAGInstrs->getSchedModel();
  if (!TSchedModel || DAGInstrs->SUnits.empty())
    return;

  LLVM_DEBUG(dbgs() << "Applying IGroupLPDAGMutation...\n");
  const GCNSubtarget &ST = DAGInstrs->MF.getSubtarget<GCNSubtarget>();
  TII = ST.getInstrInfo();
  DAG = static_cast<ScheduleDAGMI *>(DAGInstrs);
  SyncedSchedGroups.clear();
  SyncedInstrs.clear();
  bool foundSB = false;
  bool foundIGLP = false;
  for (auto R = DAG->SUnits.rbegin(), E = DAG->SUnits.rend(); R != E; ++R) {
    unsigned Opc = R->getInstr()->getOpcode();
    // SCHED_[GROUP_]BARRIER and IGLP are mutually exclusive.
    if (Opc == AMDGPU::SCHED_BARRIER) {
      addSchedBarrierEdges(*R);
      foundSB = true;
    } else if (Opc == AMDGPU::SCHED_GROUP_BARRIER) {
      initSchedGroupBarrierPipelineStage(R);
      foundSB = true;
    } else if (Opc == AMDGPU::IGLP_OPT) {
      resetEdges(*R, DAG);
      if (!foundSB && !foundIGLP)
        initIGLPOpt(*R);
      foundIGLP = true;
    }
  }

  if (foundSB || foundIGLP) {
    PipelineSolver PS(SyncedSchedGroups, SyncedInstrs, DAG);
    // PipelineSolver performs the mutation by adding the edges it
    // determined as the best
    PS.solve();
  }
}

void IGroupLPDAGMutation::addSchedBarrierEdges(SUnit &SchedBarrier) {
  MachineInstr &MI = *SchedBarrier.getInstr();
  assert(MI.getOpcode() == AMDGPU::SCHED_BARRIER);
  // Remove all existing edges from the SCHED_BARRIER that were added due to the
  // instruction having side effects.
  resetEdges(SchedBarrier, DAG);
  auto InvertedMask =
      invertSchedBarrierMask((SchedGroupMask)MI.getOperand(0).getImm());
  SchedGroup SG(InvertedMask, std::nullopt, DAG, TII);
  SG.initSchedGroup();
  // Preserve original instruction ordering relative to the SCHED_BARRIER.
  SG.link(
      SchedBarrier,
      (function_ref<bool(const SUnit *A, const SUnit *B)>)[](
          const SUnit *A, const SUnit *B) { return A->NodeNum > B->NodeNum; });
}

SchedGroupMask
IGroupLPDAGMutation::invertSchedBarrierMask(SchedGroupMask Mask) const {
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

void IGroupLPDAGMutation::initSchedGroupBarrierPipelineStage(
    std::vector<SUnit>::reverse_iterator RIter) {
  // Remove all existing edges from the SCHED_GROUP_BARRIER that were added due
  // to the instruction having side effects.
  resetEdges(*RIter, DAG);
  MachineInstr &SGB = *RIter->getInstr();
  assert(SGB.getOpcode() == AMDGPU::SCHED_GROUP_BARRIER);
  int32_t SGMask = SGB.getOperand(0).getImm();
  int32_t Size = SGB.getOperand(1).getImm();
  int32_t SyncID = SGB.getOperand(2).getImm();

  auto &SG = SyncedSchedGroups[SyncID].emplace_back((SchedGroupMask)SGMask,
                                                    Size, SyncID, DAG, TII);

  SG.initSchedGroup(RIter, SyncedInstrs[SG.getSyncID()]);
}

void IGroupLPDAGMutation::initIGLPOpt(SUnit &SU) {
  IGLPStrategyID StrategyID =
      (IGLPStrategyID)SU.getInstr()->getOperand(0).getImm();
  auto S = createIGLPStrategy(StrategyID, DAG, TII);
  if (S->shouldApplyStrategy(DAG))
    S->applyIGLPStrategy(SyncedInstrs, SyncedSchedGroups);
}

} // namespace

namespace llvm {

std::unique_ptr<ScheduleDAGMutation> createIGroupLPDAGMutation() {
  return std::make_unique<IGroupLPDAGMutation>();
}

} // end namespace llvm
