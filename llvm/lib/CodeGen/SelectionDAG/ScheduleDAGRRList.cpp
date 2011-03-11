//===----- ScheduleDAGRRList.cpp - Reg pressure reduction list scheduler --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements bottom-up and top-down register pressure reduction list
// schedulers, using standard algorithms.  The basic approach uses a priority
// queue of available nodes to schedule.  One at a time, nodes are taken from
// the priority queue (thus in priority order), checked for legality to
// schedule, and emitted if legal.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "pre-RA-sched"
#include "ScheduleDAGSDNodes.h"
#include "llvm/InlineAsm.h"
#include "llvm/CodeGen/SchedulerRegistry.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/ScheduleHazardRecognizer.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <climits>
using namespace llvm;

STATISTIC(NumBacktracks, "Number of times scheduler backtracked");
STATISTIC(NumUnfolds,    "Number of nodes unfolded");
STATISTIC(NumDups,       "Number of duplicated nodes");
STATISTIC(NumPRCopies,   "Number of physical register copies");

static RegisterScheduler
  burrListDAGScheduler("list-burr",
                       "Bottom-up register reduction list scheduling",
                       createBURRListDAGScheduler);
static RegisterScheduler
  tdrListrDAGScheduler("list-tdrr",
                       "Top-down register reduction list scheduling",
                       createTDRRListDAGScheduler);
static RegisterScheduler
  sourceListDAGScheduler("source",
                         "Similar to list-burr but schedules in source "
                         "order when possible",
                         createSourceListDAGScheduler);

static RegisterScheduler
  hybridListDAGScheduler("list-hybrid",
                         "Bottom-up register pressure aware list scheduling "
                         "which tries to balance latency and register pressure",
                         createHybridListDAGScheduler);

static RegisterScheduler
  ILPListDAGScheduler("list-ilp",
                      "Bottom-up register pressure aware list scheduling "
                      "which tries to balance ILP and register pressure",
                      createILPListDAGScheduler);

static cl::opt<bool> DisableSchedCycles(
  "disable-sched-cycles", cl::Hidden, cl::init(false),
  cl::desc("Disable cycle-level precision during preRA scheduling"));

// Temporary sched=list-ilp flags until the heuristics are robust.
static cl::opt<bool> DisableSchedRegPressure(
  "disable-sched-reg-pressure", cl::Hidden, cl::init(false),
  cl::desc("Disable regpressure priority in sched=list-ilp"));
static cl::opt<bool> DisableSchedLiveUses(
  "disable-sched-live-uses", cl::Hidden, cl::init(true),
  cl::desc("Disable live use priority in sched=list-ilp"));
static cl::opt<bool> DisableSchedStalls(
  "disable-sched-stalls", cl::Hidden, cl::init(true),
  cl::desc("Disable no-stall priority in sched=list-ilp"));
static cl::opt<bool> DisableSchedCriticalPath(
  "disable-sched-critical-path", cl::Hidden, cl::init(false),
  cl::desc("Disable critical path priority in sched=list-ilp"));
static cl::opt<bool> DisableSchedHeight(
  "disable-sched-height", cl::Hidden, cl::init(false),
  cl::desc("Disable scheduled-height priority in sched=list-ilp"));

static cl::opt<int> MaxReorderWindow(
  "max-sched-reorder", cl::Hidden, cl::init(6),
  cl::desc("Number of instructions to allow ahead of the critical path "
           "in sched=list-ilp"));

static cl::opt<unsigned> AvgIPC(
  "sched-avg-ipc", cl::Hidden, cl::init(1),
  cl::desc("Average inst/cycle whan no target itinerary exists."));

#ifndef NDEBUG
namespace {
  // For sched=list-ilp, Count the number of times each factor comes into play.
  enum { FactPressureDiff, FactRegUses, FactHeight, FactDepth, FactStatic,
         FactOther, NumFactors };
}
static const char *FactorName[NumFactors] =
{"PressureDiff", "RegUses", "Height", "Depth","Static", "Other"};
static int FactorCount[NumFactors];
#endif //!NDEBUG

namespace {
//===----------------------------------------------------------------------===//
/// ScheduleDAGRRList - The actual register reduction list scheduler
/// implementation.  This supports both top-down and bottom-up scheduling.
///
class ScheduleDAGRRList : public ScheduleDAGSDNodes {
private:
  /// isBottomUp - This is true if the scheduling problem is bottom-up, false if
  /// it is top-down.
  bool isBottomUp;

  /// NeedLatency - True if the scheduler will make use of latency information.
  ///
  bool NeedLatency;

  /// AvailableQueue - The priority queue to use for the available SUnits.
  SchedulingPriorityQueue *AvailableQueue;

  /// PendingQueue - This contains all of the instructions whose operands have
  /// been issued, but their results are not ready yet (due to the latency of
  /// the operation).  Once the operands becomes available, the instruction is
  /// added to the AvailableQueue.
  std::vector<SUnit*> PendingQueue;

  /// HazardRec - The hazard recognizer to use.
  ScheduleHazardRecognizer *HazardRec;

  /// CurCycle - The current scheduler state corresponds to this cycle.
  unsigned CurCycle;

  /// MinAvailableCycle - Cycle of the soonest available instruction.
  unsigned MinAvailableCycle;

  /// IssueCount - Count instructions issued in this cycle
  /// Currently valid only for bottom-up scheduling.
  unsigned IssueCount;

  /// LiveRegDefs - A set of physical registers and their definition
  /// that are "live". These nodes must be scheduled before any other nodes that
  /// modifies the registers can be scheduled.
  unsigned NumLiveRegs;
  std::vector<SUnit*> LiveRegDefs;
  std::vector<SUnit*> LiveRegGens;

  /// Topo - A topological ordering for SUnits which permits fast IsReachable
  /// and similar queries.
  ScheduleDAGTopologicalSort Topo;

public:
  ScheduleDAGRRList(MachineFunction &mf, bool needlatency,
                    SchedulingPriorityQueue *availqueue,
                    CodeGenOpt::Level OptLevel)
    : ScheduleDAGSDNodes(mf), isBottomUp(availqueue->isBottomUp()),
      NeedLatency(needlatency), AvailableQueue(availqueue), CurCycle(0),
      Topo(SUnits) {

    const TargetMachine &tm = mf.getTarget();
    if (DisableSchedCycles || !NeedLatency)
      HazardRec = new ScheduleHazardRecognizer();
    else
      HazardRec = tm.getInstrInfo()->CreateTargetHazardRecognizer(&tm, this);
  }

  ~ScheduleDAGRRList() {
    delete HazardRec;
    delete AvailableQueue;
  }

  void Schedule();

  ScheduleHazardRecognizer *getHazardRec() { return HazardRec; }

  /// IsReachable - Checks if SU is reachable from TargetSU.
  bool IsReachable(const SUnit *SU, const SUnit *TargetSU) {
    return Topo.IsReachable(SU, TargetSU);
  }

  /// WillCreateCycle - Returns true if adding an edge from SU to TargetSU will
  /// create a cycle.
  bool WillCreateCycle(SUnit *SU, SUnit *TargetSU) {
    return Topo.WillCreateCycle(SU, TargetSU);
  }

  /// AddPred - adds a predecessor edge to SUnit SU.
  /// This returns true if this is a new predecessor.
  /// Updates the topological ordering if required.
  void AddPred(SUnit *SU, const SDep &D) {
    Topo.AddPred(SU, D.getSUnit());
    SU->addPred(D);
  }

  /// RemovePred - removes a predecessor edge from SUnit SU.
  /// This returns true if an edge was removed.
  /// Updates the topological ordering if required.
  void RemovePred(SUnit *SU, const SDep &D) {
    Topo.RemovePred(SU, D.getSUnit());
    SU->removePred(D);
  }

private:
  bool isReady(SUnit *SU) {
    return DisableSchedCycles || !AvailableQueue->hasReadyFilter() ||
      AvailableQueue->isReady(SU);
  }

  void ReleasePred(SUnit *SU, const SDep *PredEdge);
  void ReleasePredecessors(SUnit *SU);
  void ReleaseSucc(SUnit *SU, const SDep *SuccEdge);
  void ReleaseSuccessors(SUnit *SU);
  void ReleasePending();
  void AdvanceToCycle(unsigned NextCycle);
  void AdvancePastStalls(SUnit *SU);
  void EmitNode(SUnit *SU);
  void ScheduleNodeBottomUp(SUnit*);
  void CapturePred(SDep *PredEdge);
  void UnscheduleNodeBottomUp(SUnit*);
  void RestoreHazardCheckerBottomUp();
  void BacktrackBottomUp(SUnit*, SUnit*);
  SUnit *CopyAndMoveSuccessors(SUnit*);
  void InsertCopiesAndMoveSuccs(SUnit*, unsigned,
                                const TargetRegisterClass*,
                                const TargetRegisterClass*,
                                SmallVector<SUnit*, 2>&);
  bool DelayForLiveRegsBottomUp(SUnit*, SmallVector<unsigned, 4>&);

  SUnit *PickNodeToScheduleBottomUp();
  void ListScheduleBottomUp();

  void ScheduleNodeTopDown(SUnit*);
  void ListScheduleTopDown();


  /// CreateNewSUnit - Creates a new SUnit and returns a pointer to it.
  /// Updates the topological ordering if required.
  SUnit *CreateNewSUnit(SDNode *N) {
    unsigned NumSUnits = SUnits.size();
    SUnit *NewNode = NewSUnit(N);
    // Update the topological ordering.
    if (NewNode->NodeNum >= NumSUnits)
      Topo.InitDAGTopologicalSorting();
    return NewNode;
  }

  /// CreateClone - Creates a new SUnit from an existing one.
  /// Updates the topological ordering if required.
  SUnit *CreateClone(SUnit *N) {
    unsigned NumSUnits = SUnits.size();
    SUnit *NewNode = Clone(N);
    // Update the topological ordering.
    if (NewNode->NodeNum >= NumSUnits)
      Topo.InitDAGTopologicalSorting();
    return NewNode;
  }

  /// ForceUnitLatencies - Register-pressure-reducing scheduling doesn't
  /// need actual latency information but the hybrid scheduler does.
  bool ForceUnitLatencies() const {
    return !NeedLatency;
  }
};
}  // end anonymous namespace


/// Schedule - Schedule the DAG using list scheduling.
void ScheduleDAGRRList::Schedule() {
  DEBUG(dbgs()
        << "********** List Scheduling BB#" << BB->getNumber()
        << " '" << BB->getName() << "' **********\n");
#ifndef NDEBUG
  for (int i = 0; i < NumFactors; ++i) {
    FactorCount[i] = 0;
  }
#endif //!NDEBUG

  CurCycle = 0;
  IssueCount = 0;
  MinAvailableCycle = DisableSchedCycles ? 0 : UINT_MAX;
  NumLiveRegs = 0;
  LiveRegDefs.resize(TRI->getNumRegs(), NULL);
  LiveRegGens.resize(TRI->getNumRegs(), NULL);

  // Build the scheduling graph.
  BuildSchedGraph(NULL);

  DEBUG(for (unsigned su = 0, e = SUnits.size(); su != e; ++su)
          SUnits[su].dumpAll(this));
  Topo.InitDAGTopologicalSorting();

  AvailableQueue->initNodes(SUnits);

  HazardRec->Reset();

  // Execute the actual scheduling loop Top-Down or Bottom-Up as appropriate.
  if (isBottomUp)
    ListScheduleBottomUp();
  else
    ListScheduleTopDown();

#ifndef NDEBUG
  for (int i = 0; i < NumFactors; ++i) {
    DEBUG(dbgs() << FactorName[i] << "\t" << FactorCount[i] << "\n");
  }
#endif // !NDEBUG
  AvailableQueue->releaseState();
}

//===----------------------------------------------------------------------===//
//  Bottom-Up Scheduling
//===----------------------------------------------------------------------===//

/// ReleasePred - Decrement the NumSuccsLeft count of a predecessor. Add it to
/// the AvailableQueue if the count reaches zero. Also update its cycle bound.
void ScheduleDAGRRList::ReleasePred(SUnit *SU, const SDep *PredEdge) {
  SUnit *PredSU = PredEdge->getSUnit();

#ifndef NDEBUG
  if (PredSU->NumSuccsLeft == 0) {
    dbgs() << "*** Scheduling failed! ***\n";
    PredSU->dump(this);
    dbgs() << " has been released too many times!\n";
    llvm_unreachable(0);
  }
#endif
  --PredSU->NumSuccsLeft;

  if (!ForceUnitLatencies()) {
    // Updating predecessor's height. This is now the cycle when the
    // predecessor can be scheduled without causing a pipeline stall.
    PredSU->setHeightToAtLeast(SU->getHeight() + PredEdge->getLatency());
  }

  // If all the node's successors are scheduled, this node is ready
  // to be scheduled. Ignore the special EntrySU node.
  if (PredSU->NumSuccsLeft == 0 && PredSU != &EntrySU) {
    PredSU->isAvailable = true;

    unsigned Height = PredSU->getHeight();
    if (Height < MinAvailableCycle)
      MinAvailableCycle = Height;

    if (isReady(PredSU)) {
      AvailableQueue->push(PredSU);
    }
    // CapturePred and others may have left the node in the pending queue, avoid
    // adding it twice.
    else if (!PredSU->isPending) {
      PredSU->isPending = true;
      PendingQueue.push_back(PredSU);
    }
  }
}

/// Call ReleasePred for each predecessor, then update register live def/gen.
/// Always update LiveRegDefs for a register dependence even if the current SU
/// also defines the register. This effectively create one large live range
/// across a sequence of two-address node. This is important because the
/// entire chain must be scheduled together. Example:
///
/// flags = (3) add
/// flags = (2) addc flags
/// flags = (1) addc flags
///
/// results in
///
/// LiveRegDefs[flags] = 3
/// LiveRegGens[flags] = 1
///
/// If (2) addc is unscheduled, then (1) addc must also be unscheduled to avoid
/// interference on flags.
void ScheduleDAGRRList::ReleasePredecessors(SUnit *SU) {
  // Bottom up: release predecessors
  for (SUnit::pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
       I != E; ++I) {
    ReleasePred(SU, &*I);
    if (I->isAssignedRegDep()) {
      // This is a physical register dependency and it's impossible or
      // expensive to copy the register. Make sure nothing that can
      // clobber the register is scheduled between the predecessor and
      // this node.
      SUnit *RegDef = LiveRegDefs[I->getReg()]; (void)RegDef;
      assert((!RegDef || RegDef == SU || RegDef == I->getSUnit()) &&
             "interference on register dependence");
      LiveRegDefs[I->getReg()] = I->getSUnit();
      if (!LiveRegGens[I->getReg()]) {
        ++NumLiveRegs;
        LiveRegGens[I->getReg()] = SU;
      }
    }
  }
}

/// Check to see if any of the pending instructions are ready to issue.  If
/// so, add them to the available queue.
void ScheduleDAGRRList::ReleasePending() {
  if (DisableSchedCycles) {
    assert(PendingQueue.empty() && "pending instrs not allowed in this mode");
    return;
  }

  // If the available queue is empty, it is safe to reset MinAvailableCycle.
  if (AvailableQueue->empty())
    MinAvailableCycle = UINT_MAX;

  // Check to see if any of the pending instructions are ready to issue.  If
  // so, add them to the available queue.
  for (unsigned i = 0, e = PendingQueue.size(); i != e; ++i) {
    unsigned ReadyCycle =
      isBottomUp ? PendingQueue[i]->getHeight() : PendingQueue[i]->getDepth();
    if (ReadyCycle < MinAvailableCycle)
      MinAvailableCycle = ReadyCycle;

    if (PendingQueue[i]->isAvailable) {
      if (!isReady(PendingQueue[i]))
          continue;
      AvailableQueue->push(PendingQueue[i]);
    }
    PendingQueue[i]->isPending = false;
    PendingQueue[i] = PendingQueue.back();
    PendingQueue.pop_back();
    --i; --e;
  }
}

/// Move the scheduler state forward by the specified number of Cycles.
void ScheduleDAGRRList::AdvanceToCycle(unsigned NextCycle) {
  if (NextCycle <= CurCycle)
    return;

  IssueCount = 0;
  AvailableQueue->setCurCycle(NextCycle);
  if (!HazardRec->isEnabled()) {
    // Bypass lots of virtual calls in case of long latency.
    CurCycle = NextCycle;
  }
  else {
    for (; CurCycle != NextCycle; ++CurCycle) {
      if (isBottomUp)
        HazardRec->RecedeCycle();
      else
        HazardRec->AdvanceCycle();
    }
  }
  // FIXME: Instead of visiting the pending Q each time, set a dirty flag on the
  // available Q to release pending nodes at least once before popping.
  ReleasePending();
}

/// Move the scheduler state forward until the specified node's dependents are
/// ready and can be scheduled with no resource conflicts.
void ScheduleDAGRRList::AdvancePastStalls(SUnit *SU) {
  if (DisableSchedCycles)
    return;

  unsigned ReadyCycle = isBottomUp ? SU->getHeight() : SU->getDepth();

  // Bump CurCycle to account for latency. We assume the latency of other
  // available instructions may be hidden by the stall (not a full pipe stall).
  // This updates the hazard recognizer's cycle before reserving resources for
  // this instruction.
  AdvanceToCycle(ReadyCycle);

  // Calls are scheduled in their preceding cycle, so don't conflict with
  // hazards from instructions after the call. EmitNode will reset the
  // scoreboard state before emitting the call.
  if (isBottomUp && SU->isCall)
    return;

  // FIXME: For resource conflicts in very long non-pipelined stages, we
  // should probably skip ahead here to avoid useless scoreboard checks.
  int Stalls = 0;
  while (true) {
    ScheduleHazardRecognizer::HazardType HT =
      HazardRec->getHazardType(SU, isBottomUp ? -Stalls : Stalls);

    if (HT == ScheduleHazardRecognizer::NoHazard)
      break;

    ++Stalls;
  }
  AdvanceToCycle(CurCycle + Stalls);
}

/// Record this SUnit in the HazardRecognizer.
/// Does not update CurCycle.
void ScheduleDAGRRList::EmitNode(SUnit *SU) {
  if (!HazardRec->isEnabled())
    return;

  // Check for phys reg copy.
  if (!SU->getNode())
    return;

  switch (SU->getNode()->getOpcode()) {
  default:
    assert(SU->getNode()->isMachineOpcode() &&
           "This target-independent node should not be scheduled.");
    break;
  case ISD::MERGE_VALUES:
  case ISD::TokenFactor:
  case ISD::CopyToReg:
  case ISD::CopyFromReg:
  case ISD::EH_LABEL:
    // Noops don't affect the scoreboard state. Copies are likely to be
    // removed.
    return;
  case ISD::INLINEASM:
    // For inline asm, clear the pipeline state.
    HazardRec->Reset();
    return;
  }
  if (isBottomUp && SU->isCall) {
    // Calls are scheduled with their preceding instructions. For bottom-up
    // scheduling, clear the pipeline state before emitting.
    HazardRec->Reset();
  }

  HazardRec->EmitInstruction(SU);

  if (!isBottomUp && SU->isCall) {
    HazardRec->Reset();
  }
}

/// ScheduleNodeBottomUp - Add the node to the schedule. Decrement the pending
/// count of its predecessors. If a predecessor pending count is zero, add it to
/// the Available queue.
void ScheduleDAGRRList::ScheduleNodeBottomUp(SUnit *SU) {
  DEBUG(dbgs() << "\n*** Scheduling [" << CurCycle << "]: ");
  DEBUG(SU->dump(this));

#ifndef NDEBUG
  if (CurCycle < SU->getHeight())
    DEBUG(dbgs() << "   Height [" << SU->getHeight() << "] pipeline stall!\n");
#endif

  // FIXME: Do not modify node height. It may interfere with
  // backtracking. Instead add a "ready cycle" to SUnit. Before scheduling the
  // node it's ready cycle can aid heuristics, and after scheduling it can
  // indicate the scheduled cycle.
  SU->setHeightToAtLeast(CurCycle);

  // Reserve resources for the scheduled intruction.
  EmitNode(SU);

  Sequence.push_back(SU);

  AvailableQueue->ScheduledNode(SU);

  // If HazardRec is disabled, and each inst counts as one cycle, then
  // advance CurCycle before ReleasePredecessors to avoid useles pushed to
  // PendingQueue for schedulers that implement HasReadyFilter.
  if (!HazardRec->isEnabled() && AvgIPC < 2)
    AdvanceToCycle(CurCycle + 1);

  // Update liveness of predecessors before successors to avoid treating a
  // two-address node as a live range def.
  ReleasePredecessors(SU);

  // Release all the implicit physical register defs that are live.
  for (SUnit::succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I) {
    // LiveRegDegs[I->getReg()] != SU when SU is a two-address node.
    if (I->isAssignedRegDep() && LiveRegDefs[I->getReg()] == SU) {
      assert(NumLiveRegs > 0 && "NumLiveRegs is already zero!");
      --NumLiveRegs;
      LiveRegDefs[I->getReg()] = NULL;
      LiveRegGens[I->getReg()] = NULL;
    }
  }

  SU->isScheduled = true;

  // Conditions under which the scheduler should eagerly advance the cycle:
  // (1) No available instructions
  // (2) All pipelines full, so available instructions must have hazards.
  //
  // If HazardRec is disabled, the cycle was advanced earlier.
  //
  // Check AvailableQueue after ReleasePredecessors in case of zero latency.
  ++IssueCount;
  if ((HazardRec->isEnabled() && HazardRec->atIssueLimit())
      || (!HazardRec->isEnabled() && AvgIPC > 1 && IssueCount == AvgIPC)
      || AvailableQueue->empty())
    AdvanceToCycle(CurCycle + 1);
}

/// CapturePred - This does the opposite of ReleasePred. Since SU is being
/// unscheduled, incrcease the succ left count of its predecessors. Remove
/// them from AvailableQueue if necessary.
void ScheduleDAGRRList::CapturePred(SDep *PredEdge) {
  SUnit *PredSU = PredEdge->getSUnit();
  if (PredSU->isAvailable) {
    PredSU->isAvailable = false;
    if (!PredSU->isPending)
      AvailableQueue->remove(PredSU);
  }

  assert(PredSU->NumSuccsLeft < UINT_MAX && "NumSuccsLeft will overflow!");
  ++PredSU->NumSuccsLeft;
}

/// UnscheduleNodeBottomUp - Remove the node from the schedule, update its and
/// its predecessor states to reflect the change.
void ScheduleDAGRRList::UnscheduleNodeBottomUp(SUnit *SU) {
  DEBUG(dbgs() << "*** Unscheduling [" << SU->getHeight() << "]: ");
  DEBUG(SU->dump(this));

  for (SUnit::pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
       I != E; ++I) {
    CapturePred(&*I);
    if (I->isAssignedRegDep() && SU == LiveRegGens[I->getReg()]){
      assert(NumLiveRegs > 0 && "NumLiveRegs is already zero!");
      assert(LiveRegDefs[I->getReg()] == I->getSUnit() &&
             "Physical register dependency violated?");
      --NumLiveRegs;
      LiveRegDefs[I->getReg()] = NULL;
      LiveRegGens[I->getReg()] = NULL;
    }
  }

  for (SUnit::succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I) {
    if (I->isAssignedRegDep()) {
      // This becomes the nearest def. Note that an earlier def may still be
      // pending if this is a two-address node.
      LiveRegDefs[I->getReg()] = SU;
      if (!LiveRegDefs[I->getReg()]) {
        ++NumLiveRegs;
      }
      if (LiveRegGens[I->getReg()] == NULL ||
          I->getSUnit()->getHeight() < LiveRegGens[I->getReg()]->getHeight())
        LiveRegGens[I->getReg()] = I->getSUnit();
    }
  }
  if (SU->getHeight() < MinAvailableCycle)
    MinAvailableCycle = SU->getHeight();

  SU->setHeightDirty();
  SU->isScheduled = false;
  SU->isAvailable = true;
  if (!DisableSchedCycles && AvailableQueue->hasReadyFilter()) {
    // Don't make available until backtracking is complete.
    SU->isPending = true;
    PendingQueue.push_back(SU);
  }
  else {
    AvailableQueue->push(SU);
  }
  AvailableQueue->UnscheduledNode(SU);
}

/// After backtracking, the hazard checker needs to be restored to a state
/// corresponding the the current cycle.
void ScheduleDAGRRList::RestoreHazardCheckerBottomUp() {
  HazardRec->Reset();

  unsigned LookAhead = std::min((unsigned)Sequence.size(),
                                HazardRec->getMaxLookAhead());
  if (LookAhead == 0)
    return;

  std::vector<SUnit*>::const_iterator I = (Sequence.end() - LookAhead);
  unsigned HazardCycle = (*I)->getHeight();
  for (std::vector<SUnit*>::const_iterator E = Sequence.end(); I != E; ++I) {
    SUnit *SU = *I;
    for (; SU->getHeight() > HazardCycle; ++HazardCycle) {
      HazardRec->RecedeCycle();
    }
    EmitNode(SU);
  }
}

/// BacktrackBottomUp - Backtrack scheduling to a previous cycle specified in
/// BTCycle in order to schedule a specific node.
void ScheduleDAGRRList::BacktrackBottomUp(SUnit *SU, SUnit *BtSU) {
  SUnit *OldSU = Sequence.back();
  while (true) {
    Sequence.pop_back();
    if (SU->isSucc(OldSU))
      // Don't try to remove SU from AvailableQueue.
      SU->isAvailable = false;
    // FIXME: use ready cycle instead of height
    CurCycle = OldSU->getHeight();
    UnscheduleNodeBottomUp(OldSU);
    AvailableQueue->setCurCycle(CurCycle);
    if (OldSU == BtSU)
      break;
    OldSU = Sequence.back();
  }

  assert(!SU->isSucc(OldSU) && "Something is wrong!");

  RestoreHazardCheckerBottomUp();

  ReleasePending();

  ++NumBacktracks;
}

static bool isOperandOf(const SUnit *SU, SDNode *N) {
  for (const SDNode *SUNode = SU->getNode(); SUNode;
       SUNode = SUNode->getGluedNode()) {
    if (SUNode->isOperandOf(N))
      return true;
  }
  return false;
}

/// CopyAndMoveSuccessors - Clone the specified node and move its scheduled
/// successors to the newly created node.
SUnit *ScheduleDAGRRList::CopyAndMoveSuccessors(SUnit *SU) {
  SDNode *N = SU->getNode();
  if (!N)
    return NULL;

  if (SU->getNode()->getGluedNode())
    return NULL;

  SUnit *NewSU;
  bool TryUnfold = false;
  for (unsigned i = 0, e = N->getNumValues(); i != e; ++i) {
    EVT VT = N->getValueType(i);
    if (VT == MVT::Glue)
      return NULL;
    else if (VT == MVT::Other)
      TryUnfold = true;
  }
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
    const SDValue &Op = N->getOperand(i);
    EVT VT = Op.getNode()->getValueType(Op.getResNo());
    if (VT == MVT::Glue)
      return NULL;
  }

  if (TryUnfold) {
    SmallVector<SDNode*, 2> NewNodes;
    if (!TII->unfoldMemoryOperand(*DAG, N, NewNodes))
      return NULL;

    DEBUG(dbgs() << "Unfolding SU #" << SU->NodeNum << "\n");
    assert(NewNodes.size() == 2 && "Expected a load folding node!");

    N = NewNodes[1];
    SDNode *LoadNode = NewNodes[0];
    unsigned NumVals = N->getNumValues();
    unsigned OldNumVals = SU->getNode()->getNumValues();
    for (unsigned i = 0; i != NumVals; ++i)
      DAG->ReplaceAllUsesOfValueWith(SDValue(SU->getNode(), i), SDValue(N, i));
    DAG->ReplaceAllUsesOfValueWith(SDValue(SU->getNode(), OldNumVals-1),
                                   SDValue(LoadNode, 1));

    // LoadNode may already exist. This can happen when there is another
    // load from the same location and producing the same type of value
    // but it has different alignment or volatileness.
    bool isNewLoad = true;
    SUnit *LoadSU;
    if (LoadNode->getNodeId() != -1) {
      LoadSU = &SUnits[LoadNode->getNodeId()];
      isNewLoad = false;
    } else {
      LoadSU = CreateNewSUnit(LoadNode);
      LoadNode->setNodeId(LoadSU->NodeNum);

      InitNumRegDefsLeft(LoadSU);
      ComputeLatency(LoadSU);
    }

    SUnit *NewSU = CreateNewSUnit(N);
    assert(N->getNodeId() == -1 && "Node already inserted!");
    N->setNodeId(NewSU->NodeNum);

    const TargetInstrDesc &TID = TII->get(N->getMachineOpcode());
    for (unsigned i = 0; i != TID.getNumOperands(); ++i) {
      if (TID.getOperandConstraint(i, TOI::TIED_TO) != -1) {
        NewSU->isTwoAddress = true;
        break;
      }
    }
    if (TID.isCommutable())
      NewSU->isCommutable = true;

    InitNumRegDefsLeft(NewSU);
    ComputeLatency(NewSU);

    // Record all the edges to and from the old SU, by category.
    SmallVector<SDep, 4> ChainPreds;
    SmallVector<SDep, 4> ChainSuccs;
    SmallVector<SDep, 4> LoadPreds;
    SmallVector<SDep, 4> NodePreds;
    SmallVector<SDep, 4> NodeSuccs;
    for (SUnit::pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
         I != E; ++I) {
      if (I->isCtrl())
        ChainPreds.push_back(*I);
      else if (isOperandOf(I->getSUnit(), LoadNode))
        LoadPreds.push_back(*I);
      else
        NodePreds.push_back(*I);
    }
    for (SUnit::succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
         I != E; ++I) {
      if (I->isCtrl())
        ChainSuccs.push_back(*I);
      else
        NodeSuccs.push_back(*I);
    }

    // Now assign edges to the newly-created nodes.
    for (unsigned i = 0, e = ChainPreds.size(); i != e; ++i) {
      const SDep &Pred = ChainPreds[i];
      RemovePred(SU, Pred);
      if (isNewLoad)
        AddPred(LoadSU, Pred);
    }
    for (unsigned i = 0, e = LoadPreds.size(); i != e; ++i) {
      const SDep &Pred = LoadPreds[i];
      RemovePred(SU, Pred);
      if (isNewLoad)
        AddPred(LoadSU, Pred);
    }
    for (unsigned i = 0, e = NodePreds.size(); i != e; ++i) {
      const SDep &Pred = NodePreds[i];
      RemovePred(SU, Pred);
      AddPred(NewSU, Pred);
    }
    for (unsigned i = 0, e = NodeSuccs.size(); i != e; ++i) {
      SDep D = NodeSuccs[i];
      SUnit *SuccDep = D.getSUnit();
      D.setSUnit(SU);
      RemovePred(SuccDep, D);
      D.setSUnit(NewSU);
      AddPred(SuccDep, D);
      // Balance register pressure.
      if (AvailableQueue->tracksRegPressure() && SuccDep->isScheduled
          && !D.isCtrl() && NewSU->NumRegDefsLeft > 0)
        --NewSU->NumRegDefsLeft;
    }
    for (unsigned i = 0, e = ChainSuccs.size(); i != e; ++i) {
      SDep D = ChainSuccs[i];
      SUnit *SuccDep = D.getSUnit();
      D.setSUnit(SU);
      RemovePred(SuccDep, D);
      if (isNewLoad) {
        D.setSUnit(LoadSU);
        AddPred(SuccDep, D);
      }
    }

    // Add a data dependency to reflect that NewSU reads the value defined
    // by LoadSU.
    AddPred(NewSU, SDep(LoadSU, SDep::Data, LoadSU->Latency));

    if (isNewLoad)
      AvailableQueue->addNode(LoadSU);
    AvailableQueue->addNode(NewSU);

    ++NumUnfolds;

    if (NewSU->NumSuccsLeft == 0) {
      NewSU->isAvailable = true;
      return NewSU;
    }
    SU = NewSU;
  }

  DEBUG(dbgs() << "    Duplicating SU #" << SU->NodeNum << "\n");
  NewSU = CreateClone(SU);

  // New SUnit has the exact same predecessors.
  for (SUnit::pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
       I != E; ++I)
    if (!I->isArtificial())
      AddPred(NewSU, *I);

  // Only copy scheduled successors. Cut them from old node's successor
  // list and move them over.
  SmallVector<std::pair<SUnit *, SDep>, 4> DelDeps;
  for (SUnit::succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I) {
    if (I->isArtificial())
      continue;
    SUnit *SuccSU = I->getSUnit();
    if (SuccSU->isScheduled) {
      SDep D = *I;
      D.setSUnit(NewSU);
      AddPred(SuccSU, D);
      D.setSUnit(SU);
      DelDeps.push_back(std::make_pair(SuccSU, D));
    }
  }
  for (unsigned i = 0, e = DelDeps.size(); i != e; ++i)
    RemovePred(DelDeps[i].first, DelDeps[i].second);

  AvailableQueue->updateNode(SU);
  AvailableQueue->addNode(NewSU);

  ++NumDups;
  return NewSU;
}

/// InsertCopiesAndMoveSuccs - Insert register copies and move all
/// scheduled successors of the given SUnit to the last copy.
void ScheduleDAGRRList::InsertCopiesAndMoveSuccs(SUnit *SU, unsigned Reg,
                                               const TargetRegisterClass *DestRC,
                                               const TargetRegisterClass *SrcRC,
                                               SmallVector<SUnit*, 2> &Copies) {
  SUnit *CopyFromSU = CreateNewSUnit(NULL);
  CopyFromSU->CopySrcRC = SrcRC;
  CopyFromSU->CopyDstRC = DestRC;

  SUnit *CopyToSU = CreateNewSUnit(NULL);
  CopyToSU->CopySrcRC = DestRC;
  CopyToSU->CopyDstRC = SrcRC;

  // Only copy scheduled successors. Cut them from old node's successor
  // list and move them over.
  SmallVector<std::pair<SUnit *, SDep>, 4> DelDeps;
  for (SUnit::succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I) {
    if (I->isArtificial())
      continue;
    SUnit *SuccSU = I->getSUnit();
    if (SuccSU->isScheduled) {
      SDep D = *I;
      D.setSUnit(CopyToSU);
      AddPred(SuccSU, D);
      DelDeps.push_back(std::make_pair(SuccSU, *I));
    }
  }
  for (unsigned i = 0, e = DelDeps.size(); i != e; ++i)
    RemovePred(DelDeps[i].first, DelDeps[i].second);

  AddPred(CopyFromSU, SDep(SU, SDep::Data, SU->Latency, Reg));
  AddPred(CopyToSU, SDep(CopyFromSU, SDep::Data, CopyFromSU->Latency, 0));

  AvailableQueue->updateNode(SU);
  AvailableQueue->addNode(CopyFromSU);
  AvailableQueue->addNode(CopyToSU);
  Copies.push_back(CopyFromSU);
  Copies.push_back(CopyToSU);

  ++NumPRCopies;
}

/// getPhysicalRegisterVT - Returns the ValueType of the physical register
/// definition of the specified node.
/// FIXME: Move to SelectionDAG?
static EVT getPhysicalRegisterVT(SDNode *N, unsigned Reg,
                                 const TargetInstrInfo *TII) {
  const TargetInstrDesc &TID = TII->get(N->getMachineOpcode());
  assert(TID.ImplicitDefs && "Physical reg def must be in implicit def list!");
  unsigned NumRes = TID.getNumDefs();
  for (const unsigned *ImpDef = TID.getImplicitDefs(); *ImpDef; ++ImpDef) {
    if (Reg == *ImpDef)
      break;
    ++NumRes;
  }
  return N->getValueType(NumRes);
}

/// CheckForLiveRegDef - Return true and update live register vector if the
/// specified register def of the specified SUnit clobbers any "live" registers.
static void CheckForLiveRegDef(SUnit *SU, unsigned Reg,
                               std::vector<SUnit*> &LiveRegDefs,
                               SmallSet<unsigned, 4> &RegAdded,
                               SmallVector<unsigned, 4> &LRegs,
                               const TargetRegisterInfo *TRI) {
  for (const unsigned *AliasI = TRI->getOverlaps(Reg); *AliasI; ++AliasI) {

    // Check if Ref is live.
    if (!LiveRegDefs[Reg]) continue;

    // Allow multiple uses of the same def.
    if (LiveRegDefs[Reg] == SU) continue;

    // Add Reg to the set of interfering live regs.
    if (RegAdded.insert(Reg))
      LRegs.push_back(Reg);
  }
}

/// DelayForLiveRegsBottomUp - Returns true if it is necessary to delay
/// scheduling of the given node to satisfy live physical register dependencies.
/// If the specific node is the last one that's available to schedule, do
/// whatever is necessary (i.e. backtracking or cloning) to make it possible.
bool ScheduleDAGRRList::
DelayForLiveRegsBottomUp(SUnit *SU, SmallVector<unsigned, 4> &LRegs) {
  if (NumLiveRegs == 0)
    return false;

  SmallSet<unsigned, 4> RegAdded;
  // If this node would clobber any "live" register, then it's not ready.
  //
  // If SU is the currently live definition of the same register that it uses,
  // then we are free to schedule it.
  for (SUnit::pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
       I != E; ++I) {
    if (I->isAssignedRegDep() && LiveRegDefs[I->getReg()] != SU)
      CheckForLiveRegDef(I->getSUnit(), I->getReg(), LiveRegDefs,
                         RegAdded, LRegs, TRI);
  }

  for (SDNode *Node = SU->getNode(); Node; Node = Node->getGluedNode()) {
    if (Node->getOpcode() == ISD::INLINEASM) {
      // Inline asm can clobber physical defs.
      unsigned NumOps = Node->getNumOperands();
      if (Node->getOperand(NumOps-1).getValueType() == MVT::Glue)
        --NumOps;  // Ignore the glue operand.

      for (unsigned i = InlineAsm::Op_FirstOperand; i != NumOps;) {
        unsigned Flags =
          cast<ConstantSDNode>(Node->getOperand(i))->getZExtValue();
        unsigned NumVals = InlineAsm::getNumOperandRegisters(Flags);

        ++i; // Skip the ID value.
        if (InlineAsm::isRegDefKind(Flags) ||
            InlineAsm::isRegDefEarlyClobberKind(Flags)) {
          // Check for def of register or earlyclobber register.
          for (; NumVals; --NumVals, ++i) {
            unsigned Reg = cast<RegisterSDNode>(Node->getOperand(i))->getReg();
            if (TargetRegisterInfo::isPhysicalRegister(Reg))
              CheckForLiveRegDef(SU, Reg, LiveRegDefs, RegAdded, LRegs, TRI);
          }
        } else
          i += NumVals;
      }
      continue;
    }

    if (!Node->isMachineOpcode())
      continue;
    const TargetInstrDesc &TID = TII->get(Node->getMachineOpcode());
    if (!TID.ImplicitDefs)
      continue;
    for (const unsigned *Reg = TID.ImplicitDefs; *Reg; ++Reg)
      CheckForLiveRegDef(SU, *Reg, LiveRegDefs, RegAdded, LRegs, TRI);
  }

  return !LRegs.empty();
}

/// Return a node that can be scheduled in this cycle. Requirements:
/// (1) Ready: latency has been satisfied
/// (2) No Hazards: resources are available
/// (3) No Interferences: may unschedule to break register interferences.
SUnit *ScheduleDAGRRList::PickNodeToScheduleBottomUp() {
  SmallVector<SUnit*, 4> Interferences;
  DenseMap<SUnit*, SmallVector<unsigned, 4> > LRegsMap;

  SUnit *CurSU = AvailableQueue->pop();
  while (CurSU) {
    SmallVector<unsigned, 4> LRegs;
    if (!DelayForLiveRegsBottomUp(CurSU, LRegs))
      break;
    LRegsMap.insert(std::make_pair(CurSU, LRegs));

    CurSU->isPending = true;  // This SU is not in AvailableQueue right now.
    Interferences.push_back(CurSU);
    CurSU = AvailableQueue->pop();
  }
  if (CurSU) {
    // Add the nodes that aren't ready back onto the available list.
    for (unsigned i = 0, e = Interferences.size(); i != e; ++i) {
      Interferences[i]->isPending = false;
      assert(Interferences[i]->isAvailable && "must still be available");
      AvailableQueue->push(Interferences[i]);
    }
    return CurSU;
  }

  // All candidates are delayed due to live physical reg dependencies.
  // Try backtracking, code duplication, or inserting cross class copies
  // to resolve it.
  for (unsigned i = 0, e = Interferences.size(); i != e; ++i) {
    SUnit *TrySU = Interferences[i];
    SmallVector<unsigned, 4> &LRegs = LRegsMap[TrySU];

    // Try unscheduling up to the point where it's safe to schedule
    // this node.
    SUnit *BtSU = NULL;
    unsigned LiveCycle = UINT_MAX;
    for (unsigned j = 0, ee = LRegs.size(); j != ee; ++j) {
      unsigned Reg = LRegs[j];
      if (LiveRegGens[Reg]->getHeight() < LiveCycle) {
        BtSU = LiveRegGens[Reg];
        LiveCycle = BtSU->getHeight();
      }
    }
    if (!WillCreateCycle(TrySU, BtSU))  {
      BacktrackBottomUp(TrySU, BtSU);

      // Force the current node to be scheduled before the node that
      // requires the physical reg dep.
      if (BtSU->isAvailable) {
        BtSU->isAvailable = false;
        if (!BtSU->isPending)
          AvailableQueue->remove(BtSU);
      }
      AddPred(TrySU, SDep(BtSU, SDep::Order, /*Latency=*/1,
                          /*Reg=*/0, /*isNormalMemory=*/false,
                          /*isMustAlias=*/false, /*isArtificial=*/true));

      // If one or more successors has been unscheduled, then the current
      // node is no longer avaialable. Schedule a successor that's now
      // available instead.
      if (!TrySU->isAvailable) {
        CurSU = AvailableQueue->pop();
      }
      else {
        CurSU = TrySU;
        TrySU->isPending = false;
        Interferences.erase(Interferences.begin()+i);
      }
      break;
    }
  }

  if (!CurSU) {
    // Can't backtrack. If it's too expensive to copy the value, then try
    // duplicate the nodes that produces these "too expensive to copy"
    // values to break the dependency. In case even that doesn't work,
    // insert cross class copies.
    // If it's not too expensive, i.e. cost != -1, issue copies.
    SUnit *TrySU = Interferences[0];
    SmallVector<unsigned, 4> &LRegs = LRegsMap[TrySU];
    assert(LRegs.size() == 1 && "Can't handle this yet!");
    unsigned Reg = LRegs[0];
    SUnit *LRDef = LiveRegDefs[Reg];
    EVT VT = getPhysicalRegisterVT(LRDef->getNode(), Reg, TII);
    const TargetRegisterClass *RC =
      TRI->getMinimalPhysRegClass(Reg, VT);
    const TargetRegisterClass *DestRC = TRI->getCrossCopyRegClass(RC);

    // If cross copy register class is null, then it must be possible copy
    // the value directly. Do not try duplicate the def.
    SUnit *NewDef = 0;
    if (DestRC)
      NewDef = CopyAndMoveSuccessors(LRDef);
    else
      DestRC = RC;
    if (!NewDef) {
      // Issue copies, these can be expensive cross register class copies.
      SmallVector<SUnit*, 2> Copies;
      InsertCopiesAndMoveSuccs(LRDef, Reg, DestRC, RC, Copies);
      DEBUG(dbgs() << "    Adding an edge from SU #" << TrySU->NodeNum
            << " to SU #" << Copies.front()->NodeNum << "\n");
      AddPred(TrySU, SDep(Copies.front(), SDep::Order, /*Latency=*/1,
                          /*Reg=*/0, /*isNormalMemory=*/false,
                          /*isMustAlias=*/false,
                          /*isArtificial=*/true));
      NewDef = Copies.back();
    }

    DEBUG(dbgs() << "    Adding an edge from SU #" << NewDef->NodeNum
          << " to SU #" << TrySU->NodeNum << "\n");
    LiveRegDefs[Reg] = NewDef;
    AddPred(NewDef, SDep(TrySU, SDep::Order, /*Latency=*/1,
                         /*Reg=*/0, /*isNormalMemory=*/false,
                         /*isMustAlias=*/false,
                         /*isArtificial=*/true));
    TrySU->isAvailable = false;
    CurSU = NewDef;
  }

  assert(CurSU && "Unable to resolve live physical register dependencies!");

  // Add the nodes that aren't ready back onto the available list.
  for (unsigned i = 0, e = Interferences.size(); i != e; ++i) {
    Interferences[i]->isPending = false;
    // May no longer be available due to backtracking.
    if (Interferences[i]->isAvailable) {
      AvailableQueue->push(Interferences[i]);
    }
  }
  return CurSU;
}

/// ListScheduleBottomUp - The main loop of list scheduling for bottom-up
/// schedulers.
void ScheduleDAGRRList::ListScheduleBottomUp() {
  // Release any predecessors of the special Exit node.
  ReleasePredecessors(&ExitSU);

  // Add root to Available queue.
  if (!SUnits.empty()) {
    SUnit *RootSU = &SUnits[DAG->getRoot().getNode()->getNodeId()];
    assert(RootSU->Succs.empty() && "Graph root shouldn't have successors!");
    RootSU->isAvailable = true;
    AvailableQueue->push(RootSU);
  }

  // While Available queue is not empty, grab the node with the highest
  // priority. If it is not ready put it back.  Schedule the node.
  Sequence.reserve(SUnits.size());
  while (!AvailableQueue->empty()) {
    DEBUG(dbgs() << "\n*** Examining Available\n";
          AvailableQueue->dump(this));

    // Pick the best node to schedule taking all constraints into
    // consideration.
    SUnit *SU = PickNodeToScheduleBottomUp();

    AdvancePastStalls(SU);

    ScheduleNodeBottomUp(SU);

    while (AvailableQueue->empty() && !PendingQueue.empty()) {
      // Advance the cycle to free resources. Skip ahead to the next ready SU.
      assert(MinAvailableCycle < UINT_MAX && "MinAvailableCycle uninitialized");
      AdvanceToCycle(std::max(CurCycle + 1, MinAvailableCycle));
    }
  }

  // Reverse the order if it is bottom up.
  std::reverse(Sequence.begin(), Sequence.end());

#ifndef NDEBUG
  VerifySchedule(isBottomUp);
#endif
}

//===----------------------------------------------------------------------===//
//  Top-Down Scheduling
//===----------------------------------------------------------------------===//

/// ReleaseSucc - Decrement the NumPredsLeft count of a successor. Add it to
/// the AvailableQueue if the count reaches zero. Also update its cycle bound.
void ScheduleDAGRRList::ReleaseSucc(SUnit *SU, const SDep *SuccEdge) {
  SUnit *SuccSU = SuccEdge->getSUnit();

#ifndef NDEBUG
  if (SuccSU->NumPredsLeft == 0) {
    dbgs() << "*** Scheduling failed! ***\n";
    SuccSU->dump(this);
    dbgs() << " has been released too many times!\n";
    llvm_unreachable(0);
  }
#endif
  --SuccSU->NumPredsLeft;

  // If all the node's predecessors are scheduled, this node is ready
  // to be scheduled. Ignore the special ExitSU node.
  if (SuccSU->NumPredsLeft == 0 && SuccSU != &ExitSU) {
    SuccSU->isAvailable = true;
    AvailableQueue->push(SuccSU);
  }
}

void ScheduleDAGRRList::ReleaseSuccessors(SUnit *SU) {
  // Top down: release successors
  for (SUnit::succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I) {
    assert(!I->isAssignedRegDep() &&
           "The list-tdrr scheduler doesn't yet support physreg dependencies!");

    ReleaseSucc(SU, &*I);
  }
}

/// ScheduleNodeTopDown - Add the node to the schedule. Decrement the pending
/// count of its successors. If a successor pending count is zero, add it to
/// the Available queue.
void ScheduleDAGRRList::ScheduleNodeTopDown(SUnit *SU) {
  DEBUG(dbgs() << "*** Scheduling [" << CurCycle << "]: ");
  DEBUG(SU->dump(this));

  assert(CurCycle >= SU->getDepth() && "Node scheduled above its depth!");
  SU->setDepthToAtLeast(CurCycle);
  Sequence.push_back(SU);

  ReleaseSuccessors(SU);
  SU->isScheduled = true;
  AvailableQueue->ScheduledNode(SU);
}

/// ListScheduleTopDown - The main loop of list scheduling for top-down
/// schedulers.
void ScheduleDAGRRList::ListScheduleTopDown() {
  AvailableQueue->setCurCycle(CurCycle);

  // Release any successors of the special Entry node.
  ReleaseSuccessors(&EntrySU);

  // All leaves to Available queue.
  for (unsigned i = 0, e = SUnits.size(); i != e; ++i) {
    // It is available if it has no predecessors.
    if (SUnits[i].Preds.empty()) {
      AvailableQueue->push(&SUnits[i]);
      SUnits[i].isAvailable = true;
    }
  }

  // While Available queue is not empty, grab the node with the highest
  // priority. If it is not ready put it back.  Schedule the node.
  Sequence.reserve(SUnits.size());
  while (!AvailableQueue->empty()) {
    SUnit *CurSU = AvailableQueue->pop();

    if (CurSU)
      ScheduleNodeTopDown(CurSU);
    ++CurCycle;
    AvailableQueue->setCurCycle(CurCycle);
  }

#ifndef NDEBUG
  VerifySchedule(isBottomUp);
#endif
}


//===----------------------------------------------------------------------===//
//                RegReductionPriorityQueue Definition
//===----------------------------------------------------------------------===//
//
// This is a SchedulingPriorityQueue that schedules using Sethi Ullman numbers
// to reduce register pressure.
//
namespace {
class RegReductionPQBase;

struct queue_sort : public std::binary_function<SUnit*, SUnit*, bool> {
  bool isReady(SUnit* SU, unsigned CurCycle) const { return true; }
};

/// bu_ls_rr_sort - Priority function for bottom up register pressure
// reduction scheduler.
struct bu_ls_rr_sort : public queue_sort {
  enum {
    IsBottomUp = true,
    HasReadyFilter = false
  };

  RegReductionPQBase *SPQ;
  bu_ls_rr_sort(RegReductionPQBase *spq) : SPQ(spq) {}
  bu_ls_rr_sort(const bu_ls_rr_sort &RHS) : SPQ(RHS.SPQ) {}

  bool operator()(SUnit* left, SUnit* right) const;
};

// td_ls_rr_sort - Priority function for top down register pressure reduction
// scheduler.
struct td_ls_rr_sort : public queue_sort {
  enum {
    IsBottomUp = false,
    HasReadyFilter = false
  };

  RegReductionPQBase *SPQ;
  td_ls_rr_sort(RegReductionPQBase *spq) : SPQ(spq) {}
  td_ls_rr_sort(const td_ls_rr_sort &RHS) : SPQ(RHS.SPQ) {}

  bool operator()(const SUnit* left, const SUnit* right) const;
};

// src_ls_rr_sort - Priority function for source order scheduler.
struct src_ls_rr_sort : public queue_sort {
  enum {
    IsBottomUp = true,
    HasReadyFilter = false
  };

  RegReductionPQBase *SPQ;
  src_ls_rr_sort(RegReductionPQBase *spq)
    : SPQ(spq) {}
  src_ls_rr_sort(const src_ls_rr_sort &RHS)
    : SPQ(RHS.SPQ) {}

  bool operator()(SUnit* left, SUnit* right) const;
};

// hybrid_ls_rr_sort - Priority function for hybrid scheduler.
struct hybrid_ls_rr_sort : public queue_sort {
  enum {
    IsBottomUp = true,
    HasReadyFilter = false
  };

  RegReductionPQBase *SPQ;
  hybrid_ls_rr_sort(RegReductionPQBase *spq)
    : SPQ(spq) {}
  hybrid_ls_rr_sort(const hybrid_ls_rr_sort &RHS)
    : SPQ(RHS.SPQ) {}

  bool isReady(SUnit *SU, unsigned CurCycle) const;

  bool operator()(SUnit* left, SUnit* right) const;
};

// ilp_ls_rr_sort - Priority function for ILP (instruction level parallelism)
// scheduler.
struct ilp_ls_rr_sort : public queue_sort {
  enum {
    IsBottomUp = true,
    HasReadyFilter = false
  };

  RegReductionPQBase *SPQ;
  ilp_ls_rr_sort(RegReductionPQBase *spq)
    : SPQ(spq) {}
  ilp_ls_rr_sort(const ilp_ls_rr_sort &RHS)
    : SPQ(RHS.SPQ) {}

  bool isReady(SUnit *SU, unsigned CurCycle) const;

  bool operator()(SUnit* left, SUnit* right) const;
};

class RegReductionPQBase : public SchedulingPriorityQueue {
protected:
  std::vector<SUnit*> Queue;
  unsigned CurQueueId;
  bool TracksRegPressure;

  // SUnits - The SUnits for the current graph.
  std::vector<SUnit> *SUnits;

  MachineFunction &MF;
  const TargetInstrInfo *TII;
  const TargetRegisterInfo *TRI;
  const TargetLowering *TLI;
  ScheduleDAGRRList *scheduleDAG;

  // SethiUllmanNumbers - The SethiUllman number for each node.
  std::vector<unsigned> SethiUllmanNumbers;

  /// RegPressure - Tracking current reg pressure per register class.
  ///
  std::vector<unsigned> RegPressure;

  /// RegLimit - Tracking the number of allocatable registers per register
  /// class.
  std::vector<unsigned> RegLimit;

public:
  RegReductionPQBase(MachineFunction &mf,
                     bool hasReadyFilter,
                     bool tracksrp,
                     const TargetInstrInfo *tii,
                     const TargetRegisterInfo *tri,
                     const TargetLowering *tli)
    : SchedulingPriorityQueue(hasReadyFilter),
      CurQueueId(0), TracksRegPressure(tracksrp),
      MF(mf), TII(tii), TRI(tri), TLI(tli), scheduleDAG(NULL) {
    if (TracksRegPressure) {
      unsigned NumRC = TRI->getNumRegClasses();
      RegLimit.resize(NumRC);
      RegPressure.resize(NumRC);
      std::fill(RegLimit.begin(), RegLimit.end(), 0);
      std::fill(RegPressure.begin(), RegPressure.end(), 0);
      for (TargetRegisterInfo::regclass_iterator I = TRI->regclass_begin(),
             E = TRI->regclass_end(); I != E; ++I)
        RegLimit[(*I)->getID()] = tri->getRegPressureLimit(*I, MF);
    }
  }

  void setScheduleDAG(ScheduleDAGRRList *scheduleDag) {
    scheduleDAG = scheduleDag;
  }

  ScheduleHazardRecognizer* getHazardRec() {
    return scheduleDAG->getHazardRec();
  }

  void initNodes(std::vector<SUnit> &sunits);

  void addNode(const SUnit *SU);

  void updateNode(const SUnit *SU);

  void releaseState() {
    SUnits = 0;
    SethiUllmanNumbers.clear();
    std::fill(RegPressure.begin(), RegPressure.end(), 0);
  }

  unsigned getNodePriority(const SUnit *SU) const;

  unsigned getNodeOrdering(const SUnit *SU) const {
    return scheduleDAG->DAG->GetOrdering(SU->getNode());
  }

  bool empty() const { return Queue.empty(); }

  void push(SUnit *U) {
    assert(!U->NodeQueueId && "Node in the queue already");
    U->NodeQueueId = ++CurQueueId;
    Queue.push_back(U);
  }

  void remove(SUnit *SU) {
    assert(!Queue.empty() && "Queue is empty!");
    assert(SU->NodeQueueId != 0 && "Not in queue!");
    std::vector<SUnit *>::iterator I = std::find(Queue.begin(), Queue.end(),
                                                 SU);
    if (I != prior(Queue.end()))
      std::swap(*I, Queue.back());
    Queue.pop_back();
    SU->NodeQueueId = 0;
  }

  bool tracksRegPressure() const { return TracksRegPressure; }

  void dumpRegPressure() const;

  bool HighRegPressure(const SUnit *SU) const;

  bool MayReduceRegPressure(SUnit *SU) const;

  int RegPressureDiff(SUnit *SU, unsigned &LiveUses) const;

  void ScheduledNode(SUnit *SU);

  void UnscheduledNode(SUnit *SU);

protected:
  bool canClobber(const SUnit *SU, const SUnit *Op);
  void AddPseudoTwoAddrDeps();
  void PrescheduleNodesWithMultipleUses();
  void CalculateSethiUllmanNumbers();
};

template<class SF>
class RegReductionPriorityQueue : public RegReductionPQBase {
  static SUnit *popFromQueue(std::vector<SUnit*> &Q, SF &Picker) {
    std::vector<SUnit *>::iterator Best = Q.begin();
    for (std::vector<SUnit *>::iterator I = llvm::next(Q.begin()),
           E = Q.end(); I != E; ++I)
      if (Picker(*Best, *I))
        Best = I;
    SUnit *V = *Best;
    if (Best != prior(Q.end()))
      std::swap(*Best, Q.back());
    Q.pop_back();
    return V;
  }

  SF Picker;

public:
  RegReductionPriorityQueue(MachineFunction &mf,
                            bool tracksrp,
                            const TargetInstrInfo *tii,
                            const TargetRegisterInfo *tri,
                            const TargetLowering *tli)
    : RegReductionPQBase(mf, SF::HasReadyFilter, tracksrp, tii, tri, tli),
      Picker(this) {}

  bool isBottomUp() const { return SF::IsBottomUp; }

  bool isReady(SUnit *U) const {
    return Picker.HasReadyFilter && Picker.isReady(U, getCurCycle());
  }

  SUnit *pop() {
    if (Queue.empty()) return NULL;

    SUnit *V = popFromQueue(Queue, Picker);
    V->NodeQueueId = 0;
    return V;
  }

  void dump(ScheduleDAG *DAG) const {
    // Emulate pop() without clobbering NodeQueueIds.
    std::vector<SUnit*> DumpQueue = Queue;
    SF DumpPicker = Picker;
    while (!DumpQueue.empty()) {
      SUnit *SU = popFromQueue(DumpQueue, DumpPicker);
      if (isBottomUp())
        dbgs() << "Height " << SU->getHeight() << ": ";
      else
        dbgs() << "Depth " << SU->getDepth() << ": ";
      SU->dump(DAG);
    }
  }
};

typedef RegReductionPriorityQueue<bu_ls_rr_sort>
BURegReductionPriorityQueue;

typedef RegReductionPriorityQueue<td_ls_rr_sort>
TDRegReductionPriorityQueue;

typedef RegReductionPriorityQueue<src_ls_rr_sort>
SrcRegReductionPriorityQueue;

typedef RegReductionPriorityQueue<hybrid_ls_rr_sort>
HybridBURRPriorityQueue;

typedef RegReductionPriorityQueue<ilp_ls_rr_sort>
ILPBURRPriorityQueue;
} // end anonymous namespace

//===----------------------------------------------------------------------===//
//           Static Node Priority for Register Pressure Reduction
//===----------------------------------------------------------------------===//

/// CalcNodeSethiUllmanNumber - Compute Sethi Ullman number.
/// Smaller number is the higher priority.
static unsigned
CalcNodeSethiUllmanNumber(const SUnit *SU, std::vector<unsigned> &SUNumbers) {
  unsigned &SethiUllmanNumber = SUNumbers[SU->NodeNum];
  if (SethiUllmanNumber != 0)
    return SethiUllmanNumber;

  unsigned Extra = 0;
  for (SUnit::const_pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
       I != E; ++I) {
    if (I->isCtrl()) continue;  // ignore chain preds
    SUnit *PredSU = I->getSUnit();
    unsigned PredSethiUllman = CalcNodeSethiUllmanNumber(PredSU, SUNumbers);
    if (PredSethiUllman > SethiUllmanNumber) {
      SethiUllmanNumber = PredSethiUllman;
      Extra = 0;
    } else if (PredSethiUllman == SethiUllmanNumber)
      ++Extra;
  }

  SethiUllmanNumber += Extra;

  if (SethiUllmanNumber == 0)
    SethiUllmanNumber = 1;

  return SethiUllmanNumber;
}

/// CalculateSethiUllmanNumbers - Calculate Sethi-Ullman numbers of all
/// scheduling units.
void RegReductionPQBase::CalculateSethiUllmanNumbers() {
  SethiUllmanNumbers.assign(SUnits->size(), 0);

  for (unsigned i = 0, e = SUnits->size(); i != e; ++i)
    CalcNodeSethiUllmanNumber(&(*SUnits)[i], SethiUllmanNumbers);
}

void RegReductionPQBase::initNodes(std::vector<SUnit> &sunits) {
  SUnits = &sunits;
  // Add pseudo dependency edges for two-address nodes.
  AddPseudoTwoAddrDeps();
  // Reroute edges to nodes with multiple uses.
  if (!TracksRegPressure)
    PrescheduleNodesWithMultipleUses();
  // Calculate node priorities.
  CalculateSethiUllmanNumbers();
}

void RegReductionPQBase::addNode(const SUnit *SU) {
  unsigned SUSize = SethiUllmanNumbers.size();
  if (SUnits->size() > SUSize)
    SethiUllmanNumbers.resize(SUSize*2, 0);
  CalcNodeSethiUllmanNumber(SU, SethiUllmanNumbers);
}

void RegReductionPQBase::updateNode(const SUnit *SU) {
  SethiUllmanNumbers[SU->NodeNum] = 0;
  CalcNodeSethiUllmanNumber(SU, SethiUllmanNumbers);
}

// Lower priority means schedule further down. For bottom-up scheduling, lower
// priority SUs are scheduled before higher priority SUs.
unsigned RegReductionPQBase::getNodePriority(const SUnit *SU) const {
  assert(SU->NodeNum < SethiUllmanNumbers.size());
  unsigned Opc = SU->getNode() ? SU->getNode()->getOpcode() : 0;
  if (Opc == ISD::TokenFactor || Opc == ISD::CopyToReg)
    // CopyToReg should be close to its uses to facilitate coalescing and
    // avoid spilling.
    return 0;
  if (Opc == TargetOpcode::EXTRACT_SUBREG ||
      Opc == TargetOpcode::SUBREG_TO_REG ||
      Opc == TargetOpcode::INSERT_SUBREG)
    // EXTRACT_SUBREG, INSERT_SUBREG, and SUBREG_TO_REG nodes should be
    // close to their uses to facilitate coalescing.
    return 0;
  if (SU->NumSuccs == 0 && SU->NumPreds != 0)
    // If SU does not have a register use, i.e. it doesn't produce a value
    // that would be consumed (e.g. store), then it terminates a chain of
    // computation.  Give it a large SethiUllman number so it will be
    // scheduled right before its predecessors that it doesn't lengthen
    // their live ranges.
    return 0xffff;
  if (SU->NumPreds == 0 && SU->NumSuccs != 0)
    // If SU does not have a register def, schedule it close to its uses
    // because it does not lengthen any live ranges.
    return 0;
  return SethiUllmanNumbers[SU->NodeNum];
}

//===----------------------------------------------------------------------===//
//                     Register Pressure Tracking
//===----------------------------------------------------------------------===//

void RegReductionPQBase::dumpRegPressure() const {
  for (TargetRegisterInfo::regclass_iterator I = TRI->regclass_begin(),
         E = TRI->regclass_end(); I != E; ++I) {
    const TargetRegisterClass *RC = *I;
    unsigned Id = RC->getID();
    unsigned RP = RegPressure[Id];
    if (!RP) continue;
    DEBUG(dbgs() << RC->getName() << ": " << RP << " / " << RegLimit[Id]
          << '\n');
  }
}

bool RegReductionPQBase::HighRegPressure(const SUnit *SU) const {
  if (!TLI)
    return false;

  for (SUnit::const_pred_iterator I = SU->Preds.begin(),E = SU->Preds.end();
       I != E; ++I) {
    if (I->isCtrl())
      continue;
    SUnit *PredSU = I->getSUnit();
    // NumRegDefsLeft is zero when enough uses of this node have been scheduled
    // to cover the number of registers defined (they are all live).
    if (PredSU->NumRegDefsLeft == 0) {
      continue;
    }
    for (ScheduleDAGSDNodes::RegDefIter RegDefPos(PredSU, scheduleDAG);
         RegDefPos.IsValid(); RegDefPos.Advance()) {
      EVT VT = RegDefPos.GetValue();
      unsigned RCId = TLI->getRepRegClassFor(VT)->getID();
      unsigned Cost = TLI->getRepRegClassCostFor(VT);
      if ((RegPressure[RCId] + Cost) >= RegLimit[RCId])
        return true;
    }
  }
  return false;
}

bool RegReductionPQBase::MayReduceRegPressure(SUnit *SU) const {
  const SDNode *N = SU->getNode();

  if (!N->isMachineOpcode() || !SU->NumSuccs)
    return false;

  unsigned NumDefs = TII->get(N->getMachineOpcode()).getNumDefs();
  for (unsigned i = 0; i != NumDefs; ++i) {
    EVT VT = N->getValueType(i);
    if (!N->hasAnyUseOfValue(i))
      continue;
    unsigned RCId = TLI->getRepRegClassFor(VT)->getID();
    if (RegPressure[RCId] >= RegLimit[RCId])
      return true;
  }
  return false;
}

// Compute the register pressure contribution by this instruction by count up
// for uses that are not live and down for defs. Only count register classes
// that are already under high pressure. As a side effect, compute the number of
// uses of registers that are already live.
//
// FIXME: This encompasses the logic in HighRegPressure and MayReduceRegPressure
// so could probably be factored.
int RegReductionPQBase::RegPressureDiff(SUnit *SU, unsigned &LiveUses) const {
  LiveUses = 0;
  int PDiff = 0;
  for (SUnit::const_pred_iterator I = SU->Preds.begin(),E = SU->Preds.end();
       I != E; ++I) {
    if (I->isCtrl())
      continue;
    SUnit *PredSU = I->getSUnit();
    // NumRegDefsLeft is zero when enough uses of this node have been scheduled
    // to cover the number of registers defined (they are all live).
    if (PredSU->NumRegDefsLeft == 0) {
      if (PredSU->getNode()->isMachineOpcode())
        ++LiveUses;
      continue;
    }
    for (ScheduleDAGSDNodes::RegDefIter RegDefPos(PredSU, scheduleDAG);
         RegDefPos.IsValid(); RegDefPos.Advance()) {
      EVT VT = RegDefPos.GetValue();
      unsigned RCId = TLI->getRepRegClassFor(VT)->getID();
      if (RegPressure[RCId] >= RegLimit[RCId])
        ++PDiff;
    }
  }
  const SDNode *N = SU->getNode();

  if (!N || !N->isMachineOpcode() || !SU->NumSuccs)
    return PDiff;

  unsigned NumDefs = TII->get(N->getMachineOpcode()).getNumDefs();
  for (unsigned i = 0; i != NumDefs; ++i) {
    EVT VT = N->getValueType(i);
    if (!N->hasAnyUseOfValue(i))
      continue;
    unsigned RCId = TLI->getRepRegClassFor(VT)->getID();
    if (RegPressure[RCId] >= RegLimit[RCId])
      --PDiff;
  }
  return PDiff;
}

void RegReductionPQBase::ScheduledNode(SUnit *SU) {
  if (!TracksRegPressure)
    return;

  if (!SU->getNode())
    return;
  
  for (SUnit::pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
       I != E; ++I) {
    if (I->isCtrl())
      continue;
    SUnit *PredSU = I->getSUnit();
    // NumRegDefsLeft is zero when enough uses of this node have been scheduled
    // to cover the number of registers defined (they are all live).
    if (PredSU->NumRegDefsLeft == 0) {
      continue;
    }
    // FIXME: The ScheduleDAG currently loses information about which of a
    // node's values is consumed by each dependence. Consequently, if the node
    // defines multiple register classes, we don't know which to pressurize
    // here. Instead the following loop consumes the register defs in an
    // arbitrary order. At least it handles the common case of clustered loads
    // to the same class. For precise liveness, each SDep needs to indicate the
    // result number. But that tightly couples the ScheduleDAG with the
    // SelectionDAG making updates tricky. A simpler hack would be to attach a
    // value type or register class to SDep.
    //
    // The most important aspect of register tracking is balancing the increase
    // here with the reduction further below. Note that this SU may use multiple
    // defs in PredSU. The can't be determined here, but we've already
    // compensated by reducing NumRegDefsLeft in PredSU during
    // ScheduleDAGSDNodes::AddSchedEdges.
    --PredSU->NumRegDefsLeft;
    unsigned SkipRegDefs = PredSU->NumRegDefsLeft;
    for (ScheduleDAGSDNodes::RegDefIter RegDefPos(PredSU, scheduleDAG);
         RegDefPos.IsValid(); RegDefPos.Advance(), --SkipRegDefs) {
      if (SkipRegDefs)
        continue;
      EVT VT = RegDefPos.GetValue();
      unsigned RCId = TLI->getRepRegClassFor(VT)->getID();
      RegPressure[RCId] += TLI->getRepRegClassCostFor(VT);
      break;
    }
  }

  // We should have this assert, but there may be dead SDNodes that never
  // materialize as SUnits, so they don't appear to generate liveness.
  //assert(SU->NumRegDefsLeft == 0 && "not all regdefs have scheduled uses");
  int SkipRegDefs = (int)SU->NumRegDefsLeft;
  for (ScheduleDAGSDNodes::RegDefIter RegDefPos(SU, scheduleDAG);
       RegDefPos.IsValid(); RegDefPos.Advance(), --SkipRegDefs) {
    if (SkipRegDefs > 0)
      continue;
    EVT VT = RegDefPos.GetValue();
    unsigned RCId = TLI->getRepRegClassFor(VT)->getID();
    if (RegPressure[RCId] < TLI->getRepRegClassCostFor(VT)) {
      // Register pressure tracking is imprecise. This can happen. But we try
      // hard not to let it happen because it likely results in poor scheduling.
      DEBUG(dbgs() << "  SU(" << SU->NodeNum << ") has too many regdefs\n");
      RegPressure[RCId] = 0;
    }
    else {
      RegPressure[RCId] -= TLI->getRepRegClassCostFor(VT);
    }
  }
  dumpRegPressure();
}

void RegReductionPQBase::UnscheduledNode(SUnit *SU) {
  if (!TracksRegPressure)
    return;

  const SDNode *N = SU->getNode();
  if (!N) return;
  
  if (!N->isMachineOpcode()) {
    if (N->getOpcode() != ISD::CopyToReg)
      return;
  } else {
    unsigned Opc = N->getMachineOpcode();
    if (Opc == TargetOpcode::EXTRACT_SUBREG ||
        Opc == TargetOpcode::INSERT_SUBREG ||
        Opc == TargetOpcode::SUBREG_TO_REG ||
        Opc == TargetOpcode::REG_SEQUENCE ||
        Opc == TargetOpcode::IMPLICIT_DEF)
      return;
  }

  for (SUnit::pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
       I != E; ++I) {
    if (I->isCtrl())
      continue;
    SUnit *PredSU = I->getSUnit();
    // NumSuccsLeft counts all deps. Don't compare it with NumSuccs which only
    // counts data deps.
    if (PredSU->NumSuccsLeft != PredSU->Succs.size())
      continue;
    const SDNode *PN = PredSU->getNode();
    if (!PN->isMachineOpcode()) {
      if (PN->getOpcode() == ISD::CopyFromReg) {
        EVT VT = PN->getValueType(0);
        unsigned RCId = TLI->getRepRegClassFor(VT)->getID();
        RegPressure[RCId] += TLI->getRepRegClassCostFor(VT);
      }
      continue;
    }
    unsigned POpc = PN->getMachineOpcode();
    if (POpc == TargetOpcode::IMPLICIT_DEF)
      continue;
    if (POpc == TargetOpcode::EXTRACT_SUBREG) {
      EVT VT = PN->getOperand(0).getValueType();
      unsigned RCId = TLI->getRepRegClassFor(VT)->getID();
      RegPressure[RCId] += TLI->getRepRegClassCostFor(VT);
      continue;
    } else if (POpc == TargetOpcode::INSERT_SUBREG ||
               POpc == TargetOpcode::SUBREG_TO_REG) {
      EVT VT = PN->getValueType(0);
      unsigned RCId = TLI->getRepRegClassFor(VT)->getID();
      RegPressure[RCId] += TLI->getRepRegClassCostFor(VT);
      continue;
    }
    unsigned NumDefs = TII->get(PN->getMachineOpcode()).getNumDefs();
    for (unsigned i = 0; i != NumDefs; ++i) {
      EVT VT = PN->getValueType(i);
      if (!PN->hasAnyUseOfValue(i))
        continue;
      unsigned RCId = TLI->getRepRegClassFor(VT)->getID();
      if (RegPressure[RCId] < TLI->getRepRegClassCostFor(VT))
        // Register pressure tracking is imprecise. This can happen.
        RegPressure[RCId] = 0;
      else
        RegPressure[RCId] -= TLI->getRepRegClassCostFor(VT);
    }
  }

  // Check for isMachineOpcode() as PrescheduleNodesWithMultipleUses()
  // may transfer data dependencies to CopyToReg.
  if (SU->NumSuccs && N->isMachineOpcode()) {
    unsigned NumDefs = TII->get(N->getMachineOpcode()).getNumDefs();
    for (unsigned i = NumDefs, e = N->getNumValues(); i != e; ++i) {
      EVT VT = N->getValueType(i);
      if (VT == MVT::Glue || VT == MVT::Other)
        continue;
      if (!N->hasAnyUseOfValue(i))
        continue;
      unsigned RCId = TLI->getRepRegClassFor(VT)->getID();
      RegPressure[RCId] += TLI->getRepRegClassCostFor(VT);
    }
  }

  dumpRegPressure();
}

//===----------------------------------------------------------------------===//
//           Dynamic Node Priority for Register Pressure Reduction
//===----------------------------------------------------------------------===//

/// closestSucc - Returns the scheduled cycle of the successor which is
/// closest to the current cycle.
static unsigned closestSucc(const SUnit *SU) {
  unsigned MaxHeight = 0;
  for (SUnit::const_succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I) {
    if (I->isCtrl()) continue;  // ignore chain succs
    unsigned Height = I->getSUnit()->getHeight();
    // If there are bunch of CopyToRegs stacked up, they should be considered
    // to be at the same position.
    if (I->getSUnit()->getNode() &&
        I->getSUnit()->getNode()->getOpcode() == ISD::CopyToReg)
      Height = closestSucc(I->getSUnit())+1;
    if (Height > MaxHeight)
      MaxHeight = Height;
  }
  return MaxHeight;
}

/// calcMaxScratches - Returns an cost estimate of the worse case requirement
/// for scratch registers, i.e. number of data dependencies.
static unsigned calcMaxScratches(const SUnit *SU) {
  unsigned Scratches = 0;
  for (SUnit::const_pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
       I != E; ++I) {
    if (I->isCtrl()) continue;  // ignore chain preds
    Scratches++;
  }
  return Scratches;
}

/// hasOnlyLiveOutUse - Return true if SU has a single value successor that is a
/// CopyToReg to a virtual register. This SU def is probably a liveout and
/// it has no other use. It should be scheduled closer to the terminator.
static bool hasOnlyLiveOutUses(const SUnit *SU) {
  bool RetVal = false;
  for (SUnit::const_succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I) {
    if (I->isCtrl()) continue;
    const SUnit *SuccSU = I->getSUnit();
    if (SuccSU->getNode() && SuccSU->getNode()->getOpcode() == ISD::CopyToReg) {
      unsigned Reg =
        cast<RegisterSDNode>(SuccSU->getNode()->getOperand(1))->getReg();
      if (TargetRegisterInfo::isVirtualRegister(Reg)) {
        RetVal = true;
        continue;
      }
    }
    return false;
  }
  return RetVal;
}

/// UnitsSharePred - Return true if the two scheduling units share a common
/// data predecessor.
static bool UnitsSharePred(const SUnit *left, const SUnit *right) {
  SmallSet<const SUnit*, 4> Preds;
  for (SUnit::const_pred_iterator I = left->Preds.begin(),E = left->Preds.end();
       I != E; ++I) {
    if (I->isCtrl()) continue;  // ignore chain preds
    Preds.insert(I->getSUnit());
  }
  for (SUnit::const_pred_iterator I = right->Preds.begin(),E = right->Preds.end();
       I != E; ++I) {
    if (I->isCtrl()) continue;  // ignore chain preds
    if (Preds.count(I->getSUnit()))
      return true;
  }
  return false;
}

// Check for either a dependence (latency) or resource (hazard) stall.
//
// Note: The ScheduleHazardRecognizer interface requires a non-const SU.
static bool BUHasStall(SUnit *SU, int Height, RegReductionPQBase *SPQ) {
  if ((int)SPQ->getCurCycle() < Height) return true;
  if (SPQ->getHazardRec()->getHazardType(SU, 0)
      != ScheduleHazardRecognizer::NoHazard)
    return true;
  return false;
}

// Return -1 if left has higher priority, 1 if right has higher priority.
// Return 0 if latency-based priority is equivalent.
static int BUCompareLatency(SUnit *left, SUnit *right, bool checkPref,
                            RegReductionPQBase *SPQ) {
  // If the two nodes share an operand and one of them has a single
  // use that is a live out copy, favor the one that is live out. Otherwise
  // it will be difficult to eliminate the copy if the instruction is a
  // loop induction variable update. e.g.
  // BB:
  // sub r1, r3, #1
  // str r0, [r2, r3]
  // mov r3, r1
  // cmp
  // bne BB
  bool SharePred = UnitsSharePred(left, right);
  // FIXME: Only adjust if BB is a loop back edge.
  // FIXME: What's the cost of a copy?
  int LBonus = (SharePred && hasOnlyLiveOutUses(left)) ? 1 : 0;
  int RBonus = (SharePred && hasOnlyLiveOutUses(right)) ? 1 : 0;
  int LHeight = (int)left->getHeight() - LBonus;
  int RHeight = (int)right->getHeight() - RBonus;

  bool LStall = (!checkPref || left->SchedulingPref == Sched::Latency) &&
    BUHasStall(left, LHeight, SPQ);
  bool RStall = (!checkPref || right->SchedulingPref == Sched::Latency) &&
    BUHasStall(right, RHeight, SPQ);

  // If scheduling one of the node will cause a pipeline stall, delay it.
  // If scheduling either one of the node will cause a pipeline stall, sort
  // them according to their height.
  if (LStall) {
    if (!RStall)
      return 1;
    if (LHeight != RHeight)
      return LHeight > RHeight ? 1 : -1;
  } else if (RStall)
    return -1;

  // If either node is scheduling for latency, sort them by height/depth
  // and latency.
  if (!checkPref || (left->SchedulingPref == Sched::Latency ||
                     right->SchedulingPref == Sched::Latency)) {
    if (DisableSchedCycles) {
      if (LHeight != RHeight)
        return LHeight > RHeight ? 1 : -1;
    }
    else {
      // If neither instruction stalls (!LStall && !RStall) then
      // its height is already covered so only its depth matters. We also reach
      // this if both stall but have the same height.
      unsigned LDepth = left->getDepth();
      unsigned RDepth = right->getDepth();
      if (LDepth != RDepth) {
        DEBUG(dbgs() << "  Comparing latency of SU (" << left->NodeNum
              << ") depth " << LDepth << " vs SU (" << right->NodeNum
              << ") depth " << RDepth << "\n");
        return LDepth < RDepth ? 1 : -1;
      }
    }
    if (left->Latency != right->Latency)
      return left->Latency > right->Latency ? 1 : -1;
  }
  return 0;
}

static bool BURRSort(SUnit *left, SUnit *right, RegReductionPQBase *SPQ) {
  unsigned LPriority = SPQ->getNodePriority(left);
  unsigned RPriority = SPQ->getNodePriority(right);
  if (LPriority != RPriority) {
    DEBUG(++FactorCount[FactStatic]);
    return LPriority > RPriority;
  }
  DEBUG(++FactorCount[FactOther]);

  // Try schedule def + use closer when Sethi-Ullman numbers are the same.
  // e.g.
  // t1 = op t2, c1
  // t3 = op t4, c2
  //
  // and the following instructions are both ready.
  // t2 = op c3
  // t4 = op c4
  //
  // Then schedule t2 = op first.
  // i.e.
  // t4 = op c4
  // t2 = op c3
  // t1 = op t2, c1
  // t3 = op t4, c2
  //
  // This creates more short live intervals.
  unsigned LDist = closestSucc(left);
  unsigned RDist = closestSucc(right);
  if (LDist != RDist)
    return LDist < RDist;

  // How many registers becomes live when the node is scheduled.
  unsigned LScratch = calcMaxScratches(left);
  unsigned RScratch = calcMaxScratches(right);
  if (LScratch != RScratch)
    return LScratch > RScratch;

  if (!DisableSchedCycles) {
    int result = BUCompareLatency(left, right, false /*checkPref*/, SPQ);
    if (result != 0)
      return result > 0;
  }
  else {
    if (left->getHeight() != right->getHeight())
      return left->getHeight() > right->getHeight();

    if (left->getDepth() != right->getDepth())
      return left->getDepth() < right->getDepth();
  }

  assert(left->NodeQueueId && right->NodeQueueId &&
         "NodeQueueId cannot be zero");
  return (left->NodeQueueId > right->NodeQueueId);
}

// Bottom up
bool bu_ls_rr_sort::operator()(SUnit *left, SUnit *right) const {
  return BURRSort(left, right, SPQ);
}

// Source order, otherwise bottom up.
bool src_ls_rr_sort::operator()(SUnit *left, SUnit *right) const {
  unsigned LOrder = SPQ->getNodeOrdering(left);
  unsigned ROrder = SPQ->getNodeOrdering(right);

  // Prefer an ordering where the lower the non-zero order number, the higher
  // the preference.
  if ((LOrder || ROrder) && LOrder != ROrder)
    return LOrder != 0 && (LOrder < ROrder || ROrder == 0);

  return BURRSort(left, right, SPQ);
}

// If the time between now and when the instruction will be ready can cover
// the spill code, then avoid adding it to the ready queue. This gives long
// stalls highest priority and allows hoisting across calls. It should also
// speed up processing the available queue.
bool hybrid_ls_rr_sort::isReady(SUnit *SU, unsigned CurCycle) const {
  static const unsigned ReadyDelay = 3;

  if (SPQ->MayReduceRegPressure(SU)) return true;

  if (SU->getHeight() > (CurCycle + ReadyDelay)) return false;

  if (SPQ->getHazardRec()->getHazardType(SU, -ReadyDelay)
      != ScheduleHazardRecognizer::NoHazard)
    return false;

  return true;
}

// Return true if right should be scheduled with higher priority than left.
bool hybrid_ls_rr_sort::operator()(SUnit *left, SUnit *right) const {
  if (left->isCall || right->isCall)
    // No way to compute latency of calls.
    return BURRSort(left, right, SPQ);

  bool LHigh = SPQ->HighRegPressure(left);
  bool RHigh = SPQ->HighRegPressure(right);
  // Avoid causing spills. If register pressure is high, schedule for
  // register pressure reduction.
  if (LHigh && !RHigh) {
    DEBUG(dbgs() << "  pressure SU(" << left->NodeNum << ") > SU("
          << right->NodeNum << ")\n");
    return true;
  }
  else if (!LHigh && RHigh) {
    DEBUG(dbgs() << "  pressure SU(" << right->NodeNum << ") > SU("
          << left->NodeNum << ")\n");
    return false;
  }
  else if (!LHigh && !RHigh) {
    int result = BUCompareLatency(left, right, true /*checkPref*/, SPQ);
    if (result != 0)
      return result > 0;
  }
  return BURRSort(left, right, SPQ);
}

// Schedule as many instructions in each cycle as possible. So don't make an
// instruction available unless it is ready in the current cycle.
bool ilp_ls_rr_sort::isReady(SUnit *SU, unsigned CurCycle) const {
  if (SU->getHeight() > CurCycle) return false;

  if (SPQ->getHazardRec()->getHazardType(SU, 0)
      != ScheduleHazardRecognizer::NoHazard)
    return false;

  return true;
}

bool canEnableCoaelscing(SUnit *SU) {
  unsigned Opc = SU->getNode() ? SU->getNode()->getOpcode() : 0;
  if (Opc == ISD::TokenFactor || Opc == ISD::CopyToReg)
    // CopyToReg should be close to its uses to facilitate coalescing and
    // avoid spilling.
    return true;

  if (Opc == TargetOpcode::EXTRACT_SUBREG ||
      Opc == TargetOpcode::SUBREG_TO_REG ||
      Opc == TargetOpcode::INSERT_SUBREG)
    // EXTRACT_SUBREG, INSERT_SUBREG, and SUBREG_TO_REG nodes should be
    // close to their uses to facilitate coalescing.
    return true;

  if (SU->NumPreds == 0 && SU->NumSuccs != 0)
    // If SU does not have a register def, schedule it close to its uses
    // because it does not lengthen any live ranges.
    return true;

  return false;
}

// list-ilp is currently an experimental scheduler that allows various
// heuristics to be enabled prior to the normal register reduction logic.
bool ilp_ls_rr_sort::operator()(SUnit *left, SUnit *right) const {
  if (left->isCall || right->isCall)
    // No way to compute latency of calls.
    return BURRSort(left, right, SPQ);

  unsigned LLiveUses = 0, RLiveUses = 0;
  int LPDiff = 0, RPDiff = 0;
  if (!DisableSchedRegPressure || !DisableSchedLiveUses) {
    LPDiff = SPQ->RegPressureDiff(left, LLiveUses);
    RPDiff = SPQ->RegPressureDiff(right, RLiveUses);
  }
  if (!DisableSchedRegPressure && LPDiff != RPDiff) {
    DEBUG(++FactorCount[FactPressureDiff]);
    DEBUG(dbgs() << "RegPressureDiff SU(" << left->NodeNum << "): " << LPDiff
          << " != SU(" << right->NodeNum << "): " << RPDiff << "\n");
    return LPDiff > RPDiff;
  }

  if (!DisableSchedRegPressure && (LPDiff > 0 || RPDiff > 0)) {
    bool LReduce = canEnableCoaelscing(left);
    bool RReduce = canEnableCoaelscing(right);
    DEBUG(if (LReduce != RReduce) ++FactorCount[FactPressureDiff]);
    if (LReduce && !RReduce) return false;
    if (RReduce && !LReduce) return true;
  }

  if (!DisableSchedLiveUses && (LLiveUses != RLiveUses)) {
    DEBUG(dbgs() << "Live uses SU(" << left->NodeNum << "): " << LLiveUses
          << " != SU(" << right->NodeNum << "): " << RLiveUses << "\n");
    DEBUG(++FactorCount[FactRegUses]);
    return LLiveUses < RLiveUses;
  }

  if (!DisableSchedStalls) {
    bool LStall = BUHasStall(left, left->getHeight(), SPQ);
    bool RStall = BUHasStall(right, right->getHeight(), SPQ);
    if (LStall != RStall) {
      DEBUG(++FactorCount[FactHeight]);
      return left->getHeight() > right->getHeight();
    }
  }

  if (!DisableSchedCriticalPath) {
    int spread = (int)left->getDepth() - (int)right->getDepth();
    if (std::abs(spread) > MaxReorderWindow) {
      DEBUG(dbgs() << "Depth of SU(" << left->NodeNum << "): "
            << left->getDepth() << " != SU(" << right->NodeNum << "): "
            << right->getDepth() << "\n");
      DEBUG(++FactorCount[FactDepth]);
      return left->getDepth() < right->getDepth();
    }
  }

  if (!DisableSchedHeight && left->getHeight() != right->getHeight()) {
    int spread = (int)left->getHeight() - (int)right->getHeight();
    if (std::abs(spread) > MaxReorderWindow) {
      DEBUG(++FactorCount[FactHeight]);
      return left->getHeight() > right->getHeight();
    }
  }

  return BURRSort(left, right, SPQ);
}

//===----------------------------------------------------------------------===//
//                    Preschedule for Register Pressure
//===----------------------------------------------------------------------===//

bool RegReductionPQBase::canClobber(const SUnit *SU, const SUnit *Op) {
  if (SU->isTwoAddress) {
    unsigned Opc = SU->getNode()->getMachineOpcode();
    const TargetInstrDesc &TID = TII->get(Opc);
    unsigned NumRes = TID.getNumDefs();
    unsigned NumOps = TID.getNumOperands() - NumRes;
    for (unsigned i = 0; i != NumOps; ++i) {
      if (TID.getOperandConstraint(i+NumRes, TOI::TIED_TO) != -1) {
        SDNode *DU = SU->getNode()->getOperand(i).getNode();
        if (DU->getNodeId() != -1 &&
            Op->OrigNode == &(*SUnits)[DU->getNodeId()])
          return true;
      }
    }
  }
  return false;
}

/// canClobberPhysRegDefs - True if SU would clobber one of SuccSU's
/// physical register defs.
static bool canClobberPhysRegDefs(const SUnit *SuccSU, const SUnit *SU,
                                  const TargetInstrInfo *TII,
                                  const TargetRegisterInfo *TRI) {
  SDNode *N = SuccSU->getNode();
  unsigned NumDefs = TII->get(N->getMachineOpcode()).getNumDefs();
  const unsigned *ImpDefs = TII->get(N->getMachineOpcode()).getImplicitDefs();
  assert(ImpDefs && "Caller should check hasPhysRegDefs");
  for (const SDNode *SUNode = SU->getNode(); SUNode;
       SUNode = SUNode->getGluedNode()) {
    if (!SUNode->isMachineOpcode())
      continue;
    const unsigned *SUImpDefs =
      TII->get(SUNode->getMachineOpcode()).getImplicitDefs();
    if (!SUImpDefs)
      return false;
    for (unsigned i = NumDefs, e = N->getNumValues(); i != e; ++i) {
      EVT VT = N->getValueType(i);
      if (VT == MVT::Glue || VT == MVT::Other)
        continue;
      if (!N->hasAnyUseOfValue(i))
        continue;
      unsigned Reg = ImpDefs[i - NumDefs];
      for (;*SUImpDefs; ++SUImpDefs) {
        unsigned SUReg = *SUImpDefs;
        if (TRI->regsOverlap(Reg, SUReg))
          return true;
      }
    }
  }
  return false;
}

/// PrescheduleNodesWithMultipleUses - Nodes with multiple uses
/// are not handled well by the general register pressure reduction
/// heuristics. When presented with code like this:
///
///      N
///    / |
///   /  |
///  U  store
///  |
/// ...
///
/// the heuristics tend to push the store up, but since the
/// operand of the store has another use (U), this would increase
/// the length of that other use (the U->N edge).
///
/// This function transforms code like the above to route U's
/// dependence through the store when possible, like this:
///
///      N
///      ||
///      ||
///     store
///       |
///       U
///       |
///      ...
///
/// This results in the store being scheduled immediately
/// after N, which shortens the U->N live range, reducing
/// register pressure.
///
void RegReductionPQBase::PrescheduleNodesWithMultipleUses() {
  // Visit all the nodes in topological order, working top-down.
  for (unsigned i = 0, e = SUnits->size(); i != e; ++i) {
    SUnit *SU = &(*SUnits)[i];
    // For now, only look at nodes with no data successors, such as stores.
    // These are especially important, due to the heuristics in
    // getNodePriority for nodes with no data successors.
    if (SU->NumSuccs != 0)
      continue;
    // For now, only look at nodes with exactly one data predecessor.
    if (SU->NumPreds != 1)
      continue;
    // Avoid prescheduling copies to virtual registers, which don't behave
    // like other nodes from the perspective of scheduling heuristics.
    if (SDNode *N = SU->getNode())
      if (N->getOpcode() == ISD::CopyToReg &&
          TargetRegisterInfo::isVirtualRegister
            (cast<RegisterSDNode>(N->getOperand(1))->getReg()))
        continue;

    // Locate the single data predecessor.
    SUnit *PredSU = 0;
    for (SUnit::const_pred_iterator II = SU->Preds.begin(),
         EE = SU->Preds.end(); II != EE; ++II)
      if (!II->isCtrl()) {
        PredSU = II->getSUnit();
        break;
      }
    assert(PredSU);

    // Don't rewrite edges that carry physregs, because that requires additional
    // support infrastructure.
    if (PredSU->hasPhysRegDefs)
      continue;
    // Short-circuit the case where SU is PredSU's only data successor.
    if (PredSU->NumSuccs == 1)
      continue;
    // Avoid prescheduling to copies from virtual registers, which don't behave
    // like other nodes from the perspective of scheduling heuristics.
    if (SDNode *N = SU->getNode())
      if (N->getOpcode() == ISD::CopyFromReg &&
          TargetRegisterInfo::isVirtualRegister
            (cast<RegisterSDNode>(N->getOperand(1))->getReg()))
        continue;

    // Perform checks on the successors of PredSU.
    for (SUnit::const_succ_iterator II = PredSU->Succs.begin(),
         EE = PredSU->Succs.end(); II != EE; ++II) {
      SUnit *PredSuccSU = II->getSUnit();
      if (PredSuccSU == SU) continue;
      // If PredSU has another successor with no data successors, for
      // now don't attempt to choose either over the other.
      if (PredSuccSU->NumSuccs == 0)
        goto outer_loop_continue;
      // Don't break physical register dependencies.
      if (SU->hasPhysRegClobbers && PredSuccSU->hasPhysRegDefs)
        if (canClobberPhysRegDefs(PredSuccSU, SU, TII, TRI))
          goto outer_loop_continue;
      // Don't introduce graph cycles.
      if (scheduleDAG->IsReachable(SU, PredSuccSU))
        goto outer_loop_continue;
    }

    // Ok, the transformation is safe and the heuristics suggest it is
    // profitable. Update the graph.
    DEBUG(dbgs() << "    Prescheduling SU #" << SU->NodeNum
                 << " next to PredSU #" << PredSU->NodeNum
                 << " to guide scheduling in the presence of multiple uses\n");
    for (unsigned i = 0; i != PredSU->Succs.size(); ++i) {
      SDep Edge = PredSU->Succs[i];
      assert(!Edge.isAssignedRegDep());
      SUnit *SuccSU = Edge.getSUnit();
      if (SuccSU != SU) {
        Edge.setSUnit(PredSU);
        scheduleDAG->RemovePred(SuccSU, Edge);
        scheduleDAG->AddPred(SU, Edge);
        Edge.setSUnit(SU);
        scheduleDAG->AddPred(SuccSU, Edge);
        --i;
      }
    }
  outer_loop_continue:;
  }
}

/// AddPseudoTwoAddrDeps - If two nodes share an operand and one of them uses
/// it as a def&use operand. Add a pseudo control edge from it to the other
/// node (if it won't create a cycle) so the two-address one will be scheduled
/// first (lower in the schedule). If both nodes are two-address, favor the
/// one that has a CopyToReg use (more likely to be a loop induction update).
/// If both are two-address, but one is commutable while the other is not
/// commutable, favor the one that's not commutable.
void RegReductionPQBase::AddPseudoTwoAddrDeps() {
  for (unsigned i = 0, e = SUnits->size(); i != e; ++i) {
    SUnit *SU = &(*SUnits)[i];
    if (!SU->isTwoAddress)
      continue;

    SDNode *Node = SU->getNode();
    if (!Node || !Node->isMachineOpcode() || SU->getNode()->getGluedNode())
      continue;

    bool isLiveOut = hasOnlyLiveOutUses(SU);
    unsigned Opc = Node->getMachineOpcode();
    const TargetInstrDesc &TID = TII->get(Opc);
    unsigned NumRes = TID.getNumDefs();
    unsigned NumOps = TID.getNumOperands() - NumRes;
    for (unsigned j = 0; j != NumOps; ++j) {
      if (TID.getOperandConstraint(j+NumRes, TOI::TIED_TO) == -1)
        continue;
      SDNode *DU = SU->getNode()->getOperand(j).getNode();
      if (DU->getNodeId() == -1)
        continue;
      const SUnit *DUSU = &(*SUnits)[DU->getNodeId()];
      if (!DUSU) continue;
      for (SUnit::const_succ_iterator I = DUSU->Succs.begin(),
           E = DUSU->Succs.end(); I != E; ++I) {
        if (I->isCtrl()) continue;
        SUnit *SuccSU = I->getSUnit();
        if (SuccSU == SU)
          continue;
        // Be conservative. Ignore if nodes aren't at roughly the same
        // depth and height.
        if (SuccSU->getHeight() < SU->getHeight() &&
            (SU->getHeight() - SuccSU->getHeight()) > 1)
          continue;
        // Skip past COPY_TO_REGCLASS nodes, so that the pseudo edge
        // constrains whatever is using the copy, instead of the copy
        // itself. In the case that the copy is coalesced, this
        // preserves the intent of the pseudo two-address heurietics.
        while (SuccSU->Succs.size() == 1 &&
               SuccSU->getNode()->isMachineOpcode() &&
               SuccSU->getNode()->getMachineOpcode() ==
                 TargetOpcode::COPY_TO_REGCLASS)
          SuccSU = SuccSU->Succs.front().getSUnit();
        // Don't constrain non-instruction nodes.
        if (!SuccSU->getNode() || !SuccSU->getNode()->isMachineOpcode())
          continue;
        // Don't constrain nodes with physical register defs if the
        // predecessor can clobber them.
        if (SuccSU->hasPhysRegDefs && SU->hasPhysRegClobbers) {
          if (canClobberPhysRegDefs(SuccSU, SU, TII, TRI))
            continue;
        }
        // Don't constrain EXTRACT_SUBREG, INSERT_SUBREG, and SUBREG_TO_REG;
        // these may be coalesced away. We want them close to their uses.
        unsigned SuccOpc = SuccSU->getNode()->getMachineOpcode();
        if (SuccOpc == TargetOpcode::EXTRACT_SUBREG ||
            SuccOpc == TargetOpcode::INSERT_SUBREG ||
            SuccOpc == TargetOpcode::SUBREG_TO_REG)
          continue;
        if ((!canClobber(SuccSU, DUSU) ||
             (isLiveOut && !hasOnlyLiveOutUses(SuccSU)) ||
             (!SU->isCommutable && SuccSU->isCommutable)) &&
            !scheduleDAG->IsReachable(SuccSU, SU)) {
          DEBUG(dbgs() << "    Adding a pseudo-two-addr edge from SU #"
                       << SU->NodeNum << " to SU #" << SuccSU->NodeNum << "\n");
          scheduleDAG->AddPred(SU, SDep(SuccSU, SDep::Order, /*Latency=*/0,
                                        /*Reg=*/0, /*isNormalMemory=*/false,
                                        /*isMustAlias=*/false,
                                        /*isArtificial=*/true));
        }
      }
    }
  }
}

/// LimitedSumOfUnscheduledPredsOfSuccs - Compute the sum of the unscheduled
/// predecessors of the successors of the SUnit SU. Stop when the provided
/// limit is exceeded.
static unsigned LimitedSumOfUnscheduledPredsOfSuccs(const SUnit *SU,
                                                    unsigned Limit) {
  unsigned Sum = 0;
  for (SUnit::const_succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I) {
    const SUnit *SuccSU = I->getSUnit();
    for (SUnit::const_pred_iterator II = SuccSU->Preds.begin(),
         EE = SuccSU->Preds.end(); II != EE; ++II) {
      SUnit *PredSU = II->getSUnit();
      if (!PredSU->isScheduled)
        if (++Sum > Limit)
          return Sum;
    }
  }
  return Sum;
}


// Top down
bool td_ls_rr_sort::operator()(const SUnit *left, const SUnit *right) const {
  unsigned LPriority = SPQ->getNodePriority(left);
  unsigned RPriority = SPQ->getNodePriority(right);
  bool LIsTarget = left->getNode() && left->getNode()->isMachineOpcode();
  bool RIsTarget = right->getNode() && right->getNode()->isMachineOpcode();
  bool LIsFloater = LIsTarget && left->NumPreds == 0;
  bool RIsFloater = RIsTarget && right->NumPreds == 0;
  unsigned LBonus = (LimitedSumOfUnscheduledPredsOfSuccs(left,1) == 1) ? 2 : 0;
  unsigned RBonus = (LimitedSumOfUnscheduledPredsOfSuccs(right,1) == 1) ? 2 : 0;

  if (left->NumSuccs == 0 && right->NumSuccs != 0)
    return false;
  else if (left->NumSuccs != 0 && right->NumSuccs == 0)
    return true;

  if (LIsFloater)
    LBonus -= 2;
  if (RIsFloater)
    RBonus -= 2;
  if (left->NumSuccs == 1)
    LBonus += 2;
  if (right->NumSuccs == 1)
    RBonus += 2;

  if (LPriority+LBonus != RPriority+RBonus)
    return LPriority+LBonus < RPriority+RBonus;

  if (left->getDepth() != right->getDepth())
    return left->getDepth() < right->getDepth();

  if (left->NumSuccsLeft != right->NumSuccsLeft)
    return left->NumSuccsLeft > right->NumSuccsLeft;

  assert(left->NodeQueueId && right->NodeQueueId &&
         "NodeQueueId cannot be zero");
  return (left->NodeQueueId > right->NodeQueueId);
}

//===----------------------------------------------------------------------===//
//                         Public Constructor Functions
//===----------------------------------------------------------------------===//

llvm::ScheduleDAGSDNodes *
llvm::createBURRListDAGScheduler(SelectionDAGISel *IS,
                                 CodeGenOpt::Level OptLevel) {
  const TargetMachine &TM = IS->TM;
  const TargetInstrInfo *TII = TM.getInstrInfo();
  const TargetRegisterInfo *TRI = TM.getRegisterInfo();

  BURegReductionPriorityQueue *PQ =
    new BURegReductionPriorityQueue(*IS->MF, false, TII, TRI, 0);
  ScheduleDAGRRList *SD = new ScheduleDAGRRList(*IS->MF, false, PQ, OptLevel);
  PQ->setScheduleDAG(SD);
  return SD;
}

llvm::ScheduleDAGSDNodes *
llvm::createTDRRListDAGScheduler(SelectionDAGISel *IS,
                                 CodeGenOpt::Level OptLevel) {
  const TargetMachine &TM = IS->TM;
  const TargetInstrInfo *TII = TM.getInstrInfo();
  const TargetRegisterInfo *TRI = TM.getRegisterInfo();

  TDRegReductionPriorityQueue *PQ =
    new TDRegReductionPriorityQueue(*IS->MF, false, TII, TRI, 0);
  ScheduleDAGRRList *SD = new ScheduleDAGRRList(*IS->MF, false, PQ, OptLevel);
  PQ->setScheduleDAG(SD);
  return SD;
}

llvm::ScheduleDAGSDNodes *
llvm::createSourceListDAGScheduler(SelectionDAGISel *IS,
                                   CodeGenOpt::Level OptLevel) {
  const TargetMachine &TM = IS->TM;
  const TargetInstrInfo *TII = TM.getInstrInfo();
  const TargetRegisterInfo *TRI = TM.getRegisterInfo();

  SrcRegReductionPriorityQueue *PQ =
    new SrcRegReductionPriorityQueue(*IS->MF, false, TII, TRI, 0);
  ScheduleDAGRRList *SD = new ScheduleDAGRRList(*IS->MF, false, PQ, OptLevel);
  PQ->setScheduleDAG(SD);
  return SD;
}

llvm::ScheduleDAGSDNodes *
llvm::createHybridListDAGScheduler(SelectionDAGISel *IS,
                                   CodeGenOpt::Level OptLevel) {
  const TargetMachine &TM = IS->TM;
  const TargetInstrInfo *TII = TM.getInstrInfo();
  const TargetRegisterInfo *TRI = TM.getRegisterInfo();
  const TargetLowering *TLI = &IS->getTargetLowering();

  HybridBURRPriorityQueue *PQ =
    new HybridBURRPriorityQueue(*IS->MF, true, TII, TRI, TLI);

  ScheduleDAGRRList *SD = new ScheduleDAGRRList(*IS->MF, true, PQ, OptLevel);
  PQ->setScheduleDAG(SD);
  return SD;
}

llvm::ScheduleDAGSDNodes *
llvm::createILPListDAGScheduler(SelectionDAGISel *IS,
                                CodeGenOpt::Level OptLevel) {
  const TargetMachine &TM = IS->TM;
  const TargetInstrInfo *TII = TM.getInstrInfo();
  const TargetRegisterInfo *TRI = TM.getRegisterInfo();
  const TargetLowering *TLI = &IS->getTargetLowering();

  ILPBURRPriorityQueue *PQ =
    new ILPBURRPriorityQueue(*IS->MF, true, TII, TRI, TLI);
  ScheduleDAGRRList *SD = new ScheduleDAGRRList(*IS->MF, true, PQ, OptLevel);
  PQ->setScheduleDAG(SD);
  return SD;
}
