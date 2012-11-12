//===- MachineScheduler.cpp - Machine Instruction Scheduler ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// MachineScheduler schedules machine instructions after phi elimination. It
// preserves LiveIntervals so it can be invoked before register allocation.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "misched"

#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/ScheduleDAGILP.h"
#include "llvm/CodeGen/ScheduleHazardRecognizer.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/PriorityQueue.h"

#include <queue>

using namespace llvm;

namespace llvm {
cl::opt<bool> ForceTopDown("misched-topdown", cl::Hidden,
                           cl::desc("Force top-down list scheduling"));
cl::opt<bool> ForceBottomUp("misched-bottomup", cl::Hidden,
                            cl::desc("Force bottom-up list scheduling"));
}

#ifndef NDEBUG
static cl::opt<bool> ViewMISchedDAGs("view-misched-dags", cl::Hidden,
  cl::desc("Pop up a window to show MISched dags after they are processed"));

static cl::opt<unsigned> MISchedCutoff("misched-cutoff", cl::Hidden,
  cl::desc("Stop scheduling after N instructions"), cl::init(~0U));
#else
static bool ViewMISchedDAGs = false;
#endif // NDEBUG

// Threshold to very roughly model an out-of-order processor's instruction
// buffers. If the actual value of this threshold matters much in practice, then
// it can be specified by the machine model. For now, it's an experimental
// tuning knob to determine when and if it matters.
static cl::opt<unsigned> ILPWindow("ilp-window", cl::Hidden,
  cl::desc("Allow expected latency to exceed the critical path by N cycles "
           "before attempting to balance ILP"),
  cl::init(10U));

//===----------------------------------------------------------------------===//
// Machine Instruction Scheduling Pass and Registry
//===----------------------------------------------------------------------===//

MachineSchedContext::MachineSchedContext():
    MF(0), MLI(0), MDT(0), PassConfig(0), AA(0), LIS(0) {
  RegClassInfo = new RegisterClassInfo();
}

MachineSchedContext::~MachineSchedContext() {
  delete RegClassInfo;
}

namespace {
/// MachineScheduler runs after coalescing and before register allocation.
class MachineScheduler : public MachineSchedContext,
                         public MachineFunctionPass {
public:
  MachineScheduler();

  virtual void getAnalysisUsage(AnalysisUsage &AU) const;

  virtual void releaseMemory() {}

  virtual bool runOnMachineFunction(MachineFunction&);

  virtual void print(raw_ostream &O, const Module* = 0) const;

  static char ID; // Class identification, replacement for typeinfo
};
} // namespace

char MachineScheduler::ID = 0;

char &llvm::MachineSchedulerID = MachineScheduler::ID;

INITIALIZE_PASS_BEGIN(MachineScheduler, "misched",
                      "Machine Instruction Scheduler", false, false)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_DEPENDENCY(SlotIndexes)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_END(MachineScheduler, "misched",
                    "Machine Instruction Scheduler", false, false)

MachineScheduler::MachineScheduler()
: MachineFunctionPass(ID) {
  initializeMachineSchedulerPass(*PassRegistry::getPassRegistry());
}

void MachineScheduler::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequiredID(MachineDominatorsID);
  AU.addRequired<MachineLoopInfo>();
  AU.addRequired<AliasAnalysis>();
  AU.addRequired<TargetPassConfig>();
  AU.addRequired<SlotIndexes>();
  AU.addPreserved<SlotIndexes>();
  AU.addRequired<LiveIntervals>();
  AU.addPreserved<LiveIntervals>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

MachinePassRegistry MachineSchedRegistry::Registry;

/// A dummy default scheduler factory indicates whether the scheduler
/// is overridden on the command line.
static ScheduleDAGInstrs *useDefaultMachineSched(MachineSchedContext *C) {
  return 0;
}

/// MachineSchedOpt allows command line selection of the scheduler.
static cl::opt<MachineSchedRegistry::ScheduleDAGCtor, false,
               RegisterPassParser<MachineSchedRegistry> >
MachineSchedOpt("misched",
                cl::init(&useDefaultMachineSched), cl::Hidden,
                cl::desc("Machine instruction scheduler to use"));

static MachineSchedRegistry
DefaultSchedRegistry("default", "Use the target's default scheduler choice.",
                     useDefaultMachineSched);

/// Forward declare the standard machine scheduler. This will be used as the
/// default scheduler if the target does not set a default.
static ScheduleDAGInstrs *createConvergingSched(MachineSchedContext *C);


/// Decrement this iterator until reaching the top or a non-debug instr.
static MachineBasicBlock::iterator
priorNonDebug(MachineBasicBlock::iterator I, MachineBasicBlock::iterator Beg) {
  assert(I != Beg && "reached the top of the region, cannot decrement");
  while (--I != Beg) {
    if (!I->isDebugValue())
      break;
  }
  return I;
}

/// If this iterator is a debug value, increment until reaching the End or a
/// non-debug instruction.
static MachineBasicBlock::iterator
nextIfDebug(MachineBasicBlock::iterator I, MachineBasicBlock::iterator End) {
  for(; I != End; ++I) {
    if (!I->isDebugValue())
      break;
  }
  return I;
}

/// Top-level MachineScheduler pass driver.
///
/// Visit blocks in function order. Divide each block into scheduling regions
/// and visit them bottom-up. Visiting regions bottom-up is not required, but is
/// consistent with the DAG builder, which traverses the interior of the
/// scheduling regions bottom-up.
///
/// This design avoids exposing scheduling boundaries to the DAG builder,
/// simplifying the DAG builder's support for "special" target instructions.
/// At the same time the design allows target schedulers to operate across
/// scheduling boundaries, for example to bundle the boudary instructions
/// without reordering them. This creates complexity, because the target
/// scheduler must update the RegionBegin and RegionEnd positions cached by
/// ScheduleDAGInstrs whenever adding or removing instructions. A much simpler
/// design would be to split blocks at scheduling boundaries, but LLVM has a
/// general bias against block splitting purely for implementation simplicity.
bool MachineScheduler::runOnMachineFunction(MachineFunction &mf) {
  DEBUG(dbgs() << "Before MISsched:\n"; mf.print(dbgs()));

  // Initialize the context of the pass.
  MF = &mf;
  MLI = &getAnalysis<MachineLoopInfo>();
  MDT = &getAnalysis<MachineDominatorTree>();
  PassConfig = &getAnalysis<TargetPassConfig>();
  AA = &getAnalysis<AliasAnalysis>();

  LIS = &getAnalysis<LiveIntervals>();
  const TargetInstrInfo *TII = MF->getTarget().getInstrInfo();

  RegClassInfo->runOnMachineFunction(*MF);

  // Select the scheduler, or set the default.
  MachineSchedRegistry::ScheduleDAGCtor Ctor = MachineSchedOpt;
  if (Ctor == useDefaultMachineSched) {
    // Get the default scheduler set by the target.
    Ctor = MachineSchedRegistry::getDefault();
    if (!Ctor) {
      Ctor = createConvergingSched;
      MachineSchedRegistry::setDefault(Ctor);
    }
  }
  // Instantiate the selected scheduler.
  OwningPtr<ScheduleDAGInstrs> Scheduler(Ctor(this));

  // Visit all machine basic blocks.
  //
  // TODO: Visit blocks in global postorder or postorder within the bottom-up
  // loop tree. Then we can optionally compute global RegPressure.
  for (MachineFunction::iterator MBB = MF->begin(), MBBEnd = MF->end();
       MBB != MBBEnd; ++MBB) {

    Scheduler->startBlock(MBB);

    // Break the block into scheduling regions [I, RegionEnd), and schedule each
    // region as soon as it is discovered. RegionEnd points the scheduling
    // boundary at the bottom of the region. The DAG does not include RegionEnd,
    // but the region does (i.e. the next RegionEnd is above the previous
    // RegionBegin). If the current block has no terminator then RegionEnd ==
    // MBB->end() for the bottom region.
    //
    // The Scheduler may insert instructions during either schedule() or
    // exitRegion(), even for empty regions. So the local iterators 'I' and
    // 'RegionEnd' are invalid across these calls.
    unsigned RemainingInstrs = MBB->size();
    for(MachineBasicBlock::iterator RegionEnd = MBB->end();
        RegionEnd != MBB->begin(); RegionEnd = Scheduler->begin()) {

      // Avoid decrementing RegionEnd for blocks with no terminator.
      if (RegionEnd != MBB->end()
          || TII->isSchedulingBoundary(llvm::prior(RegionEnd), MBB, *MF)) {
        --RegionEnd;
        // Count the boundary instruction.
        --RemainingInstrs;
      }

      // The next region starts above the previous region. Look backward in the
      // instruction stream until we find the nearest boundary.
      MachineBasicBlock::iterator I = RegionEnd;
      for(;I != MBB->begin(); --I, --RemainingInstrs) {
        if (TII->isSchedulingBoundary(llvm::prior(I), MBB, *MF))
          break;
      }
      // Notify the scheduler of the region, even if we may skip scheduling
      // it. Perhaps it still needs to be bundled.
      Scheduler->enterRegion(MBB, I, RegionEnd, RemainingInstrs);

      // Skip empty scheduling regions (0 or 1 schedulable instructions).
      if (I == RegionEnd || I == llvm::prior(RegionEnd)) {
        // Close the current region. Bundle the terminator if needed.
        // This invalidates 'RegionEnd' and 'I'.
        Scheduler->exitRegion();
        continue;
      }
      DEBUG(dbgs() << "********** MI Scheduling **********\n");
      DEBUG(dbgs() << MF->getName()
            << ":BB#" << MBB->getNumber() << "\n  From: " << *I << "    To: ";
            if (RegionEnd != MBB->end()) dbgs() << *RegionEnd;
            else dbgs() << "End";
            dbgs() << " Remaining: " << RemainingInstrs << "\n");

      // Schedule a region: possibly reorder instructions.
      // This invalidates 'RegionEnd' and 'I'.
      Scheduler->schedule();

      // Close the current region.
      Scheduler->exitRegion();

      // Scheduling has invalidated the current iterator 'I'. Ask the
      // scheduler for the top of it's scheduled region.
      RegionEnd = Scheduler->begin();
    }
    assert(RemainingInstrs == 0 && "Instruction count mismatch!");
    Scheduler->finishBlock();
  }
  Scheduler->finalizeSchedule();
  DEBUG(LIS->print(dbgs()));
  return true;
}

void MachineScheduler::print(raw_ostream &O, const Module* m) const {
  // unimplemented
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void ReadyQueue::dump() {
  dbgs() << Name << ": ";
  for (unsigned i = 0, e = Queue.size(); i < e; ++i)
    dbgs() << Queue[i]->NodeNum << " ";
  dbgs() << "\n";
}
#endif

//===----------------------------------------------------------------------===//
// ScheduleDAGMI - Base class for MachineInstr scheduling with LiveIntervals
// preservation.
//===----------------------------------------------------------------------===//

/// ReleaseSucc - Decrement the NumPredsLeft count of a successor. When
/// NumPredsLeft reaches zero, release the successor node.
///
/// FIXME: Adjust SuccSU height based on MinLatency.
void ScheduleDAGMI::releaseSucc(SUnit *SU, SDep *SuccEdge) {
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
  if (SuccSU->NumPredsLeft == 0 && SuccSU != &ExitSU)
    SchedImpl->releaseTopNode(SuccSU);
}

/// releaseSuccessors - Call releaseSucc on each of SU's successors.
void ScheduleDAGMI::releaseSuccessors(SUnit *SU) {
  for (SUnit::succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I) {
    releaseSucc(SU, &*I);
  }
}

/// ReleasePred - Decrement the NumSuccsLeft count of a predecessor. When
/// NumSuccsLeft reaches zero, release the predecessor node.
///
/// FIXME: Adjust PredSU height based on MinLatency.
void ScheduleDAGMI::releasePred(SUnit *SU, SDep *PredEdge) {
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
  if (PredSU->NumSuccsLeft == 0 && PredSU != &EntrySU)
    SchedImpl->releaseBottomNode(PredSU);
}

/// releasePredecessors - Call releasePred on each of SU's predecessors.
void ScheduleDAGMI::releasePredecessors(SUnit *SU) {
  for (SUnit::pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
       I != E; ++I) {
    releasePred(SU, &*I);
  }
}

void ScheduleDAGMI::moveInstruction(MachineInstr *MI,
                                    MachineBasicBlock::iterator InsertPos) {
  // Advance RegionBegin if the first instruction moves down.
  if (&*RegionBegin == MI)
    ++RegionBegin;

  // Update the instruction stream.
  BB->splice(InsertPos, BB, MI);

  // Update LiveIntervals
  LIS->handleMove(MI, /*UpdateFlags=*/true);

  // Recede RegionBegin if an instruction moves above the first.
  if (RegionBegin == InsertPos)
    RegionBegin = MI;
}

bool ScheduleDAGMI::checkSchedLimit() {
#ifndef NDEBUG
  if (NumInstrsScheduled == MISchedCutoff && MISchedCutoff != ~0U) {
    CurrentTop = CurrentBottom;
    return false;
  }
  ++NumInstrsScheduled;
#endif
  return true;
}

/// enterRegion - Called back from MachineScheduler::runOnMachineFunction after
/// crossing a scheduling boundary. [begin, end) includes all instructions in
/// the region, including the boundary itself and single-instruction regions
/// that don't get scheduled.
void ScheduleDAGMI::enterRegion(MachineBasicBlock *bb,
                                MachineBasicBlock::iterator begin,
                                MachineBasicBlock::iterator end,
                                unsigned endcount)
{
  ScheduleDAGInstrs::enterRegion(bb, begin, end, endcount);

  // For convenience remember the end of the liveness region.
  LiveRegionEnd =
    (RegionEnd == bb->end()) ? RegionEnd : llvm::next(RegionEnd);
}

// Setup the register pressure trackers for the top scheduled top and bottom
// scheduled regions.
void ScheduleDAGMI::initRegPressure() {
  TopRPTracker.init(&MF, RegClassInfo, LIS, BB, RegionBegin);
  BotRPTracker.init(&MF, RegClassInfo, LIS, BB, LiveRegionEnd);

  // Close the RPTracker to finalize live ins.
  RPTracker.closeRegion();

  DEBUG(RPTracker.getPressure().dump(TRI));

  // Initialize the live ins and live outs.
  TopRPTracker.addLiveRegs(RPTracker.getPressure().LiveInRegs);
  BotRPTracker.addLiveRegs(RPTracker.getPressure().LiveOutRegs);

  // Close one end of the tracker so we can call
  // getMaxUpward/DownwardPressureDelta before advancing across any
  // instructions. This converts currently live regs into live ins/outs.
  TopRPTracker.closeTop();
  BotRPTracker.closeBottom();

  // Account for liveness generated by the region boundary.
  if (LiveRegionEnd != RegionEnd)
    BotRPTracker.recede();

  assert(BotRPTracker.getPos() == RegionEnd && "Can't find the region bottom");

  // Cache the list of excess pressure sets in this region. This will also track
  // the max pressure in the scheduled code for these sets.
  RegionCriticalPSets.clear();
  std::vector<unsigned> RegionPressure = RPTracker.getPressure().MaxSetPressure;
  for (unsigned i = 0, e = RegionPressure.size(); i < e; ++i) {
    unsigned Limit = TRI->getRegPressureSetLimit(i);
    DEBUG(dbgs() << TRI->getRegPressureSetName(i)
          << "Limit " << Limit
          << " Actual " << RegionPressure[i] << "\n");
    if (RegionPressure[i] > Limit)
      RegionCriticalPSets.push_back(PressureElement(i, 0));
  }
  DEBUG(dbgs() << "Excess PSets: ";
        for (unsigned i = 0, e = RegionCriticalPSets.size(); i != e; ++i)
          dbgs() << TRI->getRegPressureSetName(
            RegionCriticalPSets[i].PSetID) << " ";
        dbgs() << "\n");
}

// FIXME: When the pressure tracker deals in pressure differences then we won't
// iterate over all RegionCriticalPSets[i].
void ScheduleDAGMI::
updateScheduledPressure(std::vector<unsigned> NewMaxPressure) {
  for (unsigned i = 0, e = RegionCriticalPSets.size(); i < e; ++i) {
    unsigned ID = RegionCriticalPSets[i].PSetID;
    int &MaxUnits = RegionCriticalPSets[i].UnitIncrease;
    if ((int)NewMaxPressure[ID] > MaxUnits)
      MaxUnits = NewMaxPressure[ID];
  }
}

/// schedule - Called back from MachineScheduler::runOnMachineFunction
/// after setting up the current scheduling region. [RegionBegin, RegionEnd)
/// only includes instructions that have DAG nodes, not scheduling boundaries.
///
/// This is a skeletal driver, with all the functionality pushed into helpers,
/// so that it can be easilly extended by experimental schedulers. Generally,
/// implementing MachineSchedStrategy should be sufficient to implement a new
/// scheduling algorithm. However, if a scheduler further subclasses
/// ScheduleDAGMI then it will want to override this virtual method in order to
/// update any specialized state.
void ScheduleDAGMI::schedule() {
  buildDAGWithRegPressure();

  postprocessDAG();

  DEBUG(for (unsigned su = 0, e = SUnits.size(); su != e; ++su)
          SUnits[su].dumpAll(this));

  if (ViewMISchedDAGs) viewGraph();

  initQueues();

  bool IsTopNode = false;
  while (SUnit *SU = SchedImpl->pickNode(IsTopNode)) {
    assert(!SU->isScheduled && "Node already scheduled");
    if (!checkSchedLimit())
      break;

    scheduleMI(SU, IsTopNode);

    updateQueues(SU, IsTopNode);
  }
  assert(CurrentTop == CurrentBottom && "Nonempty unscheduled zone.");

  placeDebugValues();

  DEBUG({
      unsigned BBNum = top()->getParent()->getNumber();
      dbgs() << "*** Final schedule for BB#" << BBNum << " ***\n";
      dumpSchedule();
      dbgs() << '\n';
    });
}

/// Build the DAG and setup three register pressure trackers.
void ScheduleDAGMI::buildDAGWithRegPressure() {
  // Initialize the register pressure tracker used by buildSchedGraph.
  RPTracker.init(&MF, RegClassInfo, LIS, BB, LiveRegionEnd);

  // Account for liveness generate by the region boundary.
  if (LiveRegionEnd != RegionEnd)
    RPTracker.recede();

  // Build the DAG, and compute current register pressure.
  buildSchedGraph(AA, &RPTracker);
  if (ViewMISchedDAGs) viewGraph();

  // Initialize top/bottom trackers after computing region pressure.
  initRegPressure();
}

/// Apply each ScheduleDAGMutation step in order.
void ScheduleDAGMI::postprocessDAG() {
  for (unsigned i = 0, e = Mutations.size(); i < e; ++i) {
    Mutations[i]->apply(this);
  }
}

// Release all DAG roots for scheduling.
void ScheduleDAGMI::releaseRoots() {
  SmallVector<SUnit*, 16> BotRoots;

  for (std::vector<SUnit>::iterator
         I = SUnits.begin(), E = SUnits.end(); I != E; ++I) {
    // A SUnit is ready to top schedule if it has no predecessors.
    if (I->Preds.empty())
      SchedImpl->releaseTopNode(&(*I));
    // A SUnit is ready to bottom schedule if it has no successors.
    if (I->Succs.empty())
      BotRoots.push_back(&(*I));
  }
  // Release bottom roots in reverse order so the higher priority nodes appear
  // first. This is more natural and slightly more efficient.
  for (SmallVectorImpl<SUnit*>::const_reverse_iterator
         I = BotRoots.rbegin(), E = BotRoots.rend(); I != E; ++I)
    SchedImpl->releaseBottomNode(*I);
}

/// Identify DAG roots and setup scheduler queues.
void ScheduleDAGMI::initQueues() {

  // Initialize the strategy before modifying the DAG.
  SchedImpl->initialize(this);

  // Release edges from the special Entry node or to the special Exit node.
  releaseSuccessors(&EntrySU);
  releasePredecessors(&ExitSU);

  // Release all DAG roots for scheduling.
  releaseRoots();

  SchedImpl->registerRoots();

  CurrentTop = nextIfDebug(RegionBegin, RegionEnd);
  CurrentBottom = RegionEnd;
}

/// Move an instruction and update register pressure.
void ScheduleDAGMI::scheduleMI(SUnit *SU, bool IsTopNode) {
  // Move the instruction to its new location in the instruction stream.
  MachineInstr *MI = SU->getInstr();

  if (IsTopNode) {
    assert(SU->isTopReady() && "node still has unscheduled dependencies");
    if (&*CurrentTop == MI)
      CurrentTop = nextIfDebug(++CurrentTop, CurrentBottom);
    else {
      moveInstruction(MI, CurrentTop);
      TopRPTracker.setPos(MI);
    }

    // Update top scheduled pressure.
    TopRPTracker.advance();
    assert(TopRPTracker.getPos() == CurrentTop && "out of sync");
    updateScheduledPressure(TopRPTracker.getPressure().MaxSetPressure);
  }
  else {
    assert(SU->isBottomReady() && "node still has unscheduled dependencies");
    MachineBasicBlock::iterator priorII =
      priorNonDebug(CurrentBottom, CurrentTop);
    if (&*priorII == MI)
      CurrentBottom = priorII;
    else {
      if (&*CurrentTop == MI) {
        CurrentTop = nextIfDebug(++CurrentTop, priorII);
        TopRPTracker.setPos(CurrentTop);
      }
      moveInstruction(MI, CurrentBottom);
      CurrentBottom = MI;
    }
    // Update bottom scheduled pressure.
    BotRPTracker.recede();
    assert(BotRPTracker.getPos() == CurrentBottom && "out of sync");
    updateScheduledPressure(BotRPTracker.getPressure().MaxSetPressure);
  }
}

/// Update scheduler queues after scheduling an instruction.
void ScheduleDAGMI::updateQueues(SUnit *SU, bool IsTopNode) {
  // Release dependent instructions for scheduling.
  if (IsTopNode)
    releaseSuccessors(SU);
  else
    releasePredecessors(SU);

  SU->isScheduled = true;

  // Notify the scheduling strategy after updating the DAG.
  SchedImpl->schedNode(SU, IsTopNode);
}

/// Reinsert any remaining debug_values, just like the PostRA scheduler.
void ScheduleDAGMI::placeDebugValues() {
  // If first instruction was a DBG_VALUE then put it back.
  if (FirstDbgValue) {
    BB->splice(RegionBegin, BB, FirstDbgValue);
    RegionBegin = FirstDbgValue;
  }

  for (std::vector<std::pair<MachineInstr *, MachineInstr *> >::iterator
         DI = DbgValues.end(), DE = DbgValues.begin(); DI != DE; --DI) {
    std::pair<MachineInstr *, MachineInstr *> P = *prior(DI);
    MachineInstr *DbgValue = P.first;
    MachineBasicBlock::iterator OrigPrevMI = P.second;
    BB->splice(++OrigPrevMI, BB, DbgValue);
    if (OrigPrevMI == llvm::prior(RegionEnd))
      RegionEnd = DbgValue;
  }
  DbgValues.clear();
  FirstDbgValue = NULL;
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void ScheduleDAGMI::dumpSchedule() const {
  for (MachineBasicBlock::iterator MI = begin(), ME = end(); MI != ME; ++MI) {
    if (SUnit *SU = getSUnit(&(*MI)))
      SU->dump(this);
    else
      dbgs() << "Missing SUnit\n";
  }
}
#endif

//===----------------------------------------------------------------------===//
// ConvergingScheduler - Implementation of the standard MachineSchedStrategy.
//===----------------------------------------------------------------------===//

namespace {
/// ConvergingScheduler shrinks the unscheduled zone using heuristics to balance
/// the schedule.
class ConvergingScheduler : public MachineSchedStrategy {
public:
  /// Represent the type of SchedCandidate found within a single queue.
  /// pickNodeBidirectional depends on these listed by decreasing priority.
  enum CandReason {
    NoCand, SingleExcess, SingleCritical, ResourceReduce, ResourceDemand,
    BotHeightReduce, BotPathReduce, TopDepthReduce, TopPathReduce,
    SingleMax, MultiPressure, NextDefUse, NodeOrder};

#ifndef NDEBUG
  static const char *getReasonStr(ConvergingScheduler::CandReason Reason);
#endif

  /// Policy for scheduling the next instruction in the candidate's zone.
  struct CandPolicy {
    bool ReduceLatency;
    unsigned ReduceResIdx;
    unsigned DemandResIdx;

    CandPolicy(): ReduceLatency(false), ReduceResIdx(0), DemandResIdx(0) {}
  };

  /// Status of an instruction's critical resource consumption.
  struct SchedResourceDelta {
    // Count critical resources in the scheduled region required by SU.
    unsigned CritResources;

    // Count critical resources from another region consumed by SU.
    unsigned DemandedResources;

    SchedResourceDelta(): CritResources(0), DemandedResources(0) {}

    bool operator==(const SchedResourceDelta &RHS) const {
      return CritResources == RHS.CritResources
        && DemandedResources == RHS.DemandedResources;
    }
    bool operator!=(const SchedResourceDelta &RHS) const {
      return !operator==(RHS);
    }
  };

  /// Store the state used by ConvergingScheduler heuristics, required for the
  /// lifetime of one invocation of pickNode().
  struct SchedCandidate {
    CandPolicy Policy;

    // The best SUnit candidate.
    SUnit *SU;

    // The reason for this candidate.
    CandReason Reason;

    // Register pressure values for the best candidate.
    RegPressureDelta RPDelta;

    // Critical resource consumption of the best candidate.
    SchedResourceDelta ResDelta;

    SchedCandidate(const CandPolicy &policy)
    : Policy(policy), SU(NULL), Reason(NoCand) {}

    bool isValid() const { return SU; }

    // Copy the status of another candidate without changing policy.
    void setBest(SchedCandidate &Best) {
      assert(Best.Reason != NoCand && "uninitialized Sched candidate");
      SU = Best.SU;
      Reason = Best.Reason;
      RPDelta = Best.RPDelta;
      ResDelta = Best.ResDelta;
    }

    void initResourceDelta(const ScheduleDAGMI *DAG,
                           const TargetSchedModel *SchedModel);
  };

  /// Summarize the unscheduled region.
  struct SchedRemainder {
    // Critical path through the DAG in expected latency.
    unsigned CriticalPath;

    // Unscheduled resources
    SmallVector<unsigned, 16> RemainingCounts;
    // Critical resource for the unscheduled zone.
    unsigned CritResIdx;
    // Number of micro-ops left to schedule.
    unsigned RemainingMicroOps;
    // Is the unscheduled zone resource limited.
    bool IsResourceLimited;

    unsigned MaxRemainingCount;

    void reset() {
      CriticalPath = 0;
      RemainingCounts.clear();
      CritResIdx = 0;
      RemainingMicroOps = 0;
      IsResourceLimited = false;
      MaxRemainingCount = 0;
    }

    SchedRemainder() { reset(); }

    void init(ScheduleDAGMI *DAG, const TargetSchedModel *SchedModel);
  };

  /// Each Scheduling boundary is associated with ready queues. It tracks the
  /// current cycle in the direction of movement, and maintains the state
  /// of "hazards" and other interlocks at the current cycle.
  struct SchedBoundary {
    ScheduleDAGMI *DAG;
    const TargetSchedModel *SchedModel;
    SchedRemainder *Rem;

    ReadyQueue Available;
    ReadyQueue Pending;
    bool CheckPending;

    // For heuristics, keep a list of the nodes that immediately depend on the
    // most recently scheduled node.
    SmallPtrSet<const SUnit*, 8> NextSUs;

    ScheduleHazardRecognizer *HazardRec;

    unsigned CurrCycle;
    unsigned IssueCount;

    /// MinReadyCycle - Cycle of the soonest available instruction.
    unsigned MinReadyCycle;

    // The expected latency of the critical path in this scheduled zone.
    unsigned ExpectedLatency;

    // Resources used in the scheduled zone beyond this boundary.
    SmallVector<unsigned, 16> ResourceCounts;

    // Cache the critical resources ID in this scheduled zone.
    unsigned CritResIdx;

    // Is the scheduled region resource limited vs. latency limited.
    bool IsResourceLimited;

    unsigned ExpectedCount;

    // Policy flag: attempt to find ILP until expected latency is covered.
    bool ShouldIncreaseILP;

#ifndef NDEBUG
    // Remember the greatest min operand latency.
    unsigned MaxMinLatency;
#endif

    void reset() {
      Available.clear();
      Pending.clear();
      CheckPending = false;
      NextSUs.clear();
      HazardRec = 0;
      CurrCycle = 0;
      IssueCount = 0;
      MinReadyCycle = UINT_MAX;
      ExpectedLatency = 0;
      ResourceCounts.resize(1);
      assert(!ResourceCounts[0] && "nonzero count for bad resource");
      CritResIdx = 0;
      IsResourceLimited = false;
      ExpectedCount = 0;
      ShouldIncreaseILP = false;
#ifndef NDEBUG
      MaxMinLatency = 0;
#endif
      // Reserve a zero-count for invalid CritResIdx.
      ResourceCounts.resize(1);
    }

    /// Pending queues extend the ready queues with the same ID and the
    /// PendingFlag set.
    SchedBoundary(unsigned ID, const Twine &Name):
      DAG(0), SchedModel(0), Rem(0), Available(ID, Name+".A"),
      Pending(ID << ConvergingScheduler::LogMaxQID, Name+".P") {
      reset();
    }

    ~SchedBoundary() { delete HazardRec; }

    void init(ScheduleDAGMI *dag, const TargetSchedModel *smodel,
              SchedRemainder *rem);

    bool isTop() const {
      return Available.getID() == ConvergingScheduler::TopQID;
    }

    unsigned getUnscheduledLatency(SUnit *SU) const {
      if (isTop())
        return SU->getHeight();
      return SU->getDepth();
    }

    unsigned getCriticalCount() const {
      return ResourceCounts[CritResIdx];
    }

    bool checkHazard(SUnit *SU);

    void checkILPPolicy();

    void releaseNode(SUnit *SU, unsigned ReadyCycle);

    void bumpCycle();

    void countResource(unsigned PIdx, unsigned Cycles);

    void bumpNode(SUnit *SU);

    void releasePending();

    void removeReady(SUnit *SU);

    SUnit *pickOnlyChoice();
  };

private:
  ScheduleDAGMI *DAG;
  const TargetSchedModel *SchedModel;
  const TargetRegisterInfo *TRI;

  // State of the top and bottom scheduled instruction boundaries.
  SchedRemainder Rem;
  SchedBoundary Top;
  SchedBoundary Bot;

public:
  /// SUnit::NodeQueueId: 0 (none), 1 (top), 2 (bot), 3 (both)
  enum {
    TopQID = 1,
    BotQID = 2,
    LogMaxQID = 2
  };

  ConvergingScheduler():
    DAG(0), SchedModel(0), TRI(0), Top(TopQID, "TopQ"), Bot(BotQID, "BotQ") {}

  virtual void initialize(ScheduleDAGMI *dag);

  virtual SUnit *pickNode(bool &IsTopNode);

  virtual void schedNode(SUnit *SU, bool IsTopNode);

  virtual void releaseTopNode(SUnit *SU);

  virtual void releaseBottomNode(SUnit *SU);

  virtual void registerRoots();

protected:
  void balanceZones(
    ConvergingScheduler::SchedBoundary &CriticalZone,
    ConvergingScheduler::SchedCandidate &CriticalCand,
    ConvergingScheduler::SchedBoundary &OppositeZone,
    ConvergingScheduler::SchedCandidate &OppositeCand);

  void checkResourceLimits(ConvergingScheduler::SchedCandidate &TopCand,
                           ConvergingScheduler::SchedCandidate &BotCand);

  void tryCandidate(SchedCandidate &Cand,
                    SchedCandidate &TryCand,
                    SchedBoundary &Zone,
                    const RegPressureTracker &RPTracker,
                    RegPressureTracker &TempTracker);

  SUnit *pickNodeBidirectional(bool &IsTopNode);

  void pickNodeFromQueue(SchedBoundary &Zone,
                         const RegPressureTracker &RPTracker,
                         SchedCandidate &Candidate);

#ifndef NDEBUG
  void traceCandidate(const SchedCandidate &Cand, const SchedBoundary &Zone);
#endif
};
} // namespace

void ConvergingScheduler::SchedRemainder::
init(ScheduleDAGMI *DAG, const TargetSchedModel *SchedModel) {
  reset();
  if (!SchedModel->hasInstrSchedModel())
    return;
  RemainingCounts.resize(SchedModel->getNumProcResourceKinds());
  for (std::vector<SUnit>::iterator
         I = DAG->SUnits.begin(), E = DAG->SUnits.end(); I != E; ++I) {
    const MCSchedClassDesc *SC = DAG->getSchedClass(&*I);
    RemainingMicroOps += SchedModel->getNumMicroOps(I->getInstr(), SC);
    for (TargetSchedModel::ProcResIter
           PI = SchedModel->getWriteProcResBegin(SC),
           PE = SchedModel->getWriteProcResEnd(SC); PI != PE; ++PI) {
      unsigned PIdx = PI->ProcResourceIdx;
      unsigned Factor = SchedModel->getResourceFactor(PIdx);
      RemainingCounts[PIdx] += (Factor * PI->Cycles);
    }
  }
}

void ConvergingScheduler::SchedBoundary::
init(ScheduleDAGMI *dag, const TargetSchedModel *smodel, SchedRemainder *rem) {
  reset();
  DAG = dag;
  SchedModel = smodel;
  Rem = rem;
  if (SchedModel->hasInstrSchedModel())
    ResourceCounts.resize(SchedModel->getNumProcResourceKinds());
}

void ConvergingScheduler::initialize(ScheduleDAGMI *dag) {
  DAG = dag;
  SchedModel = DAG->getSchedModel();
  TRI = DAG->TRI;
  Rem.init(DAG, SchedModel);
  Top.init(DAG, SchedModel, &Rem);
  Bot.init(DAG, SchedModel, &Rem);

  // Initialize resource counts.

  // Initialize the HazardRecognizers. If itineraries don't exist, are empty, or
  // are disabled, then these HazardRecs will be disabled.
  const InstrItineraryData *Itin = SchedModel->getInstrItineraries();
  const TargetMachine &TM = DAG->MF.getTarget();
  Top.HazardRec = TM.getInstrInfo()->CreateTargetMIHazardRecognizer(Itin, DAG);
  Bot.HazardRec = TM.getInstrInfo()->CreateTargetMIHazardRecognizer(Itin, DAG);

  assert((!ForceTopDown || !ForceBottomUp) &&
         "-misched-topdown incompatible with -misched-bottomup");
}

void ConvergingScheduler::releaseTopNode(SUnit *SU) {
  if (SU->isScheduled)
    return;

  for (SUnit::succ_iterator I = SU->Preds.begin(), E = SU->Preds.end();
       I != E; ++I) {
    unsigned PredReadyCycle = I->getSUnit()->TopReadyCycle;
    unsigned MinLatency = I->getMinLatency();
#ifndef NDEBUG
    Top.MaxMinLatency = std::max(MinLatency, Top.MaxMinLatency);
#endif
    if (SU->TopReadyCycle < PredReadyCycle + MinLatency)
      SU->TopReadyCycle = PredReadyCycle + MinLatency;
  }
  Top.releaseNode(SU, SU->TopReadyCycle);
}

void ConvergingScheduler::releaseBottomNode(SUnit *SU) {
  if (SU->isScheduled)
    return;

  assert(SU->getInstr() && "Scheduled SUnit must have instr");

  for (SUnit::succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I) {
    unsigned SuccReadyCycle = I->getSUnit()->BotReadyCycle;
    unsigned MinLatency = I->getMinLatency();
#ifndef NDEBUG
    Bot.MaxMinLatency = std::max(MinLatency, Bot.MaxMinLatency);
#endif
    if (SU->BotReadyCycle < SuccReadyCycle + MinLatency)
      SU->BotReadyCycle = SuccReadyCycle + MinLatency;
  }
  Bot.releaseNode(SU, SU->BotReadyCycle);
}

void ConvergingScheduler::registerRoots() {
  Rem.CriticalPath = DAG->ExitSU.getDepth();
  // Some roots may not feed into ExitSU. Check all of them in case.
  for (std::vector<SUnit*>::const_iterator
         I = Bot.Available.begin(), E = Bot.Available.end(); I != E; ++I) {
    if ((*I)->getDepth() > Rem.CriticalPath)
      Rem.CriticalPath = (*I)->getDepth();
  }
  DEBUG(dbgs() << "Critical Path: " << Rem.CriticalPath << '\n');
}

/// Does this SU have a hazard within the current instruction group.
///
/// The scheduler supports two modes of hazard recognition. The first is the
/// ScheduleHazardRecognizer API. It is a fully general hazard recognizer that
/// supports highly complicated in-order reservation tables
/// (ScoreboardHazardRecognizer) and arbitraty target-specific logic.
///
/// The second is a streamlined mechanism that checks for hazards based on
/// simple counters that the scheduler itself maintains. It explicitly checks
/// for instruction dispatch limitations, including the number of micro-ops that
/// can dispatch per cycle.
///
/// TODO: Also check whether the SU must start a new group.
bool ConvergingScheduler::SchedBoundary::checkHazard(SUnit *SU) {
  if (HazardRec->isEnabled())
    return HazardRec->getHazardType(SU) != ScheduleHazardRecognizer::NoHazard;

  unsigned uops = SchedModel->getNumMicroOps(SU->getInstr());
  if ((IssueCount > 0) && (IssueCount + uops > SchedModel->getIssueWidth())) {
    DEBUG(dbgs() << "  SU(" << SU->NodeNum << ") uops="
          << SchedModel->getNumMicroOps(SU->getInstr()) << '\n');
    return true;
  }
  return false;
}

/// If expected latency is covered, disable ILP policy.
void ConvergingScheduler::SchedBoundary::checkILPPolicy() {
  if (ShouldIncreaseILP
      && (IsResourceLimited || ExpectedLatency <= CurrCycle)) {
    ShouldIncreaseILP = false;
    DEBUG(dbgs() << "Disable ILP: " << Available.getName() << '\n');
  }
}

void ConvergingScheduler::SchedBoundary::releaseNode(SUnit *SU,
                                                     unsigned ReadyCycle) {

  if (ReadyCycle < MinReadyCycle)
    MinReadyCycle = ReadyCycle;

  // Check for interlocks first. For the purpose of other heuristics, an
  // instruction that cannot issue appears as if it's not in the ReadyQueue.
  if (ReadyCycle > CurrCycle || checkHazard(SU))
    Pending.push(SU);
  else
    Available.push(SU);

  // Record this node as an immediate dependent of the scheduled node.
  NextSUs.insert(SU);

  // If CriticalPath has been computed, then check if the unscheduled nodes
  // exceed the ILP window. Before registerRoots, CriticalPath==0.
  if (Rem->CriticalPath && (ExpectedLatency + getUnscheduledLatency(SU)
                            > Rem->CriticalPath + ILPWindow)) {
    ShouldIncreaseILP = true;
    DEBUG(dbgs() << "Increase ILP: " << Available.getName() << " "
          << ExpectedLatency << " + " << getUnscheduledLatency(SU) << '\n');
  }
}

/// Move the boundary of scheduled code by one cycle.
void ConvergingScheduler::SchedBoundary::bumpCycle() {
  unsigned Width = SchedModel->getIssueWidth();
  IssueCount = (IssueCount <= Width) ? 0 : IssueCount - Width;

  unsigned NextCycle = CurrCycle + 1;
  assert(MinReadyCycle < UINT_MAX && "MinReadyCycle uninitialized");
  if (MinReadyCycle > NextCycle) {
    IssueCount = 0;
    NextCycle = MinReadyCycle;
  }

  if (!HazardRec->isEnabled()) {
    // Bypass HazardRec virtual calls.
    CurrCycle = NextCycle;
  }
  else {
    // Bypass getHazardType calls in case of long latency.
    for (; CurrCycle != NextCycle; ++CurrCycle) {
      if (isTop())
        HazardRec->AdvanceCycle();
      else
        HazardRec->RecedeCycle();
    }
  }
  CheckPending = true;
  IsResourceLimited = getCriticalCount() > std::max(ExpectedLatency, CurrCycle);

  DEBUG(dbgs() << "  *** " << Available.getName() << " cycle "
        << CurrCycle << '\n');
}

/// Add the given processor resource to this scheduled zone.
void ConvergingScheduler::SchedBoundary::countResource(unsigned PIdx,
                                                       unsigned Cycles) {
  unsigned Factor = SchedModel->getResourceFactor(PIdx);
  DEBUG(dbgs() << "  " << SchedModel->getProcResource(PIdx)->Name
        << " +(" << Cycles << "x" << Factor
        << ") / " << SchedModel->getLatencyFactor() << '\n');

  unsigned Count = Factor * Cycles;
  ResourceCounts[PIdx] += Count;
  assert(Rem->RemainingCounts[PIdx] >= Count && "resource double counted");
  Rem->RemainingCounts[PIdx] -= Count;

  // Reset MaxRemainingCount for sanity.
  Rem->MaxRemainingCount = 0;

  // Check if this resource exceeds the current critical resource by a full
  // cycle. If so, it becomes the critical resource.
  if ((int)(ResourceCounts[PIdx] - ResourceCounts[CritResIdx])
      >= (int)SchedModel->getLatencyFactor()) {
    CritResIdx = PIdx;
    DEBUG(dbgs() << "  *** Critical resource "
          << SchedModel->getProcResource(PIdx)->Name << " x"
          << ResourceCounts[PIdx] << '\n');
  }
}

/// Move the boundary of scheduled code by one SUnit.
void ConvergingScheduler::SchedBoundary::bumpNode(SUnit *SU) {
  // Update the reservation table.
  if (HazardRec->isEnabled()) {
    if (!isTop() && SU->isCall) {
      // Calls are scheduled with their preceding instructions. For bottom-up
      // scheduling, clear the pipeline state before emitting.
      HazardRec->Reset();
    }
    HazardRec->EmitInstruction(SU);
  }
  // Update resource counts and critical resource.
  if (SchedModel->hasInstrSchedModel()) {
    const MCSchedClassDesc *SC = DAG->getSchedClass(SU);
    Rem->RemainingMicroOps -= SchedModel->getNumMicroOps(SU->getInstr(), SC);
    for (TargetSchedModel::ProcResIter
           PI = SchedModel->getWriteProcResBegin(SC),
           PE = SchedModel->getWriteProcResEnd(SC); PI != PE; ++PI) {
      countResource(PI->ProcResourceIdx, PI->Cycles);
    }
  }
  if (isTop()) {
    if (SU->getDepth() > ExpectedLatency)
      ExpectedLatency = SU->getDepth();
  }
  else {
    if (SU->getHeight() > ExpectedLatency)
      ExpectedLatency = SU->getHeight();
  }

  IsResourceLimited = getCriticalCount() > std::max(ExpectedLatency, CurrCycle);

  // Check the instruction group dispatch limit.
  // TODO: Check if this SU must end a dispatch group.
  IssueCount += SchedModel->getNumMicroOps(SU->getInstr());

  // checkHazard prevents scheduling multiple instructions per cycle that exceed
  // issue width. However, we commonly reach the maximum. In this case
  // opportunistically bump the cycle to avoid uselessly checking everything in
  // the readyQ. Furthermore, a single instruction may produce more than one
  // cycle's worth of micro-ops.
  if (IssueCount >= SchedModel->getIssueWidth()) {
    DEBUG(dbgs() << "  *** Max instrs at cycle " << CurrCycle << '\n');
    bumpCycle();
  }
}

/// Release pending ready nodes in to the available queue. This makes them
/// visible to heuristics.
void ConvergingScheduler::SchedBoundary::releasePending() {
  // If the available queue is empty, it is safe to reset MinReadyCycle.
  if (Available.empty())
    MinReadyCycle = UINT_MAX;

  // Check to see if any of the pending instructions are ready to issue.  If
  // so, add them to the available queue.
  for (unsigned i = 0, e = Pending.size(); i != e; ++i) {
    SUnit *SU = *(Pending.begin()+i);
    unsigned ReadyCycle = isTop() ? SU->TopReadyCycle : SU->BotReadyCycle;

    if (ReadyCycle < MinReadyCycle)
      MinReadyCycle = ReadyCycle;

    if (ReadyCycle > CurrCycle)
      continue;

    if (checkHazard(SU))
      continue;

    Available.push(SU);
    Pending.remove(Pending.begin()+i);
    --i; --e;
  }
  DEBUG(if (!Pending.empty()) Pending.dump());
  CheckPending = false;
}

/// Remove SU from the ready set for this boundary.
void ConvergingScheduler::SchedBoundary::removeReady(SUnit *SU) {
  if (Available.isInQueue(SU))
    Available.remove(Available.find(SU));
  else {
    assert(Pending.isInQueue(SU) && "bad ready count");
    Pending.remove(Pending.find(SU));
  }
}

/// If this queue only has one ready candidate, return it. As a side effect,
/// defer any nodes that now hit a hazard, and advance the cycle until at least
/// one node is ready. If multiple instructions are ready, return NULL.
SUnit *ConvergingScheduler::SchedBoundary::pickOnlyChoice() {
  if (CheckPending)
    releasePending();

  if (IssueCount > 0) {
    // Defer any ready instrs that now have a hazard.
    for (ReadyQueue::iterator I = Available.begin(); I != Available.end();) {
      if (checkHazard(*I)) {
        Pending.push(*I);
        I = Available.remove(I);
        continue;
      }
      ++I;
    }
  }
  for (unsigned i = 0; Available.empty(); ++i) {
    assert(i <= (HazardRec->getMaxLookAhead() + MaxMinLatency) &&
           "permanent hazard"); (void)i;
    bumpCycle();
    releasePending();
  }
  if (Available.size() == 1)
    return *Available.begin();
  return NULL;
}

/// Record the candidate policy for opposite zones with different critical
/// resources.
///
/// If the CriticalZone is latency limited, don't force a policy for the
/// candidates here. Instead, When releasing each candidate, releaseNode
/// compares the region's critical path to the candidate's height or depth and
/// the scheduled zone's expected latency then sets ShouldIncreaseILP.
void ConvergingScheduler::balanceZones(
  ConvergingScheduler::SchedBoundary &CriticalZone,
  ConvergingScheduler::SchedCandidate &CriticalCand,
  ConvergingScheduler::SchedBoundary &OppositeZone,
  ConvergingScheduler::SchedCandidate &OppositeCand) {

  if (!CriticalZone.IsResourceLimited)
    return;

  SchedRemainder *Rem = CriticalZone.Rem;

  // If the critical zone is overconsuming a resource relative to the
  // remainder, try to reduce it.
  unsigned RemainingCritCount =
    Rem->RemainingCounts[CriticalZone.CritResIdx];
  if ((int)(Rem->MaxRemainingCount - RemainingCritCount)
      > (int)SchedModel->getLatencyFactor()) {
    CriticalCand.Policy.ReduceResIdx = CriticalZone.CritResIdx;
    DEBUG(dbgs() << "Balance " << CriticalZone.Available.getName() << " reduce "
          << SchedModel->getProcResource(CriticalZone.CritResIdx)->Name
          << '\n');
  }
  // If the other zone is underconsuming a resource relative to the full zone,
  // try to increase it.
  unsigned OppositeCount =
    OppositeZone.ResourceCounts[CriticalZone.CritResIdx];
  if ((int)(OppositeZone.ExpectedCount - OppositeCount)
      > (int)SchedModel->getLatencyFactor()) {
    OppositeCand.Policy.DemandResIdx = CriticalZone.CritResIdx;
    DEBUG(dbgs() << "Balance " << OppositeZone.Available.getName() << " demand "
          << SchedModel->getProcResource(OppositeZone.CritResIdx)->Name
          << '\n');
  }
}

/// Determine if the scheduled zones exceed resource limits or critical path and
/// set each candidate's ReduceHeight policy accordingly.
void ConvergingScheduler::checkResourceLimits(
  ConvergingScheduler::SchedCandidate &TopCand,
  ConvergingScheduler::SchedCandidate &BotCand) {

  Bot.checkILPPolicy();
  Top.checkILPPolicy();
  if (Bot.ShouldIncreaseILP)
    BotCand.Policy.ReduceLatency = true;
  if (Top.ShouldIncreaseILP)
    TopCand.Policy.ReduceLatency = true;

  // Handle resource-limited regions.
  if (Top.IsResourceLimited && Bot.IsResourceLimited
      && Top.CritResIdx == Bot.CritResIdx) {
    // If the scheduled critical resource in both zones is no longer the
    // critical remaining resource, attempt to reduce resource height both ways.
    if (Top.CritResIdx != Rem.CritResIdx) {
      TopCand.Policy.ReduceResIdx = Top.CritResIdx;
      BotCand.Policy.ReduceResIdx = Bot.CritResIdx;
      DEBUG(dbgs() << "Reduce scheduled "
            << SchedModel->getProcResource(Top.CritResIdx)->Name << '\n');
    }
    return;
  }
  // Handle latency-limited regions.
  if (!Top.IsResourceLimited && !Bot.IsResourceLimited) {
    // If the total scheduled expected latency exceeds the region's critical
    // path then reduce latency both ways.
    //
    // Just because a zone is not resource limited does not mean it is latency
    // limited. Unbuffered resource, such as max micro-ops may cause CurrCycle
    // to exceed expected latency.
    if ((Top.ExpectedLatency + Bot.ExpectedLatency >= Rem.CriticalPath)
        && (Rem.CriticalPath > Top.CurrCycle + Bot.CurrCycle)) {
      TopCand.Policy.ReduceLatency = true;
      BotCand.Policy.ReduceLatency = true;
      DEBUG(dbgs() << "Reduce scheduled latency " << Top.ExpectedLatency
            << " + " << Bot.ExpectedLatency << '\n');
    }
    return;
  }
  // The critical resource is different in each zone, so request balancing.

  // Compute the cost of each zone.
  Rem.MaxRemainingCount = std::max(
    Rem.RemainingMicroOps * SchedModel->getMicroOpFactor(),
    Rem.RemainingCounts[Rem.CritResIdx]);
  Top.ExpectedCount = std::max(Top.ExpectedLatency, Top.CurrCycle);
  Top.ExpectedCount = std::max(
    Top.getCriticalCount(),
    Top.ExpectedCount * SchedModel->getLatencyFactor());
  Bot.ExpectedCount = std::max(Bot.ExpectedLatency, Bot.CurrCycle);
  Bot.ExpectedCount = std::max(
    Bot.getCriticalCount(),
    Bot.ExpectedCount * SchedModel->getLatencyFactor());

  balanceZones(Top, TopCand, Bot, BotCand);
  balanceZones(Bot, BotCand, Top, TopCand);
}

void ConvergingScheduler::SchedCandidate::
initResourceDelta(const ScheduleDAGMI *DAG,
                  const TargetSchedModel *SchedModel) {
  if (!Policy.ReduceResIdx && !Policy.DemandResIdx)
    return;

  const MCSchedClassDesc *SC = DAG->getSchedClass(SU);
  for (TargetSchedModel::ProcResIter
         PI = SchedModel->getWriteProcResBegin(SC),
         PE = SchedModel->getWriteProcResEnd(SC); PI != PE; ++PI) {
    if (PI->ProcResourceIdx == Policy.ReduceResIdx)
      ResDelta.CritResources += PI->Cycles;
    if (PI->ProcResourceIdx == Policy.DemandResIdx)
      ResDelta.DemandedResources += PI->Cycles;
  }
}

/// Return true if this heuristic determines order.
static bool tryLess(unsigned TryVal, unsigned CandVal,
                    ConvergingScheduler::SchedCandidate &TryCand,
                    ConvergingScheduler::SchedCandidate &Cand,
                    ConvergingScheduler::CandReason Reason) {
  if (TryVal < CandVal) {
    TryCand.Reason = Reason;
    return true;
  }
  if (TryVal > CandVal) {
    if (Cand.Reason > Reason)
      Cand.Reason = Reason;
    return true;
  }
  return false;
}
static bool tryGreater(unsigned TryVal, unsigned CandVal,
                       ConvergingScheduler::SchedCandidate &TryCand,
                       ConvergingScheduler::SchedCandidate &Cand,
                       ConvergingScheduler::CandReason Reason) {
  if (TryVal > CandVal) {
    TryCand.Reason = Reason;
    return true;
  }
  if (TryVal < CandVal) {
    if (Cand.Reason > Reason)
      Cand.Reason = Reason;
    return true;
  }
  return false;
}

/// Apply a set of heursitics to a new candidate. Heuristics are currently
/// hierarchical. This may be more efficient than a graduated cost model because
/// we don't need to evaluate all aspects of the model for each node in the
/// queue. But it's really done to make the heuristics easier to debug and
/// statistically analyze.
///
/// \param Cand provides the policy and current best candidate.
/// \param TryCand refers to the next SUnit candidate, otherwise uninitialized.
/// \param Zone describes the scheduled zone that we are extending.
/// \param RPTracker describes reg pressure within the scheduled zone.
/// \param TempTracker is a scratch pressure tracker to reuse in queries.
void ConvergingScheduler::tryCandidate(SchedCandidate &Cand,
                                       SchedCandidate &TryCand,
                                       SchedBoundary &Zone,
                                       const RegPressureTracker &RPTracker,
                                       RegPressureTracker &TempTracker) {

  // Always initialize TryCand's RPDelta.
  TempTracker.getMaxPressureDelta(TryCand.SU->getInstr(), TryCand.RPDelta,
                                  DAG->getRegionCriticalPSets(),
                                  DAG->getRegPressure().MaxSetPressure);

  // Initialize the candidate if needed.
  if (!Cand.isValid()) {
    TryCand.Reason = NodeOrder;
    return;
  }
  // Avoid exceeding the target's limit.
  if (tryLess(TryCand.RPDelta.Excess.UnitIncrease,
              Cand.RPDelta.Excess.UnitIncrease, TryCand, Cand, SingleExcess))
    return;
  if (Cand.Reason == SingleExcess)
    Cand.Reason = MultiPressure;

  // Avoid increasing the max critical pressure in the scheduled region.
  if (tryLess(TryCand.RPDelta.CriticalMax.UnitIncrease,
              Cand.RPDelta.CriticalMax.UnitIncrease,
              TryCand, Cand, SingleCritical))
    return;
  if (Cand.Reason == SingleCritical)
    Cand.Reason = MultiPressure;

  // Avoid critical resource consumption and balance the schedule.
  TryCand.initResourceDelta(DAG, SchedModel);
  if (tryLess(TryCand.ResDelta.CritResources, Cand.ResDelta.CritResources,
              TryCand, Cand, ResourceReduce))
    return;
  if (tryGreater(TryCand.ResDelta.DemandedResources,
                 Cand.ResDelta.DemandedResources,
                 TryCand, Cand, ResourceDemand))
    return;

  // Avoid serializing long latency dependence chains.
  if (Cand.Policy.ReduceLatency) {
    if (Zone.isTop()) {
      if (Cand.SU->getDepth() * SchedModel->getLatencyFactor()
          > Zone.ExpectedCount) {
        if (tryLess(TryCand.SU->getDepth(), Cand.SU->getDepth(),
                    TryCand, Cand, TopDepthReduce))
          return;
      }
      if (tryGreater(TryCand.SU->getHeight(), Cand.SU->getHeight(),
                     TryCand, Cand, TopPathReduce))
        return;
    }
    else {
      if (Cand.SU->getHeight() * SchedModel->getLatencyFactor()
          > Zone.ExpectedCount) {
        if (tryLess(TryCand.SU->getHeight(), Cand.SU->getHeight(),
                    TryCand, Cand, BotHeightReduce))
          return;
      }
      if (tryGreater(TryCand.SU->getDepth(), Cand.SU->getDepth(),
                     TryCand, Cand, BotPathReduce))
        return;
    }
  }

  // Avoid increasing the max pressure of the entire region.
  if (tryLess(TryCand.RPDelta.CurrentMax.UnitIncrease,
              Cand.RPDelta.CurrentMax.UnitIncrease, TryCand, Cand, SingleMax))
    return;
  if (Cand.Reason == SingleMax)
    Cand.Reason = MultiPressure;

  // Prefer immediate defs/users of the last scheduled instruction. This is a
  // nice pressure avoidance strategy that also conserves the processor's
  // register renaming resources and keeps the machine code readable.
  if (Zone.NextSUs.count(TryCand.SU) && !Zone.NextSUs.count(Cand.SU)) {
    TryCand.Reason = NextDefUse;
    return;
  }
  if (!Zone.NextSUs.count(TryCand.SU) && Zone.NextSUs.count(Cand.SU)) {
    if (Cand.Reason > NextDefUse)
      Cand.Reason = NextDefUse;
    return;
  }
  // Fall through to original instruction order.
  if ((Zone.isTop() && TryCand.SU->NodeNum < Cand.SU->NodeNum)
      || (!Zone.isTop() && TryCand.SU->NodeNum > Cand.SU->NodeNum)) {
    TryCand.Reason = NodeOrder;
  }
}

/// pickNodeFromQueue helper that returns true if the LHS reg pressure effect is
/// more desirable than RHS from scheduling standpoint.
static bool compareRPDelta(const RegPressureDelta &LHS,
                           const RegPressureDelta &RHS) {
  // Compare each component of pressure in decreasing order of importance
  // without checking if any are valid. Invalid PressureElements are assumed to
  // have UnitIncrease==0, so are neutral.

  // Avoid increasing the max critical pressure in the scheduled region.
  if (LHS.Excess.UnitIncrease != RHS.Excess.UnitIncrease) {
    DEBUG(dbgs() << "RP excess top - bot: "
          << (LHS.Excess.UnitIncrease - RHS.Excess.UnitIncrease) << '\n');
    return LHS.Excess.UnitIncrease < RHS.Excess.UnitIncrease;
  }
  // Avoid increasing the max critical pressure in the scheduled region.
  if (LHS.CriticalMax.UnitIncrease != RHS.CriticalMax.UnitIncrease) {
    DEBUG(dbgs() << "RP critical top - bot: "
          << (LHS.CriticalMax.UnitIncrease - RHS.CriticalMax.UnitIncrease)
          << '\n');
    return LHS.CriticalMax.UnitIncrease < RHS.CriticalMax.UnitIncrease;
  }
  // Avoid increasing the max pressure of the entire region.
  if (LHS.CurrentMax.UnitIncrease != RHS.CurrentMax.UnitIncrease) {
    DEBUG(dbgs() << "RP current top - bot: "
          << (LHS.CurrentMax.UnitIncrease - RHS.CurrentMax.UnitIncrease)
          << '\n');
    return LHS.CurrentMax.UnitIncrease < RHS.CurrentMax.UnitIncrease;
  }
  return false;
}

#ifndef NDEBUG
const char *ConvergingScheduler::getReasonStr(
  ConvergingScheduler::CandReason Reason) {
  switch (Reason) {
  case NoCand:         return "NOCAND    ";
  case SingleExcess:   return "REG-EXCESS";
  case SingleCritical: return "REG-CRIT  ";
  case SingleMax:      return "REG-MAX   ";
  case MultiPressure:  return "REG-MULTI ";
  case ResourceReduce: return "RES-REDUCE";
  case ResourceDemand: return "RES-DEMAND";
  case TopDepthReduce: return "TOP-DEPTH ";
  case TopPathReduce:  return "TOP-PATH  ";
  case BotHeightReduce:return "BOT-HEIGHT";
  case BotPathReduce:  return "BOT-PATH  ";
  case NextDefUse:     return "DEF-USE   ";
  case NodeOrder:      return "ORDER     ";
  };
  llvm_unreachable("Unknown reason!");
}

void ConvergingScheduler::traceCandidate(const SchedCandidate &Cand,
                                         const SchedBoundary &Zone) {
  const char *Label = getReasonStr(Cand.Reason);
  PressureElement P;
  unsigned ResIdx = 0;
  unsigned Latency = 0;
  switch (Cand.Reason) {
  default:
    break;
  case SingleExcess:
    P = Cand.RPDelta.Excess;
    break;
  case SingleCritical:
    P = Cand.RPDelta.CriticalMax;
    break;
  case SingleMax:
    P = Cand.RPDelta.CurrentMax;
    break;
  case ResourceReduce:
    ResIdx = Cand.Policy.ReduceResIdx;
    break;
  case ResourceDemand:
    ResIdx = Cand.Policy.DemandResIdx;
    break;
  case TopDepthReduce:
    Latency = Cand.SU->getDepth();
    break;
  case TopPathReduce:
    Latency = Cand.SU->getHeight();
    break;
  case BotHeightReduce:
    Latency = Cand.SU->getHeight();
    break;
  case BotPathReduce:
    Latency = Cand.SU->getDepth();
    break;
  }
  dbgs() << Label << " " << Zone.Available.getName() << " ";
  if (P.isValid())
    dbgs() << TRI->getRegPressureSetName(P.PSetID) << ":" << P.UnitIncrease
           << " ";
  else
    dbgs() << "     ";
  if (ResIdx)
    dbgs() << SchedModel->getProcResource(ResIdx)->Name << " ";
  else
    dbgs() << "        ";
  if (Latency)
    dbgs() << Latency << " cycles ";
  else
    dbgs() << "         ";
  Cand.SU->dump(DAG);
}
#endif

/// Pick the best candidate from the top queue.
///
/// TODO: getMaxPressureDelta results can be mostly cached for each SUnit during
/// DAG building. To adjust for the current scheduling location we need to
/// maintain the number of vreg uses remaining to be top-scheduled.
void ConvergingScheduler::pickNodeFromQueue(SchedBoundary &Zone,
                                            const RegPressureTracker &RPTracker,
                                            SchedCandidate &Cand) {
  ReadyQueue &Q = Zone.Available;

  DEBUG(Q.dump());

  // getMaxPressureDelta temporarily modifies the tracker.
  RegPressureTracker &TempTracker = const_cast<RegPressureTracker&>(RPTracker);

  for (ReadyQueue::iterator I = Q.begin(), E = Q.end(); I != E; ++I) {

    SchedCandidate TryCand(Cand.Policy);
    TryCand.SU = *I;
    tryCandidate(Cand, TryCand, Zone, RPTracker, TempTracker);
    if (TryCand.Reason != NoCand) {
      // Initialize resource delta if needed in case future heuristics query it.
      if (TryCand.ResDelta == SchedResourceDelta())
        TryCand.initResourceDelta(DAG, SchedModel);
      Cand.setBest(TryCand);
      DEBUG(traceCandidate(Cand, Zone));
    }
    TryCand.SU = *I;
  }
}

static void tracePick(const ConvergingScheduler::SchedCandidate &Cand,
                      bool IsTop) {
  DEBUG(dbgs() << "Pick " << (IsTop ? "top" : "bot")
        << " SU(" << Cand.SU->NodeNum << ") "
        << ConvergingScheduler::getReasonStr(Cand.Reason) << '\n');
}

/// Pick the best candidate node from either the top or bottom queue.
SUnit *ConvergingScheduler::pickNodeBidirectional(bool &IsTopNode) {
  // Schedule as far as possible in the direction of no choice. This is most
  // efficient, but also provides the best heuristics for CriticalPSets.
  if (SUnit *SU = Bot.pickOnlyChoice()) {
    IsTopNode = false;
    return SU;
  }
  if (SUnit *SU = Top.pickOnlyChoice()) {
    IsTopNode = true;
    return SU;
  }
  CandPolicy NoPolicy;
  SchedCandidate BotCand(NoPolicy);
  SchedCandidate TopCand(NoPolicy);
  checkResourceLimits(TopCand, BotCand);

  // Prefer bottom scheduling when heuristics are silent.
  pickNodeFromQueue(Bot, DAG->getBotRPTracker(), BotCand);
  assert(BotCand.Reason != NoCand && "failed to find the first candidate");

  // If either Q has a single candidate that provides the least increase in
  // Excess pressure, we can immediately schedule from that Q.
  //
  // RegionCriticalPSets summarizes the pressure within the scheduled region and
  // affects picking from either Q. If scheduling in one direction must
  // increase pressure for one of the excess PSets, then schedule in that
  // direction first to provide more freedom in the other direction.
  if (BotCand.Reason == SingleExcess || BotCand.Reason == SingleCritical) {
    IsTopNode = false;
    tracePick(BotCand, IsTopNode);
    return BotCand.SU;
  }
  // Check if the top Q has a better candidate.
  pickNodeFromQueue(Top, DAG->getTopRPTracker(), TopCand);
  assert(TopCand.Reason != NoCand && "failed to find the first candidate");

  // If either Q has a single candidate that minimizes pressure above the
  // original region's pressure pick it.
  if (TopCand.Reason <= SingleMax || BotCand.Reason <= SingleMax) {
    if (TopCand.Reason < BotCand.Reason) {
      IsTopNode = true;
      tracePick(TopCand, IsTopNode);
      return TopCand.SU;
    }
    IsTopNode = false;
    tracePick(BotCand, IsTopNode);
    return BotCand.SU;
  }
  // Check for a salient pressure difference and pick the best from either side.
  if (compareRPDelta(TopCand.RPDelta, BotCand.RPDelta)) {
    IsTopNode = true;
    tracePick(TopCand, IsTopNode);
    return TopCand.SU;
  }
  // Otherwise prefer the bottom candidate, in node order if all else failed.
  if (TopCand.Reason < BotCand.Reason) {
    IsTopNode = true;
    tracePick(TopCand, IsTopNode);
    return TopCand.SU;
  }
  IsTopNode = false;
  tracePick(BotCand, IsTopNode);
  return BotCand.SU;
}

/// Pick the best node to balance the schedule. Implements MachineSchedStrategy.
SUnit *ConvergingScheduler::pickNode(bool &IsTopNode) {
  if (DAG->top() == DAG->bottom()) {
    assert(Top.Available.empty() && Top.Pending.empty() &&
           Bot.Available.empty() && Bot.Pending.empty() && "ReadyQ garbage");
    return NULL;
  }
  SUnit *SU;
  do {
    if (ForceTopDown) {
      SU = Top.pickOnlyChoice();
      if (!SU) {
        CandPolicy NoPolicy;
        SchedCandidate TopCand(NoPolicy);
        pickNodeFromQueue(Top, DAG->getTopRPTracker(), TopCand);
        assert(TopCand.Reason != NoCand && "failed to find the first candidate");
        SU = TopCand.SU;
      }
      IsTopNode = true;
    }
    else if (ForceBottomUp) {
      SU = Bot.pickOnlyChoice();
      if (!SU) {
        CandPolicy NoPolicy;
        SchedCandidate BotCand(NoPolicy);
        pickNodeFromQueue(Bot, DAG->getBotRPTracker(), BotCand);
        assert(BotCand.Reason != NoCand && "failed to find the first candidate");
        SU = BotCand.SU;
      }
      IsTopNode = false;
    }
    else {
      SU = pickNodeBidirectional(IsTopNode);
    }
  } while (SU->isScheduled);

  if (SU->isTopReady())
    Top.removeReady(SU);
  if (SU->isBottomReady())
    Bot.removeReady(SU);

  DEBUG(dbgs() << "*** " << (IsTopNode ? "Top" : "Bottom")
        << " Scheduling Instruction in cycle "
        << (IsTopNode ? Top.CurrCycle : Bot.CurrCycle) << '\n';
        SU->dump(DAG));
  return SU;
}

/// Update the scheduler's state after scheduling a node. This is the same node
/// that was just returned by pickNode(). However, ScheduleDAGMI needs to update
/// it's state based on the current cycle before MachineSchedStrategy does.
void ConvergingScheduler::schedNode(SUnit *SU, bool IsTopNode) {
  if (IsTopNode) {
    SU->TopReadyCycle = Top.CurrCycle;
    Top.bumpNode(SU);
  }
  else {
    SU->BotReadyCycle = Bot.CurrCycle;
    Bot.bumpNode(SU);
  }
}

/// Create the standard converging machine scheduler. This will be used as the
/// default scheduler if the target does not set a default.
static ScheduleDAGInstrs *createConvergingSched(MachineSchedContext *C) {
  assert((!ForceTopDown || !ForceBottomUp) &&
         "-misched-topdown incompatible with -misched-bottomup");
  return new ScheduleDAGMI(C, new ConvergingScheduler());
}
static MachineSchedRegistry
ConvergingSchedRegistry("converge", "Standard converging scheduler.",
                        createConvergingSched);

//===----------------------------------------------------------------------===//
// ILP Scheduler. Currently for experimental analysis of heuristics.
//===----------------------------------------------------------------------===//

namespace {
/// \brief Order nodes by the ILP metric.
struct ILPOrder {
  ScheduleDAGILP *ILP;
  bool MaximizeILP;

  ILPOrder(ScheduleDAGILP *ilp, bool MaxILP): ILP(ilp), MaximizeILP(MaxILP) {}

  /// \brief Apply a less-than relation on node priority.
  bool operator()(const SUnit *A, const SUnit *B) const {
    // Return true if A comes after B in the Q.
    if (MaximizeILP)
      return ILP->getILP(A) < ILP->getILP(B);
    else
      return ILP->getILP(A) > ILP->getILP(B);
  }
};

/// \brief Schedule based on the ILP metric.
class ILPScheduler : public MachineSchedStrategy {
  ScheduleDAGILP ILP;
  ILPOrder Cmp;

  std::vector<SUnit*> ReadyQ;
public:
  ILPScheduler(bool MaximizeILP)
  : ILP(/*BottomUp=*/true), Cmp(&ILP, MaximizeILP) {}

  virtual void initialize(ScheduleDAGMI *DAG) {
    ReadyQ.clear();
    ILP.resize(DAG->SUnits.size());
  }

  virtual void registerRoots() {
    for (std::vector<SUnit*>::const_iterator
           I = ReadyQ.begin(), E = ReadyQ.end(); I != E; ++I) {
      ILP.computeILP(*I);
    }
  }

  /// Implement MachineSchedStrategy interface.
  /// -----------------------------------------

  virtual SUnit *pickNode(bool &IsTopNode) {
    if (ReadyQ.empty()) return NULL;
    pop_heap(ReadyQ.begin(), ReadyQ.end(), Cmp);
    SUnit *SU = ReadyQ.back();
    ReadyQ.pop_back();
    IsTopNode = false;
    DEBUG(dbgs() << "*** Scheduling " << *SU->getInstr()
          << " ILP: " << ILP.getILP(SU) << '\n');
    return SU;
  }

  virtual void schedNode(SUnit *, bool) {}

  virtual void releaseTopNode(SUnit *) { /*only called for top roots*/ }

  virtual void releaseBottomNode(SUnit *SU) {
    ReadyQ.push_back(SU);
    std::push_heap(ReadyQ.begin(), ReadyQ.end(), Cmp);
  }
};
} // namespace

static ScheduleDAGInstrs *createILPMaxScheduler(MachineSchedContext *C) {
  return new ScheduleDAGMI(C, new ILPScheduler(true));
}
static ScheduleDAGInstrs *createILPMinScheduler(MachineSchedContext *C) {
  return new ScheduleDAGMI(C, new ILPScheduler(false));
}
static MachineSchedRegistry ILPMaxRegistry(
  "ilpmax", "Schedule bottom-up for max ILP", createILPMaxScheduler);
static MachineSchedRegistry ILPMinRegistry(
  "ilpmin", "Schedule bottom-up for min ILP", createILPMinScheduler);

//===----------------------------------------------------------------------===//
// Machine Instruction Shuffler for Correctness Testing
//===----------------------------------------------------------------------===//

#ifndef NDEBUG
namespace {
/// Apply a less-than relation on the node order, which corresponds to the
/// instruction order prior to scheduling. IsReverse implements greater-than.
template<bool IsReverse>
struct SUnitOrder {
  bool operator()(SUnit *A, SUnit *B) const {
    if (IsReverse)
      return A->NodeNum > B->NodeNum;
    else
      return A->NodeNum < B->NodeNum;
  }
};

/// Reorder instructions as much as possible.
class InstructionShuffler : public MachineSchedStrategy {
  bool IsAlternating;
  bool IsTopDown;

  // Using a less-than relation (SUnitOrder<false>) for the TopQ priority
  // gives nodes with a higher number higher priority causing the latest
  // instructions to be scheduled first.
  PriorityQueue<SUnit*, std::vector<SUnit*>, SUnitOrder<false> >
    TopQ;
  // When scheduling bottom-up, use greater-than as the queue priority.
  PriorityQueue<SUnit*, std::vector<SUnit*>, SUnitOrder<true> >
    BottomQ;
public:
  InstructionShuffler(bool alternate, bool topdown)
    : IsAlternating(alternate), IsTopDown(topdown) {}

  virtual void initialize(ScheduleDAGMI *) {
    TopQ.clear();
    BottomQ.clear();
  }

  /// Implement MachineSchedStrategy interface.
  /// -----------------------------------------

  virtual SUnit *pickNode(bool &IsTopNode) {
    SUnit *SU;
    if (IsTopDown) {
      do {
        if (TopQ.empty()) return NULL;
        SU = TopQ.top();
        TopQ.pop();
      } while (SU->isScheduled);
      IsTopNode = true;
    }
    else {
      do {
        if (BottomQ.empty()) return NULL;
        SU = BottomQ.top();
        BottomQ.pop();
      } while (SU->isScheduled);
      IsTopNode = false;
    }
    if (IsAlternating)
      IsTopDown = !IsTopDown;
    return SU;
  }

  virtual void schedNode(SUnit *SU, bool IsTopNode) {}

  virtual void releaseTopNode(SUnit *SU) {
    TopQ.push(SU);
  }
  virtual void releaseBottomNode(SUnit *SU) {
    BottomQ.push(SU);
  }
};
} // namespace

static ScheduleDAGInstrs *createInstructionShuffler(MachineSchedContext *C) {
  bool Alternate = !ForceTopDown && !ForceBottomUp;
  bool TopDown = !ForceBottomUp;
  assert((TopDown || !ForceTopDown) &&
         "-misched-topdown incompatible with -misched-bottomup");
  return new ScheduleDAGMI(C, new InstructionShuffler(Alternate, TopDown));
}
static MachineSchedRegistry ShufflerRegistry(
  "shuffle", "Shuffle machine instructions alternating directions",
  createInstructionShuffler);
#endif // !NDEBUG
