//===----- SchedulePostRAList.cpp - list scheduler ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements a top-down list scheduler, using standard algorithms.
// The basic approach uses a priority queue of available nodes to schedule.
// One at a time, nodes are taken from the priority queue (thus in priority
// order), checked for legality to schedule, and emitted if legal.
//
// Nodes may not be legal to schedule either due to structural hazards (e.g.
// pipeline or resource constraints) or because an input to the instruction has
// not completed execution.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "post-RA-sched"
#include "AntiDepBreaker.h"
#include "AggressiveAntiDepBreaker.h"
#include "CriticalAntiDepBreaker.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/LatencyPriorityQueue.h"
#include "llvm/CodeGen/SchedulerRegistry.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include "llvm/CodeGen/ScheduleHazardRecognizer.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(NumNoops, "Number of noops inserted");
STATISTIC(NumStalls, "Number of pipeline stalls");
STATISTIC(NumFixedAnti, "Number of fixed anti-dependencies");

// Post-RA scheduling is enabled with
// TargetSubtargetInfo.enablePostRAScheduler(). This flag can be used to
// override the target.
static cl::opt<bool>
EnablePostRAScheduler("post-RA-scheduler",
                       cl::desc("Enable scheduling after register allocation"),
                       cl::init(false), cl::Hidden);
static cl::opt<std::string>
EnableAntiDepBreaking("break-anti-dependencies",
                      cl::desc("Break post-RA scheduling anti-dependencies: "
                               "\"critical\", \"all\", or \"none\""),
                      cl::init("none"), cl::Hidden);

// If DebugDiv > 0 then only schedule MBB with (ID % DebugDiv) == DebugMod
static cl::opt<int>
DebugDiv("postra-sched-debugdiv",
                      cl::desc("Debug control MBBs that are scheduled"),
                      cl::init(0), cl::Hidden);
static cl::opt<int>
DebugMod("postra-sched-debugmod",
                      cl::desc("Debug control MBBs that are scheduled"),
                      cl::init(0), cl::Hidden);

AntiDepBreaker::~AntiDepBreaker() { }

namespace {
  class PostRAScheduler : public MachineFunctionPass {
    const TargetInstrInfo *TII;
    RegisterClassInfo RegClassInfo;

  public:
    static char ID;
    PostRAScheduler() : MachineFunctionPass(ID) {}

    void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<AliasAnalysis>();
      AU.addRequired<TargetPassConfig>();
      AU.addRequired<MachineDominatorTree>();
      AU.addPreserved<MachineDominatorTree>();
      AU.addRequired<MachineLoopInfo>();
      AU.addPreserved<MachineLoopInfo>();
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    bool runOnMachineFunction(MachineFunction &Fn);
  };
  char PostRAScheduler::ID = 0;

  class SchedulePostRATDList : public ScheduleDAGInstrs {
    /// AvailableQueue - The priority queue to use for the available SUnits.
    ///
    LatencyPriorityQueue AvailableQueue;

    /// PendingQueue - This contains all of the instructions whose operands have
    /// been issued, but their results are not ready yet (due to the latency of
    /// the operation).  Once the operands becomes available, the instruction is
    /// added to the AvailableQueue.
    std::vector<SUnit*> PendingQueue;

    /// Topo - A topological ordering for SUnits.
    ScheduleDAGTopologicalSort Topo;

    /// HazardRec - The hazard recognizer to use.
    ScheduleHazardRecognizer *HazardRec;

    /// AntiDepBreak - Anti-dependence breaking object, or NULL if none
    AntiDepBreaker *AntiDepBreak;

    /// AA - AliasAnalysis for making memory reference queries.
    AliasAnalysis *AA;

    /// LiveRegs - true if the register is live.
    BitVector LiveRegs;

    /// The schedule. Null SUnit*'s represent noop instructions.
    std::vector<SUnit*> Sequence;

  public:
    SchedulePostRATDList(
      MachineFunction &MF, MachineLoopInfo &MLI, MachineDominatorTree &MDT,
      AliasAnalysis *AA, const RegisterClassInfo&,
      TargetSubtargetInfo::AntiDepBreakMode AntiDepMode,
      SmallVectorImpl<const TargetRegisterClass*> &CriticalPathRCs);

    ~SchedulePostRATDList();

    /// startBlock - Initialize register live-range state for scheduling in
    /// this block.
    ///
    void startBlock(MachineBasicBlock *BB);

    /// Initialize the scheduler state for the next scheduling region.
    virtual void enterRegion(MachineBasicBlock *bb,
                             MachineBasicBlock::iterator begin,
                             MachineBasicBlock::iterator end,
                             unsigned endcount);

    /// Notify that the scheduler has finished scheduling the current region.
    virtual void exitRegion();

    /// Schedule - Schedule the instruction range using list scheduling.
    ///
    void schedule();

    void EmitSchedule();

    /// Observe - Update liveness information to account for the current
    /// instruction, which will not be scheduled.
    ///
    void Observe(MachineInstr *MI, unsigned Count);

    /// finishBlock - Clean up register live-range state.
    ///
    void finishBlock();

    /// FixupKills - Fix register kill flags that have been made
    /// invalid due to scheduling
    ///
    void FixupKills(MachineBasicBlock *MBB);

  private:
    void ReleaseSucc(SUnit *SU, SDep *SuccEdge);
    void ReleaseSuccessors(SUnit *SU);
    void ScheduleNodeTopDown(SUnit *SU, unsigned CurCycle);
    void ListScheduleTopDown();
    void StartBlockForKills(MachineBasicBlock *BB);

    // ToggleKillFlag - Toggle a register operand kill flag. Other
    // adjustments may be made to the instruction if necessary. Return
    // true if the operand has been deleted, false if not.
    bool ToggleKillFlag(MachineInstr *MI, MachineOperand &MO);

    void dumpSchedule() const;
  };
}

char &llvm::PostRASchedulerID = PostRAScheduler::ID;

INITIALIZE_PASS(PostRAScheduler, "post-RA-sched",
                "Post RA top-down list latency scheduler", false, false)

SchedulePostRATDList::SchedulePostRATDList(
  MachineFunction &MF, MachineLoopInfo &MLI, MachineDominatorTree &MDT,
  AliasAnalysis *AA, const RegisterClassInfo &RCI,
  TargetSubtargetInfo::AntiDepBreakMode AntiDepMode,
  SmallVectorImpl<const TargetRegisterClass*> &CriticalPathRCs)
  : ScheduleDAGInstrs(MF, MLI, MDT, /*IsPostRA=*/true), Topo(SUnits), AA(AA),
    LiveRegs(TRI->getNumRegs())
{
  const TargetMachine &TM = MF.getTarget();
  const InstrItineraryData *InstrItins = TM.getInstrItineraryData();
  HazardRec =
    TM.getInstrInfo()->CreateTargetPostRAHazardRecognizer(InstrItins, this);

  assert((AntiDepMode == TargetSubtargetInfo::ANTIDEP_NONE ||
          MRI.tracksLiveness()) &&
         "Live-ins must be accurate for anti-dependency breaking");
  AntiDepBreak =
    ((AntiDepMode == TargetSubtargetInfo::ANTIDEP_ALL) ?
     (AntiDepBreaker *)new AggressiveAntiDepBreaker(MF, RCI, CriticalPathRCs) :
     ((AntiDepMode == TargetSubtargetInfo::ANTIDEP_CRITICAL) ?
      (AntiDepBreaker *)new CriticalAntiDepBreaker(MF, RCI) : NULL));
}

SchedulePostRATDList::~SchedulePostRATDList() {
  delete HazardRec;
  delete AntiDepBreak;
}

/// Initialize state associated with the next scheduling region.
void SchedulePostRATDList::enterRegion(MachineBasicBlock *bb,
                 MachineBasicBlock::iterator begin,
                 MachineBasicBlock::iterator end,
                 unsigned endcount) {
  ScheduleDAGInstrs::enterRegion(bb, begin, end, endcount);
  Sequence.clear();
}

/// Print the schedule before exiting the region.
void SchedulePostRATDList::exitRegion() {
  DEBUG({
      dbgs() << "*** Final schedule ***\n";
      dumpSchedule();
      dbgs() << '\n';
    });
  ScheduleDAGInstrs::exitRegion();
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
/// dumpSchedule - dump the scheduled Sequence.
void SchedulePostRATDList::dumpSchedule() const {
  for (unsigned i = 0, e = Sequence.size(); i != e; i++) {
    if (SUnit *SU = Sequence[i])
      SU->dump(this);
    else
      dbgs() << "**** NOOP ****\n";
  }
}
#endif

bool PostRAScheduler::runOnMachineFunction(MachineFunction &Fn) {
  TII = Fn.getTarget().getInstrInfo();
  MachineLoopInfo &MLI = getAnalysis<MachineLoopInfo>();
  MachineDominatorTree &MDT = getAnalysis<MachineDominatorTree>();
  AliasAnalysis *AA = &getAnalysis<AliasAnalysis>();
  TargetPassConfig *PassConfig = &getAnalysis<TargetPassConfig>();

  RegClassInfo.runOnMachineFunction(Fn);

  // Check for explicit enable/disable of post-ra scheduling.
  TargetSubtargetInfo::AntiDepBreakMode AntiDepMode =
    TargetSubtargetInfo::ANTIDEP_NONE;
  SmallVector<const TargetRegisterClass*, 4> CriticalPathRCs;
  if (EnablePostRAScheduler.getPosition() > 0) {
    if (!EnablePostRAScheduler)
      return false;
  } else {
    // Check that post-RA scheduling is enabled for this target.
    // This may upgrade the AntiDepMode.
    const TargetSubtargetInfo &ST = Fn.getTarget().getSubtarget<TargetSubtargetInfo>();
    if (!ST.enablePostRAScheduler(PassConfig->getOptLevel(), AntiDepMode,
                                  CriticalPathRCs))
      return false;
  }

  // Check for antidep breaking override...
  if (EnableAntiDepBreaking.getPosition() > 0) {
    AntiDepMode = (EnableAntiDepBreaking == "all")
      ? TargetSubtargetInfo::ANTIDEP_ALL
      : ((EnableAntiDepBreaking == "critical")
         ? TargetSubtargetInfo::ANTIDEP_CRITICAL
         : TargetSubtargetInfo::ANTIDEP_NONE);
  }

  DEBUG(dbgs() << "PostRAScheduler\n");

  SchedulePostRATDList Scheduler(Fn, MLI, MDT, AA, RegClassInfo, AntiDepMode,
                                 CriticalPathRCs);

  // Loop over all of the basic blocks
  for (MachineFunction::iterator MBB = Fn.begin(), MBBe = Fn.end();
       MBB != MBBe; ++MBB) {
#ifndef NDEBUG
    // If DebugDiv > 0 then only schedule MBB with (ID % DebugDiv) == DebugMod
    if (DebugDiv > 0) {
      static int bbcnt = 0;
      if (bbcnt++ % DebugDiv != DebugMod)
        continue;
      dbgs() << "*** DEBUG scheduling " << Fn.getName()
             << ":BB#" << MBB->getNumber() << " ***\n";
    }
#endif

    // Initialize register live-range state for scheduling in this block.
    Scheduler.startBlock(MBB);

    // Schedule each sequence of instructions not interrupted by a label
    // or anything else that effectively needs to shut down scheduling.
    MachineBasicBlock::iterator Current = MBB->end();
    unsigned Count = MBB->size(), CurrentCount = Count;
    for (MachineBasicBlock::iterator I = Current; I != MBB->begin(); ) {
      MachineInstr *MI = llvm::prior(I);
      // Calls are not scheduling boundaries before register allocation, but
      // post-ra we don't gain anything by scheduling across calls since we
      // don't need to worry about register pressure.
      if (MI->isCall() || TII->isSchedulingBoundary(MI, MBB, Fn)) {
        Scheduler.enterRegion(MBB, I, Current, CurrentCount);
        Scheduler.schedule();
        Scheduler.exitRegion();
        Scheduler.EmitSchedule();
        Current = MI;
        CurrentCount = Count - 1;
        Scheduler.Observe(MI, CurrentCount);
      }
      I = MI;
      --Count;
      if (MI->isBundle())
        Count -= MI->getBundleSize();
    }
    assert(Count == 0 && "Instruction count mismatch!");
    assert((MBB->begin() == Current || CurrentCount != 0) &&
           "Instruction count mismatch!");
    Scheduler.enterRegion(MBB, MBB->begin(), Current, CurrentCount);
    Scheduler.schedule();
    Scheduler.exitRegion();
    Scheduler.EmitSchedule();

    // Clean up register live-range state.
    Scheduler.finishBlock();

    // Update register kills
    Scheduler.FixupKills(MBB);
  }

  return true;
}

/// StartBlock - Initialize register live-range state for scheduling in
/// this block.
///
void SchedulePostRATDList::startBlock(MachineBasicBlock *BB) {
  // Call the superclass.
  ScheduleDAGInstrs::startBlock(BB);

  // Reset the hazard recognizer and anti-dep breaker.
  HazardRec->Reset();
  if (AntiDepBreak != NULL)
    AntiDepBreak->StartBlock(BB);
}

/// Schedule - Schedule the instruction range using list scheduling.
///
void SchedulePostRATDList::schedule() {
  // Build the scheduling graph.
  buildSchedGraph(AA);

  if (AntiDepBreak != NULL) {
    unsigned Broken =
      AntiDepBreak->BreakAntiDependencies(SUnits, RegionBegin, RegionEnd,
                                          EndIndex, DbgValues);

    if (Broken != 0) {
      // We made changes. Update the dependency graph.
      // Theoretically we could update the graph in place:
      // When a live range is changed to use a different register, remove
      // the def's anti-dependence *and* output-dependence edges due to
      // that register, and add new anti-dependence and output-dependence
      // edges based on the next live range of the register.
      ScheduleDAG::clearDAG();
      buildSchedGraph(AA);

      NumFixedAnti += Broken;
    }
  }

  DEBUG(dbgs() << "********** List Scheduling **********\n");
  DEBUG(for (unsigned su = 0, e = SUnits.size(); su != e; ++su)
          SUnits[su].dumpAll(this));

  AvailableQueue.initNodes(SUnits);
  ListScheduleTopDown();
  AvailableQueue.releaseState();
}

/// Observe - Update liveness information to account for the current
/// instruction, which will not be scheduled.
///
void SchedulePostRATDList::Observe(MachineInstr *MI, unsigned Count) {
  if (AntiDepBreak != NULL)
    AntiDepBreak->Observe(MI, Count, EndIndex);
}

/// FinishBlock - Clean up register live-range state.
///
void SchedulePostRATDList::finishBlock() {
  if (AntiDepBreak != NULL)
    AntiDepBreak->FinishBlock();

  // Call the superclass.
  ScheduleDAGInstrs::finishBlock();
}

/// StartBlockForKills - Initialize register live-range state for updating kills
///
void SchedulePostRATDList::StartBlockForKills(MachineBasicBlock *BB) {
  // Start with no live registers.
  LiveRegs.reset();

  // Determine the live-out physregs for this block.
  if (!BB->empty() && BB->back().isReturn()) {
    // In a return block, examine the function live-out regs.
    for (MachineRegisterInfo::liveout_iterator I = MRI.liveout_begin(),
           E = MRI.liveout_end(); I != E; ++I) {
      unsigned Reg = *I;
      LiveRegs.set(Reg);
      // Repeat, for all subregs.
      for (MCSubRegIterator SubRegs(Reg, TRI); SubRegs.isValid(); ++SubRegs)
        LiveRegs.set(*SubRegs);
    }
  }
  else {
    // In a non-return block, examine the live-in regs of all successors.
    for (MachineBasicBlock::succ_iterator SI = BB->succ_begin(),
           SE = BB->succ_end(); SI != SE; ++SI) {
      for (MachineBasicBlock::livein_iterator I = (*SI)->livein_begin(),
             E = (*SI)->livein_end(); I != E; ++I) {
        unsigned Reg = *I;
        LiveRegs.set(Reg);
        // Repeat, for all subregs.
        for (MCSubRegIterator SubRegs(Reg, TRI); SubRegs.isValid(); ++SubRegs)
          LiveRegs.set(*SubRegs);
      }
    }
  }
}

bool SchedulePostRATDList::ToggleKillFlag(MachineInstr *MI,
                                          MachineOperand &MO) {
  // Setting kill flag...
  if (!MO.isKill()) {
    MO.setIsKill(true);
    return false;
  }

  // If MO itself is live, clear the kill flag...
  if (LiveRegs.test(MO.getReg())) {
    MO.setIsKill(false);
    return false;
  }

  // If any subreg of MO is live, then create an imp-def for that
  // subreg and keep MO marked as killed.
  MO.setIsKill(false);
  bool AllDead = true;
  const unsigned SuperReg = MO.getReg();
  for (MCSubRegIterator SubRegs(SuperReg, TRI); SubRegs.isValid(); ++SubRegs) {
    if (LiveRegs.test(*SubRegs)) {
      MI->addOperand(MachineOperand::CreateReg(*SubRegs,
                                               true  /*IsDef*/,
                                               true  /*IsImp*/,
                                               false /*IsKill*/,
                                               false /*IsDead*/));
      AllDead = false;
    }
  }

  if(AllDead)
    MO.setIsKill(true);
  return false;
}

/// FixupKills - Fix the register kill flags, they may have been made
/// incorrect by instruction reordering.
///
void SchedulePostRATDList::FixupKills(MachineBasicBlock *MBB) {
  DEBUG(dbgs() << "Fixup kills for BB#" << MBB->getNumber() << '\n');

  BitVector killedRegs(TRI->getNumRegs());

  StartBlockForKills(MBB);

  // Examine block from end to start...
  unsigned Count = MBB->size();
  for (MachineBasicBlock::iterator I = MBB->end(), E = MBB->begin();
       I != E; --Count) {
    MachineInstr *MI = --I;
    if (MI->isDebugValue())
      continue;

    // Update liveness.  Registers that are defed but not used in this
    // instruction are now dead. Mark register and all subregs as they
    // are completely defined.
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (MO.isRegMask())
        LiveRegs.clearBitsNotInMask(MO.getRegMask());
      if (!MO.isReg()) continue;
      unsigned Reg = MO.getReg();
      if (Reg == 0) continue;
      if (!MO.isDef()) continue;
      // Ignore two-addr defs.
      if (MI->isRegTiedToUseOperand(i)) continue;

      LiveRegs.reset(Reg);

      // Repeat for all subregs.
      for (MCSubRegIterator SubRegs(Reg, TRI); SubRegs.isValid(); ++SubRegs)
        LiveRegs.reset(*SubRegs);
    }

    // Examine all used registers and set/clear kill flag. When a
    // register is used multiple times we only set the kill flag on
    // the first use.
    killedRegs.reset();
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (!MO.isReg() || !MO.isUse()) continue;
      unsigned Reg = MO.getReg();
      if ((Reg == 0) || MRI.isReserved(Reg)) continue;

      bool kill = false;
      if (!killedRegs.test(Reg)) {
        kill = true;
        // A register is not killed if any subregs are live...
        for (MCSubRegIterator SubRegs(Reg, TRI); SubRegs.isValid(); ++SubRegs) {
          if (LiveRegs.test(*SubRegs)) {
            kill = false;
            break;
          }
        }

        // If subreg is not live, then register is killed if it became
        // live in this instruction
        if (kill)
          kill = !LiveRegs.test(Reg);
      }

      if (MO.isKill() != kill) {
        DEBUG(dbgs() << "Fixing " << MO << " in ");
        // Warning: ToggleKillFlag may invalidate MO.
        ToggleKillFlag(MI, MO);
        DEBUG(MI->dump());
      }

      killedRegs.set(Reg);
    }

    // Mark any used register (that is not using undef) and subregs as
    // now live...
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (!MO.isReg() || !MO.isUse() || MO.isUndef()) continue;
      unsigned Reg = MO.getReg();
      if ((Reg == 0) || MRI.isReserved(Reg)) continue;

      LiveRegs.set(Reg);

      for (MCSubRegIterator SubRegs(Reg, TRI); SubRegs.isValid(); ++SubRegs)
        LiveRegs.set(*SubRegs);
    }
  }
}

//===----------------------------------------------------------------------===//
//  Top-Down Scheduling
//===----------------------------------------------------------------------===//

/// ReleaseSucc - Decrement the NumPredsLeft count of a successor. Add it to
/// the PendingQueue if the count reaches zero. Also update its cycle bound.
void SchedulePostRATDList::ReleaseSucc(SUnit *SU, SDep *SuccEdge) {
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

  // Standard scheduler algorithms will recompute the depth of the successor
  // here as such:
  //   SuccSU->setDepthToAtLeast(SU->getDepth() + SuccEdge->getLatency());
  //
  // However, we lazily compute node depth instead. Note that
  // ScheduleNodeTopDown has already updated the depth of this node which causes
  // all descendents to be marked dirty. Setting the successor depth explicitly
  // here would cause depth to be recomputed for all its ancestors. If the
  // successor is not yet ready (because of a transitively redundant edge) then
  // this causes depth computation to be quadratic in the size of the DAG.

  // If all the node's predecessors are scheduled, this node is ready
  // to be scheduled. Ignore the special ExitSU node.
  if (SuccSU->NumPredsLeft == 0 && SuccSU != &ExitSU)
    PendingQueue.push_back(SuccSU);
}

/// ReleaseSuccessors - Call ReleaseSucc on each of SU's successors.
void SchedulePostRATDList::ReleaseSuccessors(SUnit *SU) {
  for (SUnit::succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I) {
    ReleaseSucc(SU, &*I);
  }
}

/// ScheduleNodeTopDown - Add the node to the schedule. Decrement the pending
/// count of its successors. If a successor pending count is zero, add it to
/// the Available queue.
void SchedulePostRATDList::ScheduleNodeTopDown(SUnit *SU, unsigned CurCycle) {
  DEBUG(dbgs() << "*** Scheduling [" << CurCycle << "]: ");
  DEBUG(SU->dump(this));

  Sequence.push_back(SU);
  assert(CurCycle >= SU->getDepth() &&
         "Node scheduled above its depth!");
  SU->setDepthToAtLeast(CurCycle);

  ReleaseSuccessors(SU);
  SU->isScheduled = true;
  AvailableQueue.scheduledNode(SU);
}

/// ListScheduleTopDown - The main loop of list scheduling for top-down
/// schedulers.
void SchedulePostRATDList::ListScheduleTopDown() {
  unsigned CurCycle = 0;

  // We're scheduling top-down but we're visiting the regions in
  // bottom-up order, so we don't know the hazards at the start of a
  // region. So assume no hazards (this should usually be ok as most
  // blocks are a single region).
  HazardRec->Reset();

  // Release any successors of the special Entry node.
  ReleaseSuccessors(&EntrySU);

  // Add all leaves to Available queue.
  for (unsigned i = 0, e = SUnits.size(); i != e; ++i) {
    // It is available if it has no predecessors.
    bool available = SUnits[i].Preds.empty();
    if (available) {
      AvailableQueue.push(&SUnits[i]);
      SUnits[i].isAvailable = true;
    }
  }

  // In any cycle where we can't schedule any instructions, we must
  // stall or emit a noop, depending on the target.
  bool CycleHasInsts = false;

  // While Available queue is not empty, grab the node with the highest
  // priority. If it is not ready put it back.  Schedule the node.
  std::vector<SUnit*> NotReady;
  Sequence.reserve(SUnits.size());
  while (!AvailableQueue.empty() || !PendingQueue.empty()) {
    // Check to see if any of the pending instructions are ready to issue.  If
    // so, add them to the available queue.
    unsigned MinDepth = ~0u;
    for (unsigned i = 0, e = PendingQueue.size(); i != e; ++i) {
      if (PendingQueue[i]->getDepth() <= CurCycle) {
        AvailableQueue.push(PendingQueue[i]);
        PendingQueue[i]->isAvailable = true;
        PendingQueue[i] = PendingQueue.back();
        PendingQueue.pop_back();
        --i; --e;
      } else if (PendingQueue[i]->getDepth() < MinDepth)
        MinDepth = PendingQueue[i]->getDepth();
    }

    DEBUG(dbgs() << "\n*** Examining Available\n"; AvailableQueue.dump(this));

    SUnit *FoundSUnit = 0;
    bool HasNoopHazards = false;
    while (!AvailableQueue.empty()) {
      SUnit *CurSUnit = AvailableQueue.pop();

      ScheduleHazardRecognizer::HazardType HT =
        HazardRec->getHazardType(CurSUnit, 0/*no stalls*/);
      if (HT == ScheduleHazardRecognizer::NoHazard) {
        FoundSUnit = CurSUnit;
        break;
      }

      // Remember if this is a noop hazard.
      HasNoopHazards |= HT == ScheduleHazardRecognizer::NoopHazard;

      NotReady.push_back(CurSUnit);
    }

    // Add the nodes that aren't ready back onto the available list.
    if (!NotReady.empty()) {
      AvailableQueue.push_all(NotReady);
      NotReady.clear();
    }

    // If we found a node to schedule...
    if (FoundSUnit) {
      // ... schedule the node...
      ScheduleNodeTopDown(FoundSUnit, CurCycle);
      HazardRec->EmitInstruction(FoundSUnit);
      CycleHasInsts = true;
      if (HazardRec->atIssueLimit()) {
        DEBUG(dbgs() << "*** Max instructions per cycle " << CurCycle << '\n');
        HazardRec->AdvanceCycle();
        ++CurCycle;
        CycleHasInsts = false;
      }
    } else {
      if (CycleHasInsts) {
        DEBUG(dbgs() << "*** Finished cycle " << CurCycle << '\n');
        HazardRec->AdvanceCycle();
      } else if (!HasNoopHazards) {
        // Otherwise, we have a pipeline stall, but no other problem,
        // just advance the current cycle and try again.
        DEBUG(dbgs() << "*** Stall in cycle " << CurCycle << '\n');
        HazardRec->AdvanceCycle();
        ++NumStalls;
      } else {
        // Otherwise, we have no instructions to issue and we have instructions
        // that will fault if we don't do this right.  This is the case for
        // processors without pipeline interlocks and other cases.
        DEBUG(dbgs() << "*** Emitting noop in cycle " << CurCycle << '\n');
        HazardRec->EmitNoop();
        Sequence.push_back(0);   // NULL here means noop
        ++NumNoops;
      }

      ++CurCycle;
      CycleHasInsts = false;
    }
  }

#ifndef NDEBUG
  unsigned ScheduledNodes = VerifyScheduledDAG(/*isBottomUp=*/false);
  unsigned Noops = 0;
  for (unsigned i = 0, e = Sequence.size(); i != e; ++i)
    if (!Sequence[i])
      ++Noops;
  assert(Sequence.size() - Noops == ScheduledNodes &&
         "The number of nodes scheduled doesn't match the expected number!");
#endif // NDEBUG
}

// EmitSchedule - Emit the machine code in scheduled order.
void SchedulePostRATDList::EmitSchedule() {
  RegionBegin = RegionEnd;

  // If first instruction was a DBG_VALUE then put it back.
  if (FirstDbgValue)
    BB->splice(RegionEnd, BB, FirstDbgValue);

  // Then re-insert them according to the given schedule.
  for (unsigned i = 0, e = Sequence.size(); i != e; i++) {
    if (SUnit *SU = Sequence[i])
      BB->splice(RegionEnd, BB, SU->getInstr());
    else
      // Null SUnit* is a noop.
      TII->insertNoop(*BB, RegionEnd);

    // Update the Begin iterator, as the first instruction in the block
    // may have been scheduled later.
    if (i == 0)
      RegionBegin = prior(RegionEnd);
  }

  // Reinsert any remaining debug_values.
  for (std::vector<std::pair<MachineInstr *, MachineInstr *> >::iterator
         DI = DbgValues.end(), DE = DbgValues.begin(); DI != DE; --DI) {
    std::pair<MachineInstr *, MachineInstr *> P = *prior(DI);
    MachineInstr *DbgValue = P.first;
    MachineBasicBlock::iterator OrigPrivMI = P.second;
    BB->splice(++OrigPrivMI, BB, DbgValue);
  }
  DbgValues.clear();
  FirstDbgValue = NULL;
}
