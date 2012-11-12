//==- MachineScheduler.h - MachineInstr Scheduling Pass ----------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides a MachineSchedRegistry for registering alternative machine
// schedulers. A Target may provide an alternative scheduler implementation by
// implementing the following boilerplate:
//
// static ScheduleDAGInstrs *createCustomMachineSched(MachineSchedContext *C) {
//  return new CustomMachineScheduler(C);
// }
// static MachineSchedRegistry
// SchedCustomRegistry("custom", "Run my target's custom scheduler",
//                     createCustomMachineSched);
//
// Inside <Target>PassConfig:
//   enablePass(&MachineSchedulerID);
//   MachineSchedRegistry::setDefault(createCustomMachineSched);
//
//===----------------------------------------------------------------------===//

#ifndef MACHINESCHEDULER_H
#define MACHINESCHEDULER_H

#include "llvm/CodeGen/MachinePassRegistry.h"
#include "llvm/CodeGen/RegisterPressure.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include "llvm/Target/TargetInstrInfo.h"

namespace llvm {

extern cl::opt<bool> ForceTopDown;
extern cl::opt<bool> ForceBottomUp;

class AliasAnalysis;
class LiveIntervals;
class MachineDominatorTree;
class MachineLoopInfo;
class RegisterClassInfo;
class ScheduleDAGInstrs;

/// MachineSchedContext provides enough context from the MachineScheduler pass
/// for the target to instantiate a scheduler.
struct MachineSchedContext {
  MachineFunction *MF;
  const MachineLoopInfo *MLI;
  const MachineDominatorTree *MDT;
  const TargetPassConfig *PassConfig;
  AliasAnalysis *AA;
  LiveIntervals *LIS;

  RegisterClassInfo *RegClassInfo;

  MachineSchedContext();
  virtual ~MachineSchedContext();
};

/// MachineSchedRegistry provides a selection of available machine instruction
/// schedulers.
class MachineSchedRegistry : public MachinePassRegistryNode {
public:
  typedef ScheduleDAGInstrs *(*ScheduleDAGCtor)(MachineSchedContext *);

  // RegisterPassParser requires a (misnamed) FunctionPassCtor type.
  typedef ScheduleDAGCtor FunctionPassCtor;

  static MachinePassRegistry Registry;

  MachineSchedRegistry(const char *N, const char *D, ScheduleDAGCtor C)
    : MachinePassRegistryNode(N, D, (MachinePassCtor)C) {
    Registry.Add(this);
  }
  ~MachineSchedRegistry() { Registry.Remove(this); }

  // Accessors.
  //
  MachineSchedRegistry *getNext() const {
    return (MachineSchedRegistry *)MachinePassRegistryNode::getNext();
  }
  static MachineSchedRegistry *getList() {
    return (MachineSchedRegistry *)Registry.getList();
  }
  static ScheduleDAGCtor getDefault() {
    return (ScheduleDAGCtor)Registry.getDefault();
  }
  static void setDefault(ScheduleDAGCtor C) {
    Registry.setDefault((MachinePassCtor)C);
  }
  static void setDefault(StringRef Name) {
    Registry.setDefault(Name);
  }
  static void setListener(MachinePassRegistryListener *L) {
    Registry.setListener(L);
  }
};

class ScheduleDAGMI;

/// MachineSchedStrategy - Interface to the scheduling algorithm used by
/// ScheduleDAGMI.
class MachineSchedStrategy {
public:
  virtual ~MachineSchedStrategy() {}

  /// Initialize the strategy after building the DAG for a new region.
  virtual void initialize(ScheduleDAGMI *DAG) = 0;

  /// Notify this strategy that all roots have been released (including those
  /// that depend on EntrySU or ExitSU).
  virtual void registerRoots() {}

  /// Pick the next node to schedule, or return NULL. Set IsTopNode to true to
  /// schedule the node at the top of the unscheduled region. Otherwise it will
  /// be scheduled at the bottom.
  virtual SUnit *pickNode(bool &IsTopNode) = 0;

  /// Notify MachineSchedStrategy that ScheduleDAGMI has scheduled an
  /// instruction and updated scheduled/remaining flags in the DAG nodes.
  virtual void schedNode(SUnit *SU, bool IsTopNode) = 0;

  /// When all predecessor dependencies have been resolved, free this node for
  /// top-down scheduling.
  virtual void releaseTopNode(SUnit *SU) = 0;
  /// When all successor dependencies have been resolved, free this node for
  /// bottom-up scheduling.
  virtual void releaseBottomNode(SUnit *SU) = 0;
};

/// ReadyQueue encapsulates vector of "ready" SUnits with basic convenience
/// methods for pushing and removing nodes. ReadyQueue's are uniquely identified
/// by an ID. SUnit::NodeQueueId is a mask of the ReadyQueues the SUnit is in.
///
/// This is a convenience class that may be used by implementations of
/// MachineSchedStrategy.
class ReadyQueue {
  unsigned ID;
  std::string Name;
  std::vector<SUnit*> Queue;

public:
  ReadyQueue(unsigned id, const Twine &name): ID(id), Name(name.str()) {}

  unsigned getID() const { return ID; }

  StringRef getName() const { return Name; }

  // SU is in this queue if it's NodeQueueID is a superset of this ID.
  bool isInQueue(SUnit *SU) const { return (SU->NodeQueueId & ID); }

  bool empty() const { return Queue.empty(); }

  void clear() { Queue.clear(); }

  unsigned size() const { return Queue.size(); }

  typedef std::vector<SUnit*>::iterator iterator;

  iterator begin() { return Queue.begin(); }

  iterator end() { return Queue.end(); }

  iterator find(SUnit *SU) {
    return std::find(Queue.begin(), Queue.end(), SU);
  }

  void push(SUnit *SU) {
    Queue.push_back(SU);
    SU->NodeQueueId |= ID;
  }

  iterator remove(iterator I) {
    (*I)->NodeQueueId &= ~ID;
    *I = Queue.back();
    unsigned idx = I - Queue.begin();
    Queue.pop_back();
    return Queue.begin() + idx;
  }

#ifndef NDEBUG
  void dump();
#endif
};

/// Mutate the DAG as a postpass after normal DAG building.
class ScheduleDAGMutation {
public:
  virtual ~ScheduleDAGMutation() {}

  virtual void apply(ScheduleDAGMI *DAG) = 0;
};

/// ScheduleDAGMI is an implementation of ScheduleDAGInstrs that schedules
/// machine instructions while updating LiveIntervals and tracking regpressure.
class ScheduleDAGMI : public ScheduleDAGInstrs {
protected:
  AliasAnalysis *AA;
  RegisterClassInfo *RegClassInfo;
  MachineSchedStrategy *SchedImpl;

  /// Ordered list of DAG postprocessing steps.
  std::vector<ScheduleDAGMutation*> Mutations;

  MachineBasicBlock::iterator LiveRegionEnd;

  /// Register pressure in this region computed by buildSchedGraph.
  IntervalPressure RegPressure;
  RegPressureTracker RPTracker;

  /// List of pressure sets that exceed the target's pressure limit before
  /// scheduling, listed in increasing set ID order. Each pressure set is paired
  /// with its max pressure in the currently scheduled regions.
  std::vector<PressureElement> RegionCriticalPSets;

  /// The top of the unscheduled zone.
  MachineBasicBlock::iterator CurrentTop;
  IntervalPressure TopPressure;
  RegPressureTracker TopRPTracker;

  /// The bottom of the unscheduled zone.
  MachineBasicBlock::iterator CurrentBottom;
  IntervalPressure BotPressure;
  RegPressureTracker BotRPTracker;

#ifndef NDEBUG
  /// The number of instructions scheduled so far. Used to cut off the
  /// scheduler at the point determined by misched-cutoff.
  unsigned NumInstrsScheduled;
#endif

public:
  ScheduleDAGMI(MachineSchedContext *C, MachineSchedStrategy *S):
    ScheduleDAGInstrs(*C->MF, *C->MLI, *C->MDT, /*IsPostRA=*/false, C->LIS),
    AA(C->AA), RegClassInfo(C->RegClassInfo), SchedImpl(S),
    RPTracker(RegPressure), CurrentTop(), TopRPTracker(TopPressure),
    CurrentBottom(), BotRPTracker(BotPressure) {
#ifndef NDEBUG
    NumInstrsScheduled = 0;
#endif
  }

  virtual ~ScheduleDAGMI() {
    delete SchedImpl;
  }

  /// Add a postprocessing step to the DAG builder.
  /// Mutations are applied in the order that they are added after normal DAG
  /// building and before MachineSchedStrategy initialization.
  void addMutation(ScheduleDAGMutation *Mutation) {
    Mutations.push_back(Mutation);
  }

  MachineBasicBlock::iterator top() const { return CurrentTop; }
  MachineBasicBlock::iterator bottom() const { return CurrentBottom; }

  /// Implement the ScheduleDAGInstrs interface for handling the next scheduling
  /// region. This covers all instructions in a block, while schedule() may only
  /// cover a subset.
  void enterRegion(MachineBasicBlock *bb,
                   MachineBasicBlock::iterator begin,
                   MachineBasicBlock::iterator end,
                   unsigned endcount);


  /// Implement ScheduleDAGInstrs interface for scheduling a sequence of
  /// reorderable instructions.
  virtual void schedule();

  /// Get current register pressure for the top scheduled instructions.
  const IntervalPressure &getTopPressure() const { return TopPressure; }
  const RegPressureTracker &getTopRPTracker() const { return TopRPTracker; }

  /// Get current register pressure for the bottom scheduled instructions.
  const IntervalPressure &getBotPressure() const { return BotPressure; }
  const RegPressureTracker &getBotRPTracker() const { return BotRPTracker; }

  /// Get register pressure for the entire scheduling region before scheduling.
  const IntervalPressure &getRegPressure() const { return RegPressure; }

  const std::vector<PressureElement> &getRegionCriticalPSets() const {
    return RegionCriticalPSets;
  }

protected:
  // Top-Level entry points for the schedule() driver...

  /// Call ScheduleDAGInstrs::buildSchedGraph with register pressure tracking
  /// enabled. This sets up three trackers. RPTracker will cover the entire DAG
  /// region, TopTracker and BottomTracker will be initialized to the top and
  /// bottom of the DAG region without covereing any unscheduled instruction.
  void buildDAGWithRegPressure();

  /// Apply each ScheduleDAGMutation step in order. This allows different
  /// instances of ScheduleDAGMI to perform custom DAG postprocessing.
  void postprocessDAG();

  /// Identify DAG roots and setup scheduler queues.
  void initQueues();

  /// Move an instruction and update register pressure.
  void scheduleMI(SUnit *SU, bool IsTopNode);

  /// Update scheduler DAG and queues after scheduling an instruction.
  void updateQueues(SUnit *SU, bool IsTopNode);

  /// Reinsert debug_values recorded in ScheduleDAGInstrs::DbgValues.
  void placeDebugValues();

  /// \brief dump the scheduled Sequence.
  void dumpSchedule() const;

  // Lesser helpers...

  void initRegPressure();

  void updateScheduledPressure(std::vector<unsigned> NewMaxPressure);

  void moveInstruction(MachineInstr *MI, MachineBasicBlock::iterator InsertPos);
  bool checkSchedLimit();

  void releaseRoots();

  void releaseSucc(SUnit *SU, SDep *SuccEdge);
  void releaseSuccessors(SUnit *SU);
  void releasePred(SUnit *SU, SDep *PredEdge);
  void releasePredecessors(SUnit *SU);
};

} // namespace llvm

#endif
