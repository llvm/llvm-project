//===- MachinePipeliner.cpp - Machine Software Pipeliner Pass -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// An implementation of the Swing Modulo Scheduling (SMS) software pipeliner.
//
// Software pipelining (SWP) is an instruction scheduling technique for loops
// that overlap loop iterations and explioits ILP via a compiler transformation.
//
// Swing Modulo Scheduling is an implementation of software pipelining
// that generates schedules that are near optimal in terms of initiation
// interval, register requirements, and stage count. See the papers:
//
// "Swing Modulo Scheduling: A Lifetime-Sensitive Approach", by J. Llosa,
// A. Gonzalez, E. Ayguade, and M. Valero. In PACT '96 Processings of the 1996
// Conference on Parallel Architectures and Compilation Techiniques.
//
// "Lifetime-Sensitive Modulo Scheduling in a Production Environment", by J.
// Llosa, E. Ayguade, A. Gonzalez, M. Valero, and J. Eckhardt. In IEEE
// Transactions on Computers, Vol. 50, No. 3, 2001.
//
// "An Implementation of Swing Modulo Scheduling With Extensions for
// Superblocks", by T. Lattner, Master's Thesis, University of Illinois at
// Urbana-Chambpain, 2005.
//
//
// The SMS algorithm consists of three main steps after computing the minimal
// initiation interval (MII).
// 1) Analyze the dependence graph and compute information about each
//    instruction in the graph.
// 2) Order the nodes (instructions) by priority based upon the heuristics
//    described in the algorithm.
// 3) Attempt to schedule the nodes in the specified order using the MII.
//
// This SMS implementation is a target-independent back-end pass. When enabled,
// the pass runs just prior to the register allocation pass, while the machine
// IR is in SSA form. If software pipelining is successful, then the original
// loop is replaced by the optimized loop. The optimized loop contains one or
// more prolog blocks, the pipelined kernel, and one or more epilog blocks. If
// the instructions cannot be scheduled in a given MII, we increase the MII by
// one and try again.
//
// The SMS implementation is an extension of the ScheduleDAGInstrs class. We
// represent loop carried dependences in the DAG as order edges to the Phi
// nodes. We also perform several passes over the DAG to eliminate unnecessary
// edges that inhibit the ability to pipeline. The implementation uses the
// DFAPacketizer class to compute the minimum initiation interval and the check
// where an instruction may be inserted in the pipelined schedule.
//
// In order for the SMS pass to work, several target specific hooks need to be
// implemented to get information about the loop structure and to rewrite
// instructions.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PriorityQueue.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/DFAPacketizer.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/RegisterPressure.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include "llvm/CodeGen/ScheduleDAGMutation.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Function.h"
#include "llvm/MC/LaneBitmask.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrItineraries.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdint>
#include <deque>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "pipeliner"

STATISTIC(NumTrytoPipeline, "Number of loops that we attempt to pipeline");
STATISTIC(NumPipelined, "Number of loops software pipelined");

/// A command line option to turn software pipelining on or off.
static cl::opt<bool> EnableSWP("enable-pipeliner", cl::Hidden, cl::init(true),
                               cl::ZeroOrMore,
                               cl::desc("Enable Software Pipelining"));

/// A command line option to enable SWP at -Os.
static cl::opt<bool> EnableSWPOptSize("enable-pipeliner-opt-size",
                                      cl::desc("Enable SWP at Os."), cl::Hidden,
                                      cl::init(false));

/// A command line argument to limit minimum initial interval for pipelining.
static cl::opt<int> SwpMaxMii("pipeliner-max-mii",
                              cl::desc("Size limit for the the MII."),
                              cl::Hidden, cl::init(27));

/// A command line argument to limit the number of stages in the pipeline.
static cl::opt<int>
    SwpMaxStages("pipeliner-max-stages",
                 cl::desc("Maximum stages allowed in the generated scheduled."),
                 cl::Hidden, cl::init(3));

/// A command line option to disable the pruning of chain dependences due to
/// an unrelated Phi.
static cl::opt<bool>
    SwpPruneDeps("pipeliner-prune-deps",
                 cl::desc("Prune dependences between unrelated Phi nodes."),
                 cl::Hidden, cl::init(true));

/// A command line option to disable the pruning of loop carried order
/// dependences.
static cl::opt<bool>
    SwpPruneLoopCarried("pipeliner-prune-loop-carried",
                        cl::desc("Prune loop carried order dependences."),
                        cl::Hidden, cl::init(true));

#ifndef NDEBUG
static cl::opt<int> SwpLoopLimit("pipeliner-max", cl::Hidden, cl::init(-1));
#endif

static cl::opt<bool> SwpIgnoreRecMII("pipeliner-ignore-recmii",
                                     cl::ReallyHidden, cl::init(false),
                                     cl::ZeroOrMore, cl::desc("Ignore RecMII"));

namespace {

class NodeSet;
class SMSchedule;

/// The main class in the implementation of the target independent
/// software pipeliner pass.
class MachinePipeliner : public MachineFunctionPass {
public:
  MachineFunction *MF = nullptr;
  const MachineLoopInfo *MLI = nullptr;
  const MachineDominatorTree *MDT = nullptr;
  const InstrItineraryData *InstrItins;
  const TargetInstrInfo *TII = nullptr;
  RegisterClassInfo RegClassInfo;

#ifndef NDEBUG
  static int NumTries;
#endif

  /// Cache the target analysis information about the loop.
  struct LoopInfo {
    MachineBasicBlock *TBB = nullptr;
    MachineBasicBlock *FBB = nullptr;
    SmallVector<MachineOperand, 4> BrCond;
    MachineInstr *LoopInductionVar = nullptr;
    MachineInstr *LoopCompare = nullptr;
  };
  LoopInfo LI;

  static char ID;

  MachinePipeliner() : MachineFunctionPass(ID) {
    initializeMachinePipelinerPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AAResultsWrapperPass>();
    AU.addPreserved<AAResultsWrapperPass>();
    AU.addRequired<MachineLoopInfo>();
    AU.addRequired<MachineDominatorTree>();
    AU.addRequired<LiveIntervals>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  bool canPipelineLoop(MachineLoop &L);
  bool scheduleLoop(MachineLoop &L);
  bool swingModuloScheduler(MachineLoop &L);
};

/// This class builds the dependence graph for the instructions in a loop,
/// and attempts to schedule the instructions using the SMS algorithm.
class SwingSchedulerDAG : public ScheduleDAGInstrs {
  MachinePipeliner &Pass;
  /// The minimum initiation interval between iterations for this schedule.
  unsigned MII = 0;
  /// Set to true if a valid pipelined schedule is found for the loop.
  bool Scheduled = false;
  MachineLoop &Loop;
  LiveIntervals &LIS;
  const RegisterClassInfo &RegClassInfo;

  /// A toplogical ordering of the SUnits, which is needed for changing
  /// dependences and iterating over the SUnits.
  ScheduleDAGTopologicalSort Topo;

  struct NodeInfo {
    int ASAP = 0;
    int ALAP = 0;

    NodeInfo() = default;
  };
  /// Computed properties for each node in the graph.
  std::vector<NodeInfo> ScheduleInfo;

  enum OrderKind { BottomUp = 0, TopDown = 1 };
  /// Computed node ordering for scheduling.
  SetVector<SUnit *> NodeOrder;

  using NodeSetType = SmallVector<NodeSet, 8>;
  using ValueMapTy = DenseMap<unsigned, unsigned>;
  using MBBVectorTy = SmallVectorImpl<MachineBasicBlock *>;
  using InstrMapTy = DenseMap<MachineInstr *, MachineInstr *>;

  /// Instructions to change when emitting the final schedule.
  DenseMap<SUnit *, std::pair<unsigned, int64_t>> InstrChanges;

  /// We may create a new instruction, so remember it because it
  /// must be deleted when the pass is finished.
  SmallPtrSet<MachineInstr *, 4> NewMIs;

  /// Ordered list of DAG postprocessing steps.
  std::vector<std::unique_ptr<ScheduleDAGMutation>> Mutations;

  /// Helper class to implement Johnson's circuit finding algorithm.
  class Circuits {
    std::vector<SUnit> &SUnits;
    SetVector<SUnit *> Stack;
    BitVector Blocked;
    SmallVector<SmallPtrSet<SUnit *, 4>, 10> B;
    SmallVector<SmallVector<int, 4>, 16> AdjK;
    unsigned NumPaths;
    static unsigned MaxPaths;

  public:
    Circuits(std::vector<SUnit> &SUs)
        : SUnits(SUs), Blocked(SUs.size()), B(SUs.size()), AdjK(SUs.size()) {}

    /// Reset the data structures used in the circuit algorithm.
    void reset() {
      Stack.clear();
      Blocked.reset();
      B.assign(SUnits.size(), SmallPtrSet<SUnit *, 4>());
      NumPaths = 0;
    }

    void createAdjacencyStructure(SwingSchedulerDAG *DAG);
    bool circuit(int V, int S, NodeSetType &NodeSets, bool HasBackedge = false);
    void unblock(int U);
  };

public:
  SwingSchedulerDAG(MachinePipeliner &P, MachineLoop &L, LiveIntervals &lis,
                    const RegisterClassInfo &rci)
      : ScheduleDAGInstrs(*P.MF, P.MLI, false), Pass(P), Loop(L), LIS(lis),
        RegClassInfo(rci), Topo(SUnits, &ExitSU) {
    P.MF->getSubtarget().getSMSMutations(Mutations);
  }

  void schedule() override;
  void finishBlock() override;

  /// Return true if the loop kernel has been scheduled.
  bool hasNewSchedule() { return Scheduled; }

  /// Return the earliest time an instruction may be scheduled.
  int getASAP(SUnit *Node) { return ScheduleInfo[Node->NodeNum].ASAP; }

  /// Return the latest time an instruction my be scheduled.
  int getALAP(SUnit *Node) { return ScheduleInfo[Node->NodeNum].ALAP; }

  /// The mobility function, which the the number of slots in which
  /// an instruction may be scheduled.
  int getMOV(SUnit *Node) { return getALAP(Node) - getASAP(Node); }

  /// The depth, in the dependence graph, for a node.
  int getDepth(SUnit *Node) { return Node->getDepth(); }

  /// The height, in the dependence graph, for a node.
  int getHeight(SUnit *Node) { return Node->getHeight(); }

  /// Return true if the dependence is a back-edge in the data dependence graph.
  /// Since the DAG doesn't contain cycles, we represent a cycle in the graph
  /// using an anti dependence from a Phi to an instruction.
  bool isBackedge(SUnit *Source, const SDep &Dep) {
    if (Dep.getKind() != SDep::Anti)
      return false;
    return Source->getInstr()->isPHI() || Dep.getSUnit()->getInstr()->isPHI();
  }

  /// Return true if the dependence is an order dependence between non-Phis.
  static bool isOrder(SUnit *Source, const SDep &Dep) {
    if (Dep.getKind() != SDep::Order)
      return false;
    return (!Source->getInstr()->isPHI() &&
            !Dep.getSUnit()->getInstr()->isPHI());
  }

  bool isLoopCarriedOrder(SUnit *Source, const SDep &Dep, bool isSucc = true);

  /// The latency of the dependence.
  unsigned getLatency(SUnit *Source, const SDep &Dep) {
    // Anti dependences represent recurrences, so use the latency of the
    // instruction on the back-edge.
    if (Dep.getKind() == SDep::Anti) {
      if (Source->getInstr()->isPHI())
        return Dep.getSUnit()->Latency;
      if (Dep.getSUnit()->getInstr()->isPHI())
        return Source->Latency;
      return Dep.getLatency();
    }
    return Dep.getLatency();
  }

  /// The distance function, which indicates that operation V of iteration I
  /// depends on operations U of iteration I-distance.
  unsigned getDistance(SUnit *U, SUnit *V, const SDep &Dep) {
    // Instructions that feed a Phi have a distance of 1. Computing larger
    // values for arrays requires data dependence information.
    if (V->getInstr()->isPHI() && Dep.getKind() == SDep::Anti)
      return 1;
    return 0;
  }

  /// Set the Minimum Initiation Interval for this schedule attempt.
  void setMII(unsigned mii) { MII = mii; }

  void applyInstrChange(MachineInstr *MI, SMSchedule &Schedule);

  void fixupRegisterOverlaps(std::deque<SUnit *> &Instrs);

  /// Return the new base register that was stored away for the changed
  /// instruction.
  unsigned getInstrBaseReg(SUnit *SU) {
    DenseMap<SUnit *, std::pair<unsigned, int64_t>>::iterator It =
        InstrChanges.find(SU);
    if (It != InstrChanges.end())
      return It->second.first;
    return 0;
  }

  void addMutation(std::unique_ptr<ScheduleDAGMutation> Mutation) {
    Mutations.push_back(std::move(Mutation));
  }

private:
  void addLoopCarriedDependences(AliasAnalysis *AA);
  void updatePhiDependences();
  void changeDependences();
  unsigned calculateResMII();
  unsigned calculateRecMII(NodeSetType &RecNodeSets);
  void findCircuits(NodeSetType &NodeSets);
  void fuseRecs(NodeSetType &NodeSets);
  void removeDuplicateNodes(NodeSetType &NodeSets);
  void computeNodeFunctions(NodeSetType &NodeSets);
  void registerPressureFilter(NodeSetType &NodeSets);
  void colocateNodeSets(NodeSetType &NodeSets);
  void checkNodeSets(NodeSetType &NodeSets);
  void groupRemainingNodes(NodeSetType &NodeSets);
  void addConnectedNodes(SUnit *SU, NodeSet &NewSet,
                         SetVector<SUnit *> &NodesAdded);
  void computeNodeOrder(NodeSetType &NodeSets);
  bool schedulePipeline(SMSchedule &Schedule);
  void generatePipelinedLoop(SMSchedule &Schedule);
  void generateProlog(SMSchedule &Schedule, unsigned LastStage,
                      MachineBasicBlock *KernelBB, ValueMapTy *VRMap,
                      MBBVectorTy &PrologBBs);
  void generateEpilog(SMSchedule &Schedule, unsigned LastStage,
                      MachineBasicBlock *KernelBB, ValueMapTy *VRMap,
                      MBBVectorTy &EpilogBBs, MBBVectorTy &PrologBBs);
  void generateExistingPhis(MachineBasicBlock *NewBB, MachineBasicBlock *BB1,
                            MachineBasicBlock *BB2, MachineBasicBlock *KernelBB,
                            SMSchedule &Schedule, ValueMapTy *VRMap,
                            InstrMapTy &InstrMap, unsigned LastStageNum,
                            unsigned CurStageNum, bool IsLast);
  void generatePhis(MachineBasicBlock *NewBB, MachineBasicBlock *BB1,
                    MachineBasicBlock *BB2, MachineBasicBlock *KernelBB,
                    SMSchedule &Schedule, ValueMapTy *VRMap,
                    InstrMapTy &InstrMap, unsigned LastStageNum,
                    unsigned CurStageNum, bool IsLast);
  void removeDeadInstructions(MachineBasicBlock *KernelBB,
                              MBBVectorTy &EpilogBBs);
  void splitLifetimes(MachineBasicBlock *KernelBB, MBBVectorTy &EpilogBBs,
                      SMSchedule &Schedule);
  void addBranches(MBBVectorTy &PrologBBs, MachineBasicBlock *KernelBB,
                   MBBVectorTy &EpilogBBs, SMSchedule &Schedule,
                   ValueMapTy *VRMap);
  bool computeDelta(MachineInstr &MI, unsigned &Delta);
  void updateMemOperands(MachineInstr &NewMI, MachineInstr &OldMI,
                         unsigned Num);
  MachineInstr *cloneInstr(MachineInstr *OldMI, unsigned CurStageNum,
                           unsigned InstStageNum);
  MachineInstr *cloneAndChangeInstr(MachineInstr *OldMI, unsigned CurStageNum,
                                    unsigned InstStageNum,
                                    SMSchedule &Schedule);
  void updateInstruction(MachineInstr *NewMI, bool LastDef,
                         unsigned CurStageNum, unsigned InstStageNum,
                         SMSchedule &Schedule, ValueMapTy *VRMap);
  MachineInstr *findDefInLoop(unsigned Reg);
  unsigned getPrevMapVal(unsigned StageNum, unsigned PhiStage, unsigned LoopVal,
                         unsigned LoopStage, ValueMapTy *VRMap,
                         MachineBasicBlock *BB);
  void rewritePhiValues(MachineBasicBlock *NewBB, unsigned StageNum,
                        SMSchedule &Schedule, ValueMapTy *VRMap,
                        InstrMapTy &InstrMap);
  void rewriteScheduledInstr(MachineBasicBlock *BB, SMSchedule &Schedule,
                             InstrMapTy &InstrMap, unsigned CurStageNum,
                             unsigned PhiNum, MachineInstr *Phi,
                             unsigned OldReg, unsigned NewReg,
                             unsigned PrevReg = 0);
  bool canUseLastOffsetValue(MachineInstr *MI, unsigned &BasePos,
                             unsigned &OffsetPos, unsigned &NewBase,
                             int64_t &NewOffset);
  void postprocessDAG();
};

/// A NodeSet contains a set of SUnit DAG nodes with additional information
/// that assigns a priority to the set.
class NodeSet {
  SetVector<SUnit *> Nodes;
  bool HasRecurrence = false;
  unsigned RecMII = 0;
  int MaxMOV = 0;
  int MaxDepth = 0;
  unsigned Colocate = 0;
  SUnit *ExceedPressure = nullptr;

public:
  using iterator = SetVector<SUnit *>::const_iterator;

  NodeSet() = default;
  NodeSet(iterator S, iterator E) : Nodes(S, E), HasRecurrence(true) {}

  bool insert(SUnit *SU) { return Nodes.insert(SU); }

  void insert(iterator S, iterator E) { Nodes.insert(S, E); }

  template <typename UnaryPredicate> bool remove_if(UnaryPredicate P) {
    return Nodes.remove_if(P);
  }

  unsigned count(SUnit *SU) const { return Nodes.count(SU); }

  bool hasRecurrence() { return HasRecurrence; };

  unsigned size() const { return Nodes.size(); }

  bool empty() const { return Nodes.empty(); }

  SUnit *getNode(unsigned i) const { return Nodes[i]; };

  void setRecMII(unsigned mii) { RecMII = mii; };

  void setColocate(unsigned c) { Colocate = c; };

  void setExceedPressure(SUnit *SU) { ExceedPressure = SU; }

  bool isExceedSU(SUnit *SU) { return ExceedPressure == SU; }

  int compareRecMII(NodeSet &RHS) { return RecMII - RHS.RecMII; }

  int getRecMII() { return RecMII; }

  /// Summarize node functions for the entire node set.
  void computeNodeSetInfo(SwingSchedulerDAG *SSD) {
    for (SUnit *SU : *this) {
      MaxMOV = std::max(MaxMOV, SSD->getMOV(SU));
      MaxDepth = std::max(MaxDepth, SSD->getDepth(SU));
    }
  }

  void clear() {
    Nodes.clear();
    RecMII = 0;
    HasRecurrence = false;
    MaxMOV = 0;
    MaxDepth = 0;
    Colocate = 0;
    ExceedPressure = nullptr;
  }

  operator SetVector<SUnit *> &() { return Nodes; }

  /// Sort the node sets by importance. First, rank them by recurrence MII,
  /// then by mobility (least mobile done first), and finally by depth.
  /// Each node set may contain a colocate value which is used as the first
  /// tie breaker, if it's set.
  bool operator>(const NodeSet &RHS) const {
    if (RecMII == RHS.RecMII) {
      if (Colocate != 0 && RHS.Colocate != 0 && Colocate != RHS.Colocate)
        return Colocate < RHS.Colocate;
      if (MaxMOV == RHS.MaxMOV)
        return MaxDepth > RHS.MaxDepth;
      return MaxMOV < RHS.MaxMOV;
    }
    return RecMII > RHS.RecMII;
  }

  bool operator==(const NodeSet &RHS) const {
    return RecMII == RHS.RecMII && MaxMOV == RHS.MaxMOV &&
           MaxDepth == RHS.MaxDepth;
  }

  bool operator!=(const NodeSet &RHS) const { return !operator==(RHS); }

  iterator begin() { return Nodes.begin(); }
  iterator end() { return Nodes.end(); }

  void print(raw_ostream &os) const {
    os << "Num nodes " << size() << " rec " << RecMII << " mov " << MaxMOV
       << " depth " << MaxDepth << " col " << Colocate << "\n";
    for (const auto &I : Nodes)
      os << "   SU(" << I->NodeNum << ") " << *(I->getInstr());
    os << "\n";
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const { print(dbgs()); }
#endif
};

/// This class repesents the scheduled code.  The main data structure is a
/// map from scheduled cycle to instructions.  During scheduling, the
/// data structure explicitly represents all stages/iterations.   When
/// the algorithm finshes, the schedule is collapsed into a single stage,
/// which represents instructions from different loop iterations.
///
/// The SMS algorithm allows negative values for cycles, so the first cycle
/// in the schedule is the smallest cycle value.
class SMSchedule {
private:
  /// Map from execution cycle to instructions.
  DenseMap<int, std::deque<SUnit *>> ScheduledInstrs;

  /// Map from instruction to execution cycle.
  std::map<SUnit *, int> InstrToCycle;

  /// Map for each register and the max difference between its uses and def.
  /// The first element in the pair is the max difference in stages. The
  /// second is true if the register defines a Phi value and loop value is
  /// scheduled before the Phi.
  std::map<unsigned, std::pair<unsigned, bool>> RegToStageDiff;

  /// Keep track of the first cycle value in the schedule.  It starts
  /// as zero, but the algorithm allows negative values.
  int FirstCycle = 0;

  /// Keep track of the last cycle value in the schedule.
  int LastCycle = 0;

  /// The initiation interval (II) for the schedule.
  int InitiationInterval = 0;

  /// Target machine information.
  const TargetSubtargetInfo &ST;

  /// Virtual register information.
  MachineRegisterInfo &MRI;

  std::unique_ptr<DFAPacketizer> Resources;

public:
  SMSchedule(MachineFunction *mf)
      : ST(mf->getSubtarget()), MRI(mf->getRegInfo()),
        Resources(ST.getInstrInfo()->CreateTargetScheduleState(ST)) {}

  void reset() {
    ScheduledInstrs.clear();
    InstrToCycle.clear();
    RegToStageDiff.clear();
    FirstCycle = 0;
    LastCycle = 0;
    InitiationInterval = 0;
  }

  /// Set the initiation interval for this schedule.
  void setInitiationInterval(int ii) { InitiationInterval = ii; }

  /// Return the first cycle in the completed schedule.  This
  /// can be a negative value.
  int getFirstCycle() const { return FirstCycle; }

  /// Return the last cycle in the finalized schedule.
  int getFinalCycle() const { return FirstCycle + InitiationInterval - 1; }

  /// Return the cycle of the earliest scheduled instruction in the dependence
  /// chain.
  int earliestCycleInChain(const SDep &Dep);

  /// Return the cycle of the latest scheduled instruction in the dependence
  /// chain.
  int latestCycleInChain(const SDep &Dep);

  void computeStart(SUnit *SU, int *MaxEarlyStart, int *MinLateStart,
                    int *MinEnd, int *MaxStart, int II, SwingSchedulerDAG *DAG);
  bool insert(SUnit *SU, int StartCycle, int EndCycle, int II);

  /// Iterators for the cycle to instruction map.
  using sched_iterator = DenseMap<int, std::deque<SUnit *>>::iterator;
  using const_sched_iterator =
      DenseMap<int, std::deque<SUnit *>>::const_iterator;

  /// Return true if the instruction is scheduled at the specified stage.
  bool isScheduledAtStage(SUnit *SU, unsigned StageNum) {
    return (stageScheduled(SU) == (int)StageNum);
  }

  /// Return the stage for a scheduled instruction.  Return -1 if
  /// the instruction has not been scheduled.
  int stageScheduled(SUnit *SU) const {
    std::map<SUnit *, int>::const_iterator it = InstrToCycle.find(SU);
    if (it == InstrToCycle.end())
      return -1;
    return (it->second - FirstCycle) / InitiationInterval;
  }

  /// Return the cycle for a scheduled instruction. This function normalizes
  /// the first cycle to be 0.
  unsigned cycleScheduled(SUnit *SU) const {
    std::map<SUnit *, int>::const_iterator it = InstrToCycle.find(SU);
    assert(it != InstrToCycle.end() && "Instruction hasn't been scheduled.");
    return (it->second - FirstCycle) % InitiationInterval;
  }

  /// Return the maximum stage count needed for this schedule.
  unsigned getMaxStageCount() {
    return (LastCycle - FirstCycle) / InitiationInterval;
  }

  /// Return the max. number of stages/iterations that can occur between a
  /// register definition and its uses.
  unsigned getStagesForReg(int Reg, unsigned CurStage) {
    std::pair<unsigned, bool> Stages = RegToStageDiff[Reg];
    if (CurStage > getMaxStageCount() && Stages.first == 0 && Stages.second)
      return 1;
    return Stages.first;
  }

  /// The number of stages for a Phi is a little different than other
  /// instructions. The minimum value computed in RegToStageDiff is 1
  /// because we assume the Phi is needed for at least 1 iteration.
  /// This is not the case if the loop value is scheduled prior to the
  /// Phi in the same stage.  This function returns the number of stages
  /// or iterations needed between the Phi definition and any uses.
  unsigned getStagesForPhi(int Reg) {
    std::pair<unsigned, bool> Stages = RegToStageDiff[Reg];
    if (Stages.second)
      return Stages.first;
    return Stages.first - 1;
  }

  /// Return the instructions that are scheduled at the specified cycle.
  std::deque<SUnit *> &getInstructions(int cycle) {
    return ScheduledInstrs[cycle];
  }

  bool isValidSchedule(SwingSchedulerDAG *SSD);
  void finalizeSchedule(SwingSchedulerDAG *SSD);
  bool orderDependence(SwingSchedulerDAG *SSD, SUnit *SU,
                       std::deque<SUnit *> &Insts);
  bool isLoopCarried(SwingSchedulerDAG *SSD, MachineInstr &Phi);
  bool isLoopCarriedDefOfUse(SwingSchedulerDAG *SSD, MachineInstr *Inst,
                             MachineOperand &MO);
  void print(raw_ostream &os) const;
  void dump() const;
};

} // end anonymous namespace

unsigned SwingSchedulerDAG::Circuits::MaxPaths = 5;
char MachinePipeliner::ID = 0;
#ifndef NDEBUG
int MachinePipeliner::NumTries = 0;
#endif
char &llvm::MachinePipelinerID = MachinePipeliner::ID;

INITIALIZE_PASS_BEGIN(MachinePipeliner, DEBUG_TYPE,
                      "Modulo Software Pipelining", false, false)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_END(MachinePipeliner, DEBUG_TYPE,
                    "Modulo Software Pipelining", false, false)

/// The "main" function for implementing Swing Modulo Scheduling.
bool MachinePipeliner::runOnMachineFunction(MachineFunction &mf) {
  if (skipFunction(mf.getFunction()))
    return false;

  if (!EnableSWP)
    return false;

  if (mf.getFunction().getAttributes().hasAttribute(
          AttributeList::FunctionIndex, Attribute::OptimizeForSize) &&
      !EnableSWPOptSize.getPosition())
    return false;

  MF = &mf;
  MLI = &getAnalysis<MachineLoopInfo>();
  MDT = &getAnalysis<MachineDominatorTree>();
  TII = MF->getSubtarget().getInstrInfo();
  RegClassInfo.runOnMachineFunction(*MF);

  for (auto &L : *MLI)
    scheduleLoop(*L);

  return false;
}

/// Attempt to perform the SMS algorithm on the specified loop. This function is
/// the main entry point for the algorithm.  The function identifies candidate
/// loops, calculates the minimum initiation interval, and attempts to schedule
/// the loop.
bool MachinePipeliner::scheduleLoop(MachineLoop &L) {
  bool Changed = false;
  for (auto &InnerLoop : L)
    Changed |= scheduleLoop(*InnerLoop);

#ifndef NDEBUG
  // Stop trying after reaching the limit (if any).
  int Limit = SwpLoopLimit;
  if (Limit >= 0) {
    if (NumTries >= SwpLoopLimit)
      return Changed;
    NumTries++;
  }
#endif

  if (!canPipelineLoop(L))
    return Changed;

  ++NumTrytoPipeline;

  Changed = swingModuloScheduler(L);

  return Changed;
}

/// Return true if the loop can be software pipelined.  The algorithm is
/// restricted to loops with a single basic block.  Make sure that the
/// branch in the loop can be analyzed.
bool MachinePipeliner::canPipelineLoop(MachineLoop &L) {
  if (L.getNumBlocks() != 1)
    return false;

  // Check if the branch can't be understood because we can't do pipelining
  // if that's the case.
  LI.TBB = nullptr;
  LI.FBB = nullptr;
  LI.BrCond.clear();
  if (TII->analyzeBranch(*L.getHeader(), LI.TBB, LI.FBB, LI.BrCond))
    return false;

  LI.LoopInductionVar = nullptr;
  LI.LoopCompare = nullptr;
  if (TII->analyzeLoop(L, LI.LoopInductionVar, LI.LoopCompare))
    return false;

  if (!L.getLoopPreheader())
    return false;

  // If any of the Phis contain subregs, then we can't pipeline
  // because we don't know how to maintain subreg information in the
  // VMap structure.
  MachineBasicBlock *MBB = L.getHeader();
  for (auto &PHI : MBB->phis())
    for (unsigned i = 1; i != PHI.getNumOperands(); i += 2)
      if (PHI.getOperand(i).getSubReg() != 0)
        return false;

  return true;
}

/// The SMS algorithm consists of the following main steps:
/// 1. Computation and analysis of the dependence graph.
/// 2. Ordering of the nodes (instructions).
/// 3. Attempt to Schedule the loop.
bool MachinePipeliner::swingModuloScheduler(MachineLoop &L) {
  assert(L.getBlocks().size() == 1 && "SMS works on single blocks only.");

  SwingSchedulerDAG SMS(*this, L, getAnalysis<LiveIntervals>(), RegClassInfo);

  MachineBasicBlock *MBB = L.getHeader();
  // The kernel should not include any terminator instructions.  These
  // will be added back later.
  SMS.startBlock(MBB);

  // Compute the number of 'real' instructions in the basic block by
  // ignoring terminators.
  unsigned size = MBB->size();
  for (MachineBasicBlock::iterator I = MBB->getFirstTerminator(),
                                   E = MBB->instr_end();
       I != E; ++I, --size)
    ;

  SMS.enterRegion(MBB, MBB->begin(), MBB->getFirstTerminator(), size);
  SMS.schedule();
  SMS.exitRegion();

  SMS.finishBlock();
  return SMS.hasNewSchedule();
}

/// We override the schedule function in ScheduleDAGInstrs to implement the
/// scheduling part of the Swing Modulo Scheduling algorithm.
void SwingSchedulerDAG::schedule() {
  AliasAnalysis *AA = &Pass.getAnalysis<AAResultsWrapperPass>().getAAResults();
  buildSchedGraph(AA);
  addLoopCarriedDependences(AA);
  updatePhiDependences();
  Topo.InitDAGTopologicalSorting();
  postprocessDAG();
  changeDependences();
  DEBUG({
    for (unsigned su = 0, e = SUnits.size(); su != e; ++su)
      SUnits[su].dumpAll(this);
  });

  NodeSetType NodeSets;
  findCircuits(NodeSets);

  // Calculate the MII.
  unsigned ResMII = calculateResMII();
  unsigned RecMII = calculateRecMII(NodeSets);

  fuseRecs(NodeSets);

  // This flag is used for testing and can cause correctness problems.
  if (SwpIgnoreRecMII)
    RecMII = 0;

  MII = std::max(ResMII, RecMII);
  DEBUG(dbgs() << "MII = " << MII << " (rec=" << RecMII << ", res=" << ResMII
               << ")\n");

  // Can't schedule a loop without a valid MII.
  if (MII == 0)
    return;

  // Don't pipeline large loops.
  if (SwpMaxMii != -1 && (int)MII > SwpMaxMii)
    return;

  computeNodeFunctions(NodeSets);

  registerPressureFilter(NodeSets);

  colocateNodeSets(NodeSets);

  checkNodeSets(NodeSets);

  DEBUG({
    for (auto &I : NodeSets) {
      dbgs() << "  Rec NodeSet ";
      I.dump();
    }
  });

  std::sort(NodeSets.begin(), NodeSets.end(), std::greater<NodeSet>());

  groupRemainingNodes(NodeSets);

  removeDuplicateNodes(NodeSets);

  DEBUG({
    for (auto &I : NodeSets) {
      dbgs() << "  NodeSet ";
      I.dump();
    }
  });

  computeNodeOrder(NodeSets);

  SMSchedule Schedule(Pass.MF);
  Scheduled = schedulePipeline(Schedule);

  if (!Scheduled)
    return;

  unsigned numStages = Schedule.getMaxStageCount();
  // No need to generate pipeline if there are no overlapped iterations.
  if (numStages == 0)
    return;

  // Check that the maximum stage count is less than user-defined limit.
  if (SwpMaxStages > -1 && (int)numStages > SwpMaxStages)
    return;

  generatePipelinedLoop(Schedule);
  ++NumPipelined;
}

/// Clean up after the software pipeliner runs.
void SwingSchedulerDAG::finishBlock() {
  for (MachineInstr *I : NewMIs)
    MF.DeleteMachineInstr(I);
  NewMIs.clear();

  // Call the superclass.
  ScheduleDAGInstrs::finishBlock();
}

/// Return the register values for  the operands of a Phi instruction.
/// This function assume the instruction is a Phi.
static void getPhiRegs(MachineInstr &Phi, MachineBasicBlock *Loop,
                       unsigned &InitVal, unsigned &LoopVal) {
  assert(Phi.isPHI() && "Expecting a Phi.");

  InitVal = 0;
  LoopVal = 0;
  for (unsigned i = 1, e = Phi.getNumOperands(); i != e; i += 2)
    if (Phi.getOperand(i + 1).getMBB() != Loop)
      InitVal = Phi.getOperand(i).getReg();
    else
      LoopVal = Phi.getOperand(i).getReg();

  assert(InitVal != 0 && LoopVal != 0 && "Unexpected Phi structure.");
}

/// Return the Phi register value that comes from the incoming block.
static unsigned getInitPhiReg(MachineInstr &Phi, MachineBasicBlock *LoopBB) {
  for (unsigned i = 1, e = Phi.getNumOperands(); i != e; i += 2)
    if (Phi.getOperand(i + 1).getMBB() != LoopBB)
      return Phi.getOperand(i).getReg();
  return 0;
}

/// Return the Phi register value that comes the the loop block.
static unsigned getLoopPhiReg(MachineInstr &Phi, MachineBasicBlock *LoopBB) {
  for (unsigned i = 1, e = Phi.getNumOperands(); i != e; i += 2)
    if (Phi.getOperand(i + 1).getMBB() == LoopBB)
      return Phi.getOperand(i).getReg();
  return 0;
}

/// Return true if SUb can be reached from SUa following the chain edges.
static bool isSuccOrder(SUnit *SUa, SUnit *SUb) {
  SmallPtrSet<SUnit *, 8> Visited;
  SmallVector<SUnit *, 8> Worklist;
  Worklist.push_back(SUa);
  while (!Worklist.empty()) {
    const SUnit *SU = Worklist.pop_back_val();
    for (auto &SI : SU->Succs) {
      SUnit *SuccSU = SI.getSUnit();
      if (SI.getKind() == SDep::Order) {
        if (Visited.count(SuccSU))
          continue;
        if (SuccSU == SUb)
          return true;
        Worklist.push_back(SuccSU);
        Visited.insert(SuccSU);
      }
    }
  }
  return false;
}

/// Return true if the instruction causes a chain between memory
/// references before and after it.
static bool isDependenceBarrier(MachineInstr &MI, AliasAnalysis *AA) {
  return MI.isCall() || MI.hasUnmodeledSideEffects() ||
         (MI.hasOrderedMemoryRef() &&
          (!MI.mayLoad() || !MI.isDereferenceableInvariantLoad(AA)));
}

/// Return the underlying objects for the memory references of an instruction.
/// This function calls the code in ValueTracking, but first checks that the
/// instruction has a memory operand.
static void getUnderlyingObjects(MachineInstr *MI,
                                 SmallVectorImpl<Value *> &Objs,
                                 const DataLayout &DL) {
  if (!MI->hasOneMemOperand())
    return;
  MachineMemOperand *MM = *MI->memoperands_begin();
  if (!MM->getValue())
    return;
  GetUnderlyingObjects(const_cast<Value *>(MM->getValue()), Objs, DL);
}

/// Add a chain edge between a load and store if the store can be an
/// alias of the load on a subsequent iteration, i.e., a loop carried
/// dependence. This code is very similar to the code in ScheduleDAGInstrs
/// but that code doesn't create loop carried dependences.
void SwingSchedulerDAG::addLoopCarriedDependences(AliasAnalysis *AA) {
  MapVector<Value *, SmallVector<SUnit *, 4>> PendingLoads;
  for (auto &SU : SUnits) {
    MachineInstr &MI = *SU.getInstr();
    if (isDependenceBarrier(MI, AA))
      PendingLoads.clear();
    else if (MI.mayLoad()) {
      SmallVector<Value *, 4> Objs;
      getUnderlyingObjects(&MI, Objs, MF.getDataLayout());
      for (auto V : Objs) {
        SmallVector<SUnit *, 4> &SUs = PendingLoads[V];
        SUs.push_back(&SU);
      }
    } else if (MI.mayStore()) {
      SmallVector<Value *, 4> Objs;
      getUnderlyingObjects(&MI, Objs, MF.getDataLayout());
      for (auto V : Objs) {
        MapVector<Value *, SmallVector<SUnit *, 4>>::iterator I =
            PendingLoads.find(V);
        if (I == PendingLoads.end())
          continue;
        for (auto Load : I->second) {
          if (isSuccOrder(Load, &SU))
            continue;
          MachineInstr &LdMI = *Load->getInstr();
          // First, perform the cheaper check that compares the base register.
          // If they are the same and the load offset is less than the store
          // offset, then mark the dependence as loop carried potentially.
          unsigned BaseReg1, BaseReg2;
          int64_t Offset1, Offset2;
          if (!TII->getMemOpBaseRegImmOfs(LdMI, BaseReg1, Offset1, TRI) ||
              !TII->getMemOpBaseRegImmOfs(MI, BaseReg2, Offset2, TRI)) {
            SU.addPred(SDep(Load, SDep::Barrier));
            continue;            
          }
          if (BaseReg1 == BaseReg2 && (int)Offset1 < (int)Offset2) {
            assert(TII->areMemAccessesTriviallyDisjoint(LdMI, MI, AA) &&
                   "What happened to the chain edge?");
            SU.addPred(SDep(Load, SDep::Barrier));
            continue;
          }
          // Second, the more expensive check that uses alias analysis on the
          // base registers. If they alias, and the load offset is less than
          // the store offset, the mark the dependence as loop carried.
          if (!AA) {
            SU.addPred(SDep(Load, SDep::Barrier));
            continue;
          }
          MachineMemOperand *MMO1 = *LdMI.memoperands_begin();
          MachineMemOperand *MMO2 = *MI.memoperands_begin();
          if (!MMO1->getValue() || !MMO2->getValue()) {
            SU.addPred(SDep(Load, SDep::Barrier));
            continue;
          }
          if (MMO1->getValue() == MMO2->getValue() &&
              MMO1->getOffset() <= MMO2->getOffset()) {
            SU.addPred(SDep(Load, SDep::Barrier));
            continue;
          }
          AliasResult AAResult = AA->alias(
              MemoryLocation(MMO1->getValue(), MemoryLocation::UnknownSize,
                             MMO1->getAAInfo()),
              MemoryLocation(MMO2->getValue(), MemoryLocation::UnknownSize,
                             MMO2->getAAInfo()));

          if (AAResult != NoAlias)
            SU.addPred(SDep(Load, SDep::Barrier));
        }
      }
    }
  }
}

/// Update the phi dependences to the DAG because ScheduleDAGInstrs no longer
/// processes dependences for PHIs. This function adds true dependences
/// from a PHI to a use, and a loop carried dependence from the use to the
/// PHI. The loop carried dependence is represented as an anti dependence
/// edge. This function also removes chain dependences between unrelated
/// PHIs.
void SwingSchedulerDAG::updatePhiDependences() {
  SmallVector<SDep, 4> RemoveDeps;
  const TargetSubtargetInfo &ST = MF.getSubtarget<TargetSubtargetInfo>();

  // Iterate over each DAG node.
  for (SUnit &I : SUnits) {
    RemoveDeps.clear();
    // Set to true if the instruction has an operand defined by a Phi.
    unsigned HasPhiUse = 0;
    unsigned HasPhiDef = 0;
    MachineInstr *MI = I.getInstr();
    // Iterate over each operand, and we process the definitions.
    for (MachineInstr::mop_iterator MOI = MI->operands_begin(),
                                    MOE = MI->operands_end();
         MOI != MOE; ++MOI) {
      if (!MOI->isReg())
        continue;
      unsigned Reg = MOI->getReg();
      if (MOI->isDef()) {
        // If the register is used by a Phi, then create an anti dependence.
        for (MachineRegisterInfo::use_instr_iterator
                 UI = MRI.use_instr_begin(Reg),
                 UE = MRI.use_instr_end();
             UI != UE; ++UI) {
          MachineInstr *UseMI = &*UI;
          SUnit *SU = getSUnit(UseMI);
          if (SU != nullptr && UseMI->isPHI()) {
            if (!MI->isPHI()) {
              SDep Dep(SU, SDep::Anti, Reg);
              I.addPred(Dep);
            } else {
              HasPhiDef = Reg;
              // Add a chain edge to a dependent Phi that isn't an existing
              // predecessor.
              if (SU->NodeNum < I.NodeNum && !I.isPred(SU))
                I.addPred(SDep(SU, SDep::Barrier));
            }
          }
        }
      } else if (MOI->isUse()) {
        // If the register is defined by a Phi, then create a true dependence.
        MachineInstr *DefMI = MRI.getUniqueVRegDef(Reg);
        if (DefMI == nullptr)
          continue;
        SUnit *SU = getSUnit(DefMI);
        if (SU != nullptr && DefMI->isPHI()) {
          if (!MI->isPHI()) {
            SDep Dep(SU, SDep::Data, Reg);
            Dep.setLatency(0);
            ST.adjustSchedDependency(SU, &I, Dep);
            I.addPred(Dep);
          } else {
            HasPhiUse = Reg;
            // Add a chain edge to a dependent Phi that isn't an existing
            // predecessor.
            if (SU->NodeNum < I.NodeNum && !I.isPred(SU))
              I.addPred(SDep(SU, SDep::Barrier));
          }
        }
      }
    }
    // Remove order dependences from an unrelated Phi.
    if (!SwpPruneDeps)
      continue;
    for (auto &PI : I.Preds) {
      MachineInstr *PMI = PI.getSUnit()->getInstr();
      if (PMI->isPHI() && PI.getKind() == SDep::Order) {
        if (I.getInstr()->isPHI()) {
          if (PMI->getOperand(0).getReg() == HasPhiUse)
            continue;
          if (getLoopPhiReg(*PMI, PMI->getParent()) == HasPhiDef)
            continue;
        }
        RemoveDeps.push_back(PI);
      }
    }
    for (int i = 0, e = RemoveDeps.size(); i != e; ++i)
      I.removePred(RemoveDeps[i]);
  }
}

/// Iterate over each DAG node and see if we can change any dependences
/// in order to reduce the recurrence MII.
void SwingSchedulerDAG::changeDependences() {
  // See if an instruction can use a value from the previous iteration.
  // If so, we update the base and offset of the instruction and change
  // the dependences.
  for (SUnit &I : SUnits) {
    unsigned BasePos = 0, OffsetPos = 0, NewBase = 0;
    int64_t NewOffset = 0;
    if (!canUseLastOffsetValue(I.getInstr(), BasePos, OffsetPos, NewBase,
                               NewOffset))
      continue;

    // Get the MI and SUnit for the instruction that defines the original base.
    unsigned OrigBase = I.getInstr()->getOperand(BasePos).getReg();
    MachineInstr *DefMI = MRI.getUniqueVRegDef(OrigBase);
    if (!DefMI)
      continue;
    SUnit *DefSU = getSUnit(DefMI);
    if (!DefSU)
      continue;
    // Get the MI and SUnit for the instruction that defins the new base.
    MachineInstr *LastMI = MRI.getUniqueVRegDef(NewBase);
    if (!LastMI)
      continue;
    SUnit *LastSU = getSUnit(LastMI);
    if (!LastSU)
      continue;

    if (Topo.IsReachable(&I, LastSU))
      continue;

    // Remove the dependence. The value now depends on a prior iteration.
    SmallVector<SDep, 4> Deps;
    for (SUnit::pred_iterator P = I.Preds.begin(), E = I.Preds.end(); P != E;
         ++P)
      if (P->getSUnit() == DefSU)
        Deps.push_back(*P);
    for (int i = 0, e = Deps.size(); i != e; i++) {
      Topo.RemovePred(&I, Deps[i].getSUnit());
      I.removePred(Deps[i]);
    }
    // Remove the chain dependence between the instructions.
    Deps.clear();
    for (auto &P : LastSU->Preds)
      if (P.getSUnit() == &I && P.getKind() == SDep::Order)
        Deps.push_back(P);
    for (int i = 0, e = Deps.size(); i != e; i++) {
      Topo.RemovePred(LastSU, Deps[i].getSUnit());
      LastSU->removePred(Deps[i]);
    }

    // Add a dependence between the new instruction and the instruction
    // that defines the new base.
    SDep Dep(&I, SDep::Anti, NewBase);
    LastSU->addPred(Dep);

    // Remember the base and offset information so that we can update the
    // instruction during code generation.
    InstrChanges[&I] = std::make_pair(NewBase, NewOffset);
  }
}

namespace {

// FuncUnitSorter - Comparison operator used to sort instructions by
// the number of functional unit choices.
struct FuncUnitSorter {
  const InstrItineraryData *InstrItins;
  DenseMap<unsigned, unsigned> Resources;

  FuncUnitSorter(const InstrItineraryData *IID) : InstrItins(IID) {}

  // Compute the number of functional unit alternatives needed
  // at each stage, and take the minimum value. We prioritize the
  // instructions by the least number of choices first.
  unsigned minFuncUnits(const MachineInstr *Inst, unsigned &F) const {
    unsigned schedClass = Inst->getDesc().getSchedClass();
    unsigned min = UINT_MAX;
    for (const InstrStage *IS = InstrItins->beginStage(schedClass),
                          *IE = InstrItins->endStage(schedClass);
         IS != IE; ++IS) {
      unsigned funcUnits = IS->getUnits();
      unsigned numAlternatives = countPopulation(funcUnits);
      if (numAlternatives < min) {
        min = numAlternatives;
        F = funcUnits;
      }
    }
    return min;
  }

  // Compute the critical resources needed by the instruction. This
  // function records the functional units needed by instructions that
  // must use only one functional unit. We use this as a tie breaker
  // for computing the resource MII. The instrutions that require
  // the same, highly used, functional unit have high priority.
  void calcCriticalResources(MachineInstr &MI) {
    unsigned SchedClass = MI.getDesc().getSchedClass();
    for (const InstrStage *IS = InstrItins->beginStage(SchedClass),
                          *IE = InstrItins->endStage(SchedClass);
         IS != IE; ++IS) {
      unsigned FuncUnits = IS->getUnits();
      if (countPopulation(FuncUnits) == 1)
        Resources[FuncUnits]++;
    }
  }

  /// Return true if IS1 has less priority than IS2.
  bool operator()(const MachineInstr *IS1, const MachineInstr *IS2) const {
    unsigned F1 = 0, F2 = 0;
    unsigned MFUs1 = minFuncUnits(IS1, F1);
    unsigned MFUs2 = minFuncUnits(IS2, F2);
    if (MFUs1 == 1 && MFUs2 == 1)
      return Resources.lookup(F1) < Resources.lookup(F2);
    return MFUs1 > MFUs2;
  }
};

} // end anonymous namespace

/// Calculate the resource constrained minimum initiation interval for the
/// specified loop. We use the DFA to model the resources needed for
/// each instruction, and we ignore dependences. A different DFA is created
/// for each cycle that is required. When adding a new instruction, we attempt
/// to add it to each existing DFA, until a legal space is found. If the
/// instruction cannot be reserved in an existing DFA, we create a new one.
unsigned SwingSchedulerDAG::calculateResMII() {
  SmallVector<DFAPacketizer *, 8> Resources;
  MachineBasicBlock *MBB = Loop.getHeader();
  Resources.push_back(TII->CreateTargetScheduleState(MF.getSubtarget()));

  // Sort the instructions by the number of available choices for scheduling,
  // least to most. Use the number of critical resources as the tie breaker.
  FuncUnitSorter FUS =
      FuncUnitSorter(MF.getSubtarget().getInstrItineraryData());
  for (MachineBasicBlock::iterator I = MBB->getFirstNonPHI(),
                                   E = MBB->getFirstTerminator();
       I != E; ++I)
    FUS.calcCriticalResources(*I);
  PriorityQueue<MachineInstr *, std::vector<MachineInstr *>, FuncUnitSorter>
      FuncUnitOrder(FUS);

  for (MachineBasicBlock::iterator I = MBB->getFirstNonPHI(),
                                   E = MBB->getFirstTerminator();
       I != E; ++I)
    FuncUnitOrder.push(&*I);

  while (!FuncUnitOrder.empty()) {
    MachineInstr *MI = FuncUnitOrder.top();
    FuncUnitOrder.pop();
    if (TII->isZeroCost(MI->getOpcode()))
      continue;
    // Attempt to reserve the instruction in an existing DFA. At least one
    // DFA is needed for each cycle.
    unsigned NumCycles = getSUnit(MI)->Latency;
    unsigned ReservedCycles = 0;
    SmallVectorImpl<DFAPacketizer *>::iterator RI = Resources.begin();
    SmallVectorImpl<DFAPacketizer *>::iterator RE = Resources.end();
    for (unsigned C = 0; C < NumCycles; ++C)
      while (RI != RE) {
        if ((*RI++)->canReserveResources(*MI)) {
          ++ReservedCycles;
          break;
        }
      }
    // Start reserving resources using existing DFAs.
    for (unsigned C = 0; C < ReservedCycles; ++C) {
      --RI;
      (*RI)->reserveResources(*MI);
    }
    // Add new DFAs, if needed, to reserve resources.
    for (unsigned C = ReservedCycles; C < NumCycles; ++C) {
      DFAPacketizer *NewResource =
          TII->CreateTargetScheduleState(MF.getSubtarget());
      assert(NewResource->canReserveResources(*MI) && "Reserve error.");
      NewResource->reserveResources(*MI);
      Resources.push_back(NewResource);
    }
  }
  int Resmii = Resources.size();
  // Delete the memory for each of the DFAs that were created earlier.
  for (DFAPacketizer *RI : Resources) {
    DFAPacketizer *D = RI;
    delete D;
  }
  Resources.clear();
  return Resmii;
}

/// Calculate the recurrence-constrainted minimum initiation interval.
/// Iterate over each circuit.  Compute the delay(c) and distance(c)
/// for each circuit. The II needs to satisfy the inequality
/// delay(c) - II*distance(c) <= 0. For each circuit, choose the smallest
/// II that satistifies the inequality, and the RecMII is the maximum
/// of those values.
unsigned SwingSchedulerDAG::calculateRecMII(NodeSetType &NodeSets) {
  unsigned RecMII = 0;

  for (NodeSet &Nodes : NodeSets) {
    if (Nodes.empty())
      continue;

    unsigned Delay = Nodes.size() - 1;
    unsigned Distance = 1;

    // ii = ceil(delay / distance)
    unsigned CurMII = (Delay + Distance - 1) / Distance;
    Nodes.setRecMII(CurMII);
    if (CurMII > RecMII)
      RecMII = CurMII;
  }

  return RecMII;
}

/// Swap all the anti dependences in the DAG. That means it is no longer a DAG,
/// but we do this to find the circuits, and then change them back.
static void swapAntiDependences(std::vector<SUnit> &SUnits) {
  SmallVector<std::pair<SUnit *, SDep>, 8> DepsAdded;
  for (unsigned i = 0, e = SUnits.size(); i != e; ++i) {
    SUnit *SU = &SUnits[i];
    for (SUnit::pred_iterator IP = SU->Preds.begin(), EP = SU->Preds.end();
         IP != EP; ++IP) {
      if (IP->getKind() != SDep::Anti)
        continue;
      DepsAdded.push_back(std::make_pair(SU, *IP));
    }
  }
  for (SmallVector<std::pair<SUnit *, SDep>, 8>::iterator I = DepsAdded.begin(),
                                                          E = DepsAdded.end();
       I != E; ++I) {
    // Remove this anti dependency and add one in the reverse direction.
    SUnit *SU = I->first;
    SDep &D = I->second;
    SUnit *TargetSU = D.getSUnit();
    unsigned Reg = D.getReg();
    unsigned Lat = D.getLatency();
    SU->removePred(D);
    SDep Dep(SU, SDep::Anti, Reg);
    Dep.setLatency(Lat);
    TargetSU->addPred(Dep);
  }
}

/// Create the adjacency structure of the nodes in the graph.
void SwingSchedulerDAG::Circuits::createAdjacencyStructure(
    SwingSchedulerDAG *DAG) {
  BitVector Added(SUnits.size());
  for (int i = 0, e = SUnits.size(); i != e; ++i) {
    Added.reset();
    // Add any successor to the adjacency matrix and exclude duplicates.
    for (auto &SI : SUnits[i].Succs) {
      // Do not process a boundary node and a back-edge is processed only
      // if it goes to a Phi.
      if (SI.getSUnit()->isBoundaryNode() ||
          (SI.getKind() == SDep::Anti && !SI.getSUnit()->getInstr()->isPHI()))
        continue;
      int N = SI.getSUnit()->NodeNum;
      if (!Added.test(N)) {
        AdjK[i].push_back(N);
        Added.set(N);
      }
    }
    // A chain edge between a store and a load is treated as a back-edge in the
    // adjacency matrix.
    for (auto &PI : SUnits[i].Preds) {
      if (!SUnits[i].getInstr()->mayStore() ||
          !DAG->isLoopCarriedOrder(&SUnits[i], PI, false))
        continue;
      if (PI.getKind() == SDep::Order && PI.getSUnit()->getInstr()->mayLoad()) {
        int N = PI.getSUnit()->NodeNum;
        if (!Added.test(N)) {
          AdjK[i].push_back(N);
          Added.set(N);
        }
      }
    }
  }
}

/// Identify an elementary circuit in the dependence graph starting at the
/// specified node.
bool SwingSchedulerDAG::Circuits::circuit(int V, int S, NodeSetType &NodeSets,
                                          bool HasBackedge) {
  SUnit *SV = &SUnits[V];
  bool F = false;
  Stack.insert(SV);
  Blocked.set(V);

  for (auto W : AdjK[V]) {
    if (NumPaths > MaxPaths)
      break;
    if (W < S)
      continue;
    if (W == S) {
      if (!HasBackedge)
        NodeSets.push_back(NodeSet(Stack.begin(), Stack.end()));
      F = true;
      ++NumPaths;
      break;
    } else if (!Blocked.test(W)) {
      if (circuit(W, S, NodeSets, W < V ? true : HasBackedge))
        F = true;
    }
  }

  if (F)
    unblock(V);
  else {
    for (auto W : AdjK[V]) {
      if (W < S)
        continue;
      if (B[W].count(SV) == 0)
        B[W].insert(SV);
    }
  }
  Stack.pop_back();
  return F;
}

/// Unblock a node in the circuit finding algorithm.
void SwingSchedulerDAG::Circuits::unblock(int U) {
  Blocked.reset(U);
  SmallPtrSet<SUnit *, 4> &BU = B[U];
  while (!BU.empty()) {
    SmallPtrSet<SUnit *, 4>::iterator SI = BU.begin();
    assert(SI != BU.end() && "Invalid B set.");
    SUnit *W = *SI;
    BU.erase(W);
    if (Blocked.test(W->NodeNum))
      unblock(W->NodeNum);
  }
}

/// Identify all the elementary circuits in the dependence graph using
/// Johnson's circuit algorithm.
void SwingSchedulerDAG::findCircuits(NodeSetType &NodeSets) {
  // Swap all the anti dependences in the DAG. That means it is no longer a DAG,
  // but we do this to find the circuits, and then change them back.
  swapAntiDependences(SUnits);

  Circuits Cir(SUnits);
  // Create the adjacency structure.
  Cir.createAdjacencyStructure(this);
  for (int i = 0, e = SUnits.size(); i != e; ++i) {
    Cir.reset();
    Cir.circuit(i, i, NodeSets);
  }

  // Change the dependences back so that we've created a DAG again.
  swapAntiDependences(SUnits);
}

/// Return true for DAG nodes that we ignore when computing the cost functions.
/// We ignore the back-edge recurrence in order to avoid unbounded recurison
/// in the calculation of the ASAP, ALAP, etc functions.
static bool ignoreDependence(const SDep &D, bool isPred) {
  if (D.isArtificial())
    return true;
  return D.getKind() == SDep::Anti && isPred;
}

/// Compute several functions need to order the nodes for scheduling.
///  ASAP - Earliest time to schedule a node.
///  ALAP - Latest time to schedule a node.
///  MOV - Mobility function, difference between ALAP and ASAP.
///  D - Depth of each node.
///  H - Height of each node.
void SwingSchedulerDAG::computeNodeFunctions(NodeSetType &NodeSets) {
  ScheduleInfo.resize(SUnits.size());

  DEBUG({
    for (ScheduleDAGTopologicalSort::const_iterator I = Topo.begin(),
                                                    E = Topo.end();
         I != E; ++I) {
      SUnit *SU = &SUnits[*I];
      SU->dump(this);
    }
  });

  int maxASAP = 0;
  // Compute ASAP.
  for (ScheduleDAGTopologicalSort::const_iterator I = Topo.begin(),
                                                  E = Topo.end();
       I != E; ++I) {
    int asap = 0;
    SUnit *SU = &SUnits[*I];
    for (SUnit::const_pred_iterator IP = SU->Preds.begin(),
                                    EP = SU->Preds.end();
         IP != EP; ++IP) {
      if (ignoreDependence(*IP, true))
        continue;
      SUnit *pred = IP->getSUnit();
      asap = std::max(asap, (int)(getASAP(pred) + getLatency(SU, *IP) -
                                  getDistance(pred, SU, *IP) * MII));
    }
    maxASAP = std::max(maxASAP, asap);
    ScheduleInfo[*I].ASAP = asap;
  }

  // Compute ALAP and MOV.
  for (ScheduleDAGTopologicalSort::const_reverse_iterator I = Topo.rbegin(),
                                                          E = Topo.rend();
       I != E; ++I) {
    int alap = maxASAP;
    SUnit *SU = &SUnits[*I];
    for (SUnit::const_succ_iterator IS = SU->Succs.begin(),
                                    ES = SU->Succs.end();
         IS != ES; ++IS) {
      if (ignoreDependence(*IS, true))
        continue;
      SUnit *succ = IS->getSUnit();
      alap = std::min(alap, (int)(getALAP(succ) - getLatency(SU, *IS) +
                                  getDistance(SU, succ, *IS) * MII));
    }

    ScheduleInfo[*I].ALAP = alap;
  }

  // After computing the node functions, compute the summary for each node set.
  for (NodeSet &I : NodeSets)
    I.computeNodeSetInfo(this);

  DEBUG({
    for (unsigned i = 0; i < SUnits.size(); i++) {
      dbgs() << "\tNode " << i << ":\n";
      dbgs() << "\t   ASAP = " << getASAP(&SUnits[i]) << "\n";
      dbgs() << "\t   ALAP = " << getALAP(&SUnits[i]) << "\n";
      dbgs() << "\t   MOV  = " << getMOV(&SUnits[i]) << "\n";
      dbgs() << "\t   D    = " << getDepth(&SUnits[i]) << "\n";
      dbgs() << "\t   H    = " << getHeight(&SUnits[i]) << "\n";
    }
  });
}

/// Compute the Pred_L(O) set, as defined in the paper. The set is defined
/// as the predecessors of the elements of NodeOrder that are not also in
/// NodeOrder.
static bool pred_L(SetVector<SUnit *> &NodeOrder,
                   SmallSetVector<SUnit *, 8> &Preds,
                   const NodeSet *S = nullptr) {
  Preds.clear();
  for (SetVector<SUnit *>::iterator I = NodeOrder.begin(), E = NodeOrder.end();
       I != E; ++I) {
    for (SUnit::pred_iterator PI = (*I)->Preds.begin(), PE = (*I)->Preds.end();
         PI != PE; ++PI) {
      if (S && S->count(PI->getSUnit()) == 0)
        continue;
      if (ignoreDependence(*PI, true))
        continue;
      if (NodeOrder.count(PI->getSUnit()) == 0)
        Preds.insert(PI->getSUnit());
    }
    // Back-edges are predecessors with an anti-dependence.
    for (SUnit::const_succ_iterator IS = (*I)->Succs.begin(),
                                    ES = (*I)->Succs.end();
         IS != ES; ++IS) {
      if (IS->getKind() != SDep::Anti)
        continue;
      if (S && S->count(IS->getSUnit()) == 0)
        continue;
      if (NodeOrder.count(IS->getSUnit()) == 0)
        Preds.insert(IS->getSUnit());
    }
  }
  return !Preds.empty();
}

/// Compute the Succ_L(O) set, as defined in the paper. The set is defined
/// as the successors of the elements of NodeOrder that are not also in
/// NodeOrder.
static bool succ_L(SetVector<SUnit *> &NodeOrder,
                   SmallSetVector<SUnit *, 8> &Succs,
                   const NodeSet *S = nullptr) {
  Succs.clear();
  for (SetVector<SUnit *>::iterator I = NodeOrder.begin(), E = NodeOrder.end();
       I != E; ++I) {
    for (SUnit::succ_iterator SI = (*I)->Succs.begin(), SE = (*I)->Succs.end();
         SI != SE; ++SI) {
      if (S && S->count(SI->getSUnit()) == 0)
        continue;
      if (ignoreDependence(*SI, false))
        continue;
      if (NodeOrder.count(SI->getSUnit()) == 0)
        Succs.insert(SI->getSUnit());
    }
    for (SUnit::const_pred_iterator PI = (*I)->Preds.begin(),
                                    PE = (*I)->Preds.end();
         PI != PE; ++PI) {
      if (PI->getKind() != SDep::Anti)
        continue;
      if (S && S->count(PI->getSUnit()) == 0)
        continue;
      if (NodeOrder.count(PI->getSUnit()) == 0)
        Succs.insert(PI->getSUnit());
    }
  }
  return !Succs.empty();
}

/// Return true if there is a path from the specified node to any of the nodes
/// in DestNodes. Keep track and return the nodes in any path.
static bool computePath(SUnit *Cur, SetVector<SUnit *> &Path,
                        SetVector<SUnit *> &DestNodes,
                        SetVector<SUnit *> &Exclude,
                        SmallPtrSet<SUnit *, 8> &Visited) {
  if (Cur->isBoundaryNode())
    return false;
  if (Exclude.count(Cur) != 0)
    return false;
  if (DestNodes.count(Cur) != 0)
    return true;
  if (!Visited.insert(Cur).second)
    return Path.count(Cur) != 0;
  bool FoundPath = false;
  for (auto &SI : Cur->Succs)
    FoundPath |= computePath(SI.getSUnit(), Path, DestNodes, Exclude, Visited);
  for (auto &PI : Cur->Preds)
    if (PI.getKind() == SDep::Anti)
      FoundPath |=
          computePath(PI.getSUnit(), Path, DestNodes, Exclude, Visited);
  if (FoundPath)
    Path.insert(Cur);
  return FoundPath;
}

/// Return true if Set1 is a subset of Set2.
template <class S1Ty, class S2Ty> static bool isSubset(S1Ty &Set1, S2Ty &Set2) {
  for (typename S1Ty::iterator I = Set1.begin(), E = Set1.end(); I != E; ++I)
    if (Set2.count(*I) == 0)
      return false;
  return true;
}

/// Compute the live-out registers for the instructions in a node-set.
/// The live-out registers are those that are defined in the node-set,
/// but not used. Except for use operands of Phis.
static void computeLiveOuts(MachineFunction &MF, RegPressureTracker &RPTracker,
                            NodeSet &NS) {
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  SmallVector<RegisterMaskPair, 8> LiveOutRegs;
  SmallSet<unsigned, 4> Uses;
  for (SUnit *SU : NS) {
    const MachineInstr *MI = SU->getInstr();
    if (MI->isPHI())
      continue;
    for (const MachineOperand &MO : MI->operands())
      if (MO.isReg() && MO.isUse()) {
        unsigned Reg = MO.getReg();
        if (TargetRegisterInfo::isVirtualRegister(Reg))
          Uses.insert(Reg);
        else if (MRI.isAllocatable(Reg))
          for (MCRegUnitIterator Units(Reg, TRI); Units.isValid(); ++Units)
            Uses.insert(*Units);
      }
  }
  for (SUnit *SU : NS)
    for (const MachineOperand &MO : SU->getInstr()->operands())
      if (MO.isReg() && MO.isDef() && !MO.isDead()) {
        unsigned Reg = MO.getReg();
        if (TargetRegisterInfo::isVirtualRegister(Reg)) {
          if (!Uses.count(Reg))
            LiveOutRegs.push_back(RegisterMaskPair(Reg,
                                                   LaneBitmask::getNone()));
        } else if (MRI.isAllocatable(Reg)) {
          for (MCRegUnitIterator Units(Reg, TRI); Units.isValid(); ++Units)
            if (!Uses.count(*Units))
              LiveOutRegs.push_back(RegisterMaskPair(*Units,
                                                     LaneBitmask::getNone()));
        }
      }
  RPTracker.addLiveRegs(LiveOutRegs);
}

/// A heuristic to filter nodes in recurrent node-sets if the register
/// pressure of a set is too high.
void SwingSchedulerDAG::registerPressureFilter(NodeSetType &NodeSets) {
  for (auto &NS : NodeSets) {
    // Skip small node-sets since they won't cause register pressure problems.
    if (NS.size() <= 2)
      continue;
    IntervalPressure RecRegPressure;
    RegPressureTracker RecRPTracker(RecRegPressure);
    RecRPTracker.init(&MF, &RegClassInfo, &LIS, BB, BB->end(), false, true);
    computeLiveOuts(MF, RecRPTracker, NS);
    RecRPTracker.closeBottom();

    std::vector<SUnit *> SUnits(NS.begin(), NS.end());
    std::sort(SUnits.begin(), SUnits.end(), [](const SUnit *A, const SUnit *B) {
      return A->NodeNum > B->NodeNum;
    });

    for (auto &SU : SUnits) {
      // Since we're computing the register pressure for a subset of the
      // instructions in a block, we need to set the tracker for each
      // instruction in the node-set. The tracker is set to the instruction
      // just after the one we're interested in.
      MachineBasicBlock::const_iterator CurInstI = SU->getInstr();
      RecRPTracker.setPos(std::next(CurInstI));

      RegPressureDelta RPDelta;
      ArrayRef<PressureChange> CriticalPSets;
      RecRPTracker.getMaxUpwardPressureDelta(SU->getInstr(), nullptr, RPDelta,
                                             CriticalPSets,
                                             RecRegPressure.MaxSetPressure);
      if (RPDelta.Excess.isValid()) {
        DEBUG(dbgs() << "Excess register pressure: SU(" << SU->NodeNum << ") "
                     << TRI->getRegPressureSetName(RPDelta.Excess.getPSet())
                     << ":" << RPDelta.Excess.getUnitInc());
        NS.setExceedPressure(SU);
        break;
      }
      RecRPTracker.recede();
    }
  }
}

/// A heuristic to colocate node sets that have the same set of
/// successors.
void SwingSchedulerDAG::colocateNodeSets(NodeSetType &NodeSets) {
  unsigned Colocate = 0;
  for (int i = 0, e = NodeSets.size(); i < e; ++i) {
    NodeSet &N1 = NodeSets[i];
    SmallSetVector<SUnit *, 8> S1;
    if (N1.empty() || !succ_L(N1, S1))
      continue;
    for (int j = i + 1; j < e; ++j) {
      NodeSet &N2 = NodeSets[j];
      if (N1.compareRecMII(N2) != 0)
        continue;
      SmallSetVector<SUnit *, 8> S2;
      if (N2.empty() || !succ_L(N2, S2))
        continue;
      if (isSubset(S1, S2) && S1.size() == S2.size()) {
        N1.setColocate(++Colocate);
        N2.setColocate(Colocate);
        break;
      }
    }
  }
}

/// Check if the existing node-sets are profitable. If not, then ignore the
/// recurrent node-sets, and attempt to schedule all nodes together. This is
/// a heuristic. If the MII is large and there is a non-recurrent node with
/// a large depth compared to the MII, then it's best to try and schedule
/// all instruction together instead of starting with the recurrent node-sets.
void SwingSchedulerDAG::checkNodeSets(NodeSetType &NodeSets) {
  // Look for loops with a large MII.
  if (MII <= 20)
    return;
  // Check if the node-set contains only a simple add recurrence.
  for (auto &NS : NodeSets)
    if (NS.size() > 2)
      return;
  // If the depth of any instruction is significantly larger than the MII, then
  // ignore the recurrent node-sets and treat all instructions equally.
  for (auto &SU : SUnits)
    if (SU.getDepth() > MII * 1.5) {
      NodeSets.clear();
      DEBUG(dbgs() << "Clear recurrence node-sets\n");
      return;
    }
}

/// Add the nodes that do not belong to a recurrence set into groups
/// based upon connected componenets.
void SwingSchedulerDAG::groupRemainingNodes(NodeSetType &NodeSets) {
  SetVector<SUnit *> NodesAdded;
  SmallPtrSet<SUnit *, 8> Visited;
  // Add the nodes that are on a path between the previous node sets and
  // the current node set.
  for (NodeSet &I : NodeSets) {
    SmallSetVector<SUnit *, 8> N;
    // Add the nodes from the current node set to the previous node set.
    if (succ_L(I, N)) {
      SetVector<SUnit *> Path;
      for (SUnit *NI : N) {
        Visited.clear();
        computePath(NI, Path, NodesAdded, I, Visited);
      }
      if (!Path.empty())
        I.insert(Path.begin(), Path.end());
    }
    // Add the nodes from the previous node set to the current node set.
    N.clear();
    if (succ_L(NodesAdded, N)) {
      SetVector<SUnit *> Path;
      for (SUnit *NI : N) {
        Visited.clear();
        computePath(NI, Path, I, NodesAdded, Visited);
      }
      if (!Path.empty())
        I.insert(Path.begin(), Path.end());
    }
    NodesAdded.insert(I.begin(), I.end());
  }

  // Create a new node set with the connected nodes of any successor of a node
  // in a recurrent set.
  NodeSet NewSet;
  SmallSetVector<SUnit *, 8> N;
  if (succ_L(NodesAdded, N))
    for (SUnit *I : N)
      addConnectedNodes(I, NewSet, NodesAdded);
  if (!NewSet.empty())
    NodeSets.push_back(NewSet);

  // Create a new node set with the connected nodes of any predecessor of a node
  // in a recurrent set.
  NewSet.clear();
  if (pred_L(NodesAdded, N))
    for (SUnit *I : N)
      addConnectedNodes(I, NewSet, NodesAdded);
  if (!NewSet.empty())
    NodeSets.push_back(NewSet);

  // Create new nodes sets with the connected nodes any any remaining node that
  // has no predecessor.
  for (unsigned i = 0; i < SUnits.size(); ++i) {
    SUnit *SU = &SUnits[i];
    if (NodesAdded.count(SU) == 0) {
      NewSet.clear();
      addConnectedNodes(SU, NewSet, NodesAdded);
      if (!NewSet.empty())
        NodeSets.push_back(NewSet);
    }
  }
}

/// Add the node to the set, and add all is its connected nodes to the set.
void SwingSchedulerDAG::addConnectedNodes(SUnit *SU, NodeSet &NewSet,
                                          SetVector<SUnit *> &NodesAdded) {
  NewSet.insert(SU);
  NodesAdded.insert(SU);
  for (auto &SI : SU->Succs) {
    SUnit *Successor = SI.getSUnit();
    if (!SI.isArtificial() && NodesAdded.count(Successor) == 0)
      addConnectedNodes(Successor, NewSet, NodesAdded);
  }
  for (auto &PI : SU->Preds) {
    SUnit *Predecessor = PI.getSUnit();
    if (!PI.isArtificial() && NodesAdded.count(Predecessor) == 0)
      addConnectedNodes(Predecessor, NewSet, NodesAdded);
  }
}

/// Return true if Set1 contains elements in Set2. The elements in common
/// are returned in a different container.
static bool isIntersect(SmallSetVector<SUnit *, 8> &Set1, const NodeSet &Set2,
                        SmallSetVector<SUnit *, 8> &Result) {
  Result.clear();
  for (unsigned i = 0, e = Set1.size(); i != e; ++i) {
    SUnit *SU = Set1[i];
    if (Set2.count(SU) != 0)
      Result.insert(SU);
  }
  return !Result.empty();
}

/// Merge the recurrence node sets that have the same initial node.
void SwingSchedulerDAG::fuseRecs(NodeSetType &NodeSets) {
  for (NodeSetType::iterator I = NodeSets.begin(), E = NodeSets.end(); I != E;
       ++I) {
    NodeSet &NI = *I;
    for (NodeSetType::iterator J = I + 1; J != E;) {
      NodeSet &NJ = *J;
      if (NI.getNode(0)->NodeNum == NJ.getNode(0)->NodeNum) {
        if (NJ.compareRecMII(NI) > 0)
          NI.setRecMII(NJ.getRecMII());
        for (NodeSet::iterator NII = J->begin(), ENI = J->end(); NII != ENI;
             ++NII)
          I->insert(*NII);
        NodeSets.erase(J);
        E = NodeSets.end();
      } else {
        ++J;
      }
    }
  }
}

/// Remove nodes that have been scheduled in previous NodeSets.
void SwingSchedulerDAG::removeDuplicateNodes(NodeSetType &NodeSets) {
  for (NodeSetType::iterator I = NodeSets.begin(), E = NodeSets.end(); I != E;
       ++I)
    for (NodeSetType::iterator J = I + 1; J != E;) {
      J->remove_if([&](SUnit *SUJ) { return I->count(SUJ); });

      if (J->empty()) {
        NodeSets.erase(J);
        E = NodeSets.end();
      } else {
        ++J;
      }
    }
}

/// Return true if Inst1 defines a value that is used in Inst2.
static bool hasDataDependence(SUnit *Inst1, SUnit *Inst2) {
  for (auto &SI : Inst1->Succs)
    if (SI.getSUnit() == Inst2 && SI.getKind() == SDep::Data)
      return true;
  return false;
}

/// Compute an ordered list of the dependence graph nodes, which
/// indicates the order that the nodes will be scheduled.  This is a
/// two-level algorithm. First, a partial order is created, which
/// consists of a list of sets ordered from highest to lowest priority.
void SwingSchedulerDAG::computeNodeOrder(NodeSetType &NodeSets) {
  SmallSetVector<SUnit *, 8> R;
  NodeOrder.clear();

  for (auto &Nodes : NodeSets) {
    DEBUG(dbgs() << "NodeSet size " << Nodes.size() << "\n");
    OrderKind Order;
    SmallSetVector<SUnit *, 8> N;
    if (pred_L(NodeOrder, N) && isSubset(N, Nodes)) {
      R.insert(N.begin(), N.end());
      Order = BottomUp;
      DEBUG(dbgs() << "  Bottom up (preds) ");
    } else if (succ_L(NodeOrder, N) && isSubset(N, Nodes)) {
      R.insert(N.begin(), N.end());
      Order = TopDown;
      DEBUG(dbgs() << "  Top down (succs) ");
    } else if (isIntersect(N, Nodes, R)) {
      // If some of the successors are in the existing node-set, then use the
      // top-down ordering.
      Order = TopDown;
      DEBUG(dbgs() << "  Top down (intersect) ");
    } else if (NodeSets.size() == 1) {
      for (auto &N : Nodes)
        if (N->Succs.size() == 0)
          R.insert(N);
      Order = BottomUp;
      DEBUG(dbgs() << "  Bottom up (all) ");
    } else {
      // Find the node with the highest ASAP.
      SUnit *maxASAP = nullptr;
      for (SUnit *SU : Nodes) {
        if (maxASAP == nullptr || getASAP(SU) >= getASAP(maxASAP))
          maxASAP = SU;
      }
      R.insert(maxASAP);
      Order = BottomUp;
      DEBUG(dbgs() << "  Bottom up (default) ");
    }

    while (!R.empty()) {
      if (Order == TopDown) {
        // Choose the node with the maximum height.  If more than one, choose
        // the node with the lowest MOV. If still more than one, check if there
        // is a dependence between the instructions.
        while (!R.empty()) {
          SUnit *maxHeight = nullptr;
          for (SUnit *I : R) {
            if (maxHeight == nullptr || getHeight(I) > getHeight(maxHeight))
              maxHeight = I;
            else if (getHeight(I) == getHeight(maxHeight) &&
                     getMOV(I) < getMOV(maxHeight) &&
                     !hasDataDependence(maxHeight, I))
              maxHeight = I;
            else if (hasDataDependence(I, maxHeight))
              maxHeight = I;
          }
          NodeOrder.insert(maxHeight);
          DEBUG(dbgs() << maxHeight->NodeNum << " ");
          R.remove(maxHeight);
          for (const auto &I : maxHeight->Succs) {
            if (Nodes.count(I.getSUnit()) == 0)
              continue;
            if (NodeOrder.count(I.getSUnit()) != 0)
              continue;
            if (ignoreDependence(I, false))
              continue;
            R.insert(I.getSUnit());
          }
          // Back-edges are predecessors with an anti-dependence.
          for (const auto &I : maxHeight->Preds) {
            if (I.getKind() != SDep::Anti)
              continue;
            if (Nodes.count(I.getSUnit()) == 0)
              continue;
            if (NodeOrder.count(I.getSUnit()) != 0)
              continue;
            R.insert(I.getSUnit());
          }
        }
        Order = BottomUp;
        DEBUG(dbgs() << "\n   Switching order to bottom up ");
        SmallSetVector<SUnit *, 8> N;
        if (pred_L(NodeOrder, N, &Nodes))
          R.insert(N.begin(), N.end());
      } else {
        // Choose the node with the maximum depth.  If more than one, choose
        // the node with the lowest MOV. If there is still more than one, check
        // for a dependence between the instructions.
        while (!R.empty()) {
          SUnit *maxDepth = nullptr;
          for (SUnit *I : R) {
            if (maxDepth == nullptr || getDepth(I) > getDepth(maxDepth))
              maxDepth = I;
            else if (getDepth(I) == getDepth(maxDepth) &&
                     getMOV(I) < getMOV(maxDepth) &&
                     !hasDataDependence(I, maxDepth))
              maxDepth = I;
            else if (hasDataDependence(maxDepth, I))
              maxDepth = I;
          }
          NodeOrder.insert(maxDepth);
          DEBUG(dbgs() << maxDepth->NodeNum << " ");
          R.remove(maxDepth);
          if (Nodes.isExceedSU(maxDepth)) {
            Order = TopDown;
            R.clear();
            R.insert(Nodes.getNode(0));
            break;
          }
          for (const auto &I : maxDepth->Preds) {
            if (Nodes.count(I.getSUnit()) == 0)
              continue;
            if (NodeOrder.count(I.getSUnit()) != 0)
              continue;
            if (I.getKind() == SDep::Anti)
              continue;
            R.insert(I.getSUnit());
          }
          // Back-edges are predecessors with an anti-dependence.
          for (const auto &I : maxDepth->Succs) {
            if (I.getKind() != SDep::Anti)
              continue;
            if (Nodes.count(I.getSUnit()) == 0)
              continue;
            if (NodeOrder.count(I.getSUnit()) != 0)
              continue;
            R.insert(I.getSUnit());
          }
        }
        Order = TopDown;
        DEBUG(dbgs() << "\n   Switching order to top down ");
        SmallSetVector<SUnit *, 8> N;
        if (succ_L(NodeOrder, N, &Nodes))
          R.insert(N.begin(), N.end());
      }
    }
    DEBUG(dbgs() << "\nDone with Nodeset\n");
  }

  DEBUG({
    dbgs() << "Node order: ";
    for (SUnit *I : NodeOrder)
      dbgs() << " " << I->NodeNum << " ";
    dbgs() << "\n";
  });
}

/// Process the nodes in the computed order and create the pipelined schedule
/// of the instructions, if possible. Return true if a schedule is found.
bool SwingSchedulerDAG::schedulePipeline(SMSchedule &Schedule) {
  if (NodeOrder.empty())
    return false;

  bool scheduleFound = false;
  // Keep increasing II until a valid schedule is found.
  for (unsigned II = MII; II < MII + 10 && !scheduleFound; ++II) {
    Schedule.reset();
    Schedule.setInitiationInterval(II);
    DEBUG(dbgs() << "Try to schedule with " << II << "\n");

    SetVector<SUnit *>::iterator NI = NodeOrder.begin();
    SetVector<SUnit *>::iterator NE = NodeOrder.end();
    do {
      SUnit *SU = *NI;

      // Compute the schedule time for the instruction, which is based
      // upon the scheduled time for any predecessors/successors.
      int EarlyStart = INT_MIN;
      int LateStart = INT_MAX;
      // These values are set when the size of the schedule window is limited
      // due to chain dependences.
      int SchedEnd = INT_MAX;
      int SchedStart = INT_MIN;
      Schedule.computeStart(SU, &EarlyStart, &LateStart, &SchedEnd, &SchedStart,
                            II, this);
      DEBUG({
        dbgs() << "Inst (" << SU->NodeNum << ") ";
        SU->getInstr()->dump();
        dbgs() << "\n";
      });
      DEBUG({
        dbgs() << "\tes: " << EarlyStart << " ls: " << LateStart
               << " me: " << SchedEnd << " ms: " << SchedStart << "\n";
      });

      if (EarlyStart > LateStart || SchedEnd < EarlyStart ||
          SchedStart > LateStart)
        scheduleFound = false;
      else if (EarlyStart != INT_MIN && LateStart == INT_MAX) {
        SchedEnd = std::min(SchedEnd, EarlyStart + (int)II - 1);
        scheduleFound = Schedule.insert(SU, EarlyStart, SchedEnd, II);
      } else if (EarlyStart == INT_MIN && LateStart != INT_MAX) {
        SchedStart = std::max(SchedStart, LateStart - (int)II + 1);
        scheduleFound = Schedule.insert(SU, LateStart, SchedStart, II);
      } else if (EarlyStart != INT_MIN && LateStart != INT_MAX) {
        SchedEnd =
            std::min(SchedEnd, std::min(LateStart, EarlyStart + (int)II - 1));
        // When scheduling a Phi it is better to start at the late cycle and go
        // backwards. The default order may insert the Phi too far away from
        // its first dependence.
        if (SU->getInstr()->isPHI())
          scheduleFound = Schedule.insert(SU, SchedEnd, EarlyStart, II);
        else
          scheduleFound = Schedule.insert(SU, EarlyStart, SchedEnd, II);
      } else {
        int FirstCycle = Schedule.getFirstCycle();
        scheduleFound = Schedule.insert(SU, FirstCycle + getASAP(SU),
                                        FirstCycle + getASAP(SU) + II - 1, II);
      }
      // Even if we find a schedule, make sure the schedule doesn't exceed the
      // allowable number of stages. We keep trying if this happens.
      if (scheduleFound)
        if (SwpMaxStages > -1 &&
            Schedule.getMaxStageCount() > (unsigned)SwpMaxStages)
          scheduleFound = false;

      DEBUG({
        if (!scheduleFound)
          dbgs() << "\tCan't schedule\n";
      });
    } while (++NI != NE && scheduleFound);

    // If a schedule is found, check if it is a valid schedule too.
    if (scheduleFound)
      scheduleFound = Schedule.isValidSchedule(this);
  }

  DEBUG(dbgs() << "Schedule Found? " << scheduleFound << "\n");

  if (scheduleFound)
    Schedule.finalizeSchedule(this);
  else
    Schedule.reset();

  return scheduleFound && Schedule.getMaxStageCount() > 0;
}

/// Given a schedule for the loop, generate a new version of the loop,
/// and replace the old version.  This function generates a prolog
/// that contains the initial iterations in the pipeline, and kernel
/// loop, and the epilogue that contains the code for the final
/// iterations.
void SwingSchedulerDAG::generatePipelinedLoop(SMSchedule &Schedule) {
  // Create a new basic block for the kernel and add it to the CFG.
  MachineBasicBlock *KernelBB = MF.CreateMachineBasicBlock(BB->getBasicBlock());

  unsigned MaxStageCount = Schedule.getMaxStageCount();

  // Remember the registers that are used in different stages. The index is
  // the iteration, or stage, that the instruction is scheduled in.  This is
  // a map between register names in the orignal block and the names created
  // in each stage of the pipelined loop.
  ValueMapTy *VRMap = new ValueMapTy[(MaxStageCount + 1) * 2];
  InstrMapTy InstrMap;

  SmallVector<MachineBasicBlock *, 4> PrologBBs;
  // Generate the prolog instructions that set up the pipeline.
  generateProlog(Schedule, MaxStageCount, KernelBB, VRMap, PrologBBs);
  MF.insert(BB->getIterator(), KernelBB);

  // Rearrange the instructions to generate the new, pipelined loop,
  // and update register names as needed.
  for (int Cycle = Schedule.getFirstCycle(),
           LastCycle = Schedule.getFinalCycle();
       Cycle <= LastCycle; ++Cycle) {
    std::deque<SUnit *> &CycleInstrs = Schedule.getInstructions(Cycle);
    // This inner loop schedules each instruction in the cycle.
    for (SUnit *CI : CycleInstrs) {
      if (CI->getInstr()->isPHI())
        continue;
      unsigned StageNum = Schedule.stageScheduled(getSUnit(CI->getInstr()));
      MachineInstr *NewMI = cloneInstr(CI->getInstr(), MaxStageCount, StageNum);
      updateInstruction(NewMI, false, MaxStageCount, StageNum, Schedule, VRMap);
      KernelBB->push_back(NewMI);
      InstrMap[NewMI] = CI->getInstr();
    }
  }

  // Copy any terminator instructions to the new kernel, and update
  // names as needed.
  for (MachineBasicBlock::iterator I = BB->getFirstTerminator(),
                                   E = BB->instr_end();
       I != E; ++I) {
    MachineInstr *NewMI = MF.CloneMachineInstr(&*I);
    updateInstruction(NewMI, false, MaxStageCount, 0, Schedule, VRMap);
    KernelBB->push_back(NewMI);
    InstrMap[NewMI] = &*I;
  }

  KernelBB->transferSuccessors(BB);
  KernelBB->replaceSuccessor(BB, KernelBB);

  generateExistingPhis(KernelBB, PrologBBs.back(), KernelBB, KernelBB, Schedule,
                       VRMap, InstrMap, MaxStageCount, MaxStageCount, false);
  generatePhis(KernelBB, PrologBBs.back(), KernelBB, KernelBB, Schedule, VRMap,
               InstrMap, MaxStageCount, MaxStageCount, false);

  DEBUG(dbgs() << "New block\n"; KernelBB->dump(););

  SmallVector<MachineBasicBlock *, 4> EpilogBBs;
  // Generate the epilog instructions to complete the pipeline.
  generateEpilog(Schedule, MaxStageCount, KernelBB, VRMap, EpilogBBs,
                 PrologBBs);

  // We need this step because the register allocation doesn't handle some
  // situations well, so we insert copies to help out.
  splitLifetimes(KernelBB, EpilogBBs, Schedule);

  // Remove dead instructions due to loop induction variables.
  removeDeadInstructions(KernelBB, EpilogBBs);

  // Add branches between prolog and epilog blocks.
  addBranches(PrologBBs, KernelBB, EpilogBBs, Schedule, VRMap);

  // Remove the original loop since it's no longer referenced.
  BB->clear();
  BB->eraseFromParent();

  delete[] VRMap;
}

/// Generate the pipeline prolog code.
void SwingSchedulerDAG::generateProlog(SMSchedule &Schedule, unsigned LastStage,
                                       MachineBasicBlock *KernelBB,
                                       ValueMapTy *VRMap,
                                       MBBVectorTy &PrologBBs) {
  MachineBasicBlock *PreheaderBB = MLI->getLoopFor(BB)->getLoopPreheader();
  assert(PreheaderBB != nullptr &&
         "Need to add code to handle loops w/o preheader");
  MachineBasicBlock *PredBB = PreheaderBB;
  InstrMapTy InstrMap;

  // Generate a basic block for each stage, not including the last stage,
  // which will be generated in the kernel. Each basic block may contain
  // instructions from multiple stages/iterations.
  for (unsigned i = 0; i < LastStage; ++i) {
    // Create and insert the prolog basic block prior to the original loop
    // basic block.  The original loop is removed later.
    MachineBasicBlock *NewBB = MF.CreateMachineBasicBlock(BB->getBasicBlock());
    PrologBBs.push_back(NewBB);
    MF.insert(BB->getIterator(), NewBB);
    NewBB->transferSuccessors(PredBB);
    PredBB->addSuccessor(NewBB);
    PredBB = NewBB;

    // Generate instructions for each appropriate stage. Process instructions
    // in original program order.
    for (int StageNum = i; StageNum >= 0; --StageNum) {
      for (MachineBasicBlock::iterator BBI = BB->instr_begin(),
                                       BBE = BB->getFirstTerminator();
           BBI != BBE; ++BBI) {
        if (Schedule.isScheduledAtStage(getSUnit(&*BBI), (unsigned)StageNum)) {
          if (BBI->isPHI())
            continue;
          MachineInstr *NewMI =
              cloneAndChangeInstr(&*BBI, i, (unsigned)StageNum, Schedule);
          updateInstruction(NewMI, false, i, (unsigned)StageNum, Schedule,
                            VRMap);
          NewBB->push_back(NewMI);
          InstrMap[NewMI] = &*BBI;
        }
      }
    }
    rewritePhiValues(NewBB, i, Schedule, VRMap, InstrMap);
    DEBUG({
      dbgs() << "prolog:\n";
      NewBB->dump();
    });
  }

  PredBB->replaceSuccessor(BB, KernelBB);

  // Check if we need to remove the branch from the preheader to the original
  // loop, and replace it with a branch to the new loop.
  unsigned numBranches = TII->removeBranch(*PreheaderBB);
  if (numBranches) {
    SmallVector<MachineOperand, 0> Cond;
    TII->insertBranch(*PreheaderBB, PrologBBs[0], nullptr, Cond, DebugLoc());
  }
}

/// Generate the pipeline epilog code. The epilog code finishes the iterations
/// that were started in either the prolog or the kernel.  We create a basic
/// block for each stage that needs to complete.
void SwingSchedulerDAG::generateEpilog(SMSchedule &Schedule, unsigned LastStage,
                                       MachineBasicBlock *KernelBB,
                                       ValueMapTy *VRMap,
                                       MBBVectorTy &EpilogBBs,
                                       MBBVectorTy &PrologBBs) {
  // We need to change the branch from the kernel to the first epilog block, so
  // this call to analyze branch uses the kernel rather than the original BB.
  MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
  SmallVector<MachineOperand, 4> Cond;
  bool checkBranch = TII->analyzeBranch(*KernelBB, TBB, FBB, Cond);
  assert(!checkBranch && "generateEpilog must be able to analyze the branch");
  if (checkBranch)
    return;

  MachineBasicBlock::succ_iterator LoopExitI = KernelBB->succ_begin();
  if (*LoopExitI == KernelBB)
    ++LoopExitI;
  assert(LoopExitI != KernelBB->succ_end() && "Expecting a successor");
  MachineBasicBlock *LoopExitBB = *LoopExitI;

  MachineBasicBlock *PredBB = KernelBB;
  MachineBasicBlock *EpilogStart = LoopExitBB;
  InstrMapTy InstrMap;

  // Generate a basic block for each stage, not including the last stage,
  // which was generated for the kernel.  Each basic block may contain
  // instructions from multiple stages/iterations.
  int EpilogStage = LastStage + 1;
  for (unsigned i = LastStage; i >= 1; --i, ++EpilogStage) {
    MachineBasicBlock *NewBB = MF.CreateMachineBasicBlock();
    EpilogBBs.push_back(NewBB);
    MF.insert(BB->getIterator(), NewBB);

    PredBB->replaceSuccessor(LoopExitBB, NewBB);
    NewBB->addSuccessor(LoopExitBB);

    if (EpilogStart == LoopExitBB)
      EpilogStart = NewBB;

    // Add instructions to the epilog depending on the current block.
    // Process instructions in original program order.
    for (unsigned StageNum = i; StageNum <= LastStage; ++StageNum) {
      for (auto &BBI : *BB) {
        if (BBI.isPHI())
          continue;
        MachineInstr *In = &BBI;
        if (Schedule.isScheduledAtStage(getSUnit(In), StageNum)) {
          MachineInstr *NewMI = cloneInstr(In, EpilogStage - LastStage, 0);
          updateInstruction(NewMI, i == 1, EpilogStage, 0, Schedule, VRMap);
          NewBB->push_back(NewMI);
          InstrMap[NewMI] = In;
        }
      }
    }
    generateExistingPhis(NewBB, PrologBBs[i - 1], PredBB, KernelBB, Schedule,
                         VRMap, InstrMap, LastStage, EpilogStage, i == 1);
    generatePhis(NewBB, PrologBBs[i - 1], PredBB, KernelBB, Schedule, VRMap,
                 InstrMap, LastStage, EpilogStage, i == 1);
    PredBB = NewBB;

    DEBUG({
      dbgs() << "epilog:\n";
      NewBB->dump();
    });
  }

  // Fix any Phi nodes in the loop exit block.
  for (MachineInstr &MI : *LoopExitBB) {
    if (!MI.isPHI())
      break;
    for (unsigned i = 2, e = MI.getNumOperands() + 1; i != e; i += 2) {
      MachineOperand &MO = MI.getOperand(i);
      if (MO.getMBB() == BB)
        MO.setMBB(PredBB);
    }
  }

  // Create a branch to the new epilog from the kernel.
  // Remove the original branch and add a new branch to the epilog.
  TII->removeBranch(*KernelBB);
  TII->insertBranch(*KernelBB, KernelBB, EpilogStart, Cond, DebugLoc());
  // Add a branch to the loop exit.
  if (EpilogBBs.size() > 0) {
    MachineBasicBlock *LastEpilogBB = EpilogBBs.back();
    SmallVector<MachineOperand, 4> Cond1;
    TII->insertBranch(*LastEpilogBB, LoopExitBB, nullptr, Cond1, DebugLoc());
  }
}

/// Replace all uses of FromReg that appear outside the specified
/// basic block with ToReg.
static void replaceRegUsesAfterLoop(unsigned FromReg, unsigned ToReg,
                                    MachineBasicBlock *MBB,
                                    MachineRegisterInfo &MRI,
                                    LiveIntervals &LIS) {
  for (MachineRegisterInfo::use_iterator I = MRI.use_begin(FromReg),
                                         E = MRI.use_end();
       I != E;) {
    MachineOperand &O = *I;
    ++I;
    if (O.getParent()->getParent() != MBB)
      O.setReg(ToReg);
  }
  if (!LIS.hasInterval(ToReg))
    LIS.createEmptyInterval(ToReg);
}

/// Return true if the register has a use that occurs outside the
/// specified loop.
static bool hasUseAfterLoop(unsigned Reg, MachineBasicBlock *BB,
                            MachineRegisterInfo &MRI) {
  for (MachineRegisterInfo::use_iterator I = MRI.use_begin(Reg),
                                         E = MRI.use_end();
       I != E; ++I)
    if (I->getParent()->getParent() != BB)
      return true;
  return false;
}

/// Generate Phis for the specific block in the generated pipelined code.
/// This function looks at the Phis from the original code to guide the
/// creation of new Phis.
void SwingSchedulerDAG::generateExistingPhis(
    MachineBasicBlock *NewBB, MachineBasicBlock *BB1, MachineBasicBlock *BB2,
    MachineBasicBlock *KernelBB, SMSchedule &Schedule, ValueMapTy *VRMap,
    InstrMapTy &InstrMap, unsigned LastStageNum, unsigned CurStageNum,
    bool IsLast) {
  // Compute the stage number for the initial value of the Phi, which
  // comes from the prolog. The prolog to use depends on to which kernel/
  // epilog that we're adding the Phi.
  unsigned PrologStage = 0;
  unsigned PrevStage = 0;
  bool InKernel = (LastStageNum == CurStageNum);
  if (InKernel) {
    PrologStage = LastStageNum - 1;
    PrevStage = CurStageNum;
  } else {
    PrologStage = LastStageNum - (CurStageNum - LastStageNum);
    PrevStage = LastStageNum + (CurStageNum - LastStageNum) - 1;
  }

  for (MachineBasicBlock::iterator BBI = BB->instr_begin(),
                                   BBE = BB->getFirstNonPHI();
       BBI != BBE; ++BBI) {
    unsigned Def = BBI->getOperand(0).getReg();

    unsigned InitVal = 0;
    unsigned LoopVal = 0;
    getPhiRegs(*BBI, BB, InitVal, LoopVal);

    unsigned PhiOp1 = 0;
    // The Phi value from the loop body typically is defined in the loop, but
    // not always. So, we need to check if the value is defined in the loop.
    unsigned PhiOp2 = LoopVal;
    if (VRMap[LastStageNum].count(LoopVal))
      PhiOp2 = VRMap[LastStageNum][LoopVal];

    int StageScheduled = Schedule.stageScheduled(getSUnit(&*BBI));
    int LoopValStage =
        Schedule.stageScheduled(getSUnit(MRI.getVRegDef(LoopVal)));
    unsigned NumStages = Schedule.getStagesForReg(Def, CurStageNum);
    if (NumStages == 0) {
      // We don't need to generate a Phi anymore, but we need to rename any uses
      // of the Phi value.
      unsigned NewReg = VRMap[PrevStage][LoopVal];
      rewriteScheduledInstr(NewBB, Schedule, InstrMap, CurStageNum, 0, &*BBI,
                            Def, NewReg);
      if (VRMap[CurStageNum].count(LoopVal))
        VRMap[CurStageNum][Def] = VRMap[CurStageNum][LoopVal];
    }
    // Adjust the number of Phis needed depending on the number of prologs left,
    // and the distance from where the Phi is first scheduled.
    unsigned NumPhis = NumStages;
    if (!InKernel && (int)PrologStage < LoopValStage)
      // The NumPhis is the maximum number of new Phis needed during the steady
      // state. If the Phi has not been scheduled in current prolog, then we
      // need to generate less Phis.
      NumPhis = std::max((int)NumPhis - (int)(LoopValStage - PrologStage), 1);
    // The number of Phis cannot exceed the number of prolog stages. Each
    // stage can potentially define two values.
    NumPhis = std::min(NumPhis, PrologStage + 2);

    unsigned NewReg = 0;

    unsigned AccessStage = (LoopValStage != -1) ? LoopValStage : StageScheduled;
    // In the epilog, we may need to look back one stage to get the correct
    // Phi name because the epilog and prolog blocks execute the same stage.
    // The correct name is from the previous block only when the Phi has
    // been completely scheduled prior to the epilog, and Phi value is not
    // needed in multiple stages.
    int StageDiff = 0;
    if (!InKernel && StageScheduled >= LoopValStage && AccessStage == 0 &&
        NumPhis == 1)
      StageDiff = 1;
    // Adjust the computations below when the phi and the loop definition
    // are scheduled in different stages.
    if (InKernel && LoopValStage != -1 && StageScheduled > LoopValStage)
      StageDiff = StageScheduled - LoopValStage;
    for (unsigned np = 0; np < NumPhis; ++np) {
      // If the Phi hasn't been scheduled, then use the initial Phi operand
      // value. Otherwise, use the scheduled version of the instruction. This
      // is a little complicated when a Phi references another Phi.
      if (np > PrologStage || StageScheduled >= (int)LastStageNum)
        PhiOp1 = InitVal;
      // Check if the Phi has already been scheduled in a prolog stage.
      else if (PrologStage >= AccessStage + StageDiff + np &&
               VRMap[PrologStage - StageDiff - np].count(LoopVal) != 0)
        PhiOp1 = VRMap[PrologStage - StageDiff - np][LoopVal];
      // Check if the Phi has already been scheduled, but the loop intruction
      // is either another Phi, or doesn't occur in the loop.
      else if (PrologStage >= AccessStage + StageDiff + np) {
        // If the Phi references another Phi, we need to examine the other
        // Phi to get the correct value.
        PhiOp1 = LoopVal;
        MachineInstr *InstOp1 = MRI.getVRegDef(PhiOp1);
        int Indirects = 1;
        while (InstOp1 && InstOp1->isPHI() && InstOp1->getParent() == BB) {
          int PhiStage = Schedule.stageScheduled(getSUnit(InstOp1));
          if ((int)(PrologStage - StageDiff - np) < PhiStage + Indirects)
            PhiOp1 = getInitPhiReg(*InstOp1, BB);
          else
            PhiOp1 = getLoopPhiReg(*InstOp1, BB);
          InstOp1 = MRI.getVRegDef(PhiOp1);
          int PhiOpStage = Schedule.stageScheduled(getSUnit(InstOp1));
          int StageAdj = (PhiOpStage != -1 ? PhiStage - PhiOpStage : 0);
          if (PhiOpStage != -1 && PrologStage - StageAdj >= Indirects + np &&
              VRMap[PrologStage - StageAdj - Indirects - np].count(PhiOp1)) {
            PhiOp1 = VRMap[PrologStage - StageAdj - Indirects - np][PhiOp1];
            break;
          }
          ++Indirects;
        }
      } else
        PhiOp1 = InitVal;
      // If this references a generated Phi in the kernel, get the Phi operand
      // from the incoming block.
      if (MachineInstr *InstOp1 = MRI.getVRegDef(PhiOp1))
        if (InstOp1->isPHI() && InstOp1->getParent() == KernelBB)
          PhiOp1 = getInitPhiReg(*InstOp1, KernelBB);

      MachineInstr *PhiInst = MRI.getVRegDef(LoopVal);
      bool LoopDefIsPhi = PhiInst && PhiInst->isPHI();
      // In the epilog, a map lookup is needed to get the value from the kernel,
      // or previous epilog block. How is does this depends on if the
      // instruction is scheduled in the previous block.
      if (!InKernel) {
        int StageDiffAdj = 0;
        if (LoopValStage != -1 && StageScheduled > LoopValStage)
          StageDiffAdj = StageScheduled - LoopValStage;
        // Use the loop value defined in the kernel, unless the kernel
        // contains the last definition of the Phi.
        if (np == 0 && PrevStage == LastStageNum &&
            (StageScheduled != 0 || LoopValStage != 0) &&
            VRMap[PrevStage - StageDiffAdj].count(LoopVal))
          PhiOp2 = VRMap[PrevStage - StageDiffAdj][LoopVal];
        // Use the value defined by the Phi. We add one because we switch
        // from looking at the loop value to the Phi definition.
        else if (np > 0 && PrevStage == LastStageNum &&
                 VRMap[PrevStage - np + 1].count(Def))
          PhiOp2 = VRMap[PrevStage - np + 1][Def];
        // Use the loop value defined in the kernel.
        else if ((unsigned)LoopValStage + StageDiffAdj > PrologStage + 1 &&
                 VRMap[PrevStage - StageDiffAdj - np].count(LoopVal))
          PhiOp2 = VRMap[PrevStage - StageDiffAdj - np][LoopVal];
        // Use the value defined by the Phi, unless we're generating the first
        // epilog and the Phi refers to a Phi in a different stage.
        else if (VRMap[PrevStage - np].count(Def) &&
                 (!LoopDefIsPhi || PrevStage != LastStageNum))
          PhiOp2 = VRMap[PrevStage - np][Def];
      }

      // Check if we can reuse an existing Phi. This occurs when a Phi
      // references another Phi, and the other Phi is scheduled in an
      // earlier stage. We can try to reuse an existing Phi up until the last
      // stage of the current Phi.
      if (LoopDefIsPhi && (int)PrologStage >= StageScheduled) {
        int LVNumStages = Schedule.getStagesForPhi(LoopVal);
        int StageDiff = (StageScheduled - LoopValStage);
        LVNumStages -= StageDiff;
        if (LVNumStages > (int)np) {
          NewReg = PhiOp2;
          unsigned ReuseStage = CurStageNum;
          if (Schedule.isLoopCarried(this, *PhiInst))
            ReuseStage -= LVNumStages;
          // Check if the Phi to reuse has been generated yet. If not, then
          // there is nothing to reuse.
          if (VRMap[ReuseStage].count(LoopVal)) {
            NewReg = VRMap[ReuseStage][LoopVal];

            rewriteScheduledInstr(NewBB, Schedule, InstrMap, CurStageNum, np,
                                  &*BBI, Def, NewReg);
            // Update the map with the new Phi name.
            VRMap[CurStageNum - np][Def] = NewReg;
            PhiOp2 = NewReg;
            if (VRMap[LastStageNum - np - 1].count(LoopVal))
              PhiOp2 = VRMap[LastStageNum - np - 1][LoopVal];

            if (IsLast && np == NumPhis - 1)
              replaceRegUsesAfterLoop(Def, NewReg, BB, MRI, LIS);
            continue;
          }
        } else if (InKernel && StageDiff > 0 &&
                   VRMap[CurStageNum - StageDiff - np].count(LoopVal))
          PhiOp2 = VRMap[CurStageNum - StageDiff - np][LoopVal];
      }

      const TargetRegisterClass *RC = MRI.getRegClass(Def);
      NewReg = MRI.createVirtualRegister(RC);

      MachineInstrBuilder NewPhi =
          BuildMI(*NewBB, NewBB->getFirstNonPHI(), DebugLoc(),
                  TII->get(TargetOpcode::PHI), NewReg);
      NewPhi.addReg(PhiOp1).addMBB(BB1);
      NewPhi.addReg(PhiOp2).addMBB(BB2);
      if (np == 0)
        InstrMap[NewPhi] = &*BBI;

      // We define the Phis after creating the new pipelined code, so
      // we need to rename the Phi values in scheduled instructions.

      unsigned PrevReg = 0;
      if (InKernel && VRMap[PrevStage - np].count(LoopVal))
        PrevReg = VRMap[PrevStage - np][LoopVal];
      rewriteScheduledInstr(NewBB, Schedule, InstrMap, CurStageNum, np, &*BBI,
                            Def, NewReg, PrevReg);
      // If the Phi has been scheduled, use the new name for rewriting.
      if (VRMap[CurStageNum - np].count(Def)) {
        unsigned R = VRMap[CurStageNum - np][Def];
        rewriteScheduledInstr(NewBB, Schedule, InstrMap, CurStageNum, np, &*BBI,
                              R, NewReg);
      }

      // Check if we need to rename any uses that occurs after the loop. The
      // register to replace depends on whether the Phi is scheduled in the
      // epilog.
      if (IsLast && np == NumPhis - 1)
        replaceRegUsesAfterLoop(Def, NewReg, BB, MRI, LIS);

      // In the kernel, a dependent Phi uses the value from this Phi.
      if (InKernel)
        PhiOp2 = NewReg;

      // Update the map with the new Phi name.
      VRMap[CurStageNum - np][Def] = NewReg;
    }

    while (NumPhis++ < NumStages) {
      rewriteScheduledInstr(NewBB, Schedule, InstrMap, CurStageNum, NumPhis,
                            &*BBI, Def, NewReg, 0);
    }

    // Check if we need to rename a Phi that has been eliminated due to
    // scheduling.
    if (NumStages == 0 && IsLast && VRMap[CurStageNum].count(LoopVal))
      replaceRegUsesAfterLoop(Def, VRMap[CurStageNum][LoopVal], BB, MRI, LIS);
  }
}

/// Generate Phis for the specified block in the generated pipelined code.
/// These are new Phis needed because the definition is scheduled after the
/// use in the pipelened sequence.
void SwingSchedulerDAG::generatePhis(
    MachineBasicBlock *NewBB, MachineBasicBlock *BB1, MachineBasicBlock *BB2,
    MachineBasicBlock *KernelBB, SMSchedule &Schedule, ValueMapTy *VRMap,
    InstrMapTy &InstrMap, unsigned LastStageNum, unsigned CurStageNum,
    bool IsLast) {
  // Compute the stage number that contains the initial Phi value, and
  // the Phi from the previous stage.
  unsigned PrologStage = 0;
  unsigned PrevStage = 0;
  unsigned StageDiff = CurStageNum - LastStageNum;
  bool InKernel = (StageDiff == 0);
  if (InKernel) {
    PrologStage = LastStageNum - 1;
    PrevStage = CurStageNum;
  } else {
    PrologStage = LastStageNum - StageDiff;
    PrevStage = LastStageNum + StageDiff - 1;
  }

  for (MachineBasicBlock::iterator BBI = BB->getFirstNonPHI(),
                                   BBE = BB->instr_end();
       BBI != BBE; ++BBI) {
    for (unsigned i = 0, e = BBI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = BBI->getOperand(i);
      if (!MO.isReg() || !MO.isDef() ||
          !TargetRegisterInfo::isVirtualRegister(MO.getReg()))
        continue;

      int StageScheduled = Schedule.stageScheduled(getSUnit(&*BBI));
      assert(StageScheduled != -1 && "Expecting scheduled instruction.");
      unsigned Def = MO.getReg();
      unsigned NumPhis = Schedule.getStagesForReg(Def, CurStageNum);
      // An instruction scheduled in stage 0 and is used after the loop
      // requires a phi in the epilog for the last definition from either
      // the kernel or prolog.
      if (!InKernel && NumPhis == 0 && StageScheduled == 0 &&
          hasUseAfterLoop(Def, BB, MRI))
        NumPhis = 1;
      if (!InKernel && (unsigned)StageScheduled > PrologStage)
        continue;

      unsigned PhiOp2 = VRMap[PrevStage][Def];
      if (MachineInstr *InstOp2 = MRI.getVRegDef(PhiOp2))
        if (InstOp2->isPHI() && InstOp2->getParent() == NewBB)
          PhiOp2 = getLoopPhiReg(*InstOp2, BB2);
      // The number of Phis can't exceed the number of prolog stages. The
      // prolog stage number is zero based.
      if (NumPhis > PrologStage + 1 - StageScheduled)
        NumPhis = PrologStage + 1 - StageScheduled;
      for (unsigned np = 0; np < NumPhis; ++np) {
        unsigned PhiOp1 = VRMap[PrologStage][Def];
        if (np <= PrologStage)
          PhiOp1 = VRMap[PrologStage - np][Def];
        if (MachineInstr *InstOp1 = MRI.getVRegDef(PhiOp1)) {
          if (InstOp1->isPHI() && InstOp1->getParent() == KernelBB)
            PhiOp1 = getInitPhiReg(*InstOp1, KernelBB);
          if (InstOp1->isPHI() && InstOp1->getParent() == NewBB)
            PhiOp1 = getInitPhiReg(*InstOp1, NewBB);
        }
        if (!InKernel)
          PhiOp2 = VRMap[PrevStage - np][Def];

        const TargetRegisterClass *RC = MRI.getRegClass(Def);
        unsigned NewReg = MRI.createVirtualRegister(RC);

        MachineInstrBuilder NewPhi =
            BuildMI(*NewBB, NewBB->getFirstNonPHI(), DebugLoc(),
                    TII->get(TargetOpcode::PHI), NewReg);
        NewPhi.addReg(PhiOp1).addMBB(BB1);
        NewPhi.addReg(PhiOp2).addMBB(BB2);
        if (np == 0)
          InstrMap[NewPhi] = &*BBI;

        // Rewrite uses and update the map. The actions depend upon whether
        // we generating code for the kernel or epilog blocks.
        if (InKernel) {
          rewriteScheduledInstr(NewBB, Schedule, InstrMap, CurStageNum, np,
                                &*BBI, PhiOp1, NewReg);
          rewriteScheduledInstr(NewBB, Schedule, InstrMap, CurStageNum, np,
                                &*BBI, PhiOp2, NewReg);

          PhiOp2 = NewReg;
          VRMap[PrevStage - np - 1][Def] = NewReg;
        } else {
          VRMap[CurStageNum - np][Def] = NewReg;
          if (np == NumPhis - 1)
            rewriteScheduledInstr(NewBB, Schedule, InstrMap, CurStageNum, np,
                                  &*BBI, Def, NewReg);
        }
        if (IsLast && np == NumPhis - 1)
          replaceRegUsesAfterLoop(Def, NewReg, BB, MRI, LIS);
      }
    }
  }
}

/// Remove instructions that generate values with no uses.
/// Typically, these are induction variable operations that generate values
/// used in the loop itself.  A dead instruction has a definition with
/// no uses, or uses that occur in the original loop only.
void SwingSchedulerDAG::removeDeadInstructions(MachineBasicBlock *KernelBB,
                                               MBBVectorTy &EpilogBBs) {
  // For each epilog block, check that the value defined by each instruction
  // is used.  If not, delete it.
  for (MBBVectorTy::reverse_iterator MBB = EpilogBBs.rbegin(),
                                     MBE = EpilogBBs.rend();
       MBB != MBE; ++MBB)
    for (MachineBasicBlock::reverse_instr_iterator MI = (*MBB)->instr_rbegin(),
                                                   ME = (*MBB)->instr_rend();
         MI != ME;) {
      // From DeadMachineInstructionElem. Don't delete inline assembly.
      if (MI->isInlineAsm()) {
        ++MI;
        continue;
      }
      bool SawStore = false;
      // Check if it's safe to remove the instruction due to side effects.
      // We can, and want to, remove Phis here.
      if (!MI->isSafeToMove(nullptr, SawStore) && !MI->isPHI()) {
        ++MI;
        continue;
      }
      bool used = true;
      for (MachineInstr::mop_iterator MOI = MI->operands_begin(),
                                      MOE = MI->operands_end();
           MOI != MOE; ++MOI) {
        if (!MOI->isReg() || !MOI->isDef())
          continue;
        unsigned reg = MOI->getReg();
        unsigned realUses = 0;
        for (MachineRegisterInfo::use_iterator UI = MRI.use_begin(reg),
                                               EI = MRI.use_end();
             UI != EI; ++UI) {
          // Check if there are any uses that occur only in the original
          // loop.  If so, that's not a real use.
          if (UI->getParent()->getParent() != BB) {
            realUses++;
            used = true;
            break;
          }
        }
        if (realUses > 0)
          break;
        used = false;
      }
      if (!used) {
        MI++->eraseFromParent();
        continue;
      }
      ++MI;
    }
  // In the kernel block, check if we can remove a Phi that generates a value
  // used in an instruction removed in the epilog block.
  for (MachineBasicBlock::iterator BBI = KernelBB->instr_begin(),
                                   BBE = KernelBB->getFirstNonPHI();
       BBI != BBE;) {
    MachineInstr *MI = &*BBI;
    ++BBI;
    unsigned reg = MI->getOperand(0).getReg();
    if (MRI.use_begin(reg) == MRI.use_end()) {
      MI->eraseFromParent();
    }
  }
}

/// For loop carried definitions, we split the lifetime of a virtual register
/// that has uses past the definition in the next iteration. A copy with a new
/// virtual register is inserted before the definition, which helps with
/// generating a better register assignment.
///
///   v1 = phi(a, v2)     v1 = phi(a, v2)
///   v2 = phi(b, v3)     v2 = phi(b, v3)
///   v3 = ..             v4 = copy v1
///   .. = V1             v3 = ..
///                       .. = v4
void SwingSchedulerDAG::splitLifetimes(MachineBasicBlock *KernelBB,
                                       MBBVectorTy &EpilogBBs,
                                       SMSchedule &Schedule) {
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  for (auto &PHI : KernelBB->phis()) {
    unsigned Def = PHI.getOperand(0).getReg();
    // Check for any Phi definition that used as an operand of another Phi
    // in the same block.
    for (MachineRegisterInfo::use_instr_iterator I = MRI.use_instr_begin(Def),
                                                 E = MRI.use_instr_end();
         I != E; ++I) {
      if (I->isPHI() && I->getParent() == KernelBB) {
        // Get the loop carried definition.
        unsigned LCDef = getLoopPhiReg(PHI, KernelBB);
        if (!LCDef)
          continue;
        MachineInstr *MI = MRI.getVRegDef(LCDef);
        if (!MI || MI->getParent() != KernelBB || MI->isPHI())
          continue;
        // Search through the rest of the block looking for uses of the Phi
        // definition. If one occurs, then split the lifetime.
        unsigned SplitReg = 0;
        for (auto &BBJ : make_range(MachineBasicBlock::instr_iterator(MI),
                                    KernelBB->instr_end()))
          if (BBJ.readsRegister(Def)) {
            // We split the lifetime when we find the first use.
            if (SplitReg == 0) {
              SplitReg = MRI.createVirtualRegister(MRI.getRegClass(Def));
              BuildMI(*KernelBB, MI, MI->getDebugLoc(),
                      TII->get(TargetOpcode::COPY), SplitReg)
                  .addReg(Def);
            }
            BBJ.substituteRegister(Def, SplitReg, 0, *TRI);
          }
        if (!SplitReg)
          continue;
        // Search through each of the epilog blocks for any uses to be renamed.
        for (auto &Epilog : EpilogBBs)
          for (auto &I : *Epilog)
            if (I.readsRegister(Def))
              I.substituteRegister(Def, SplitReg, 0, *TRI);
        break;
      }
    }
  }
}

/// Remove the incoming block from the Phis in a basic block.
static void removePhis(MachineBasicBlock *BB, MachineBasicBlock *Incoming) {
  for (MachineInstr &MI : *BB) {
    if (!MI.isPHI())
      break;
    for (unsigned i = 1, e = MI.getNumOperands(); i != e; i += 2)
      if (MI.getOperand(i + 1).getMBB() == Incoming) {
        MI.RemoveOperand(i + 1);
        MI.RemoveOperand(i);
        break;
      }
  }
}

/// Create branches from each prolog basic block to the appropriate epilog
/// block.  These edges are needed if the loop ends before reaching the
/// kernel.
void SwingSchedulerDAG::addBranches(MBBVectorTy &PrologBBs,
                                    MachineBasicBlock *KernelBB,
                                    MBBVectorTy &EpilogBBs,
                                    SMSchedule &Schedule, ValueMapTy *VRMap) {
  assert(PrologBBs.size() == EpilogBBs.size() && "Prolog/Epilog mismatch");
  MachineInstr *IndVar = Pass.LI.LoopInductionVar;
  MachineInstr *Cmp = Pass.LI.LoopCompare;
  MachineBasicBlock *LastPro = KernelBB;
  MachineBasicBlock *LastEpi = KernelBB;

  // Start from the blocks connected to the kernel and work "out"
  // to the first prolog and the last epilog blocks.
  SmallVector<MachineInstr *, 4> PrevInsts;
  unsigned MaxIter = PrologBBs.size() - 1;
  unsigned LC = UINT_MAX;
  unsigned LCMin = UINT_MAX;
  for (unsigned i = 0, j = MaxIter; i <= MaxIter; ++i, --j) {
    // Add branches to the prolog that go to the corresponding
    // epilog, and the fall-thru prolog/kernel block.
    MachineBasicBlock *Prolog = PrologBBs[j];
    MachineBasicBlock *Epilog = EpilogBBs[i];
    // We've executed one iteration, so decrement the loop count and check for
    // the loop end.
    SmallVector<MachineOperand, 4> Cond;
    // Check if the LOOP0 has already been removed. If so, then there is no need
    // to reduce the trip count.
    if (LC != 0)
      LC = TII->reduceLoopCount(*Prolog, IndVar, *Cmp, Cond, PrevInsts, j,
                                MaxIter);

    // Record the value of the first trip count, which is used to determine if
    // branches and blocks can be removed for constant trip counts.
    if (LCMin == UINT_MAX)
      LCMin = LC;

    unsigned numAdded = 0;
    if (TargetRegisterInfo::isVirtualRegister(LC)) {
      Prolog->addSuccessor(Epilog);
      numAdded = TII->insertBranch(*Prolog, Epilog, LastPro, Cond, DebugLoc());
    } else if (j >= LCMin) {
      Prolog->addSuccessor(Epilog);
      Prolog->removeSuccessor(LastPro);
      LastEpi->removeSuccessor(Epilog);
      numAdded = TII->insertBranch(*Prolog, Epilog, nullptr, Cond, DebugLoc());
      removePhis(Epilog, LastEpi);
      // Remove the blocks that are no longer referenced.
      if (LastPro != LastEpi) {
        LastEpi->clear();
        LastEpi->eraseFromParent();
      }
      LastPro->clear();
      LastPro->eraseFromParent();
    } else {
      numAdded = TII->insertBranch(*Prolog, LastPro, nullptr, Cond, DebugLoc());
      removePhis(Epilog, Prolog);
    }
    LastPro = Prolog;
    LastEpi = Epilog;
    for (MachineBasicBlock::reverse_instr_iterator I = Prolog->instr_rbegin(),
                                                   E = Prolog->instr_rend();
         I != E && numAdded > 0; ++I, --numAdded)
      updateInstruction(&*I, false, j, 0, Schedule, VRMap);
  }
}

/// Return true if we can compute the amount the instruction changes
/// during each iteration. Set Delta to the amount of the change.
bool SwingSchedulerDAG::computeDelta(MachineInstr &MI, unsigned &Delta) {
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  unsigned BaseReg;
  int64_t Offset;
  if (!TII->getMemOpBaseRegImmOfs(MI, BaseReg, Offset, TRI))
    return false;

  MachineRegisterInfo &MRI = MF.getRegInfo();
  // Check if there is a Phi. If so, get the definition in the loop.
  MachineInstr *BaseDef = MRI.getVRegDef(BaseReg);
  if (BaseDef && BaseDef->isPHI()) {
    BaseReg = getLoopPhiReg(*BaseDef, MI.getParent());
    BaseDef = MRI.getVRegDef(BaseReg);
  }
  if (!BaseDef)
    return false;

  int D = 0;
  if (!TII->getIncrementValue(*BaseDef, D) && D >= 0)
    return false;

  Delta = D;
  return true;
}

/// Update the memory operand with a new offset when the pipeliner
/// generates a new copy of the instruction that refers to a
/// different memory location.
void SwingSchedulerDAG::updateMemOperands(MachineInstr &NewMI,
                                          MachineInstr &OldMI, unsigned Num) {
  if (Num == 0)
    return;
  // If the instruction has memory operands, then adjust the offset
  // when the instruction appears in different stages.
  unsigned NumRefs = NewMI.memoperands_end() - NewMI.memoperands_begin();
  if (NumRefs == 0)
    return;
  MachineInstr::mmo_iterator NewMemRefs = MF.allocateMemRefsArray(NumRefs);
  unsigned Refs = 0;
  for (MachineMemOperand *MMO : NewMI.memoperands()) {
    if (MMO->isVolatile() || (MMO->isInvariant() && MMO->isDereferenceable()) ||
        (!MMO->getValue())) {
      NewMemRefs[Refs++] = MMO;
      continue;
    }
    unsigned Delta;
    if (computeDelta(OldMI, Delta)) {
      int64_t AdjOffset = Delta * Num;
      NewMemRefs[Refs++] =
          MF.getMachineMemOperand(MMO, AdjOffset, MMO->getSize());
    } else
      NewMemRefs[Refs++] = MF.getMachineMemOperand(MMO, 0, UINT64_MAX);
  }
  NewMI.setMemRefs(NewMemRefs, NewMemRefs + NumRefs);
}

/// Clone the instruction for the new pipelined loop and update the
/// memory operands, if needed.
MachineInstr *SwingSchedulerDAG::cloneInstr(MachineInstr *OldMI,
                                            unsigned CurStageNum,
                                            unsigned InstStageNum) {
  MachineInstr *NewMI = MF.CloneMachineInstr(OldMI);
  // Check for tied operands in inline asm instructions. This should be handled
  // elsewhere, but I'm not sure of the best solution.
  if (OldMI->isInlineAsm())
    for (unsigned i = 0, e = OldMI->getNumOperands(); i != e; ++i) {
      const auto &MO = OldMI->getOperand(i);
      if (MO.isReg() && MO.isUse())
        break;
      unsigned UseIdx;
      if (OldMI->isRegTiedToUseOperand(i, &UseIdx))
        NewMI->tieOperands(i, UseIdx);
    }
  updateMemOperands(*NewMI, *OldMI, CurStageNum - InstStageNum);
  return NewMI;
}

/// Clone the instruction for the new pipelined loop. If needed, this
/// function updates the instruction using the values saved in the
/// InstrChanges structure.
MachineInstr *SwingSchedulerDAG::cloneAndChangeInstr(MachineInstr *OldMI,
                                                     unsigned CurStageNum,
                                                     unsigned InstStageNum,
                                                     SMSchedule &Schedule) {
  MachineInstr *NewMI = MF.CloneMachineInstr(OldMI);
  DenseMap<SUnit *, std::pair<unsigned, int64_t>>::iterator It =
      InstrChanges.find(getSUnit(OldMI));
  if (It != InstrChanges.end()) {
    std::pair<unsigned, int64_t> RegAndOffset = It->second;
    unsigned BasePos, OffsetPos;
    if (!TII->getBaseAndOffsetPosition(*OldMI, BasePos, OffsetPos))
      return nullptr;
    int64_t NewOffset = OldMI->getOperand(OffsetPos).getImm();
    MachineInstr *LoopDef = findDefInLoop(RegAndOffset.first);
    if (Schedule.stageScheduled(getSUnit(LoopDef)) > (signed)InstStageNum)
      NewOffset += RegAndOffset.second * (CurStageNum - InstStageNum);
    NewMI->getOperand(OffsetPos).setImm(NewOffset);
  }
  updateMemOperands(*NewMI, *OldMI, CurStageNum - InstStageNum);
  return NewMI;
}

/// Update the machine instruction with new virtual registers.  This
/// function may change the defintions and/or uses.
void SwingSchedulerDAG::updateInstruction(MachineInstr *NewMI, bool LastDef,
                                          unsigned CurStageNum,
                                          unsigned InstrStageNum,
                                          SMSchedule &Schedule,
                                          ValueMapTy *VRMap) {
  for (unsigned i = 0, e = NewMI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = NewMI->getOperand(i);
    if (!MO.isReg() || !TargetRegisterInfo::isVirtualRegister(MO.getReg()))
      continue;
    unsigned reg = MO.getReg();
    if (MO.isDef()) {
      // Create a new virtual register for the definition.
      const TargetRegisterClass *RC = MRI.getRegClass(reg);
      unsigned NewReg = MRI.createVirtualRegister(RC);
      MO.setReg(NewReg);
      VRMap[CurStageNum][reg] = NewReg;
      if (LastDef)
        replaceRegUsesAfterLoop(reg, NewReg, BB, MRI, LIS);
    } else if (MO.isUse()) {
      MachineInstr *Def = MRI.getVRegDef(reg);
      // Compute the stage that contains the last definition for instruction.
      int DefStageNum = Schedule.stageScheduled(getSUnit(Def));
      unsigned StageNum = CurStageNum;
      if (DefStageNum != -1 && (int)InstrStageNum > DefStageNum) {
        // Compute the difference in stages between the defintion and the use.
        unsigned StageDiff = (InstrStageNum - DefStageNum);
        // Make an adjustment to get the last definition.
        StageNum -= StageDiff;
      }
      if (VRMap[StageNum].count(reg))
        MO.setReg(VRMap[StageNum][reg]);
    }
  }
}

/// Return the instruction in the loop that defines the register.
/// If the definition is a Phi, then follow the Phi operand to
/// the instruction in the loop.
MachineInstr *SwingSchedulerDAG::findDefInLoop(unsigned Reg) {
  SmallPtrSet<MachineInstr *, 8> Visited;
  MachineInstr *Def = MRI.getVRegDef(Reg);
  while (Def->isPHI()) {
    if (!Visited.insert(Def).second)
      break;
    for (unsigned i = 1, e = Def->getNumOperands(); i < e; i += 2)
      if (Def->getOperand(i + 1).getMBB() == BB) {
        Def = MRI.getVRegDef(Def->getOperand(i).getReg());
        break;
      }
  }
  return Def;
}

/// Return the new name for the value from the previous stage.
unsigned SwingSchedulerDAG::getPrevMapVal(unsigned StageNum, unsigned PhiStage,
                                          unsigned LoopVal, unsigned LoopStage,
                                          ValueMapTy *VRMap,
                                          MachineBasicBlock *BB) {
  unsigned PrevVal = 0;
  if (StageNum > PhiStage) {
    MachineInstr *LoopInst = MRI.getVRegDef(LoopVal);
    if (PhiStage == LoopStage && VRMap[StageNum - 1].count(LoopVal))
      // The name is defined in the previous stage.
      PrevVal = VRMap[StageNum - 1][LoopVal];
    else if (VRMap[StageNum].count(LoopVal))
      // The previous name is defined in the current stage when the instruction
      // order is swapped.
      PrevVal = VRMap[StageNum][LoopVal];
    else if (!LoopInst->isPHI() || LoopInst->getParent() != BB)
      // The loop value hasn't yet been scheduled.
      PrevVal = LoopVal;
    else if (StageNum == PhiStage + 1)
      // The loop value is another phi, which has not been scheduled.
      PrevVal = getInitPhiReg(*LoopInst, BB);
    else if (StageNum > PhiStage + 1 && LoopInst->getParent() == BB)
      // The loop value is another phi, which has been scheduled.
      PrevVal =
          getPrevMapVal(StageNum - 1, PhiStage, getLoopPhiReg(*LoopInst, BB),
                        LoopStage, VRMap, BB);
  }
  return PrevVal;
}

/// Rewrite the Phi values in the specified block to use the mappings
/// from the initial operand. Once the Phi is scheduled, we switch
/// to using the loop value instead of the Phi value, so those names
/// do not need to be rewritten.
void SwingSchedulerDAG::rewritePhiValues(MachineBasicBlock *NewBB,
                                         unsigned StageNum,
                                         SMSchedule &Schedule,
                                         ValueMapTy *VRMap,
                                         InstrMapTy &InstrMap) {
  for (auto &PHI : BB->phis()) {
    unsigned InitVal = 0;
    unsigned LoopVal = 0;
    getPhiRegs(PHI, BB, InitVal, LoopVal);
    unsigned PhiDef = PHI.getOperand(0).getReg();

    unsigned PhiStage =
        (unsigned)Schedule.stageScheduled(getSUnit(MRI.getVRegDef(PhiDef)));
    unsigned LoopStage =
        (unsigned)Schedule.stageScheduled(getSUnit(MRI.getVRegDef(LoopVal)));
    unsigned NumPhis = Schedule.getStagesForPhi(PhiDef);
    if (NumPhis > StageNum)
      NumPhis = StageNum;
    for (unsigned np = 0; np <= NumPhis; ++np) {
      unsigned NewVal =
          getPrevMapVal(StageNum - np, PhiStage, LoopVal, LoopStage, VRMap, BB);
      if (!NewVal)
        NewVal = InitVal;
      rewriteScheduledInstr(NewBB, Schedule, InstrMap, StageNum - np, np, &PHI,
                            PhiDef, NewVal);
    }
  }
}

/// Rewrite a previously scheduled instruction to use the register value
/// from the new instruction. Make sure the instruction occurs in the
/// basic block, and we don't change the uses in the new instruction.
void SwingSchedulerDAG::rewriteScheduledInstr(
    MachineBasicBlock *BB, SMSchedule &Schedule, InstrMapTy &InstrMap,
    unsigned CurStageNum, unsigned PhiNum, MachineInstr *Phi, unsigned OldReg,
    unsigned NewReg, unsigned PrevReg) {
  bool InProlog = (CurStageNum < Schedule.getMaxStageCount());
  int StagePhi = Schedule.stageScheduled(getSUnit(Phi)) + PhiNum;
  // Rewrite uses that have been scheduled already to use the new
  // Phi register.
  for (MachineRegisterInfo::use_iterator UI = MRI.use_begin(OldReg),
                                         EI = MRI.use_end();
       UI != EI;) {
    MachineOperand &UseOp = *UI;
    MachineInstr *UseMI = UseOp.getParent();
    ++UI;
    if (UseMI->getParent() != BB)
      continue;
    if (UseMI->isPHI()) {
      if (!Phi->isPHI() && UseMI->getOperand(0).getReg() == NewReg)
        continue;
      if (getLoopPhiReg(*UseMI, BB) != OldReg)
        continue;
    }
    InstrMapTy::iterator OrigInstr = InstrMap.find(UseMI);
    assert(OrigInstr != InstrMap.end() && "Instruction not scheduled.");
    SUnit *OrigMISU = getSUnit(OrigInstr->second);
    int StageSched = Schedule.stageScheduled(OrigMISU);
    int CycleSched = Schedule.cycleScheduled(OrigMISU);
    unsigned ReplaceReg = 0;
    // This is the stage for the scheduled instruction.
    if (StagePhi == StageSched && Phi->isPHI()) {
      int CyclePhi = Schedule.cycleScheduled(getSUnit(Phi));
      if (PrevReg && InProlog)
        ReplaceReg = PrevReg;
      else if (PrevReg && !Schedule.isLoopCarried(this, *Phi) &&
               (CyclePhi <= CycleSched || OrigMISU->getInstr()->isPHI()))
        ReplaceReg = PrevReg;
      else
        ReplaceReg = NewReg;
    }
    // The scheduled instruction occurs before the scheduled Phi, and the
    // Phi is not loop carried.
    if (!InProlog && StagePhi + 1 == StageSched &&
        !Schedule.isLoopCarried(this, *Phi))
      ReplaceReg = NewReg;
    if (StagePhi > StageSched && Phi->isPHI())
      ReplaceReg = NewReg;
    if (!InProlog && !Phi->isPHI() && StagePhi < StageSched)
      ReplaceReg = NewReg;
    if (ReplaceReg) {
      MRI.constrainRegClass(ReplaceReg, MRI.getRegClass(OldReg));
      UseOp.setReg(ReplaceReg);
    }
  }
}

/// Check if we can change the instruction to use an offset value from the
/// previous iteration. If so, return true and set the base and offset values
/// so that we can rewrite the load, if necessary.
///   v1 = Phi(v0, v3)
///   v2 = load v1, 0
///   v3 = post_store v1, 4, x
/// This function enables the load to be rewritten as v2 = load v3, 4.
bool SwingSchedulerDAG::canUseLastOffsetValue(MachineInstr *MI,
                                              unsigned &BasePos,
                                              unsigned &OffsetPos,
                                              unsigned &NewBase,
                                              int64_t &Offset) {
  // Get the load instruction.
  if (TII->isPostIncrement(*MI))
    return false;
  unsigned BasePosLd, OffsetPosLd;
  if (!TII->getBaseAndOffsetPosition(*MI, BasePosLd, OffsetPosLd))
    return false;
  unsigned BaseReg = MI->getOperand(BasePosLd).getReg();

  // Look for the Phi instruction.
  MachineRegisterInfo &MRI = MI->getMF()->getRegInfo();
  MachineInstr *Phi = MRI.getVRegDef(BaseReg);
  if (!Phi || !Phi->isPHI())
    return false;
  // Get the register defined in the loop block.
  unsigned PrevReg = getLoopPhiReg(*Phi, MI->getParent());
  if (!PrevReg)
    return false;

  // Check for the post-increment load/store instruction.
  MachineInstr *PrevDef = MRI.getVRegDef(PrevReg);
  if (!PrevDef || PrevDef == MI)
    return false;

  if (!TII->isPostIncrement(*PrevDef))
    return false;

  unsigned BasePos1 = 0, OffsetPos1 = 0;
  if (!TII->getBaseAndOffsetPosition(*PrevDef, BasePos1, OffsetPos1))
    return false;

  // Make sure offset values are both positive or both negative.
  int64_t LoadOffset = MI->getOperand(OffsetPosLd).getImm();
  int64_t StoreOffset = PrevDef->getOperand(OffsetPos1).getImm();
  if ((LoadOffset >= 0) != (StoreOffset >= 0))
    return false;

  // Set the return value once we determine that we return true.
  BasePos = BasePosLd;
  OffsetPos = OffsetPosLd;
  NewBase = PrevReg;
  Offset = StoreOffset;
  return true;
}

/// Apply changes to the instruction if needed. The changes are need
/// to improve the scheduling and depend up on the final schedule.
void SwingSchedulerDAG::applyInstrChange(MachineInstr *MI,
                                         SMSchedule &Schedule) {
  SUnit *SU = getSUnit(MI);
  DenseMap<SUnit *, std::pair<unsigned, int64_t>>::iterator It =
      InstrChanges.find(SU);
  if (It != InstrChanges.end()) {
    std::pair<unsigned, int64_t> RegAndOffset = It->second;
    unsigned BasePos, OffsetPos;
    if (!TII->getBaseAndOffsetPosition(*MI, BasePos, OffsetPos))
      return;
    unsigned BaseReg = MI->getOperand(BasePos).getReg();
    MachineInstr *LoopDef = findDefInLoop(BaseReg);
    int DefStageNum = Schedule.stageScheduled(getSUnit(LoopDef));
    int DefCycleNum = Schedule.cycleScheduled(getSUnit(LoopDef));
    int BaseStageNum = Schedule.stageScheduled(SU);
    int BaseCycleNum = Schedule.cycleScheduled(SU);
    if (BaseStageNum < DefStageNum) {
      MachineInstr *NewMI = MF.CloneMachineInstr(MI);
      int OffsetDiff = DefStageNum - BaseStageNum;
      if (DefCycleNum < BaseCycleNum) {
        NewMI->getOperand(BasePos).setReg(RegAndOffset.first);
        if (OffsetDiff > 0)
          --OffsetDiff;
      }
      int64_t NewOffset =
          MI->getOperand(OffsetPos).getImm() + RegAndOffset.second * OffsetDiff;
      NewMI->getOperand(OffsetPos).setImm(NewOffset);
      SU->setInstr(NewMI);
      MISUnitMap[NewMI] = SU;
      NewMIs.insert(NewMI);
    }
  }
}

/// Return true for an order dependence that is loop carried potentially.
/// An order dependence is loop carried if the destination defines a value
/// that may be used by the source in a subsequent iteration.
bool SwingSchedulerDAG::isLoopCarriedOrder(SUnit *Source, const SDep &Dep,
                                           bool isSucc) {
  if (!isOrder(Source, Dep) || Dep.isArtificial())
    return false;

  if (!SwpPruneLoopCarried)
    return true;

  MachineInstr *SI = Source->getInstr();
  MachineInstr *DI = Dep.getSUnit()->getInstr();
  if (!isSucc)
    std::swap(SI, DI);
  assert(SI != nullptr && DI != nullptr && "Expecting SUnit with an MI.");

  // Assume ordered loads and stores may have a loop carried dependence.
  if (SI->hasUnmodeledSideEffects() || DI->hasUnmodeledSideEffects() ||
      SI->hasOrderedMemoryRef() || DI->hasOrderedMemoryRef())
    return true;

  // Only chain dependences between a load and store can be loop carried.
  if (!DI->mayStore() || !SI->mayLoad())
    return false;

  unsigned DeltaS, DeltaD;
  if (!computeDelta(*SI, DeltaS) || !computeDelta(*DI, DeltaD))
    return true;

  unsigned BaseRegS, BaseRegD;
  int64_t OffsetS, OffsetD;
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  if (!TII->getMemOpBaseRegImmOfs(*SI, BaseRegS, OffsetS, TRI) ||
      !TII->getMemOpBaseRegImmOfs(*DI, BaseRegD, OffsetD, TRI))
    return true;

  if (BaseRegS != BaseRegD)
    return true;

  uint64_t AccessSizeS = (*SI->memoperands_begin())->getSize();
  uint64_t AccessSizeD = (*DI->memoperands_begin())->getSize();

  // This is the main test, which checks the offset values and the loop
  // increment value to determine if the accesses may be loop carried.
  if (OffsetS >= OffsetD)
    return OffsetS + AccessSizeS > DeltaS;
  else
    return OffsetD + AccessSizeD > DeltaD;

  return true;
}

void SwingSchedulerDAG::postprocessDAG() {
  for (auto &M : Mutations)
    M->apply(this);
}

/// Try to schedule the node at the specified StartCycle and continue
/// until the node is schedule or the EndCycle is reached.  This function
/// returns true if the node is scheduled.  This routine may search either
/// forward or backward for a place to insert the instruction based upon
/// the relative values of StartCycle and EndCycle.
bool SMSchedule::insert(SUnit *SU, int StartCycle, int EndCycle, int II) {
  bool forward = true;
  if (StartCycle > EndCycle)
    forward = false;

  // The terminating condition depends on the direction.
  int termCycle = forward ? EndCycle + 1 : EndCycle - 1;
  for (int curCycle = StartCycle; curCycle != termCycle;
       forward ? ++curCycle : --curCycle) {

    // Add the already scheduled instructions at the specified cycle to the DFA.
    Resources->clearResources();
    for (int checkCycle = FirstCycle + ((curCycle - FirstCycle) % II);
         checkCycle <= LastCycle; checkCycle += II) {
      std::deque<SUnit *> &cycleInstrs = ScheduledInstrs[checkCycle];

      for (std::deque<SUnit *>::iterator I = cycleInstrs.begin(),
                                         E = cycleInstrs.end();
           I != E; ++I) {
        if (ST.getInstrInfo()->isZeroCost((*I)->getInstr()->getOpcode()))
          continue;
        assert(Resources->canReserveResources(*(*I)->getInstr()) &&
               "These instructions have already been scheduled.");
        Resources->reserveResources(*(*I)->getInstr());
      }
    }
    if (ST.getInstrInfo()->isZeroCost(SU->getInstr()->getOpcode()) ||
        Resources->canReserveResources(*SU->getInstr())) {
      DEBUG({
        dbgs() << "\tinsert at cycle " << curCycle << " ";
        SU->getInstr()->dump();
      });

      ScheduledInstrs[curCycle].push_back(SU);
      InstrToCycle.insert(std::make_pair(SU, curCycle));
      if (curCycle > LastCycle)
        LastCycle = curCycle;
      if (curCycle < FirstCycle)
        FirstCycle = curCycle;
      return true;
    }
    DEBUG({
      dbgs() << "\tfailed to insert at cycle " << curCycle << " ";
      SU->getInstr()->dump();
    });
  }
  return false;
}

// Return the cycle of the earliest scheduled instruction in the chain.
int SMSchedule::earliestCycleInChain(const SDep &Dep) {
  SmallPtrSet<SUnit *, 8> Visited;
  SmallVector<SDep, 8> Worklist;
  Worklist.push_back(Dep);
  int EarlyCycle = INT_MAX;
  while (!Worklist.empty()) {
    const SDep &Cur = Worklist.pop_back_val();
    SUnit *PrevSU = Cur.getSUnit();
    if (Visited.count(PrevSU))
      continue;
    std::map<SUnit *, int>::const_iterator it = InstrToCycle.find(PrevSU);
    if (it == InstrToCycle.end())
      continue;
    EarlyCycle = std::min(EarlyCycle, it->second);
    for (const auto &PI : PrevSU->Preds)
      if (SwingSchedulerDAG::isOrder(PrevSU, PI))
        Worklist.push_back(PI);
    Visited.insert(PrevSU);
  }
  return EarlyCycle;
}

// Return the cycle of the latest scheduled instruction in the chain.
int SMSchedule::latestCycleInChain(const SDep &Dep) {
  SmallPtrSet<SUnit *, 8> Visited;
  SmallVector<SDep, 8> Worklist;
  Worklist.push_back(Dep);
  int LateCycle = INT_MIN;
  while (!Worklist.empty()) {
    const SDep &Cur = Worklist.pop_back_val();
    SUnit *SuccSU = Cur.getSUnit();
    if (Visited.count(SuccSU))
      continue;
    std::map<SUnit *, int>::const_iterator it = InstrToCycle.find(SuccSU);
    if (it == InstrToCycle.end())
      continue;
    LateCycle = std::max(LateCycle, it->second);
    for (const auto &SI : SuccSU->Succs)
      if (SwingSchedulerDAG::isOrder(SuccSU, SI))
        Worklist.push_back(SI);
    Visited.insert(SuccSU);
  }
  return LateCycle;
}

/// If an instruction has a use that spans multiple iterations, then
/// return true. These instructions are characterized by having a back-ege
/// to a Phi, which contains a reference to another Phi.
static SUnit *multipleIterations(SUnit *SU, SwingSchedulerDAG *DAG) {
  for (auto &P : SU->Preds)
    if (DAG->isBackedge(SU, P) && P.getSUnit()->getInstr()->isPHI())
      for (auto &S : P.getSUnit()->Succs)
        if (S.getKind() == SDep::Order && S.getSUnit()->getInstr()->isPHI())
          return P.getSUnit();
  return nullptr;
}

/// Compute the scheduling start slot for the instruction.  The start slot
/// depends on any predecessor or successor nodes scheduled already.
void SMSchedule::computeStart(SUnit *SU, int *MaxEarlyStart, int *MinLateStart,
                              int *MinEnd, int *MaxStart, int II,
                              SwingSchedulerDAG *DAG) {
  // Iterate over each instruction that has been scheduled already.  The start
  // slot computuation depends on whether the previously scheduled instruction
  // is a predecessor or successor of the specified instruction.
  for (int cycle = getFirstCycle(); cycle <= LastCycle; ++cycle) {

    // Iterate over each instruction in the current cycle.
    for (SUnit *I : getInstructions(cycle)) {
      // Because we're processing a DAG for the dependences, we recognize
      // the back-edge in recurrences by anti dependences.
      for (unsigned i = 0, e = (unsigned)SU->Preds.size(); i != e; ++i) {
        const SDep &Dep = SU->Preds[i];
        if (Dep.getSUnit() == I) {
          if (!DAG->isBackedge(SU, Dep)) {
            int EarlyStart = cycle + DAG->getLatency(SU, Dep) -
                             DAG->getDistance(Dep.getSUnit(), SU, Dep) * II;
            *MaxEarlyStart = std::max(*MaxEarlyStart, EarlyStart);
            if (DAG->isLoopCarriedOrder(SU, Dep, false)) {
              int End = earliestCycleInChain(Dep) + (II - 1);
              *MinEnd = std::min(*MinEnd, End);
            }
          } else {
            int LateStart = cycle - DAG->getLatency(SU, Dep) +
                            DAG->getDistance(SU, Dep.getSUnit(), Dep) * II;
            *MinLateStart = std::min(*MinLateStart, LateStart);
          }
        }
        // For instruction that requires multiple iterations, make sure that
        // the dependent instruction is not scheduled past the definition.
        SUnit *BE = multipleIterations(I, DAG);
        if (BE && Dep.getSUnit() == BE && !SU->getInstr()->isPHI() &&
            !SU->isPred(I))
          *MinLateStart = std::min(*MinLateStart, cycle);
      }
      for (unsigned i = 0, e = (unsigned)SU->Succs.size(); i != e; ++i)
        if (SU->Succs[i].getSUnit() == I) {
          const SDep &Dep = SU->Succs[i];
          if (!DAG->isBackedge(SU, Dep)) {
            int LateStart = cycle - DAG->getLatency(SU, Dep) +
                            DAG->getDistance(SU, Dep.getSUnit(), Dep) * II;
            *MinLateStart = std::min(*MinLateStart, LateStart);
            if (DAG->isLoopCarriedOrder(SU, Dep)) {
              int Start = latestCycleInChain(Dep) + 1 - II;
              *MaxStart = std::max(*MaxStart, Start);
            }
          } else {
            int EarlyStart = cycle + DAG->getLatency(SU, Dep) -
                             DAG->getDistance(Dep.getSUnit(), SU, Dep) * II;
            *MaxEarlyStart = std::max(*MaxEarlyStart, EarlyStart);
          }
        }
    }
  }
}

/// Order the instructions within a cycle so that the definitions occur
/// before the uses. Returns true if the instruction is added to the start
/// of the list, or false if added to the end.
bool SMSchedule::orderDependence(SwingSchedulerDAG *SSD, SUnit *SU,
                                 std::deque<SUnit *> &Insts) {
  MachineInstr *MI = SU->getInstr();
  bool OrderBeforeUse = false;
  bool OrderAfterDef = false;
  bool OrderBeforeDef = false;
  unsigned MoveDef = 0;
  unsigned MoveUse = 0;
  int StageInst1 = stageScheduled(SU);

  unsigned Pos = 0;
  for (std::deque<SUnit *>::iterator I = Insts.begin(), E = Insts.end(); I != E;
       ++I, ++Pos) {
    // Relative order of Phis does not matter.
    if (MI->isPHI() && (*I)->getInstr()->isPHI())
      continue;
    for (unsigned i = 0, e = MI->getNumOperands(); i < e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (!MO.isReg() || !TargetRegisterInfo::isVirtualRegister(MO.getReg()))
        continue;
      unsigned Reg = MO.getReg();
      unsigned BasePos, OffsetPos;
      if (ST.getInstrInfo()->getBaseAndOffsetPosition(*MI, BasePos, OffsetPos))
        if (MI->getOperand(BasePos).getReg() == Reg)
          if (unsigned NewReg = SSD->getInstrBaseReg(SU))
            Reg = NewReg;
      bool Reads, Writes;
      std::tie(Reads, Writes) =
          (*I)->getInstr()->readsWritesVirtualRegister(Reg);
      if (MO.isDef() && Reads && stageScheduled(*I) <= StageInst1) {
        OrderBeforeUse = true;
        MoveUse = Pos;
      } else if (MO.isDef() && Reads && stageScheduled(*I) > StageInst1) {
        // Add the instruction after the scheduled instruction.
        OrderAfterDef = true;
        MoveDef = Pos;
      } else if (MO.isUse() && Writes && stageScheduled(*I) == StageInst1) {
        if (cycleScheduled(*I) == cycleScheduled(SU) && !(*I)->isSucc(SU)) {
          OrderBeforeUse = true;
          MoveUse = Pos;
        } else {
          OrderAfterDef = true;
          MoveDef = Pos;
        }
      } else if (MO.isUse() && Writes && stageScheduled(*I) > StageInst1) {
        OrderBeforeUse = true;
        MoveUse = Pos;
        if (MoveUse != 0) {
          OrderAfterDef = true;
          MoveDef = Pos - 1;
        }
      } else if (MO.isUse() && Writes && stageScheduled(*I) < StageInst1) {
        // Add the instruction before the scheduled instruction.
        OrderBeforeUse = true;
        MoveUse = Pos;
      } else if (MO.isUse() && stageScheduled(*I) == StageInst1 &&
                 isLoopCarriedDefOfUse(SSD, (*I)->getInstr(), MO)) {
        OrderBeforeDef = true;
        MoveUse = Pos;
      }
    }
    // Check for order dependences between instructions. Make sure the source
    // is ordered before the destination.
    for (auto &S : SU->Succs)
      if (S.getKind() == SDep::Order) {
        if (S.getSUnit() == *I && stageScheduled(*I) == StageInst1) {
          OrderBeforeUse = true;
          MoveUse = Pos;
        }
      } else if (TargetRegisterInfo::isPhysicalRegister(S.getReg())) {
        if (cycleScheduled(SU) != cycleScheduled(S.getSUnit())) {
          if (S.isAssignedRegDep()) {
            OrderAfterDef = true;
            MoveDef = Pos;
          }
        } else {
          OrderBeforeUse = true;
          MoveUse = Pos;
        }
      }
    for (auto &P : SU->Preds)
      if (P.getKind() == SDep::Order) {
        if (P.getSUnit() == *I && stageScheduled(*I) == StageInst1) {
          OrderAfterDef = true;
          MoveDef = Pos;
        }
      } else if (TargetRegisterInfo::isPhysicalRegister(P.getReg())) {
        if (cycleScheduled(SU) != cycleScheduled(P.getSUnit())) {
          if (P.isAssignedRegDep()) {
            OrderBeforeUse = true;
            MoveUse = Pos;
          }
        } else {
          OrderAfterDef = true;
          MoveDef = Pos;
        }
      }
  }

  // A circular dependence.
  if (OrderAfterDef && OrderBeforeUse && MoveUse == MoveDef)
    OrderBeforeUse = false;

  // OrderAfterDef takes precedences over OrderBeforeDef. The latter is due
  // to a loop-carried dependence.
  if (OrderBeforeDef)
    OrderBeforeUse = !OrderAfterDef || (MoveUse > MoveDef);

  // The uncommon case when the instruction order needs to be updated because
  // there is both a use and def.
  if (OrderBeforeUse && OrderAfterDef) {
    SUnit *UseSU = Insts.at(MoveUse);
    SUnit *DefSU = Insts.at(MoveDef);
    if (MoveUse > MoveDef) {
      Insts.erase(Insts.begin() + MoveUse);
      Insts.erase(Insts.begin() + MoveDef);
    } else {
      Insts.erase(Insts.begin() + MoveDef);
      Insts.erase(Insts.begin() + MoveUse);
    }
    if (orderDependence(SSD, UseSU, Insts)) {
      Insts.push_front(SU);
      orderDependence(SSD, DefSU, Insts);
      return true;
    }
    Insts.pop_back();
    Insts.push_back(SU);
    Insts.push_back(UseSU);
    orderDependence(SSD, DefSU, Insts);
    return false;
  }
  // Put the new instruction first if there is a use in the list. Otherwise,
  // put it at the end of the list.
  if (OrderBeforeUse)
    Insts.push_front(SU);
  else
    Insts.push_back(SU);
  return OrderBeforeUse;
}

/// Return true if the scheduled Phi has a loop carried operand.
bool SMSchedule::isLoopCarried(SwingSchedulerDAG *SSD, MachineInstr &Phi) {
  if (!Phi.isPHI())
    return false;
  assert(Phi.isPHI() && "Expecing a Phi.");
  SUnit *DefSU = SSD->getSUnit(&Phi);
  unsigned DefCycle = cycleScheduled(DefSU);
  int DefStage = stageScheduled(DefSU);

  unsigned InitVal = 0;
  unsigned LoopVal = 0;
  getPhiRegs(Phi, Phi.getParent(), InitVal, LoopVal);
  SUnit *UseSU = SSD->getSUnit(MRI.getVRegDef(LoopVal));
  if (!UseSU)
    return true;
  if (UseSU->getInstr()->isPHI())
    return true;
  unsigned LoopCycle = cycleScheduled(UseSU);
  int LoopStage = stageScheduled(UseSU);
  return (LoopCycle > DefCycle) || (LoopStage <= DefStage);
}

/// Return true if the instruction is a definition that is loop carried
/// and defines the use on the next iteration.
///        v1 = phi(v2, v3)
///  (Def) v3 = op v1
///  (MO)   = v1
/// If MO appears before Def, then then v1 and v3 may get assigned to the same
/// register.
bool SMSchedule::isLoopCarriedDefOfUse(SwingSchedulerDAG *SSD,
                                       MachineInstr *Def, MachineOperand &MO) {
  if (!MO.isReg())
    return false;
  if (Def->isPHI())
    return false;
  MachineInstr *Phi = MRI.getVRegDef(MO.getReg());
  if (!Phi || !Phi->isPHI() || Phi->getParent() != Def->getParent())
    return false;
  if (!isLoopCarried(SSD, *Phi))
    return false;
  unsigned LoopReg = getLoopPhiReg(*Phi, Phi->getParent());
  for (unsigned i = 0, e = Def->getNumOperands(); i != e; ++i) {
    MachineOperand &DMO = Def->getOperand(i);
    if (!DMO.isReg() || !DMO.isDef())
      continue;
    if (DMO.getReg() == LoopReg)
      return true;
  }
  return false;
}

// Check if the generated schedule is valid. This function checks if
// an instruction that uses a physical register is scheduled in a
// different stage than the definition. The pipeliner does not handle
// physical register values that may cross a basic block boundary.
bool SMSchedule::isValidSchedule(SwingSchedulerDAG *SSD) {
  for (int i = 0, e = SSD->SUnits.size(); i < e; ++i) {
    SUnit &SU = SSD->SUnits[i];
    if (!SU.hasPhysRegDefs)
      continue;
    int StageDef = stageScheduled(&SU);
    assert(StageDef != -1 && "Instruction should have been scheduled.");
    for (auto &SI : SU.Succs)
      if (SI.isAssignedRegDep())
        if (ST.getRegisterInfo()->isPhysicalRegister(SI.getReg()))
          if (stageScheduled(SI.getSUnit()) != StageDef)
            return false;
  }
  return true;
}

/// Attempt to fix the degenerate cases when the instruction serialization
/// causes the register lifetimes to overlap. For example,
///   p' = store_pi(p, b)
///      = load p, offset
/// In this case p and p' overlap, which means that two registers are needed.
/// Instead, this function changes the load to use p' and updates the offset.
void SwingSchedulerDAG::fixupRegisterOverlaps(std::deque<SUnit *> &Instrs) {
  unsigned OverlapReg = 0;
  unsigned NewBaseReg = 0;
  for (SUnit *SU : Instrs) {
    MachineInstr *MI = SU->getInstr();
    for (unsigned i = 0, e = MI->getNumOperands(); i < e; ++i) {
      const MachineOperand &MO = MI->getOperand(i);
      // Look for an instruction that uses p. The instruction occurs in the
      // same cycle but occurs later in the serialized order.
      if (MO.isReg() && MO.isUse() && MO.getReg() == OverlapReg) {
        // Check that the instruction appears in the InstrChanges structure,
        // which contains instructions that can have the offset updated.
        DenseMap<SUnit *, std::pair<unsigned, int64_t>>::iterator It =
          InstrChanges.find(SU);
        if (It != InstrChanges.end()) {
          unsigned BasePos, OffsetPos;
          // Update the base register and adjust the offset.
          if (TII->getBaseAndOffsetPosition(*MI, BasePos, OffsetPos)) {
            MachineInstr *NewMI = MF.CloneMachineInstr(MI);
            NewMI->getOperand(BasePos).setReg(NewBaseReg);
            int64_t NewOffset =
                MI->getOperand(OffsetPos).getImm() - It->second.second;
            NewMI->getOperand(OffsetPos).setImm(NewOffset);
            SU->setInstr(NewMI);
            MISUnitMap[NewMI] = SU;
            NewMIs.insert(NewMI);
          }
        }
        OverlapReg = 0;
        NewBaseReg = 0;
        break;
      }
      // Look for an instruction of the form p' = op(p), which uses and defines
      // two virtual registers that get allocated to the same physical register.
      unsigned TiedUseIdx = 0;
      if (MI->isRegTiedToUseOperand(i, &TiedUseIdx)) {
        // OverlapReg is p in the example above.
        OverlapReg = MI->getOperand(TiedUseIdx).getReg();
        // NewBaseReg is p' in the example above.
        NewBaseReg = MI->getOperand(i).getReg();
        break;
      }
    }
  }
}

/// After the schedule has been formed, call this function to combine
/// the instructions from the different stages/cycles.  That is, this
/// function creates a schedule that represents a single iteration.
void SMSchedule::finalizeSchedule(SwingSchedulerDAG *SSD) {
  // Move all instructions to the first stage from later stages.
  for (int cycle = getFirstCycle(); cycle <= getFinalCycle(); ++cycle) {
    for (int stage = 1, lastStage = getMaxStageCount(); stage <= lastStage;
         ++stage) {
      std::deque<SUnit *> &cycleInstrs =
          ScheduledInstrs[cycle + (stage * InitiationInterval)];
      for (std::deque<SUnit *>::reverse_iterator I = cycleInstrs.rbegin(),
                                                 E = cycleInstrs.rend();
           I != E; ++I)
        ScheduledInstrs[cycle].push_front(*I);
    }
  }
  // Iterate over the definitions in each instruction, and compute the
  // stage difference for each use.  Keep the maximum value.
  for (auto &I : InstrToCycle) {
    int DefStage = stageScheduled(I.first);
    MachineInstr *MI = I.first->getInstr();
    for (unsigned i = 0, e = MI->getNumOperands(); i < e; ++i) {
      MachineOperand &Op = MI->getOperand(i);
      if (!Op.isReg() || !Op.isDef())
        continue;

      unsigned Reg = Op.getReg();
      unsigned MaxDiff = 0;
      bool PhiIsSwapped = false;
      for (MachineRegisterInfo::use_iterator UI = MRI.use_begin(Reg),
                                             EI = MRI.use_end();
           UI != EI; ++UI) {
        MachineOperand &UseOp = *UI;
        MachineInstr *UseMI = UseOp.getParent();
        SUnit *SUnitUse = SSD->getSUnit(UseMI);
        int UseStage = stageScheduled(SUnitUse);
        unsigned Diff = 0;
        if (UseStage != -1 && UseStage >= DefStage)
          Diff = UseStage - DefStage;
        if (MI->isPHI()) {
          if (isLoopCarried(SSD, *MI))
            ++Diff;
          else
            PhiIsSwapped = true;
        }
        MaxDiff = std::max(Diff, MaxDiff);
      }
      RegToStageDiff[Reg] = std::make_pair(MaxDiff, PhiIsSwapped);
    }
  }

  // Erase all the elements in the later stages. Only one iteration should
  // remain in the scheduled list, and it contains all the instructions.
  for (int cycle = getFinalCycle() + 1; cycle <= LastCycle; ++cycle)
    ScheduledInstrs.erase(cycle);

  // Change the registers in instruction as specified in the InstrChanges
  // map. We need to use the new registers to create the correct order.
  for (int i = 0, e = SSD->SUnits.size(); i != e; ++i) {
    SUnit *SU = &SSD->SUnits[i];
    SSD->applyInstrChange(SU->getInstr(), *this);
  }

  // Reorder the instructions in each cycle to fix and improve the
  // generated code.
  for (int Cycle = getFirstCycle(), E = getFinalCycle(); Cycle <= E; ++Cycle) {
    std::deque<SUnit *> &cycleInstrs = ScheduledInstrs[Cycle];
    std::deque<SUnit *> newOrderZC;
    // Put the zero-cost, pseudo instructions at the start of the cycle.
    for (unsigned i = 0, e = cycleInstrs.size(); i < e; ++i) {
      SUnit *SU = cycleInstrs[i];
      if (ST.getInstrInfo()->isZeroCost(SU->getInstr()->getOpcode()))
        orderDependence(SSD, SU, newOrderZC);
    }
    std::deque<SUnit *> newOrderI;
    // Then, add the regular instructions back.
    for (unsigned i = 0, e = cycleInstrs.size(); i < e; ++i) {
      SUnit *SU = cycleInstrs[i];
      if (!ST.getInstrInfo()->isZeroCost(SU->getInstr()->getOpcode()))
        orderDependence(SSD, SU, newOrderI);
    }
    // Replace the old order with the new order.
    cycleInstrs.swap(newOrderZC);
    cycleInstrs.insert(cycleInstrs.end(), newOrderI.begin(), newOrderI.end());
    SSD->fixupRegisterOverlaps(cycleInstrs);
  }

  DEBUG(dump(););
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
/// Print the schedule information to the given output.
void SMSchedule::print(raw_ostream &os) const {
  // Iterate over each cycle.
  for (int cycle = getFirstCycle(); cycle <= getFinalCycle(); ++cycle) {
    // Iterate over each instruction in the cycle.
    const_sched_iterator cycleInstrs = ScheduledInstrs.find(cycle);
    for (SUnit *CI : cycleInstrs->second) {
      os << "cycle " << cycle << " (" << stageScheduled(CI) << ") ";
      os << "(" << CI->NodeNum << ") ";
      CI->getInstr()->print(os);
      os << "\n";
    }
  }
}

/// Utility function used for debugging to print the schedule.
LLVM_DUMP_METHOD void SMSchedule::dump() const { print(dbgs()); }
#endif
