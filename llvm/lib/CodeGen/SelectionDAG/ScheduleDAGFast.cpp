//===----- ScheduleDAGFast.cpp - Fast poor list scheduler -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements a fast scheduler.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "pre-RA-sched"
#include "llvm/CodeGen/ScheduleDAGSDNodes.h"
#include "llvm/CodeGen/SchedulerRegistry.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Compiler.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
using namespace llvm;

STATISTIC(NumUnfolds,    "Number of nodes unfolded");
STATISTIC(NumDups,       "Number of duplicated nodes");
STATISTIC(NumPRCopies,   "Number of physical copies");

static RegisterScheduler
  fastDAGScheduler("fast", "Fast suboptimal list scheduling",
                   createFastDAGScheduler);

namespace {
  /// FastPriorityQueue - A degenerate priority queue that considers
  /// all nodes to have the same priority.
  ///
  struct VISIBILITY_HIDDEN FastPriorityQueue {
    SmallVector<SUnit *, 16> Queue;

    bool empty() const { return Queue.empty(); }
    
    void push(SUnit *U) {
      Queue.push_back(U);
    }

    SUnit *pop() {
      if (empty()) return NULL;
      SUnit *V = Queue.back();
      Queue.pop_back();
      return V;
    }
  };

//===----------------------------------------------------------------------===//
/// ScheduleDAGFast - The actual "fast" list scheduler implementation.
///
class VISIBILITY_HIDDEN ScheduleDAGFast : public ScheduleDAGSDNodes {
private:
  /// AvailableQueue - The priority queue to use for the available SUnits.
  FastPriorityQueue AvailableQueue;

  /// LiveRegDefs - A set of physical registers and their definition
  /// that are "live". These nodes must be scheduled before any other nodes that
  /// modifies the registers can be scheduled.
  unsigned NumLiveRegs;
  std::vector<SUnit*> LiveRegDefs;
  std::vector<unsigned> LiveRegCycles;

public:
  ScheduleDAGFast(MachineFunction &mf)
    : ScheduleDAGSDNodes(mf) {}

  void Schedule();

  /// AddPred - adds a predecessor edge to SUnit SU.
  /// This returns true if this is a new predecessor.
  void AddPred(SUnit *SU, const SDep &D) {
    SU->addPred(D);
  }

  /// RemovePred - removes a predecessor edge from SUnit SU.
  /// This returns true if an edge was removed.
  void RemovePred(SUnit *SU, const SDep &D) {
    SU->removePred(D);
  }

private:
  void ReleasePred(SUnit *SU, SDep *PredEdge);
  void ScheduleNodeBottomUp(SUnit*, unsigned);
  SUnit *CopyAndMoveSuccessors(SUnit*);
  void InsertCopiesAndMoveSuccs(SUnit*, unsigned,
                                const TargetRegisterClass*,
                                const TargetRegisterClass*,
                                SmallVector<SUnit*, 2>&);
  bool DelayForLiveRegsBottomUp(SUnit*, SmallVector<unsigned, 4>&);
  void ListScheduleBottomUp();

  /// ForceUnitLatencies - The fast scheduler doesn't care about real latencies.
  bool ForceUnitLatencies() const { return true; }
};
}  // end anonymous namespace


/// Schedule - Schedule the DAG using list scheduling.
void ScheduleDAGFast::Schedule() {
  DOUT << "********** List Scheduling **********\n";

  NumLiveRegs = 0;
  LiveRegDefs.resize(TRI->getNumRegs(), NULL);  
  LiveRegCycles.resize(TRI->getNumRegs(), 0);

  // Build the scheduling graph.
  BuildSchedGraph();

  DEBUG(for (unsigned su = 0, e = SUnits.size(); su != e; ++su)
          SUnits[su].dumpAll(this));

  // Execute the actual scheduling loop.
  ListScheduleBottomUp();
}

//===----------------------------------------------------------------------===//
//  Bottom-Up Scheduling
//===----------------------------------------------------------------------===//

/// ReleasePred - Decrement the NumSuccsLeft count of a predecessor. Add it to
/// the AvailableQueue if the count reaches zero. Also update its cycle bound.
void ScheduleDAGFast::ReleasePred(SUnit *SU, SDep *PredEdge) {
  SUnit *PredSU = PredEdge->getSUnit();
  --PredSU->NumSuccsLeft;
  
#ifndef NDEBUG
  if (PredSU->NumSuccsLeft < 0) {
    cerr << "*** Scheduling failed! ***\n";
    PredSU->dump(this);
    cerr << " has been released too many times!\n";
    assert(0);
  }
#endif
  
  if (PredSU->NumSuccsLeft == 0) {
    PredSU->isAvailable = true;
    AvailableQueue.push(PredSU);
  }
}

/// ScheduleNodeBottomUp - Add the node to the schedule. Decrement the pending
/// count of its predecessors. If a predecessor pending count is zero, add it to
/// the Available queue.
void ScheduleDAGFast::ScheduleNodeBottomUp(SUnit *SU, unsigned CurCycle) {
  DOUT << "*** Scheduling [" << CurCycle << "]: ";
  DEBUG(SU->dump(this));

  assert(CurCycle >= SU->getHeight() && "Node scheduled below its height!");
  SU->setHeightToAtLeast(CurCycle);
  Sequence.push_back(SU);

  // Bottom up: release predecessors
  for (SUnit::pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
       I != E; ++I) {
    ReleasePred(SU, &*I);
    if (I->isAssignedRegDep()) {
      // This is a physical register dependency and it's impossible or
      // expensive to copy the register. Make sure nothing that can 
      // clobber the register is scheduled between the predecessor and
      // this node.
      if (!LiveRegDefs[I->getReg()]) {
        ++NumLiveRegs;
        LiveRegDefs[I->getReg()] = I->getSUnit();
        LiveRegCycles[I->getReg()] = CurCycle;
      }
    }
  }

  // Release all the implicit physical register defs that are live.
  for (SUnit::succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I) {
    if (I->isAssignedRegDep()) {
      if (LiveRegCycles[I->getReg()] == I->getSUnit()->getHeight()) {
        assert(NumLiveRegs > 0 && "NumLiveRegs is already zero!");
        assert(LiveRegDefs[I->getReg()] == SU &&
               "Physical register dependency violated?");
        --NumLiveRegs;
        LiveRegDefs[I->getReg()] = NULL;
        LiveRegCycles[I->getReg()] = 0;
      }
    }
  }

  SU->isScheduled = true;
}

/// CopyAndMoveSuccessors - Clone the specified node and move its scheduled
/// successors to the newly created node.
SUnit *ScheduleDAGFast::CopyAndMoveSuccessors(SUnit *SU) {
  if (SU->getNode()->getFlaggedNode())
    return NULL;

  SDNode *N = SU->getNode();
  if (!N)
    return NULL;

  SUnit *NewSU;
  bool TryUnfold = false;
  for (unsigned i = 0, e = N->getNumValues(); i != e; ++i) {
    MVT VT = N->getValueType(i);
    if (VT == MVT::Flag)
      return NULL;
    else if (VT == MVT::Other)
      TryUnfold = true;
  }
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
    const SDValue &Op = N->getOperand(i);
    MVT VT = Op.getNode()->getValueType(Op.getResNo());
    if (VT == MVT::Flag)
      return NULL;
  }

  if (TryUnfold) {
    SmallVector<SDNode*, 2> NewNodes;
    if (!TII->unfoldMemoryOperand(*DAG, N, NewNodes))
      return NULL;

    DOUT << "Unfolding SU # " << SU->NodeNum << "\n";
    assert(NewNodes.size() == 2 && "Expected a load folding node!");

    N = NewNodes[1];
    SDNode *LoadNode = NewNodes[0];
    unsigned NumVals = N->getNumValues();
    unsigned OldNumVals = SU->getNode()->getNumValues();
    for (unsigned i = 0; i != NumVals; ++i)
      DAG->ReplaceAllUsesOfValueWith(SDValue(SU->getNode(), i), SDValue(N, i));
    DAG->ReplaceAllUsesOfValueWith(SDValue(SU->getNode(), OldNumVals-1),
                                   SDValue(LoadNode, 1));

    SUnit *NewSU = NewSUnit(N);
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

    // LoadNode may already exist. This can happen when there is another
    // load from the same location and producing the same type of value
    // but it has different alignment or volatileness.
    bool isNewLoad = true;
    SUnit *LoadSU;
    if (LoadNode->getNodeId() != -1) {
      LoadSU = &SUnits[LoadNode->getNodeId()];
      isNewLoad = false;
    } else {
      LoadSU = NewSUnit(LoadNode);
      LoadNode->setNodeId(LoadSU->NodeNum);
    }

    SDep ChainPred;
    SmallVector<SDep, 4> ChainSuccs;
    SmallVector<SDep, 4> LoadPreds;
    SmallVector<SDep, 4> NodePreds;
    SmallVector<SDep, 4> NodeSuccs;
    for (SUnit::pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
         I != E; ++I) {
      if (I->isCtrl())
        ChainPred = *I;
      else if (I->getSUnit()->getNode() &&
               I->getSUnit()->getNode()->isOperandOf(LoadNode))
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

    if (ChainPred.getSUnit()) {
      RemovePred(SU, ChainPred);
      if (isNewLoad)
        AddPred(LoadSU, ChainPred);
    }
    for (unsigned i = 0, e = LoadPreds.size(); i != e; ++i) {
      const SDep &Pred = LoadPreds[i];
      RemovePred(SU, Pred);
      if (isNewLoad) {
        AddPred(LoadSU, Pred);
      }
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
    if (isNewLoad) {
      AddPred(NewSU, SDep(LoadSU, SDep::Order, LoadSU->Latency));
    }

    ++NumUnfolds;

    if (NewSU->NumSuccsLeft == 0) {
      NewSU->isAvailable = true;
      return NewSU;
    }
    SU = NewSU;
  }

  DOUT << "Duplicating SU # " << SU->NodeNum << "\n";
  NewSU = Clone(SU);

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

  ++NumDups;
  return NewSU;
}

/// InsertCopiesAndMoveSuccs - Insert register copies and move all
/// scheduled successors of the given SUnit to the last copy.
void ScheduleDAGFast::InsertCopiesAndMoveSuccs(SUnit *SU, unsigned Reg,
                                              const TargetRegisterClass *DestRC,
                                              const TargetRegisterClass *SrcRC,
                                               SmallVector<SUnit*, 2> &Copies) {
  SUnit *CopyFromSU = NewSUnit(static_cast<SDNode *>(NULL));
  CopyFromSU->CopySrcRC = SrcRC;
  CopyFromSU->CopyDstRC = DestRC;

  SUnit *CopyToSU = NewSUnit(static_cast<SDNode *>(NULL));
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
  for (unsigned i = 0, e = DelDeps.size(); i != e; ++i) {
    RemovePred(DelDeps[i].first, DelDeps[i].second);
  }

  AddPred(CopyFromSU, SDep(SU, SDep::Data, SU->Latency, Reg));
  AddPred(CopyToSU, SDep(CopyFromSU, SDep::Data, CopyFromSU->Latency, 0));

  Copies.push_back(CopyFromSU);
  Copies.push_back(CopyToSU);

  ++NumPRCopies;
}

/// getPhysicalRegisterVT - Returns the ValueType of the physical register
/// definition of the specified node.
/// FIXME: Move to SelectionDAG?
static MVT getPhysicalRegisterVT(SDNode *N, unsigned Reg,
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

/// DelayForLiveRegsBottomUp - Returns true if it is necessary to delay
/// scheduling of the given node to satisfy live physical register dependencies.
/// If the specific node is the last one that's available to schedule, do
/// whatever is necessary (i.e. backtracking or cloning) to make it possible.
bool ScheduleDAGFast::DelayForLiveRegsBottomUp(SUnit *SU,
                                               SmallVector<unsigned, 4> &LRegs){
  if (NumLiveRegs == 0)
    return false;

  SmallSet<unsigned, 4> RegAdded;
  // If this node would clobber any "live" register, then it's not ready.
  for (SUnit::pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
       I != E; ++I) {
    if (I->isAssignedRegDep()) {
      unsigned Reg = I->getReg();
      if (LiveRegDefs[Reg] && LiveRegDefs[Reg] != I->getSUnit()) {
        if (RegAdded.insert(Reg))
          LRegs.push_back(Reg);
      }
      for (const unsigned *Alias = TRI->getAliasSet(Reg);
           *Alias; ++Alias)
        if (LiveRegDefs[*Alias] && LiveRegDefs[*Alias] != I->getSUnit()) {
          if (RegAdded.insert(*Alias))
            LRegs.push_back(*Alias);
        }
    }
  }

  for (SDNode *Node = SU->getNode(); Node; Node = Node->getFlaggedNode()) {
    if (!Node->isMachineOpcode())
      continue;
    const TargetInstrDesc &TID = TII->get(Node->getMachineOpcode());
    if (!TID.ImplicitDefs)
      continue;
    for (const unsigned *Reg = TID.ImplicitDefs; *Reg; ++Reg) {
      if (LiveRegDefs[*Reg] && LiveRegDefs[*Reg] != SU) {
        if (RegAdded.insert(*Reg))
          LRegs.push_back(*Reg);
      }
      for (const unsigned *Alias = TRI->getAliasSet(*Reg);
           *Alias; ++Alias)
        if (LiveRegDefs[*Alias] && LiveRegDefs[*Alias] != SU) {
          if (RegAdded.insert(*Alias))
            LRegs.push_back(*Alias);
        }
    }
  }
  return !LRegs.empty();
}


/// ListScheduleBottomUp - The main loop of list scheduling for bottom-up
/// schedulers.
void ScheduleDAGFast::ListScheduleBottomUp() {
  unsigned CurCycle = 0;
  // Add root to Available queue.
  if (!SUnits.empty()) {
    SUnit *RootSU = &SUnits[DAG->getRoot().getNode()->getNodeId()];
    assert(RootSU->Succs.empty() && "Graph root shouldn't have successors!");
    RootSU->isAvailable = true;
    AvailableQueue.push(RootSU);
  }

  // While Available queue is not empty, grab the node with the highest
  // priority. If it is not ready put it back.  Schedule the node.
  SmallVector<SUnit*, 4> NotReady;
  DenseMap<SUnit*, SmallVector<unsigned, 4> > LRegsMap;
  Sequence.reserve(SUnits.size());
  while (!AvailableQueue.empty()) {
    bool Delayed = false;
    LRegsMap.clear();
    SUnit *CurSU = AvailableQueue.pop();
    while (CurSU) {
      SmallVector<unsigned, 4> LRegs;
      if (!DelayForLiveRegsBottomUp(CurSU, LRegs))
        break;
      Delayed = true;
      LRegsMap.insert(std::make_pair(CurSU, LRegs));

      CurSU->isPending = true;  // This SU is not in AvailableQueue right now.
      NotReady.push_back(CurSU);
      CurSU = AvailableQueue.pop();
    }

    // All candidates are delayed due to live physical reg dependencies.
    // Try code duplication or inserting cross class copies
    // to resolve it.
    if (Delayed && !CurSU) {
      if (!CurSU) {
        // Try duplicating the nodes that produces these
        // "expensive to copy" values to break the dependency. In case even
        // that doesn't work, insert cross class copies.
        SUnit *TrySU = NotReady[0];
        SmallVector<unsigned, 4> &LRegs = LRegsMap[TrySU];
        assert(LRegs.size() == 1 && "Can't handle this yet!");
        unsigned Reg = LRegs[0];
        SUnit *LRDef = LiveRegDefs[Reg];
        MVT VT = getPhysicalRegisterVT(LRDef->getNode(), Reg, TII);
        const TargetRegisterClass *RC =
          TRI->getPhysicalRegisterRegClass(Reg, VT);
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
          DOUT << "Adding an edge from SU # " << TrySU->NodeNum
               << " to SU #" << Copies.front()->NodeNum << "\n";
          AddPred(TrySU, SDep(Copies.front(), SDep::Order, /*Latency=*/1,
                              /*Reg=*/0, /*isNormalMemory=*/false,
                              /*isMustAlias=*/false, /*isArtificial=*/true));
          NewDef = Copies.back();
        }

        DOUT << "Adding an edge from SU # " << NewDef->NodeNum
             << " to SU #" << TrySU->NodeNum << "\n";
        LiveRegDefs[Reg] = NewDef;
        AddPred(NewDef, SDep(TrySU, SDep::Order, /*Latency=*/1,
                             /*Reg=*/0, /*isNormalMemory=*/false,
                             /*isMustAlias=*/false, /*isArtificial=*/true));
        TrySU->isAvailable = false;
        CurSU = NewDef;
      }

      if (!CurSU) {
        assert(false && "Unable to resolve live physical register dependencies!");
        abort();
      }
    }

    // Add the nodes that aren't ready back onto the available list.
    for (unsigned i = 0, e = NotReady.size(); i != e; ++i) {
      NotReady[i]->isPending = false;
      // May no longer be available due to backtracking.
      if (NotReady[i]->isAvailable)
        AvailableQueue.push(NotReady[i]);
    }
    NotReady.clear();

    if (CurSU)
      ScheduleNodeBottomUp(CurSU, CurCycle);
    ++CurCycle;
  }

  // Reverse the order if it is bottom up.
  std::reverse(Sequence.begin(), Sequence.end());
  
  
#ifndef NDEBUG
  // Verify that all SUnits were scheduled.
  bool AnyNotSched = false;
  unsigned DeadNodes = 0;
  unsigned Noops = 0;
  for (unsigned i = 0, e = SUnits.size(); i != e; ++i) {
    if (!SUnits[i].isScheduled) {
      if (SUnits[i].NumPreds == 0 && SUnits[i].NumSuccs == 0) {
        ++DeadNodes;
        continue;
      }
      if (!AnyNotSched)
        cerr << "*** List scheduling failed! ***\n";
      SUnits[i].dump(this);
      cerr << "has not been scheduled!\n";
      AnyNotSched = true;
    }
    if (SUnits[i].NumSuccsLeft != 0) {
      if (!AnyNotSched)
        cerr << "*** List scheduling failed! ***\n";
      SUnits[i].dump(this);
      cerr << "has successors left!\n";
      AnyNotSched = true;
    }
  }
  for (unsigned i = 0, e = Sequence.size(); i != e; ++i)
    if (!Sequence[i])
      ++Noops;
  assert(!AnyNotSched);
  assert(Sequence.size() + DeadNodes - Noops == SUnits.size() &&
         "The number of nodes scheduled doesn't match the expected number!");
#endif
}

//===----------------------------------------------------------------------===//
//                         Public Constructor Functions
//===----------------------------------------------------------------------===//

llvm::ScheduleDAG* llvm::createFastDAGScheduler(SelectionDAGISel *IS, bool) {
  return new ScheduleDAGFast(*IS->MF);
}
