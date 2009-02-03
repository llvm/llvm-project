//===---- ScheduleDAG.cpp - Implement the ScheduleDAG class ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements the ScheduleDAG class, which is a base class used by
// scheduling implementation classes.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "pre-RA-sched"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/ScheduleHazardRecognizer.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include <climits>
using namespace llvm;

ScheduleDAG::ScheduleDAG(MachineFunction &mf)
  : DAG(0), BB(0), TM(mf.getTarget()),
    TII(TM.getInstrInfo()),
    TRI(TM.getRegisterInfo()),
    TLI(TM.getTargetLowering()),
    MF(mf), MRI(mf.getRegInfo()),
    ConstPool(MF.getConstantPool()) {
}

ScheduleDAG::~ScheduleDAG() {}

/// dump - dump the schedule.
void ScheduleDAG::dumpSchedule() const {
  for (unsigned i = 0, e = Sequence.size(); i != e; i++) {
    if (SUnit *SU = Sequence[i])
      SU->dump(this);
    else
      cerr << "**** NOOP ****\n";
  }
}


/// Run - perform scheduling.
///
void ScheduleDAG::Run(SelectionDAG *dag, MachineBasicBlock *bb,
                      MachineBasicBlock::iterator begin,
                      MachineBasicBlock::iterator end) {
  assert((!dag || begin == end) &&
         "An instruction range was given for SelectionDAG scheduling!");

  SUnits.clear();
  Sequence.clear();
  DAG = dag;
  BB = bb;
  Begin = begin;
  End = end;

  Schedule();
  
  DOUT << "*** Final schedule ***\n";
  DEBUG(dumpSchedule());
  DOUT << "\n";
}

/// addPred - This adds the specified edge as a pred of the current node if
/// not already.  It also adds the current node as a successor of the
/// specified node.
void SUnit::addPred(const SDep &D) {
  // If this node already has this depenence, don't add a redundant one.
  for (unsigned i = 0, e = (unsigned)Preds.size(); i != e; ++i)
    if (Preds[i] == D)
      return;
  // Now add a corresponding succ to N.
  SDep P = D;
  P.setSUnit(this);
  SUnit *N = D.getSUnit();
  // Update the bookkeeping.
  if (D.getKind() == SDep::Data) {
    ++NumPreds;
    ++N->NumSuccs;
  }
  if (!N->isScheduled)
    ++NumPredsLeft;
  if (!isScheduled)
    ++N->NumSuccsLeft;
  Preds.push_back(D);
  N->Succs.push_back(P);
  if (P.getLatency() != 0) {
    this->setDepthDirty();
    N->setHeightDirty();
  }
}

/// removePred - This removes the specified edge as a pred of the current
/// node if it exists.  It also removes the current node as a successor of
/// the specified node.
void SUnit::removePred(const SDep &D) {
  // Find the matching predecessor.
  for (SmallVector<SDep, 4>::iterator I = Preds.begin(), E = Preds.end();
       I != E; ++I)
    if (*I == D) {
      bool FoundSucc = false;
      // Find the corresponding successor in N.
      SDep P = D;
      P.setSUnit(this);
      SUnit *N = D.getSUnit();
      for (SmallVector<SDep, 4>::iterator II = N->Succs.begin(),
             EE = N->Succs.end(); II != EE; ++II)
        if (*II == P) {
          FoundSucc = true;
          N->Succs.erase(II);
          break;
        }
      assert(FoundSucc && "Mismatching preds / succs lists!");
      Preds.erase(I);
      // Update the bookkeeping.
      if (P.getKind() == SDep::Data) {
        --NumPreds;
        --N->NumSuccs;
      }
      if (!N->isScheduled)
        --NumPredsLeft;
      if (!isScheduled)
        --N->NumSuccsLeft;
      if (P.getLatency() != 0) {
        this->setDepthDirty();
        N->setHeightDirty();
      }
      return;
    }
}

void SUnit::setDepthDirty() {
  if (!isDepthCurrent) return;
  SmallVector<SUnit*, 8> WorkList;
  WorkList.push_back(this);
  do {
    SUnit *SU = WorkList.pop_back_val();
    SU->isDepthCurrent = false;
    for (SUnit::const_succ_iterator I = SU->Succs.begin(),
         E = SU->Succs.end(); I != E; ++I) {
      SUnit *SuccSU = I->getSUnit();
      if (SuccSU->isDepthCurrent)
        WorkList.push_back(SuccSU);
    }
  } while (!WorkList.empty());
}

void SUnit::setHeightDirty() {
  if (!isHeightCurrent) return;
  SmallVector<SUnit*, 8> WorkList;
  WorkList.push_back(this);
  do {
    SUnit *SU = WorkList.pop_back_val();
    SU->isHeightCurrent = false;
    for (SUnit::const_pred_iterator I = SU->Preds.begin(),
         E = SU->Preds.end(); I != E; ++I) {
      SUnit *PredSU = I->getSUnit();
      if (PredSU->isHeightCurrent)
        WorkList.push_back(PredSU);
    }
  } while (!WorkList.empty());
}

/// setDepthToAtLeast - Update this node's successors to reflect the
/// fact that this node's depth just increased.
///
void SUnit::setDepthToAtLeast(unsigned NewDepth) {
  if (NewDepth <= getDepth())
    return;
  setDepthDirty();
  Depth = NewDepth;
  isDepthCurrent = true;
}

/// setHeightToAtLeast - Update this node's predecessors to reflect the
/// fact that this node's height just increased.
///
void SUnit::setHeightToAtLeast(unsigned NewHeight) {
  if (NewHeight <= getHeight())
    return;
  setHeightDirty();
  Height = NewHeight;
  isHeightCurrent = true;
}

/// ComputeDepth - Calculate the maximal path from the node to the exit.
///
void SUnit::ComputeDepth() {
  SmallVector<SUnit*, 8> WorkList;
  WorkList.push_back(this);
  do {
    SUnit *Cur = WorkList.back();

    bool Done = true;
    unsigned MaxPredDepth = 0;
    for (SUnit::const_pred_iterator I = Cur->Preds.begin(),
         E = Cur->Preds.end(); I != E; ++I) {
      SUnit *PredSU = I->getSUnit();
      if (PredSU->isDepthCurrent)
        MaxPredDepth = std::max(MaxPredDepth,
                                PredSU->Depth + I->getLatency());
      else {
        Done = false;
        WorkList.push_back(PredSU);
      }
    }

    if (Done) {
      WorkList.pop_back();
      if (MaxPredDepth != Cur->Depth) {
        Cur->setDepthDirty();
        Cur->Depth = MaxPredDepth;
      }
      Cur->isDepthCurrent = true;
    }
  } while (!WorkList.empty());
}

/// ComputeHeight - Calculate the maximal path from the node to the entry.
///
void SUnit::ComputeHeight() {
  SmallVector<SUnit*, 8> WorkList;
  WorkList.push_back(this);
  do {
    SUnit *Cur = WorkList.back();

    bool Done = true;
    unsigned MaxSuccHeight = 0;
    for (SUnit::const_succ_iterator I = Cur->Succs.begin(),
         E = Cur->Succs.end(); I != E; ++I) {
      SUnit *SuccSU = I->getSUnit();
      if (SuccSU->isHeightCurrent)
        MaxSuccHeight = std::max(MaxSuccHeight,
                                 SuccSU->Height + I->getLatency());
      else {
        Done = false;
        WorkList.push_back(SuccSU);
      }
    }

    if (Done) {
      WorkList.pop_back();
      if (MaxSuccHeight != Cur->Height) {
        Cur->setHeightDirty();
        Cur->Height = MaxSuccHeight;
      }
      Cur->isHeightCurrent = true;
    }
  } while (!WorkList.empty());
}

/// SUnit - Scheduling unit. It's an wrapper around either a single SDNode or
/// a group of nodes flagged together.
void SUnit::dump(const ScheduleDAG *G) const {
  cerr << "SU(" << NodeNum << "): ";
  G->dumpNode(this);
}

void SUnit::dumpAll(const ScheduleDAG *G) const {
  dump(G);

  cerr << "  # preds left       : " << NumPredsLeft << "\n";
  cerr << "  # succs left       : " << NumSuccsLeft << "\n";
  cerr << "  Latency            : " << Latency << "\n";
  cerr << "  Depth              : " << Depth << "\n";
  cerr << "  Height             : " << Height << "\n";

  if (Preds.size() != 0) {
    cerr << "  Predecessors:\n";
    for (SUnit::const_succ_iterator I = Preds.begin(), E = Preds.end();
         I != E; ++I) {
      cerr << "   ";
      switch (I->getKind()) {
      case SDep::Data:        cerr << "val "; break;
      case SDep::Anti:        cerr << "anti"; break;
      case SDep::Output:      cerr << "out "; break;
      case SDep::Order:       cerr << "ch  "; break;
      }
      cerr << "#";
      cerr << I->getSUnit() << " - SU(" << I->getSUnit()->NodeNum << ")";
      if (I->isArtificial())
        cerr << " *";
      cerr << "\n";
    }
  }
  if (Succs.size() != 0) {
    cerr << "  Successors:\n";
    for (SUnit::const_succ_iterator I = Succs.begin(), E = Succs.end();
         I != E; ++I) {
      cerr << "   ";
      switch (I->getKind()) {
      case SDep::Data:        cerr << "val "; break;
      case SDep::Anti:        cerr << "anti"; break;
      case SDep::Output:      cerr << "out "; break;
      case SDep::Order:       cerr << "ch  "; break;
      }
      cerr << "#";
      cerr << I->getSUnit() << " - SU(" << I->getSUnit()->NodeNum << ")";
      if (I->isArtificial())
        cerr << " *";
      cerr << "\n";
    }
  }
  cerr << "\n";
}

#ifndef NDEBUG
/// VerifySchedule - Verify that all SUnits were scheduled and that
/// their state is consistent.
///
void ScheduleDAG::VerifySchedule(bool isBottomUp) {
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
        cerr << "*** Scheduling failed! ***\n";
      SUnits[i].dump(this);
      cerr << "has not been scheduled!\n";
      AnyNotSched = true;
    }
    if (SUnits[i].isScheduled &&
        (isBottomUp ? SUnits[i].getHeight() : SUnits[i].getHeight()) >
          unsigned(INT_MAX)) {
      if (!AnyNotSched)
        cerr << "*** Scheduling failed! ***\n";
      SUnits[i].dump(this);
      cerr << "has an unexpected "
           << (isBottomUp ? "Height" : "Depth") << " value!\n";
      AnyNotSched = true;
    }
    if (isBottomUp) {
      if (SUnits[i].NumSuccsLeft != 0) {
        if (!AnyNotSched)
          cerr << "*** Scheduling failed! ***\n";
        SUnits[i].dump(this);
        cerr << "has successors left!\n";
        AnyNotSched = true;
      }
    } else {
      if (SUnits[i].NumPredsLeft != 0) {
        if (!AnyNotSched)
          cerr << "*** Scheduling failed! ***\n";
        SUnits[i].dump(this);
        cerr << "has predecessors left!\n";
        AnyNotSched = true;
      }
    }
  }
  for (unsigned i = 0, e = Sequence.size(); i != e; ++i)
    if (!Sequence[i])
      ++Noops;
  assert(!AnyNotSched);
  assert(Sequence.size() + DeadNodes - Noops == SUnits.size() &&
         "The number of nodes scheduled doesn't match the expected number!");
}
#endif

/// InitDAGTopologicalSorting - create the initial topological 
/// ordering from the DAG to be scheduled.
///
/// The idea of the algorithm is taken from 
/// "Online algorithms for managing the topological order of
/// a directed acyclic graph" by David J. Pearce and Paul H.J. Kelly
/// This is the MNR algorithm, which was first introduced by 
/// A. Marchetti-Spaccamela, U. Nanni and H. Rohnert in  
/// "Maintaining a topological order under edge insertions".
///
/// Short description of the algorithm: 
///
/// Topological ordering, ord, of a DAG maps each node to a topological
/// index so that for all edges X->Y it is the case that ord(X) < ord(Y).
///
/// This means that if there is a path from the node X to the node Z, 
/// then ord(X) < ord(Z).
///
/// This property can be used to check for reachability of nodes:
/// if Z is reachable from X, then an insertion of the edge Z->X would 
/// create a cycle.
///
/// The algorithm first computes a topological ordering for the DAG by
/// initializing the Index2Node and Node2Index arrays and then tries to keep
/// the ordering up-to-date after edge insertions by reordering the DAG.
///
/// On insertion of the edge X->Y, the algorithm first marks by calling DFS
/// the nodes reachable from Y, and then shifts them using Shift to lie
/// immediately after X in Index2Node.
void ScheduleDAGTopologicalSort::InitDAGTopologicalSorting() {
  unsigned DAGSize = SUnits.size();
  std::vector<SUnit*> WorkList;
  WorkList.reserve(DAGSize);

  Index2Node.resize(DAGSize);
  Node2Index.resize(DAGSize);

  // Initialize the data structures.
  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &SUnits[i];
    int NodeNum = SU->NodeNum;
    unsigned Degree = SU->Succs.size();
    // Temporarily use the Node2Index array as scratch space for degree counts.
    Node2Index[NodeNum] = Degree;

    // Is it a node without dependencies?
    if (Degree == 0) {
      assert(SU->Succs.empty() && "SUnit should have no successors");
      // Collect leaf nodes.
      WorkList.push_back(SU);
    }
  }  

  int Id = DAGSize;
  while (!WorkList.empty()) {
    SUnit *SU = WorkList.back();
    WorkList.pop_back();
    Allocate(SU->NodeNum, --Id);
    for (SUnit::const_pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
         I != E; ++I) {
      SUnit *SU = I->getSUnit();
      if (!--Node2Index[SU->NodeNum])
        // If all dependencies of the node are processed already,
        // then the node can be computed now.
        WorkList.push_back(SU);
    }
  }

  Visited.resize(DAGSize);

#ifndef NDEBUG
  // Check correctness of the ordering
  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &SUnits[i];
    for (SUnit::const_pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
         I != E; ++I) {
      assert(Node2Index[SU->NodeNum] > Node2Index[I->getSUnit()->NodeNum] && 
      "Wrong topological sorting");
    }
  }
#endif
}

/// AddPred - Updates the topological ordering to accomodate an edge
/// to be added from SUnit X to SUnit Y.
void ScheduleDAGTopologicalSort::AddPred(SUnit *Y, SUnit *X) {
  int UpperBound, LowerBound;
  LowerBound = Node2Index[Y->NodeNum];
  UpperBound = Node2Index[X->NodeNum];
  bool HasLoop = false;
  // Is Ord(X) < Ord(Y) ?
  if (LowerBound < UpperBound) {
    // Update the topological order.
    Visited.reset();
    DFS(Y, UpperBound, HasLoop);
    assert(!HasLoop && "Inserted edge creates a loop!");
    // Recompute topological indexes.
    Shift(Visited, LowerBound, UpperBound);
  }
}

/// RemovePred - Updates the topological ordering to accomodate an
/// an edge to be removed from the specified node N from the predecessors
/// of the current node M.
void ScheduleDAGTopologicalSort::RemovePred(SUnit *M, SUnit *N) {
  // InitDAGTopologicalSorting();
}

/// DFS - Make a DFS traversal to mark all nodes reachable from SU and mark
/// all nodes affected by the edge insertion. These nodes will later get new
/// topological indexes by means of the Shift method.
void ScheduleDAGTopologicalSort::DFS(const SUnit *SU, int UpperBound,
                                     bool& HasLoop) {
  std::vector<const SUnit*> WorkList;
  WorkList.reserve(SUnits.size()); 

  WorkList.push_back(SU);
  do {
    SU = WorkList.back();
    WorkList.pop_back();
    Visited.set(SU->NodeNum);
    for (int I = SU->Succs.size()-1; I >= 0; --I) {
      int s = SU->Succs[I].getSUnit()->NodeNum;
      if (Node2Index[s] == UpperBound) {
        HasLoop = true; 
        return;
      }
      // Visit successors if not already and in affected region.
      if (!Visited.test(s) && Node2Index[s] < UpperBound) {
        WorkList.push_back(SU->Succs[I].getSUnit());
      } 
    } 
  } while (!WorkList.empty());
}

/// Shift - Renumber the nodes so that the topological ordering is 
/// preserved.
void ScheduleDAGTopologicalSort::Shift(BitVector& Visited, int LowerBound, 
                                       int UpperBound) {
  std::vector<int> L;
  int shift = 0;
  int i;

  for (i = LowerBound; i <= UpperBound; ++i) {
    // w is node at topological index i.
    int w = Index2Node[i];
    if (Visited.test(w)) {
      // Unmark.
      Visited.reset(w);
      L.push_back(w);
      shift = shift + 1;
    } else {
      Allocate(w, i - shift);
    }
  }

  for (unsigned j = 0; j < L.size(); ++j) {
    Allocate(L[j], i - shift);
    i = i + 1;
  }
}


/// WillCreateCycle - Returns true if adding an edge from SU to TargetSU will
/// create a cycle.
bool ScheduleDAGTopologicalSort::WillCreateCycle(SUnit *SU, SUnit *TargetSU) {
  if (IsReachable(TargetSU, SU))
    return true;
  for (SUnit::pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
       I != E; ++I)
    if (I->isAssignedRegDep() &&
        IsReachable(TargetSU, I->getSUnit()))
      return true;
  return false;
}

/// IsReachable - Checks if SU is reachable from TargetSU.
bool ScheduleDAGTopologicalSort::IsReachable(const SUnit *SU,
                                             const SUnit *TargetSU) {
  // If insertion of the edge SU->TargetSU would create a cycle
  // then there is a path from TargetSU to SU.
  int UpperBound, LowerBound;
  LowerBound = Node2Index[TargetSU->NodeNum];
  UpperBound = Node2Index[SU->NodeNum];
  bool HasLoop = false;
  // Is Ord(TargetSU) < Ord(SU) ?
  if (LowerBound < UpperBound) {
    Visited.reset();
    // There may be a path from TargetSU to SU. Check for it. 
    DFS(TargetSU, UpperBound, HasLoop);
  }
  return HasLoop;
}

/// Allocate - assign the topological index to the node n.
void ScheduleDAGTopologicalSort::Allocate(int n, int index) {
  Node2Index[n] = index;
  Index2Node[index] = n;
}

ScheduleDAGTopologicalSort::ScheduleDAGTopologicalSort(
                                                     std::vector<SUnit> &sunits)
 : SUnits(sunits) {}

ScheduleHazardRecognizer::~ScheduleHazardRecognizer() {}
