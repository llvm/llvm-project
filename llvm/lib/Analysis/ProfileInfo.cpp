//===- ProfileInfo.cpp - Profile Info Interface ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the abstract ProfileInfo interface, and the default
// "no profile" implementation.
//
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "profile-info"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/ProfileInfo.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Pass.h"
#include "llvm/Support/CFG.h"
#include "llvm/ADT/SmallSet.h"
#include <set>
#include <queue>
#include <limits>
using namespace llvm;

namespace llvm {
  template<> char ProfileInfoT<Function,BasicBlock>::ID = 0;
}

// Register the ProfileInfo interface, providing a nice name to refer to.
INITIALIZE_ANALYSIS_GROUP(ProfileInfo, "Profile Information", NoProfileInfo)

namespace llvm {

template <>
ProfileInfoT<MachineFunction, MachineBasicBlock>::ProfileInfoT() {}
template <>
ProfileInfoT<MachineFunction, MachineBasicBlock>::~ProfileInfoT() {}

template <>
ProfileInfoT<Function, BasicBlock>::ProfileInfoT() {
  MachineProfile = 0;
}
template <>
ProfileInfoT<Function, BasicBlock>::~ProfileInfoT() {
  if (MachineProfile) delete MachineProfile;
}

template<>
char ProfileInfoT<MachineFunction, MachineBasicBlock>::ID = 0;

template<>
const double ProfileInfoT<Function,BasicBlock>::MissingValue = -1;

template<> const
double ProfileInfoT<MachineFunction, MachineBasicBlock>::MissingValue = -1;

template<> double
ProfileInfoT<Function,BasicBlock>::getExecutionCount(const BasicBlock *BB) {
  std::map<const Function*, BlockCounts>::iterator J =
    BlockInformation.find(BB->getParent());
  if (J != BlockInformation.end()) {
    BlockCounts::iterator I = J->second.find(BB);
    if (I != J->second.end())
      return I->second;
  }

  double Count = MissingValue;

  const_pred_iterator PI = pred_begin(BB), PE = pred_end(BB);

  // Are there zero predecessors of this block?
  if (PI == PE) {
    Edge e = getEdge(0, BB);
    Count = getEdgeWeight(e);
  } else {
    // Otherwise, if there are predecessors, the execution count of this block is
    // the sum of the edge frequencies from the incoming edges.
    std::set<const BasicBlock*> ProcessedPreds;
    Count = 0;
    for (; PI != PE; ++PI) {
      const BasicBlock *P = *PI;
      if (ProcessedPreds.insert(P).second) {
        double w = getEdgeWeight(getEdge(P, BB));
        if (w == MissingValue) {
          Count = MissingValue;
          break;
        }
        Count += w;
      }
    }
  }

  // If the predecessors did not suffice to get block weight, try successors.
  if (Count == MissingValue) {

    succ_const_iterator SI = succ_begin(BB), SE = succ_end(BB);

    // Are there zero successors of this block?
    if (SI == SE) {
      Edge e = getEdge(BB,0);
      Count = getEdgeWeight(e);
    } else {
      std::set<const BasicBlock*> ProcessedSuccs;
      Count = 0;
      for (; SI != SE; ++SI)
        if (ProcessedSuccs.insert(*SI).second) {
          double w = getEdgeWeight(getEdge(BB, *SI));
          if (w == MissingValue) {
            Count = MissingValue;
            break;
          }
          Count += w;
        }
    }
  }

  if (Count != MissingValue) BlockInformation[BB->getParent()][BB] = Count;
  return Count;
}

template<>
double ProfileInfoT<MachineFunction, MachineBasicBlock>::
        getExecutionCount(const MachineBasicBlock *MBB) {
  std::map<const MachineFunction*, BlockCounts>::iterator J =
    BlockInformation.find(MBB->getParent());
  if (J != BlockInformation.end()) {
    BlockCounts::iterator I = J->second.find(MBB);
    if (I != J->second.end())
      return I->second;
  }

  return MissingValue;
}

template<>
double ProfileInfoT<Function,BasicBlock>::getExecutionCount(const Function *F) {
  std::map<const Function*, double>::iterator J =
    FunctionInformation.find(F);
  if (J != FunctionInformation.end())
    return J->second;

  // isDeclaration() is checked here and not at start of function to allow
  // functions without a body still to have a execution count.
  if (F->isDeclaration()) return MissingValue;

  double Count = getExecutionCount(&F->getEntryBlock());
  if (Count != MissingValue) FunctionInformation[F] = Count;
  return Count;
}

template<>
double ProfileInfoT<MachineFunction, MachineBasicBlock>::
        getExecutionCount(const MachineFunction *MF) {
  std::map<const MachineFunction*, double>::iterator J =
    FunctionInformation.find(MF);
  if (J != FunctionInformation.end())
    return J->second;

  double Count = getExecutionCount(&MF->front());
  if (Count != MissingValue) FunctionInformation[MF] = Count;
  return Count;
}

template<>
void ProfileInfoT<Function,BasicBlock>::
        setExecutionCount(const BasicBlock *BB, double w) {
  DEBUG(dbgs() << "Creating Block " << BB->getName() 
               << " (weight: " << format("%.20g",w) << ")\n");
  BlockInformation[BB->getParent()][BB] = w;
}

template<>
void ProfileInfoT<MachineFunction, MachineBasicBlock>::
        setExecutionCount(const MachineBasicBlock *MBB, double w) {
  DEBUG(dbgs() << "Creating Block " << MBB->getBasicBlock()->getName()
               << " (weight: " << format("%.20g",w) << ")\n");
  BlockInformation[MBB->getParent()][MBB] = w;
}

template<>
void ProfileInfoT<Function,BasicBlock>::addEdgeWeight(Edge e, double w) {
  double oldw = getEdgeWeight(e);
  assert (oldw != MissingValue && "Adding weight to Edge with no previous weight");
  DEBUG(dbgs() << "Adding to Edge " << e
               << " (new weight: " << format("%.20g",oldw + w) << ")\n");
  EdgeInformation[getFunction(e)][e] = oldw + w;
}

template<>
void ProfileInfoT<Function,BasicBlock>::
        addExecutionCount(const BasicBlock *BB, double w) {
  double oldw = getExecutionCount(BB);
  assert (oldw != MissingValue && "Adding weight to Block with no previous weight");
  DEBUG(dbgs() << "Adding to Block " << BB->getName()
               << " (new weight: " << format("%.20g",oldw + w) << ")\n");
  BlockInformation[BB->getParent()][BB] = oldw + w;
}

template<>
void ProfileInfoT<Function,BasicBlock>::removeBlock(const BasicBlock *BB) {
  std::map<const Function*, BlockCounts>::iterator J =
    BlockInformation.find(BB->getParent());
  if (J == BlockInformation.end()) return;

  DEBUG(dbgs() << "Deleting " << BB->getName() << "\n");
  J->second.erase(BB);
}

template<>
void ProfileInfoT<Function,BasicBlock>::removeEdge(Edge e) {
  std::map<const Function*, EdgeWeights>::iterator J =
    EdgeInformation.find(getFunction(e));
  if (J == EdgeInformation.end()) return;

  DEBUG(dbgs() << "Deleting" << e << "\n");
  J->second.erase(e);
}

template<>
void ProfileInfoT<Function,BasicBlock>::
        replaceEdge(const Edge &oldedge, const Edge &newedge) {
  double w;
  if ((w = getEdgeWeight(newedge)) == MissingValue) {
    w = getEdgeWeight(oldedge);
    DEBUG(dbgs() << "Replacing " << oldedge << " with " << newedge  << "\n");
  } else {
    w += getEdgeWeight(oldedge);
    DEBUG(dbgs() << "Adding " << oldedge << " to " << newedge  << "\n");
  }
  setEdgeWeight(newedge,w);
  removeEdge(oldedge);
}

template<>
const BasicBlock *ProfileInfoT<Function,BasicBlock>::
        GetPath(const BasicBlock *Src, const BasicBlock *Dest,
                Path &P, unsigned Mode) {
  const BasicBlock *BB = 0;
  bool hasFoundPath = false;

  std::queue<const BasicBlock *> BFS;
  BFS.push(Src);

  while(BFS.size() && !hasFoundPath) {
    BB = BFS.front();
    BFS.pop();

    succ_const_iterator Succ = succ_begin(BB), End = succ_end(BB);
    if (Succ == End) {
      P[0] = BB;
      if (Mode & GetPathToExit) {
        hasFoundPath = true;
        BB = 0;
      }
    }
    for(;Succ != End; ++Succ) {
      if (P.find(*Succ) != P.end()) continue;
      Edge e = getEdge(BB,*Succ);
      if ((Mode & GetPathWithNewEdges) && (getEdgeWeight(e) != MissingValue)) continue;
      P[*Succ] = BB;
      BFS.push(*Succ);
      if ((Mode & GetPathToDest) && *Succ == Dest) {
        hasFoundPath = true;
        BB = *Succ;
        break;
      }
      if ((Mode & GetPathToValue) && (getExecutionCount(*Succ) != MissingValue)) {
        hasFoundPath = true;
        BB = *Succ;
        break;
      }
    }
  }

  return BB;
}

template<>
void ProfileInfoT<Function,BasicBlock>::
        divertFlow(const Edge &oldedge, const Edge &newedge) {
  DEBUG(dbgs() << "Diverting " << oldedge << " via " << newedge );

  // First check if the old edge was taken, if not, just delete it...
  if (getEdgeWeight(oldedge) == 0) {
    removeEdge(oldedge);
    return;
  }

  Path P;
  P[newedge.first] = 0;
  P[newedge.second] = newedge.first;
  const BasicBlock *BB = GetPath(newedge.second,oldedge.second,P,GetPathToExit | GetPathToDest);

  double w = getEdgeWeight (oldedge);
  DEBUG(dbgs() << ", Weight: " << format("%.20g",w) << "\n");
  do {
    const BasicBlock *Parent = P.find(BB)->second;
    Edge e = getEdge(Parent,BB);
    double oldw = getEdgeWeight(e);
    double oldc = getExecutionCount(e.first);
    setEdgeWeight(e, w+oldw);
    if (Parent != oldedge.first) {
      setExecutionCount(e.first, w+oldc);
    }
    BB = Parent;
  } while (BB != newedge.first);
  removeEdge(oldedge);
}

/// Replaces all occurrences of RmBB in the ProfilingInfo with DestBB.
/// This checks all edges of the function the blocks reside in and replaces the
/// occurrences of RmBB with DestBB.
template<>
void ProfileInfoT<Function,BasicBlock>::
        replaceAllUses(const BasicBlock *RmBB, const BasicBlock *DestBB) {
  DEBUG(dbgs() << "Replacing " << RmBB->getName()
               << " with " << DestBB->getName() << "\n");
  const Function *F = DestBB->getParent();
  std::map<const Function*, EdgeWeights>::iterator J =
    EdgeInformation.find(F);
  if (J == EdgeInformation.end()) return;

  Edge e, newedge;
  bool erasededge = false;
  EdgeWeights::iterator I = J->second.begin(), E = J->second.end();
  while(I != E) {
    e = (I++)->first;
    bool foundedge = false; bool eraseedge = false;
    if (e.first == RmBB) {
      if (e.second == DestBB) {
        eraseedge = true;
      } else {
        newedge = getEdge(DestBB, e.second);
        foundedge = true;
      }
    }
    if (e.second == RmBB) {
      if (e.first == DestBB) {
        eraseedge = true;
      } else {
        newedge = getEdge(e.first, DestBB);
        foundedge = true;
      }
    }
    if (foundedge) {
      replaceEdge(e, newedge);
    }
    if (eraseedge) {
      if (erasededge) {
        Edge newedge = getEdge(DestBB, DestBB);
        replaceEdge(e, newedge);
      } else {
        removeEdge(e);
        erasededge = true;
      }
    }
  }
}

/// Splits an edge in the ProfileInfo and redirects flow over NewBB.
/// Since its possible that there is more than one edge in the CFG from FristBB
/// to SecondBB its necessary to redirect the flow proporionally.
template<>
void ProfileInfoT<Function,BasicBlock>::splitEdge(const BasicBlock *FirstBB,
                                                  const BasicBlock *SecondBB,
                                                  const BasicBlock *NewBB,
                                                  bool MergeIdenticalEdges) {
  const Function *F = FirstBB->getParent();
  std::map<const Function*, EdgeWeights>::iterator J =
    EdgeInformation.find(F);
  if (J == EdgeInformation.end()) return;

  // Generate edges and read current weight.
  Edge e  = getEdge(FirstBB, SecondBB);
  Edge n1 = getEdge(FirstBB, NewBB);
  Edge n2 = getEdge(NewBB, SecondBB);
  EdgeWeights &ECs = J->second;
  double w = ECs[e];

  int succ_count = 0;
  if (!MergeIdenticalEdges) {
    // First count the edges from FristBB to SecondBB, if there is more than
    // one, only slice out a proporional part for NewBB.
    for(succ_const_iterator BBI = succ_begin(FirstBB), BBE = succ_end(FirstBB);
        BBI != BBE; ++BBI) {
      if (*BBI == SecondBB) succ_count++;  
    }
    // When the NewBB is completely new, increment the count by one so that
    // the counts are properly distributed.
    if (getExecutionCount(NewBB) == ProfileInfo::MissingValue) succ_count++;
  } else {
    // When the edges are merged anyway, then redirect all flow.
    succ_count = 1;
  }

  // We know now how many edges there are from FirstBB to SecondBB, reroute a
  // proportional part of the edge weight over NewBB.
  double neww = floor(w / succ_count);
  ECs[n1] += neww;
  ECs[n2] += neww;
  BlockInformation[F][NewBB] += neww;
  if (succ_count == 1) {
    ECs.erase(e);
  } else {
    ECs[e] -= neww;
  }
}

template<>
void ProfileInfoT<Function,BasicBlock>::splitBlock(const BasicBlock *Old,
                                                   const BasicBlock* New) {
  const Function *F = Old->getParent();
  std::map<const Function*, EdgeWeights>::iterator J =
    EdgeInformation.find(F);
  if (J == EdgeInformation.end()) return;

  DEBUG(dbgs() << "Splitting " << Old->getName() << " to " << New->getName() << "\n");

  std::set<Edge> Edges;
  for (EdgeWeights::iterator ewi = J->second.begin(), ewe = J->second.end(); 
       ewi != ewe; ++ewi) {
    Edge old = ewi->first;
    if (old.first == Old) {
      Edges.insert(old);
    }
  }
  for (std::set<Edge>::iterator EI = Edges.begin(), EE = Edges.end(); 
       EI != EE; ++EI) {
    Edge newedge = getEdge(New, EI->second);
    replaceEdge(*EI, newedge);
  }

  double w = getExecutionCount(Old);
  setEdgeWeight(getEdge(Old, New), w);
  setExecutionCount(New, w);
}

template<>
void ProfileInfoT<Function,BasicBlock>::splitBlock(const BasicBlock *BB,
                                                   const BasicBlock* NewBB,
                                                   BasicBlock *const *Preds,
                                                   unsigned NumPreds) {
  const Function *F = BB->getParent();
  std::map<const Function*, EdgeWeights>::iterator J =
    EdgeInformation.find(F);
  if (J == EdgeInformation.end()) return;

  DEBUG(dbgs() << "Splitting " << NumPreds << " Edges from " << BB->getName() 
               << " to " << NewBB->getName() << "\n");

  // Collect weight that was redirected over NewBB.
  double newweight = 0;
  
  std::set<const BasicBlock *> ProcessedPreds;
  // For all requestes Predecessors.
  for (unsigned pred = 0; pred < NumPreds; ++pred) {
    const BasicBlock * Pred = Preds[pred];
    if (ProcessedPreds.insert(Pred).second) {
      // Create edges and read old weight.
      Edge oldedge = getEdge(Pred, BB);
      Edge newedge = getEdge(Pred, NewBB);

      // Remember how much weight was redirected.
      newweight += getEdgeWeight(oldedge);
    
      replaceEdge(oldedge,newedge);
    }
  }

  Edge newedge = getEdge(NewBB,BB);
  setEdgeWeight(newedge, newweight);
  setExecutionCount(NewBB, newweight);
}

template<>
void ProfileInfoT<Function,BasicBlock>::transfer(const Function *Old,
                                                 const Function *New) {
  DEBUG(dbgs() << "Replacing Function " << Old->getName() << " with "
               << New->getName() << "\n");
  std::map<const Function*, EdgeWeights>::iterator J =
    EdgeInformation.find(Old);
  if(J != EdgeInformation.end()) {
    EdgeInformation[New] = J->second;
  }
  EdgeInformation.erase(Old);
  BlockInformation.erase(Old);
  FunctionInformation.erase(Old);
}

static double readEdgeOrRemember(ProfileInfo::Edge edge, double w,
                                 ProfileInfo::Edge &tocalc, unsigned &uncalc) {
  if (w == ProfileInfo::MissingValue) {
    tocalc = edge;
    uncalc++;
    return 0;
  } else {
    return w;
  }
}

template<>
bool ProfileInfoT<Function,BasicBlock>::
        CalculateMissingEdge(const BasicBlock *BB, Edge &removed,
                             bool assumeEmptySelf) {
  Edge edgetocalc;
  unsigned uncalculated = 0;

  // collect weights of all incoming and outgoing edges, rememer edges that
  // have no value
  double incount = 0;
  SmallSet<const BasicBlock*,8> pred_visited;
  const_pred_iterator bbi = pred_begin(BB), bbe = pred_end(BB);
  if (bbi==bbe) {
    Edge e = getEdge(0,BB);
    incount += readEdgeOrRemember(e, getEdgeWeight(e) ,edgetocalc,uncalculated);
  }
  for (;bbi != bbe; ++bbi) {
    if (pred_visited.insert(*bbi)) {
      Edge e = getEdge(*bbi,BB);
      incount += readEdgeOrRemember(e, getEdgeWeight(e) ,edgetocalc,uncalculated);
    }
  }

  double outcount = 0;
  SmallSet<const BasicBlock*,8> succ_visited;
  succ_const_iterator sbbi = succ_begin(BB), sbbe = succ_end(BB);
  if (sbbi==sbbe) {
    Edge e = getEdge(BB,0);
    if (getEdgeWeight(e) == MissingValue) {
      double w = getExecutionCount(BB);
      if (w != MissingValue) {
        setEdgeWeight(e,w);
        removed = e;
      }
    }
    outcount += readEdgeOrRemember(e, getEdgeWeight(e), edgetocalc, uncalculated);
  }
  for (;sbbi != sbbe; ++sbbi) {
    if (succ_visited.insert(*sbbi)) {
      Edge e = getEdge(BB,*sbbi);
      outcount += readEdgeOrRemember(e, getEdgeWeight(e), edgetocalc, uncalculated);
    }
  }

  // if exactly one edge weight was missing, calculate it and remove it from
  // spanning tree
  if (uncalculated == 0 ) {
    return true;
  } else
  if (uncalculated == 1) {
    if (incount < outcount) {
      EdgeInformation[BB->getParent()][edgetocalc] = outcount-incount;
    } else {
      EdgeInformation[BB->getParent()][edgetocalc] = incount-outcount;
    }
    DEBUG(dbgs() << "--Calc Edge Counter for " << edgetocalc << ": "
                 << format("%.20g", getEdgeWeight(edgetocalc)) << "\n");
    removed = edgetocalc;
    return true;
  } else 
  if (uncalculated == 2 && assumeEmptySelf && edgetocalc.first == edgetocalc.second && incount == outcount) {
    setEdgeWeight(edgetocalc, incount * 10);
    removed = edgetocalc;
    return true;
  } else {
    return false;
  }
}

static void readEdge(ProfileInfo *PI, ProfileInfo::Edge e, double &calcw, std::set<ProfileInfo::Edge> &misscount) {
  double w = PI->getEdgeWeight(e);
  if (w != ProfileInfo::MissingValue) {
    calcw += w;
  } else {
    misscount.insert(e);
  }
}

template<>
bool ProfileInfoT<Function,BasicBlock>::EstimateMissingEdges(const BasicBlock *BB) {
  double inWeight = 0;
  std::set<Edge> inMissing;
  std::set<const BasicBlock*> ProcessedPreds;
  const_pred_iterator bbi = pred_begin(BB), bbe = pred_end(BB);
  if (bbi == bbe) {
    readEdge(this,getEdge(0,BB),inWeight,inMissing);
  }
  for( ; bbi != bbe; ++bbi ) {
    if (ProcessedPreds.insert(*bbi).second) {
      readEdge(this,getEdge(*bbi,BB),inWeight,inMissing);
    }
  }

  double outWeight = 0;
  std::set<Edge> outMissing;
  std::set<const BasicBlock*> ProcessedSuccs;
  succ_const_iterator sbbi = succ_begin(BB), sbbe = succ_end(BB);
  if (sbbi == sbbe)
    readEdge(this,getEdge(BB,0),outWeight,outMissing);
  for ( ; sbbi != sbbe; ++sbbi ) {
    if (ProcessedSuccs.insert(*sbbi).second) {
      readEdge(this,getEdge(BB,*sbbi),outWeight,outMissing);
    }
  }

  double share;
  std::set<Edge>::iterator ei,ee;
  if (inMissing.size() == 0 && outMissing.size() > 0) {
    ei = outMissing.begin();
    ee = outMissing.end();
    share = inWeight/outMissing.size();
    setExecutionCount(BB,inWeight);
  } else
  if (inMissing.size() > 0 && outMissing.size() == 0 && outWeight == 0) {
    ei = inMissing.begin();
    ee = inMissing.end();
    share = 0;
    setExecutionCount(BB,0);
  } else
  if (inMissing.size() == 0 && outMissing.size() == 0) {
    setExecutionCount(BB,outWeight);
    return true;
  } else {
    return false;
  }
  for ( ; ei != ee; ++ei ) {
    setEdgeWeight(*ei,share);
  }
  return true;
}

template<>
void ProfileInfoT<Function,BasicBlock>::repair(const Function *F) {
//  if (getExecutionCount(&(F->getEntryBlock())) == 0) {
//    for (Function::const_iterator FI = F->begin(), FE = F->end();
//         FI != FE; ++FI) {
//      const BasicBlock* BB = &(*FI);
//      {
//        const_pred_iterator NBB = pred_begin(BB), End = pred_end(BB);
//        if (NBB == End) {
//          setEdgeWeight(getEdge(0,BB),0);
//        }
//        for(;NBB != End; ++NBB) {
//          setEdgeWeight(getEdge(*NBB,BB),0);
//        }
//      }
//      {
//        succ_const_iterator NBB = succ_begin(BB), End = succ_end(BB);
//        if (NBB == End) {
//          setEdgeWeight(getEdge(0,BB),0);
//        }
//        for(;NBB != End; ++NBB) {
//          setEdgeWeight(getEdge(*NBB,BB),0);
//        }
//      }
//    }
//    return;
//  }
  // The set of BasicBlocks that are still unvisited.
  std::set<const BasicBlock*> Unvisited;

  // The set of return edges (Edges with no successors).
  std::set<Edge> ReturnEdges;
  double ReturnWeight = 0;
  
  // First iterate over the whole function and collect:
  // 1) The blocks in this function in the Unvisited set.
  // 2) The return edges in the ReturnEdges set.
  // 3) The flow that is leaving the function already via return edges.

  // Data structure for searching the function.
  std::queue<const BasicBlock *> BFS;
  const BasicBlock *BB = &(F->getEntryBlock());
  BFS.push(BB);
  Unvisited.insert(BB);

  while (BFS.size()) {
    BB = BFS.front(); BFS.pop();
    succ_const_iterator NBB = succ_begin(BB), End = succ_end(BB);
    if (NBB == End) {
      Edge e = getEdge(BB,0);
      double w = getEdgeWeight(e);
      if (w == MissingValue) {
        // If the return edge has no value, try to read value from block.
        double bw = getExecutionCount(BB);
        if (bw != MissingValue) {
          setEdgeWeight(e,bw);
          ReturnWeight += bw;
        } else {
          // If both return edge and block provide no value, collect edge.
          ReturnEdges.insert(e);
        }
      } else {
        // If the return edge has a proper value, collect it.
        ReturnWeight += w;
      }
    }
    for (;NBB != End; ++NBB) {
      if (Unvisited.insert(*NBB).second) {
        BFS.push(*NBB);
      }
    }
  }

  while (Unvisited.size() > 0) {
    unsigned oldUnvisitedCount = Unvisited.size();
    bool FoundPath = false;

    // If there is only one edge left, calculate it.
    if (ReturnEdges.size() == 1) {
      ReturnWeight = getExecutionCount(&(F->getEntryBlock())) - ReturnWeight;

      Edge e = *ReturnEdges.begin();
      setEdgeWeight(e,ReturnWeight);
      setExecutionCount(e.first,ReturnWeight);

      Unvisited.erase(e.first);
      ReturnEdges.erase(e);
      continue;
    }

    // Calculate all blocks where only one edge is missing, this may also
    // resolve furhter return edges.
    std::set<const BasicBlock *>::iterator FI = Unvisited.begin(), FE = Unvisited.end();
    while(FI != FE) {
      const BasicBlock *BB = *FI; ++FI;
      Edge e;
      if(CalculateMissingEdge(BB,e,true)) {
        if (BlockInformation[F].find(BB) == BlockInformation[F].end()) {
          setExecutionCount(BB,getExecutionCount(BB));
        }
        Unvisited.erase(BB);
        if (e.first != 0 && e.second == 0) {
          ReturnEdges.erase(e);
          ReturnWeight += getEdgeWeight(e);
        }
      }
    }
    if (oldUnvisitedCount > Unvisited.size()) continue;

    // Estimate edge weights by dividing the flow proportionally.
    FI = Unvisited.begin(), FE = Unvisited.end();
    while(FI != FE) {
      const BasicBlock *BB = *FI; ++FI;
      const BasicBlock *Dest = 0;
      bool AllEdgesHaveSameReturn = true;
      // Check each Successor, these must all end up in the same or an empty
      // return block otherwise its dangerous to do an estimation on them.
      for (succ_const_iterator Succ = succ_begin(BB), End = succ_end(BB);
           Succ != End; ++Succ) {
        Path P;
        GetPath(*Succ, 0, P, GetPathToExit);
        if (Dest && Dest != P[0]) {
          AllEdgesHaveSameReturn = false;
        }
        Dest = P[0];
      }
      if (AllEdgesHaveSameReturn) {
        if(EstimateMissingEdges(BB)) {
          Unvisited.erase(BB);
          break;
        }
      }
    }
    if (oldUnvisitedCount > Unvisited.size()) continue;

    // Check if there is a path to an block that has a known value and redirect
    // flow accordingly.
    FI = Unvisited.begin(), FE = Unvisited.end();
    while(FI != FE && !FoundPath) {
      // Fetch path.
      const BasicBlock *BB = *FI; ++FI;
      Path P;
      const BasicBlock *Dest = GetPath(BB, 0, P, GetPathToValue);

      // Calculate incoming flow.
      double iw = 0; unsigned inmissing = 0; unsigned incount = 0; unsigned invalid = 0;
      std::set<const BasicBlock *> Processed;
      for (const_pred_iterator NBB = pred_begin(BB), End = pred_end(BB);
           NBB != End; ++NBB) {
        if (Processed.insert(*NBB).second) {
          Edge e = getEdge(*NBB, BB);
          double ew = getEdgeWeight(e);
          if (ew != MissingValue) {
            iw += ew;
            invalid++;
          } else {
            // If the path contains the successor, this means its a backedge,
            // do not count as missing.
            if (P.find(*NBB) == P.end())
              inmissing++;
          }
          incount++;
        }
      }
      if (inmissing == incount) continue;
      if (invalid == 0) continue;

      // Subtract (already) outgoing flow.
      Processed.clear();
      for (succ_const_iterator NBB = succ_begin(BB), End = succ_end(BB);
           NBB != End; ++NBB) {
        if (Processed.insert(*NBB).second) {
          Edge e = getEdge(BB, *NBB);
          double ew = getEdgeWeight(e);
          if (ew != MissingValue) {
            iw -= ew;
          }
        }
      }
      if (iw < 0) continue;

      // Check the receiving end of the path if it can handle the flow.
      double ow = getExecutionCount(Dest);
      Processed.clear();
      for (succ_const_iterator NBB = succ_begin(BB), End = succ_end(BB);
           NBB != End; ++NBB) {
        if (Processed.insert(*NBB).second) {
          Edge e = getEdge(BB, *NBB);
          double ew = getEdgeWeight(e);
          if (ew != MissingValue) {
            ow -= ew;
          }
        }
      }
      if (ow < 0) continue;

      // Determine how much flow shall be used.
      double ew = getEdgeWeight(getEdge(P[Dest],Dest));
      if (ew != MissingValue) {
        ew = ew<ow?ew:ow;
        ew = ew<iw?ew:iw;
      } else {
        if (inmissing == 0)
          ew = iw;
      }

      // Create flow.
      if (ew != MissingValue) {
        do {
          Edge e = getEdge(P[Dest],Dest);
          if (getEdgeWeight(e) == MissingValue) {
            setEdgeWeight(e,ew);
            FoundPath = true;
          }
          Dest = P[Dest];
        } while (Dest != BB);
      }
    }
    if (FoundPath) continue;

    // Calculate a block with self loop.
    FI = Unvisited.begin(), FE = Unvisited.end();
    while(FI != FE && !FoundPath) {
      const BasicBlock *BB = *FI; ++FI;
      bool SelfEdgeFound = false;
      for (succ_const_iterator NBB = succ_begin(BB), End = succ_end(BB);
           NBB != End; ++NBB) {
        if (*NBB == BB) {
          SelfEdgeFound = true;
          break;
        }
      }
      if (SelfEdgeFound) {
        Edge e = getEdge(BB,BB);
        if (getEdgeWeight(e) == MissingValue) {
          double iw = 0;
          std::set<const BasicBlock *> Processed;
          for (const_pred_iterator NBB = pred_begin(BB), End = pred_end(BB);
               NBB != End; ++NBB) {
            if (Processed.insert(*NBB).second) {
              Edge e = getEdge(*NBB, BB);
              double ew = getEdgeWeight(e);
              if (ew != MissingValue) {
                iw += ew;
              }
            }
          }
          setEdgeWeight(e,iw * 10);
          FoundPath = true;
        }
      }
    }
    if (FoundPath) continue;

    // Determine backedges, set them to zero.
    FI = Unvisited.begin(), FE = Unvisited.end();
    while(FI != FE && !FoundPath) {
      const BasicBlock *BB = *FI; ++FI;
      const BasicBlock *Dest = 0;
      Path P;
      bool BackEdgeFound = false;
      for (const_pred_iterator NBB = pred_begin(BB), End = pred_end(BB);
           NBB != End; ++NBB) {
        Dest = GetPath(BB, *NBB, P, GetPathToDest | GetPathWithNewEdges);
        if (Dest == *NBB) {
          BackEdgeFound = true;
          break;
        }
      }
      if (BackEdgeFound) {
        Edge e = getEdge(Dest,BB);
        double w = getEdgeWeight(e);
        if (w == MissingValue) {
          setEdgeWeight(e,0);
          FoundPath = true;
        }
        do {
          Edge e = getEdge(P[Dest], Dest);
          double w = getEdgeWeight(e);
          if (w == MissingValue) {
            setEdgeWeight(e,0);
            FoundPath = true;
          }
          Dest = P[Dest];
        } while (Dest != BB);
      }
    }
    if (FoundPath) continue;

    // Channel flow to return block.
    FI = Unvisited.begin(), FE = Unvisited.end();
    while(FI != FE && !FoundPath) {
      const BasicBlock *BB = *FI; ++FI;

      Path P;
      const BasicBlock *Dest = GetPath(BB, 0, P, GetPathToExit | GetPathWithNewEdges);
      Dest = P[0];
      if (!Dest) continue;

      if (getEdgeWeight(getEdge(Dest,0)) == MissingValue) {
        // Calculate incoming flow.
        double iw = 0;
        std::set<const BasicBlock *> Processed;
        for (const_pred_iterator NBB = pred_begin(BB), End = pred_end(BB);
             NBB != End; ++NBB) {
          if (Processed.insert(*NBB).second) {
            Edge e = getEdge(*NBB, BB);
            double ew = getEdgeWeight(e);
            if (ew != MissingValue) {
              iw += ew;
            }
          }
        }
        do {
          Edge e = getEdge(P[Dest], Dest);
          double w = getEdgeWeight(e);
          if (w == MissingValue) {
            setEdgeWeight(e,iw);
            FoundPath = true;
          } else {
            assert(0 && "Edge should not have value already!");
          }
          Dest = P[Dest];
        } while (Dest != BB);
      }
    }
    if (FoundPath) continue;

    // Speculatively set edges to zero.
    FI = Unvisited.begin(), FE = Unvisited.end();
    while(FI != FE && !FoundPath) {
      const BasicBlock *BB = *FI; ++FI;

      for (const_pred_iterator NBB = pred_begin(BB), End = pred_end(BB);
           NBB != End; ++NBB) {
        Edge e = getEdge(*NBB,BB);
        double w = getEdgeWeight(e);
        if (w == MissingValue) {
          setEdgeWeight(e,0);
          FoundPath = true;
          break;
        }
      }
    }
    if (FoundPath) continue;

    errs() << "{";
    FI = Unvisited.begin(), FE = Unvisited.end();
    while(FI != FE) {
      const BasicBlock *BB = *FI; ++FI;
      dbgs() << BB->getName();
      if (FI != FE)
        dbgs() << ",";
    }
    errs() << "}";

    errs() << "ASSERT: could not repair function";
    assert(0 && "could not repair function");
  }

  EdgeWeights J = EdgeInformation[F];
  for (EdgeWeights::iterator EI = J.begin(), EE = J.end(); EI != EE; ++EI) {
    Edge e = EI->first;

    bool SuccFound = false;
    if (e.first != 0) {
      succ_const_iterator NBB = succ_begin(e.first), End = succ_end(e.first);
      if (NBB == End) {
        if (0 == e.second) {
          SuccFound = true;
        }
      }
      for (;NBB != End; ++NBB) {
        if (*NBB == e.second) {
          SuccFound = true;
          break;
        }
      }
      if (!SuccFound) {
        removeEdge(e);
      }
    }
  }
}

raw_ostream& operator<<(raw_ostream &O, const MachineFunction *MF) {
  return O << MF->getFunction()->getName() << "(MF)";
}

raw_ostream& operator<<(raw_ostream &O, const MachineBasicBlock *MBB) {
  return O << MBB->getBasicBlock()->getName() << "(MB)";
}

raw_ostream& operator<<(raw_ostream &O, std::pair<const MachineBasicBlock *, const MachineBasicBlock *> E) {
  O << "(";

  if (E.first)
    O << E.first;
  else
    O << "0";

  O << ",";

  if (E.second)
    O << E.second;
  else
    O << "0";

  return O << ")";
}

} // namespace llvm

//===----------------------------------------------------------------------===//
//  NoProfile ProfileInfo implementation
//

namespace {
  struct NoProfileInfo : public ImmutablePass, public ProfileInfo {
    static char ID; // Class identification, replacement for typeinfo
    NoProfileInfo() : ImmutablePass(ID) {
      initializeNoProfileInfoPass(*PassRegistry::getPassRegistry());
    }
    
    /// getAdjustedAnalysisPointer - This method is used when a pass implements
    /// an analysis interface through multiple inheritance.  If needed, it
    /// should override this to adjust the this pointer as needed for the
    /// specified pass info.
    virtual void *getAdjustedAnalysisPointer(AnalysisID PI) {
      if (PI == &ProfileInfo::ID)
        return (ProfileInfo*)this;
      return this;
    }
    
    virtual const char *getPassName() const {
      return "NoProfileInfo";
    }
  };
}  // End of anonymous namespace

char NoProfileInfo::ID = 0;
// Register this pass...
INITIALIZE_AG_PASS(NoProfileInfo, ProfileInfo, "no-profile",
                   "No Profile Information", false, true, true)

ImmutablePass *llvm::createNoProfileInfoPass() { return new NoProfileInfo(); }
