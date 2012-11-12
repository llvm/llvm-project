//===- DominanceFrontier.cpp - Dominance Frontier Calculation -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DominanceFrontier.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

char DominanceFrontier::ID = 0;
INITIALIZE_PASS_BEGIN(DominanceFrontier, "domfrontier",
                "Dominance Frontier Construction", true, true)
INITIALIZE_PASS_DEPENDENCY(DominatorTree)
INITIALIZE_PASS_END(DominanceFrontier, "domfrontier",
                "Dominance Frontier Construction", true, true)

namespace {
  class DFCalculateWorkObject {
  public:
    DFCalculateWorkObject(BasicBlock *B, BasicBlock *P, 
                          const DomTreeNode *N,
                          const DomTreeNode *PN)
    : currentBB(B), parentBB(P), Node(N), parentNode(PN) {}
    BasicBlock *currentBB;
    BasicBlock *parentBB;
    const DomTreeNode *Node;
    const DomTreeNode *parentNode;
  };
}

void DominanceFrontier::anchor() { }

const DominanceFrontier::DomSetType &
DominanceFrontier::calculate(const DominatorTree &DT,
                             const DomTreeNode *Node) {
  BasicBlock *BB = Node->getBlock();
  DomSetType *Result = NULL;

  std::vector<DFCalculateWorkObject> workList;
  SmallPtrSet<BasicBlock *, 32> visited;

  workList.push_back(DFCalculateWorkObject(BB, NULL, Node, NULL));
  do {
    DFCalculateWorkObject *currentW = &workList.back();
    assert (currentW && "Missing work object.");

    BasicBlock *currentBB = currentW->currentBB;
    BasicBlock *parentBB = currentW->parentBB;
    const DomTreeNode *currentNode = currentW->Node;
    const DomTreeNode *parentNode = currentW->parentNode;
    assert (currentBB && "Invalid work object. Missing current Basic Block");
    assert (currentNode && "Invalid work object. Missing current Node");
    DomSetType &S = Frontiers[currentBB];

    // Visit each block only once.
    if (visited.count(currentBB) == 0) {
      visited.insert(currentBB);

      // Loop over CFG successors to calculate DFlocal[currentNode]
      for (succ_iterator SI = succ_begin(currentBB), SE = succ_end(currentBB);
           SI != SE; ++SI) {
        // Does Node immediately dominate this successor?
        if (DT[*SI]->getIDom() != currentNode)
          S.insert(*SI);
      }
    }

    // At this point, S is DFlocal.  Now we union in DFup's of our children...
    // Loop through and visit the nodes that Node immediately dominates (Node's
    // children in the IDomTree)
    bool visitChild = false;
    for (DomTreeNode::const_iterator NI = currentNode->begin(), 
           NE = currentNode->end(); NI != NE; ++NI) {
      DomTreeNode *IDominee = *NI;
      BasicBlock *childBB = IDominee->getBlock();
      if (visited.count(childBB) == 0) {
        workList.push_back(DFCalculateWorkObject(childBB, currentBB,
                                                 IDominee, currentNode));
        visitChild = true;
      }
    }

    // If all children are visited or there is any child then pop this block
    // from the workList.
    if (!visitChild) {

      if (!parentBB) {
        Result = &S;
        break;
      }

      DomSetType::const_iterator CDFI = S.begin(), CDFE = S.end();
      DomSetType &parentSet = Frontiers[parentBB];
      for (; CDFI != CDFE; ++CDFI) {
        if (!DT.properlyDominates(parentNode, DT[*CDFI]))
          parentSet.insert(*CDFI);
      }
      workList.pop_back();
    }

  } while (!workList.empty());

  return *Result;
}

void DominanceFrontierBase::print(raw_ostream &OS, const Module* ) const {
  for (const_iterator I = begin(), E = end(); I != E; ++I) {
    OS << "  DomFrontier for BB ";
    if (I->first)
      WriteAsOperand(OS, I->first, false);
    else
      OS << " <<exit node>>";
    OS << " is:\t";
    
    const std::set<BasicBlock*> &BBs = I->second;
    
    for (std::set<BasicBlock*>::const_iterator I = BBs.begin(), E = BBs.end();
         I != E; ++I) {
      OS << ' ';
      if (*I)
        WriteAsOperand(OS, *I, false);
      else
        OS << "<<exit node>>";
    }
    OS << "\n";
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void DominanceFrontierBase::dump() const {
  print(dbgs());
}
#endif

