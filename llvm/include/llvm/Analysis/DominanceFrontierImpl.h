//===- llvm/Analysis/DominanceFrontier.h - Dominator Frontiers --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the generic implementation of the DominanceFrontier class, which
// calculate and holds the dominance frontier for a function for.
//
// This should be considered deprecated, don't add any more uses of this data
// structure.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DOMINANCEFRONTIERIMPL_H
#define LLVM_ANALYSIS_DOMINANCEFRONTIERIMPL_H

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/DominanceFrontier.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GenericDomTree.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <vector>

namespace llvm {

template <class BlockT>
class DFCalculateWorkObject {
public:
  using DomTreeNodeT = DomTreeNodeBase<BlockT>;

  DFCalculateWorkObject(BlockT *B, BlockT *P, const DomTreeNodeT *N,
                        const DomTreeNodeT *PN)
      : currentBB(B), parentBB(P), Node(N), parentNode(PN) {}

  BlockT *currentBB;
  BlockT *parentBB;
  const DomTreeNodeT *Node;
  const DomTreeNodeT *parentNode;
};

template <class BlockT, bool IsPostDom>
void DominanceFrontierBase<BlockT, IsPostDom>::print(raw_ostream &OS) const {
  for (const_iterator I = begin(), E = end(); I != E; ++I) {
    OS << "  DomFrontier for BB ";
    if (I->first)
      I->first->printAsOperand(OS, false);
    else
      OS << " <<exit node>>";
    OS << " is:\t";

    const SetVector<BlockT *> &BBs = I->second;

    for (const BlockT *BB : BBs) {
      OS << ' ';
      if (BB)
        BB->printAsOperand(OS, false);
      else
        OS << "<<exit node>>";
    }
    OS << '\n';
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
template <class BlockT, bool IsPostDom>
void DominanceFrontierBase<BlockT, IsPostDom>::dump() const {
  print(dbgs());
}
#endif

template <class BlockT, bool IsPostDom>
void DominanceFrontierBase<BlockT, IsPostDom>::calculate(const DomTreeT &DT) {
  // NOTE: RootNode might be virtual for `IsPostDom == true`.
  const DomTreeNodeT *RootNode = DT.getRootNode();
  BlockT *BB = RootNode->getBlock();

  std::vector<DFCalculateWorkObject<BlockT>> workList;
  SmallPtrSet<BlockT *, 32> visited;

  workList.push_back(
      DFCalculateWorkObject<BlockT>(BB, nullptr, RootNode, nullptr));
  do {
    DFCalculateWorkObject<BlockT> *currentW = &workList.back();
    assert(currentW && "Missing work object.");

    BlockT *currentBB = currentW->currentBB;
    BlockT *parentBB = currentW->parentBB;
    const DomTreeNodeT *currentNode = currentW->Node;
    const DomTreeNodeT *parentNode = currentW->parentNode;
    assert(currentNode && "Invalid work object. Missing current Node");

    // Visit each block only once.
    if (visited.insert(currentBB).second) {
      // Loop over CFG successors to calculate DFlocal[currentNode].
      //
      // Note that for `IsPostDom == true`, virtual root node (empty currentBB)
      // is an immediate post-dominator for all the exit nodes (which are
      // virtual node's CFG successors).
      if (currentBB) {
        DomSetType &S = this->Frontiers[currentBB];
        for (const auto Child : children<GraphTy>(currentBB)) {
          // Does Node immediately dominate this successor?
          if (DT[Child]->getIDom() != currentNode)
            S.insert(Child);
        }
      }
    }

    // At this point, S is DFlocal.  Now we union in DFup's of our children...
    // Loop through and visit the nodes that Node immediately dominates (Node's
    // children in the IDomTree)
    bool visitChild = false;
    for (typename DomTreeNodeT::const_iterator NI = currentNode->begin(),
                                               NE = currentNode->end();
         NI != NE; ++NI) {
      DomTreeNodeT *IDominee = *NI;
      BlockT *childBB = IDominee->getBlock();
      if (visited.count(childBB) == 0) {
        workList.push_back(DFCalculateWorkObject<BlockT>(
            childBB, currentBB, IDominee, currentNode));
        visitChild = true;
      }
    }

    // If all children are visited or there is any child then pop this block
    // from the workList.
    if (!visitChild) {
      if (RootNode == currentNode) {
        break;
      }

      workList.pop_back();
      if (!parentBB) {
        assert(IsPostDom && "For forward frontiers only root node (processed "
                            "above) might not have a parent.");
        // Processing below isn't necessary for the virtual root node in case
        // of post-dominance frontier.
        continue;
      }

      DomSetType &S = this->Frontiers[currentBB];
      typename DomSetType::const_iterator CDFI = S.begin(), CDFE = S.end();
      DomSetType &parentSet = this->Frontiers[parentBB];
      for (; CDFI != CDFE; ++CDFI) {
        if (!DT.properlyDominates(parentNode, DT[*CDFI]))
          parentSet.insert(*CDFI);
      }
    }

  } while (!workList.empty());
}

} // end namespace llvm

#endif // LLVM_ANALYSIS_DOMINANCEFRONTIERIMPL_H
