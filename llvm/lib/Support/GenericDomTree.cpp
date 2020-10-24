//===- GenericDomTree.cpp - Generic dominator trees for graphs --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/GenericDomTree.h"

#include "llvm/ADT/SmallSet.h"

using namespace llvm;

bool GenericDomTreeNodeBase::compare(
    const GenericDomTreeNodeBase *Other) const {
  if (getNumChildren() != Other->getNumChildren())
    return true;

  if (Level != Other->Level)
    return true;

  SmallSet<CfgBlockRef, 4> OtherChildren;
  for (const GenericDomTreeNodeBase *I : *Other) {
    CfgBlockRef Nd = I->getBlock();
    OtherChildren.insert(Nd);
  }

  for (const GenericDomTreeNodeBase *I : *this) {
    CfgBlockRef N = I->getBlock();
    if (OtherChildren.count(N) == 0)
      return true;
  }
  return false;
}

void GenericDomTreeNodeBase::setIDom(GenericDomTreeNodeBase *NewIDom) {
  assert(IDom && "No immediate dominator?");
  if (IDom == NewIDom)
    return;

  auto I = find(IDom->Children, this);
  assert(I != IDom->Children.end() &&
         "Not in immediate dominator children set!");
  // I am no longer your child...
  IDom->Children.erase(I);

  // Switch to new dominator
  IDom = NewIDom;
  IDom->Children.push_back(this);

  UpdateLevel();
}

void GenericDomTreeNodeBase::UpdateLevel() {
  assert(IDom);
  if (Level == IDom->Level + 1)
    return;

  SmallVector<GenericDomTreeNodeBase *, 64> WorkStack = {this};

  while (!WorkStack.empty()) {
    GenericDomTreeNodeBase *Current = WorkStack.pop_back_val();
    Current->Level = Current->IDom->Level + 1;

    for (GenericDomTreeNodeBase *C : *Current) {
      assert(C->IDom);
      if (C->Level != C->IDom->Level + 1)
        WorkStack.push_back(C);
    }
  }
}

/// compare - Return false if the other dominator tree base matches this
/// dominator tree base. Otherwise return true.
bool GenericDominatorTreeBase::compare(
    const GenericDominatorTreeBase &Other) const {
  if (DomTreeNodes.size() != Other.DomTreeNodes.size())
    return true;

  for (const auto &DomTreeNode : DomTreeNodes) {
    CfgBlockRef BB = DomTreeNode.first;
    auto OI = Other.DomTreeNodes.find(BB);
    if (OI == Other.DomTreeNodes.end())
      return true;

    GenericDomTreeNodeBase &MyNd = *DomTreeNode.second;
    GenericDomTreeNodeBase &OtherNd = *OI->second;

    if (MyNd.compare(&OtherNd))
      return true;
  }

  return false;
}

void GenericDominatorTreeBase::reset() {
  DomTreeNodes.clear();
  RootNode = nullptr;
  DFSInfoValid = false;
  SlowQueries = 0;
}

/// properlyDominates - Returns true iff A dominates B and A != B.
/// Note that this is not a constant time operation!
bool GenericDominatorTreeBase::properlyDominates(
    const GenericDomTreeNodeBase *A, const GenericDomTreeNodeBase *B) const {
  if (!A || !B)
    return false;
  if (A == B)
    return false;
  return dominates(A, B);
}

bool GenericDominatorTreeBase::properlyDominatesBlock(CfgBlockRef A,
                                                      CfgBlockRef B) const {
  if (A == B)
    return false;

  return dominates(getNode(A), getNode(B));
}

/// dominates - Returns true iff A dominates B.  Note that this is not a
/// constant time operation!
bool GenericDominatorTreeBase::dominates(
    const GenericDomTreeNodeBase *A, const GenericDomTreeNodeBase *B) const {
  // A node trivially dominates itself.
  if (B == A)
    return true;

  // An unreachable node is dominated by anything.
  if (!isReachableFromEntry(B))
    return true;

  // And dominates nothing.
  if (!isReachableFromEntry(A))
    return false;

  if (B->getIDom() == A)
    return true;

  if (A->getIDom() == B)
    return false;

  // A can only dominate B if it is higher in the tree.
  if (A->getLevel() >= B->getLevel())
    return false;

  // Compare the result of the tree walk and the dfs numbers, if expensive
  // checks are enabled.
#ifdef EXPENSIVE_CHECKS
  assert(
      (!DFSInfoValid || (dominatedBySlowTreeWalk(A, B) == B->DominatedBy(A))) &&
      "Tree walk disagrees with dfs numbers!");
#endif

  if (DFSInfoValid)
    return B->DominatedBy(A);

  // If we end up with too many slow queries, just update the
  // DFS numbers on the theory that we are going to keep querying.
  SlowQueries++;
  if (SlowQueries > 32) {
    updateDFSNumbers();
    return B->DominatedBy(A);
  }

  return dominatedBySlowTreeWalk(A, B);
}

bool GenericDominatorTreeBase::dominatesBlock(CfgBlockRef A,
                                              CfgBlockRef B) const {
  if (A == B)
    return true;

  // Cast away the const qualifiers here. This is ok since
  // this function doesn't actually return the values returned
  // from getNode.
  return dominates(getNode(A), getNode(B));
}

/// findNearestCommonDominator - Find nearest common dominator of A and B.
const GenericDomTreeNodeBase *
GenericDominatorTreeBase::findNearestCommonDominator(
    const GenericDomTreeNodeBase *A, const GenericDomTreeNodeBase *B) const {
  if (A == RootNode || B == RootNode)
    return RootNode;

  assert(A && "A muset be in the tree");
  assert(B && "B muset be in the tree");

  // Use level information to go up the tree until the levels match. Then
  // continue going up til we arrive at the same node.
  while (A != B) {
    if (A->getLevel() < B->getLevel())
      std::swap(A, B);

    A = A->IDom;
    assert(A != nullptr && "nodes in different dominator trees?");
  }

  return A;
}

CfgBlockRef
GenericDominatorTreeBase::findNearestCommonDominatorBlock(CfgBlockRef A,
                                                          CfgBlockRef B) const {
  assert(A && B && "Pointers are not valid");

  const GenericDomTreeNodeBase *Dom =
      findNearestCommonDominator(getNode(A), getNode(B));

  return Dom ? Dom->getBlock() : CfgBlockRef();
}

/// updateDFSNumbers - Assign In and Out numbers to the nodes while walking
/// dominator tree in dfs order.
void GenericDominatorTreeBase::updateDFSNumbers() const {
  if (DFSInfoValid) {
    SlowQueries = 0;
    return;
  }

  SmallVector<std::pair<const GenericDomTreeNodeBase *,
                        GenericDomTreeNodeBase::const_iterator>,
              32>
      WorkStack;

  const GenericDomTreeNodeBase *ThisRoot = getRootNode();
  if (!ThisRoot)
    return;

  // Both dominators and postdominators have a single root node. In the case
  // case of PostDominatorTree, this node is a virtual root.
  WorkStack.push_back({ThisRoot, ThisRoot->begin()});

  unsigned DFSNum = 0;
  ThisRoot->DFSNumIn = DFSNum++;

  while (!WorkStack.empty()) {
    const GenericDomTreeNodeBase *Node = WorkStack.back().first;
    const auto ChildIt = WorkStack.back().second;

    // If we visited all of the children of this node, "recurse" back up the
    // stack setting the DFOutNum.
    if (ChildIt == Node->end()) {
      Node->DFSNumOut = DFSNum++;
      WorkStack.pop_back();
    } else {
      // Otherwise, recursively visit this child.
      const GenericDomTreeNodeBase *Child = *ChildIt;
      ++WorkStack.back().second;

      WorkStack.push_back({Child, Child->begin()});
      Child->DFSNumIn = DFSNum++;
    }
  }

  SlowQueries = 0;
  DFSInfoValid = true;
}

bool GenericDominatorTreeBase::dominatedBySlowTreeWalk(
    const GenericDomTreeNodeBase *A, const GenericDomTreeNodeBase *B) const {
  assert(A != B);
  assert(isReachableFromEntry(B));
  assert(isReachableFromEntry(A));

  const unsigned ALevel = A->getLevel();
  const GenericDomTreeNodeBase *IDom;

  // Don't walk nodes above A's subtree. When we reach A's level, we must
  // either find A or be in some other subtree not dominated by A.
  while ((IDom = B->getIDom()) != nullptr && IDom->getLevel() >= ALevel)
    B = IDom; // Walk up the tree

  return B == A;
}
