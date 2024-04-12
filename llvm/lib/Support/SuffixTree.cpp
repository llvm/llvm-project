//===- llvm/Support/SuffixTree.cpp - Implement Suffix Tree ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Suffix Tree class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/SuffixTree.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SuffixTreeNode.h"
#include <stack>

using namespace llvm;

/// \returns the number of elements in the substring associated with \p N.
static size_t numElementsInSubstring(const SuffixTreeNode *N) {
  assert(N && "Got a null node?");
  if (auto *Internal = dyn_cast<SuffixTreeInternalNode>(N))
    if (Internal->isRoot())
      return 0;
  return N->getEndIdx() - N->getStartIdx() + 1;
}

SuffixTree::SuffixTree(const ArrayRef<unsigned> &Str,
                       bool OutlinerLeafDescendants)
    : Str(Str), OutlinerLeafDescendants(OutlinerLeafDescendants) {
  Root = insertRoot();
  Active.Node = Root;

  // Keep track of the number of suffixes we have to add of the current
  // prefix.
  unsigned SuffixesToAdd = 0;

  // Construct the suffix tree iteratively on each prefix of the string.
  // PfxEndIdx is the end index of the current prefix.
  // End is one past the last element in the string.
  for (unsigned PfxEndIdx = 0, End = Str.size(); PfxEndIdx < End; PfxEndIdx++) {
    SuffixesToAdd++;
    LeafEndIdx = PfxEndIdx; // Extend each of the leaves.
    SuffixesToAdd = extend(PfxEndIdx, SuffixesToAdd);
  }

  // Set the suffix indices of each leaf.
  assert(Root && "Root node can't be nullptr!");
  setSuffixIndices();

  // Collect all leaf nodes of the suffix tree. And for each internal node,
  // record the range of leaf nodes that are descendants of it.
  if (OutlinerLeafDescendants)
    setLeafNodes();
}

SuffixTreeNode *SuffixTree::insertLeaf(SuffixTreeInternalNode &Parent,
                                       unsigned StartIdx, unsigned Edge) {
  assert(StartIdx <= LeafEndIdx && "String can't start after it ends!");
  auto *N = new (LeafNodeAllocator.Allocate())
      SuffixTreeLeafNode(StartIdx, &LeafEndIdx);
  Parent.Children[Edge] = N;
  return N;
}

SuffixTreeInternalNode *
SuffixTree::insertInternalNode(SuffixTreeInternalNode *Parent,
                               unsigned StartIdx, unsigned EndIdx,
                               unsigned Edge) {
  assert(StartIdx <= EndIdx && "String can't start after it ends!");
  assert(!(!Parent && StartIdx != SuffixTreeNode::EmptyIdx) &&
         "Non-root internal nodes must have parents!");
  auto *N = new (InternalNodeAllocator.Allocate())
      SuffixTreeInternalNode(StartIdx, EndIdx, Root);
  if (Parent)
    Parent->Children[Edge] = N;
  return N;
}

SuffixTreeInternalNode *SuffixTree::insertRoot() {
  return insertInternalNode(/*Parent = */ nullptr, SuffixTreeNode::EmptyIdx,
                            SuffixTreeNode::EmptyIdx, /*Edge = */ 0);
}

void SuffixTree::setSuffixIndices() {
  // List of nodes we need to visit along with the current length of the
  // string.
  SmallVector<std::pair<SuffixTreeNode *, unsigned>> ToVisit;

  // Current node being visited.
  SuffixTreeNode *CurrNode = Root;

  // Sum of the lengths of the nodes down the path to the current one.
  unsigned CurrNodeLen = 0;
  ToVisit.push_back({CurrNode, CurrNodeLen});
  while (!ToVisit.empty()) {
    std::tie(CurrNode, CurrNodeLen) = ToVisit.back();
    ToVisit.pop_back();
    // Length of the current node from the root down to here.
    CurrNode->setConcatLen(CurrNodeLen);
    if (auto *InternalNode = dyn_cast<SuffixTreeInternalNode>(CurrNode))
      for (auto &ChildPair : InternalNode->Children) {
        assert(ChildPair.second && "Node had a null child!");
        ToVisit.push_back(
            {ChildPair.second,
             CurrNodeLen + numElementsInSubstring(ChildPair.second)});
      }
    // No children, so we are at the end of the string.
    if (auto *LeafNode = dyn_cast<SuffixTreeLeafNode>(CurrNode))
      LeafNode->setSuffixIdx(Str.size() - CurrNodeLen);
  }
}

void SuffixTree::setLeafNodes() {
  // A stack that keeps track of nodes to visit for post-order DFS traversal.
  std::stack<SuffixTreeNode *> ToVisit;
  ToVisit.push(Root);

  // This keeps track of the index of the next leaf node to be added to
  // the LeafNodes vector of the suffix tree.
  unsigned LeafCounter = 0;

  // This keeps track of nodes whose children have been added to the stack
  // during the post-order depth-first traversal of the tree.
  llvm::SmallPtrSet<SuffixTreeInternalNode *, 32> ChildrenAddedToStack;

  // Traverse the tree in post-order.
  while (!ToVisit.empty()) {
    SuffixTreeNode *CurrNode = ToVisit.top();
    ToVisit.pop();
    if (auto *CurrInternalNode = dyn_cast<SuffixTreeInternalNode>(CurrNode)) {
      // The current node is an internal node.
      if (ChildrenAddedToStack.find(CurrInternalNode) !=
          ChildrenAddedToStack.end()) {
        // If the children of the current node has been added to the stack,
        // then this is the second time we visit this node and at this point,
        // all of its children have already been processed. Now, we can
        // set its LeftLeafIdx and RightLeafIdx;
        auto it = CurrInternalNode->Children.begin();
        if (it != CurrInternalNode->Children.end()) {
          // Get the first child to use its RightLeafIdx. The RightLeafIdx is
          // used as the first child is the initial one added to the stack, so
          // it's the last one to be processed. This implies that the leaf
          // descendants of the first child are assigned the largest index
          // numbers.
          CurrNode->setRightLeafIdx(it->second->getRightLeafIdx());
          // get the last child to use its LeftLeafIdx.
          while (std::next(it) != CurrInternalNode->Children.end())
            it = std::next(it);
          CurrNode->setLeftLeafIdx(it->second->getLeftLeafIdx());
          assert(CurrNode->getLeftLeafIdx() <= CurrNode->getRightLeafIdx() &&
                 "LeftLeafIdx should not be larger than RightLeafIdx");
        }
      } else {
        // This is the first time we visit this node. This means that its
        // children have not been added to the stack yet. Hence, we will add
        // the current node back to the stack and add its children to the
        // stack for processing.
        ToVisit.push(CurrNode);
        for (auto &ChildPair : CurrInternalNode->Children)
          ToVisit.push(ChildPair.second);
        ChildrenAddedToStack.insert(CurrInternalNode);
      }
    } else {
      // The current node is a leaf node.
      // We can simplyset its LeftLeafIdx and RightLeafIdx.
      CurrNode->setLeftLeafIdx(LeafCounter);
      CurrNode->setRightLeafIdx(LeafCounter);
      LeafCounter++;
      auto *CurrLeafNode = cast<SuffixTreeLeafNode>(CurrNode);
      LeafNodes.push_back(CurrLeafNode);
    }
  }
}

unsigned SuffixTree::extend(unsigned EndIdx, unsigned SuffixesToAdd) {
  SuffixTreeInternalNode *NeedsLink = nullptr;

  while (SuffixesToAdd > 0) {

    // Are we waiting to add anything other than just the last character?
    if (Active.Len == 0) {
      // If not, then say the active index is the end index.
      Active.Idx = EndIdx;
    }

    assert(Active.Idx <= EndIdx && "Start index can't be after end index!");

    // The first character in the current substring we're looking at.
    unsigned FirstChar = Str[Active.Idx];

    // Have we inserted anything starting with FirstChar at the current node?
    if (Active.Node->Children.count(FirstChar) == 0) {
      // If not, then we can just insert a leaf and move to the next step.
      insertLeaf(*Active.Node, EndIdx, FirstChar);

      // The active node is an internal node, and we visited it, so it must
      // need a link if it doesn't have one.
      if (NeedsLink) {
        NeedsLink->setLink(Active.Node);
        NeedsLink = nullptr;
      }
    } else {
      // There's a match with FirstChar, so look for the point in the tree to
      // insert a new node.
      SuffixTreeNode *NextNode = Active.Node->Children[FirstChar];

      unsigned SubstringLen = numElementsInSubstring(NextNode);

      // Is the current suffix we're trying to insert longer than the size of
      // the child we want to move to?
      if (Active.Len >= SubstringLen) {
        // If yes, then consume the characters we've seen and move to the next
        // node.
        assert(isa<SuffixTreeInternalNode>(NextNode) &&
               "Expected an internal node?");
        Active.Idx += SubstringLen;
        Active.Len -= SubstringLen;
        Active.Node = cast<SuffixTreeInternalNode>(NextNode);
        continue;
      }

      // Otherwise, the suffix we're trying to insert must be contained in the
      // next node we want to move to.
      unsigned LastChar = Str[EndIdx];

      // Is the string we're trying to insert a substring of the next node?
      if (Str[NextNode->getStartIdx() + Active.Len] == LastChar) {
        // If yes, then we're done for this step. Remember our insertion point
        // and move to the next end index. At this point, we have an implicit
        // suffix tree.
        if (NeedsLink && !Active.Node->isRoot()) {
          NeedsLink->setLink(Active.Node);
          NeedsLink = nullptr;
        }

        Active.Len++;
        break;
      }

      // The string we're trying to insert isn't a substring of the next node,
      // but matches up to a point. Split the node.
      //
      // For example, say we ended our search at a node n and we're trying to
      // insert ABD. Then we'll create a new node s for AB, reduce n to just
      // representing C, and insert a new leaf node l to represent d. This
      // allows us to ensure that if n was a leaf, it remains a leaf.
      //
      //   | ABC  ---split--->  | AB
      //   n                    s
      //                     C / \ D
      //                      n   l

      // The node s from the diagram
      SuffixTreeInternalNode *SplitNode = insertInternalNode(
          Active.Node, NextNode->getStartIdx(),
          NextNode->getStartIdx() + Active.Len - 1, FirstChar);

      // Insert the new node representing the new substring into the tree as
      // a child of the split node. This is the node l from the diagram.
      insertLeaf(*SplitNode, EndIdx, LastChar);

      // Make the old node a child of the split node and update its start
      // index. This is the node n from the diagram.
      NextNode->incrementStartIdx(Active.Len);
      SplitNode->Children[Str[NextNode->getStartIdx()]] = NextNode;

      // SplitNode is an internal node, update the suffix link.
      if (NeedsLink)
        NeedsLink->setLink(SplitNode);

      NeedsLink = SplitNode;
    }

    // We've added something new to the tree, so there's one less suffix to
    // add.
    SuffixesToAdd--;

    if (Active.Node->isRoot()) {
      if (Active.Len > 0) {
        Active.Len--;
        Active.Idx = EndIdx - SuffixesToAdd + 1;
      }
    } else {
      // Start the next phase at the next smallest suffix.
      Active.Node = Active.Node->getLink();
    }
  }

  return SuffixesToAdd;
}

void SuffixTree::RepeatedSubstringIterator::advance() {
  // Clear the current state. If we're at the end of the range, then this
  // is the state we want to be in.
  RS = RepeatedSubstring();
  N = nullptr;

  // Each leaf node represents a repeat of a string.
  SmallVector<unsigned> RepeatedSubstringStarts;
  SmallVector<SuffixTreeLeafNode *> LeafDescendants;

  // Continue visiting nodes until we find one which repeats more than once.
  while (!InternalNodesToVisit.empty()) {
    RepeatedSubstringStarts.clear();
    auto *Curr = InternalNodesToVisit.back();
    InternalNodesToVisit.pop_back();

    // Keep track of the length of the string associated with the node. If
    // it's too short, we'll quit.
    unsigned Length = Curr->getConcatLen();

    // Iterate over each child, saving internal nodes for visiting, and
    // leaf nodes in LeafChildren. Internal nodes represent individual
    // strings, which may repeat.
    for (auto &ChildPair : Curr->Children) {
      // Save all of this node's children for processing.
      if (auto *InternalChild =
              dyn_cast<SuffixTreeInternalNode>(ChildPair.second)) {
        InternalNodesToVisit.push_back(InternalChild);
        continue;
      }

      if (Length < MinLength || OutlinerLeafDescendants)
        continue;

      // Have an occurrence of a potentially repeated string. Save it.
      auto *Leaf = cast<SuffixTreeLeafNode>(ChildPair.second);
      RepeatedSubstringStarts.push_back(Leaf->getSuffixIdx());
    }

    if (OutlinerLeafDescendants && Length >= MinLength) {
      LeafDescendants.assign(LeafNodes.begin() + Curr->getLeftLeafIdx(),
                             LeafNodes.begin() + Curr->getRightLeafIdx() + 1);
      for (SuffixTreeLeafNode *Leaf : LeafDescendants)
        RepeatedSubstringStarts.push_back(Leaf->getSuffixIdx());
    }

    // The root never represents a repeated substring. If we're looking at
    // that, then skip it.
    if (Curr->isRoot())
      continue;

    // Do we have any repeated substrings?
    if (RepeatedSubstringStarts.size() < 2)
      continue;

    // Yes. Update the state to reflect this, and then bail out.
    N = Curr;
    RS.Length = Length;
    for (unsigned StartIdx : RepeatedSubstringStarts)
      RS.StartIndices.push_back(StartIdx);

    // Sort the start indices so that we can efficiently check if candidates
    // overlap with each other in MachineOutliner::findCandidates().
    llvm::sort(RS.StartIndices);

    break;
  }
  // At this point, either NewRS is an empty RepeatedSubstring, or it was
  // set in the above loop. Similarly, N is either nullptr, or the node
  // associated with NewRS.
}
