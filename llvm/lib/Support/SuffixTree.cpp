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
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SuffixTreeNode.h"

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
                       bool OutlinerLeafDescendants, unsigned BoundaryStart)
    : Str(Str), OutlinerLeafDescendants(OutlinerLeafDescendants),
      BoundaryStart(BoundaryStart) {
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
  // Subtract the parent path length to find the start of the suffix.
  N->setSuffixIdx(StartIdx - Parent.getConcatLen());
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
  if (Parent) {
    // Path length = parent path length + edge length.
    N->setConcatLen(Parent->getConcatLen() + (EndIdx - StartIdx + 1));
    Parent->Children[Edge] = N;
  }
  return N;
}

SuffixTreeInternalNode *SuffixTree::insertRoot() {
  return insertInternalNode(/*Parent = */ nullptr, SuffixTreeNode::EmptyIdx,
                            SuffixTreeNode::EmptyIdx, /*Edge = */ 0);
}

void SuffixTree::setLeafNodes() {
  // A stack that keeps track of nodes to visit for post-order DFS traversal.
  SmallVector<SuffixTreeInternalNode *> ToVisit;
  ToVisit.push_back(Root);

  // This keeps track of the index of the next leaf node to be added to
  // the LeafNodes vector of the suffix tree.
  unsigned LeafCounter = 0;

  // This keeps track of nodes whose children have been added to the stack.
  // The value is the index of its first leaf in LeafNodes.
  DenseMap<SuffixTreeInternalNode *, unsigned> ChildrenMap;

  // Traverse the tree in post-order.
  while (!ToVisit.empty()) {
    SuffixTreeInternalNode *CurrInternalNode = ToVisit.pop_back_val();
    auto I = ChildrenMap.find(CurrInternalNode);
    if (I == ChildrenMap.end()) {
      // This is the first time we visit this node.
      // Its children have not been added to the stack yet.
      // We add current node back, and add its internal children to the stack.
      auto J = CurrInternalNode->Children.begin();
      if (J != CurrInternalNode->Children.end()) {
        ToVisit.push_back(CurrInternalNode);
        ChildrenMap[CurrInternalNode] = LeafCounter;
        for (; J != CurrInternalNode->Children.end(); ++J) {
          if (auto *InternalChild =
                  dyn_cast<SuffixTreeInternalNode>(J->second)) {
            ToVisit.push_back(InternalChild);
          } else {
            LeafNodes.push_back(
                cast<SuffixTreeLeafNode>(J->second)->getSuffixIdx());
            ++LeafCounter;
          }
        }
      }
    } else {
      // This is the second time we visit this node.
      // All of its children have already been processed.
      // The root does not need a leaf range.
      if (CurrInternalNode->isRoot())
        continue;
      CurrInternalNode->setLeftLeafIdx(I->second);
      CurrInternalNode->setRightLeafIdx(LeafCounter - 1);
      assert(CurrInternalNode->getLeftLeafIdx() <=
                 CurrInternalNode->getRightLeafIdx() &&
             "LeftLeafIdx should not be larger than RightLeafIdx");
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
    // Each boundary symbol is unique, so it cannot already label an outgoing
    // edge at the active node. Skip the child lookup.
    if (auto It = isBoundary(FirstChar) ? Active.Node->Children.end()
                                        : Active.Node->Children.find(FirstChar);
        It == Active.Node->Children.end()) {
      // If not, then we can just insert a leaf and move to the next step.
      // Short boundary leaves cannot contribute to repeated substrings.
      if (!isBoundary(FirstChar) || Active.Node->getConcatLen() >= MinLength)
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
      SuffixTreeNode *NextNode = It->second;
      unsigned NextStartIdx = NextNode->getStartIdx();

      // A leaf edge reaches the current prefix end, so it cannot be consumed.
      if (auto *NextInternalNode = dyn_cast<SuffixTreeInternalNode>(NextNode)) {
        unsigned SubstringLen = numElementsInSubstring(NextInternalNode);

        // Consume the edge only when the active suffix reaches its end.
        if (Active.Len >= SubstringLen) {
          Active.Idx += SubstringLen;
          Active.Len -= SubstringLen;
          Active.Node = NextInternalNode;
          continue;
        }
      }

      // Otherwise, the suffix we're trying to insert must be contained in the
      // next node we want to move to.
      unsigned LastChar = Str[EndIdx];

      // Is the string we're trying to insert a substring of the next node?
      if (Str[NextStartIdx + Active.Len] == LastChar) {
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
          Active.Node, NextStartIdx, NextStartIdx + Active.Len - 1, FirstChar);

      // Insert the new node representing the new substring into the tree as
      // a child of the split node. This is the node l from the diagram.
      if (!isBoundary(LastChar) || SplitNode->getConcatLen() >= MinLength)
        insertLeaf(*SplitNode, EndIdx, LastChar);

      // Make the old node a child of the split node and update its start
      // index. This is the node n from the diagram.
      NextStartIdx += Active.Len;
      NextNode->incrementStartIdx(Active.Len);
      unsigned Edge = Str[NextStartIdx];
      if (!isBoundary(Edge) || SplitNode->getConcatLen() >= MinLength)
        SplitNode->Children[Edge] = NextNode;

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

  // Continue visiting nodes until we find one which repeats more than once.
  while (!InternalNodesToVisit.empty()) {
    RepeatedSubstringStarts.clear();
    auto *Curr = InternalNodesToVisit.pop_back_val();

    // Keep track of the length of the string associated with the node. If
    // it's too short, we'll quit.
    unsigned Length = Curr->getConcatLen();

    // Iterate over each child, saving internal nodes for visiting.
    // Internal nodes represent individual strings, which may repeat.
    for (auto &ChildPair : Curr->Children)
      // Save all of this node's children for processing.
      if (auto *InternalChild =
              dyn_cast<SuffixTreeInternalNode>(ChildPair.second))
        InternalNodesToVisit.push_back(InternalChild);

    // If length of repeated substring is below threshold, then skip it.
    if (Length < MinLength)
      continue;

    // The root never represents a repeated substring. If we're looking at
    // that, then skip it.
    if (Curr->isRoot())
      continue;

    // Collect leaf children or leaf descendants by OutlinerLeafDescendants.
    if (OutlinerLeafDescendants) {
      // Leaf descendants occupy a contiguous range.
      RepeatedSubstringStarts.append(LeafNodes.begin() + Curr->getLeftLeafIdx(),
                                     LeafNodes.begin() +
                                         Curr->getRightLeafIdx() + 1);
    } else {
      for (auto &ChildPair : Curr->Children)
        if (auto *Leaf = dyn_cast<SuffixTreeLeafNode>(ChildPair.second))
          RepeatedSubstringStarts.push_back(Leaf->getSuffixIdx());
    }

    // Do we have any repeated substrings?
    if (RepeatedSubstringStarts.size() < 2)
      continue;

    // Yes. Update the state to reflect this, and then bail out.
    N = Curr;
    RS.Length = Length;
    llvm::append_range(RS.StartIndices, RepeatedSubstringStarts);
    break;
  }
  // At this point, either NewRS is an empty RepeatedSubstring, or it was
  // set in the above loop. Similarly, N is either nullptr, or the node
  // associated with NewRS.
}
