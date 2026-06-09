//===-- OutlinedHashTree.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// An OutlinedHashTree is a Trie that contains sequences of stable hash values
// of instructions that have been outlined. This OutlinedHashTree can be used
// to understand the outlined instruction sequences collected across modules.
//
//===----------------------------------------------------------------------===//

#include "llvm/CGData/OutlinedHashTree.h"
#include "llvm/CGData/OutlinedHashTreeRecord.h"

#define DEBUG_TYPE "outlined-hash-tree"

using namespace llvm;

void MutableOutlinedHashTree::walkGraph(NodeCallbackFn CallbackNode,
                                        EdgeCallbackFn CallbackEdge,
                                        bool SortedWalk) const {
  SmallVector<const HashNode *> Stack;
  Stack.emplace_back(getRoot());

  while (!Stack.empty()) {
    const auto *Current = Stack.pop_back_val();
    if (CallbackNode)
      CallbackNode(Current);

    auto HandleNext = [&](const HashNode *Next) {
      if (CallbackEdge)
        CallbackEdge(Current, Next);
      Stack.emplace_back(Next);
    };
    if (SortedWalk) {
      SmallVector<std::pair<stable_hash, const HashNode *>> SortedSuccessors;
      for (const auto &[Hash, Successor] : Current->Successors)
        SortedSuccessors.emplace_back(Hash, Successor.get());
      llvm::sort(SortedSuccessors);
      for (const auto &P : SortedSuccessors)
        HandleNext(P.second);
    } else {
      for (const auto &P : Current->Successors)
        HandleNext(P.second.get());
    }
  }
}

size_t MutableOutlinedHashTree::size(bool GetTerminalCountOnly) const {
  size_t Size = 0;
  walkGraph([&Size, GetTerminalCountOnly](const HashNode *N) {
    Size += (N && (!GetTerminalCountOnly || N->Terminals));
  });
  return Size;
}

size_t MutableOutlinedHashTree::depth() const {
  size_t Size = 0;
  DenseMap<const HashNode *, size_t> DepthMap;
  walkGraph([&Size, &DepthMap](
                const HashNode *N) { Size = std::max(Size, DepthMap[N]); },
            [&DepthMap](const HashNode *Src, const HashNode *Dst) {
              size_t Depth = DepthMap[Src];
              DepthMap[Dst] = Depth + 1;
            });
  return Size;
}

void MutableOutlinedHashTree::insert(const HashSequencePair &SequencePair) {
  auto &[Sequence, Count] = SequencePair;
  HashNode *Current = getRoot();

  for (stable_hash StableHash : Sequence) {
    auto I = Current->Successors.find(StableHash);
    if (I == Current->Successors.end()) {
      std::unique_ptr<HashNode> Next = std::make_unique<HashNode>();
      HashNode *NextPtr = Next.get();
      NextPtr->Hash = StableHash;
      Current->Successors.try_emplace(StableHash, std::move(Next));
      Current = NextPtr;
    } else
      Current = I->second.get();
  }
  if (Count)
    Current->Terminals = Current->Terminals.value_or(0) + Count;
}

void MutableOutlinedHashTree::merge(const MutableOutlinedHashTree *Tree) {
  HashNode *Dst = getRoot();
  const HashNode *Src = Tree->getRoot();
  SmallVector<std::pair<HashNode *, const HashNode *>> Stack;
  Stack.emplace_back(Dst, Src);

  while (!Stack.empty()) {
    auto [DstNode, SrcNode] = Stack.pop_back_val();
    if (!SrcNode)
      continue;
    if (SrcNode->Terminals)
      DstNode->Terminals = DstNode->Terminals.value_or(0) + *SrcNode->Terminals;
    for (auto &[Hash, NextSrcNode] : SrcNode->Successors) {
      HashNode *NextDstNode;
      auto I = DstNode->Successors.find(Hash);
      if (I == DstNode->Successors.end()) {
        auto NextDst = std::make_unique<HashNode>();
        NextDstNode = NextDst.get();
        NextDstNode->Hash = Hash;
        DstNode->Successors.try_emplace(Hash, std::move(NextDst));
      } else
        NextDstNode = I->second.get();

      Stack.emplace_back(NextDstNode, NextSrcNode.get());
    }
  }
}

std::optional<unsigned>
OutlinedHashTree::find(const HashSequence &Sequence) const {
  HashNodeCursor Cursor = getRootCursor();
  for (stable_hash StableHash : Sequence) {
    auto Next = Cursor.getSuccessor(*this, StableHash);
    if (!Next)
      return 0;
    Cursor = *Next;
  }
  return Cursor.getTerminals(*this);
}

OutlinedHashTree::HashNodeCursor OutlinedHashTree::getRootCursor() const {
  if (isReadInPlace())
    // root block at region offset 0
    return HashNodeCursor(uint64_t(0));
  return HashNodeCursor(&Root);
}

std::optional<OutlinedHashTree::HashNodeCursor>
OutlinedHashTree::HashNodeCursor::getSuccessor(const OutlinedHashTree &Tree,
                                               stable_hash H) const {
  if (Tree.isReadInPlace()) {
    if (auto SuccessorOffset = OutlinedHashTreeRecord::readSuccessor(
            Tree.BlockBase + BlockOffset, H))
      return HashNodeCursor(*SuccessorOffset);
    return std::nullopt;
  }
  auto It = Node->Successors.find(H);
  if (It == Node->Successors.end())
    return std::nullopt;
  return HashNodeCursor(It->second.get());
}

std::optional<unsigned> OutlinedHashTree::HashNodeCursor::getTerminals(
    const OutlinedHashTree &Tree) const {
  if (Tree.isReadInPlace()) {
    unsigned T =
        OutlinedHashTreeRecord::readTerminals(Tree.BlockBase + BlockOffset);
    if (!T)
      return std::nullopt;
    return T;
  }
  return Node->Terminals;
}
