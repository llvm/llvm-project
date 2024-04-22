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

#include "llvm/CodeGenData/OutlinedHashTree.h"

#include <stack>
#include <tuple>

#define DEBUG_TYPE "outlined-hash-tree"

using namespace llvm;

void OutlinedHashTree::walkGraph(NodeCallbackFn CallbackNode,
                                 EdgeCallbackFn CallbackEdge,
                                 bool SortedWalk) const {
  std::stack<const HashNode *> Stack;
  Stack.push(getRoot());

  while (!Stack.empty()) {
    const auto *Current = Stack.top();
    Stack.pop();
    if (CallbackNode)
      CallbackNode(Current);

    auto HandleNext = [&](const HashNode *Next) {
      if (CallbackEdge)
        CallbackEdge(Current, Next);
      Stack.push(Next);
    };
    if (SortedWalk) {
      std::map<stable_hash, const HashNode *> SortedSuccessors;
      for (const auto &P : Current->Successors)
        SortedSuccessors[P.first] = P.second.get();
      for (const auto &P : SortedSuccessors)
        HandleNext(P.second);
    } else {
      for (const auto &P : Current->Successors)
        HandleNext(P.second.get());
    }
  }
}

size_t OutlinedHashTree::size(bool GetTerminalCountOnly) const {
  size_t Size = 0;
  walkGraph([&Size, GetTerminalCountOnly](const HashNode *N) {
    Size += (N && (!GetTerminalCountOnly || N->Terminals));
  });
  return Size;
}

size_t OutlinedHashTree::depth() const {
  size_t Size = 0;
  std::unordered_map<const HashNode *, size_t> DepthMap;
  walkGraph([&Size, &DepthMap](
                const HashNode *N) { Size = std::max(Size, DepthMap[N]); },
            [&DepthMap](const HashNode *Src, const HashNode *Dst) {
              size_t Depth = DepthMap[Src];
              DepthMap[Dst] = Depth + 1;
            });
  return Size;
}

void OutlinedHashTree::insert(const HashSequencePair &SequencePair) {
  const auto &Sequence = SequencePair.first;
  unsigned Count = SequencePair.second;
  HashNode *Current = getRoot();

  for (stable_hash StableHash : Sequence) {
    auto I = Current->Successors.find(StableHash);
    if (I == Current->Successors.end()) {
      std::unique_ptr<HashNode> Next = std::make_unique<HashNode>();
      HashNode *NextPtr = Next.get();
      NextPtr->Hash = StableHash;
      Current->Successors.emplace(StableHash, std::move(Next));
      Current = NextPtr;
    } else
      Current = I->second.get();
  }
  Current->Terminals += Count;
}

void OutlinedHashTree::merge(const OutlinedHashTree *Tree) {
  HashNode *Dst = getRoot();
  const HashNode *Src = Tree->getRoot();
  std::stack<std::pair<HashNode *, const HashNode *>> Stack;
  Stack.push({Dst, Src});

  while (!Stack.empty()) {
    auto [DstNode, SrcNode] = Stack.top();
    Stack.pop();
    if (!SrcNode)
      continue;
    DstNode->Terminals += SrcNode->Terminals;

    for (auto &[Hash, NextSrcNode] : SrcNode->Successors) {
      HashNode *NextDstNode;
      auto I = DstNode->Successors.find(Hash);
      if (I == DstNode->Successors.end()) {
        auto NextDst = std::make_unique<HashNode>();
        NextDstNode = NextDst.get();
        NextDstNode->Hash = Hash;
        DstNode->Successors.emplace(Hash, std::move(NextDst));
      } else
        NextDstNode = I->second.get();

      Stack.push({NextDstNode, NextSrcNode.get()});
    }
  }
}

unsigned OutlinedHashTree::find(const HashSequence &Sequence) const {
  const HashNode *Current = getRoot();
  for (stable_hash StableHash : Sequence) {
    const auto I = Current->Successors.find(StableHash);
    if (I == Current->Successors.end())
      return 0;
    Current = I->second.get();
  }
  return Current->Terminals;
}
