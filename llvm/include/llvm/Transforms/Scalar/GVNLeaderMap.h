//===- GVNLeaderMap.h - Leader map for GVN --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file / This file provides a data structure for mapping value numbers to
/// lists of values that have that value number.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_GVNLEADERMAP_H
#define LLVM_TRANSFORMS_SCALAR_GVNLEADERMAP_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/Allocator.h"

namespace llvm {

class GVNLeaderMap {
public:
  struct LeaderTableEntry {
    // Use AssertingVH here to catch dangling Value*'s in the leader table.
    // Will crash if the value gets deleted before the AssertingVH is
    // destroyed.
    AssertingVH<Value> Val;
    const BasicBlock *BB;
    LeaderTableEntry(Value *V, const BasicBlock *BB) : Val(V), BB(BB) {}
  };

private:
  struct LeaderListNode {
    LeaderTableEntry Entry;
    LeaderListNode *Next;
    LeaderListNode(Value *V, const BasicBlock *BB, LeaderListNode *Next)
        : Entry(V, BB), Next(Next) {}
  };
  DenseMap<uint32_t, LeaderListNode> NumToLeaders;
  BumpPtrAllocator TableAllocator;

public:
  class leader_iterator {
    const LeaderListNode *Current;

  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = const LeaderTableEntry;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type *;
    using reference = value_type &;

    leader_iterator(const LeaderListNode *C) : Current(C) {}
    leader_iterator &operator++() {
      assert(Current && "Dereferenced end of leader list!");
      Current = Current->Next;
      return *this;
    }
    bool operator==(const leader_iterator &Other) const {
      return Current == Other.Current;
    }
    bool operator!=(const leader_iterator &Other) const {
      return Current != Other.Current;
    }
    reference operator*() const { return Current->Entry; }
  };

  iterator_range<leader_iterator> getLeaders(uint32_t N) {
    auto I = NumToLeaders.find(N);
    if (I == NumToLeaders.end()) {
      return iterator_range(leader_iterator(nullptr), leader_iterator(nullptr));
    }

    return iterator_range(leader_iterator(&I->second),
                          leader_iterator(nullptr));
  }

  LLVM_ABI void insert(uint32_t N, Value *V, const BasicBlock *BB);
  LLVM_ABI void erase(uint32_t N, Instruction *I, const BasicBlock *BB);
  void clear() {
    // Manually destroy non-head nodes (in BumpPtrAllocator) to properly
    // clean up AssertingVH handles before Reset(). Head nodes are destroyed
    // by NumToLeaders.clear() below.
    for (auto &[_, HeadNode] : NumToLeaders) {
      LeaderListNode *N = HeadNode.Next;
      while (N) {
        auto *Next = N->Next;
        N->~LeaderListNode();
        N = Next;
      }
    }
    NumToLeaders.clear();
    TableAllocator.Reset();
  }
};
} // namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_GVNLEADERMAP_H