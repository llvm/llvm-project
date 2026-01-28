//===- Semilattice.h - A semilattice of KnownBits -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Builds a semilattice structure from the integral values in a given function,
// and holds an associated KnownBits for each value, representing the dataflow
// of KnownBits. Intended to be used to cache KnownBits for the entire function,
// with invalidation APIs.
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_SEMILATTICE_H
#define LLVM_ANALYSIS_SEMILATTICE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/KnownBits.h"

namespace llvm {
class Semilattice;
class Value;
class User;
class Function;
class DataLayout;

class SemilatticeNode {
  friend class Semilattice;
  PointerIntPair<Value *, 1> ValHasKnownBits;
  KnownBits Known;
  SmallVector<SemilatticeNode *, 4> Parents;
  SmallVector<SemilatticeNode *, 4> Children;

public:
  using iterator = SmallVectorImpl<SemilatticeNode *>::iterator;
  LLVM_ABI_FOR_TEST iterator child_begin() { // NOLINT
    return Children.begin();
  }
  LLVM_ABI_FOR_TEST iterator child_end() { return Children.end(); } // NOLINT
  LLVM_ABI_FOR_TEST iterator_range<iterator> children() {
    return make_range(child_begin(), child_end());
  }
  LLVM_ABI_FOR_TEST bool isLeaf() const { return Children.empty(); }
  LLVM_ABI_FOR_TEST bool isRoot() const { return Parents.empty(); }
  LLVM_ABI_FOR_TEST size_t getNumParents() const { return Parents.size(); }
  LLVM_ABI_FOR_TEST size_t getNumChildren() const { return Children.size(); }
  LLVM_ABI_FOR_TEST Value *getValue() const {
    return ValHasKnownBits.getPointer();
  }
  LLVM_ABI_FOR_TEST bool hasKnownBits() const {
    return ValHasKnownBits.getInt();
  }
  LLVM_ABI_FOR_TEST void setHasKnownBits() { ValHasKnownBits.setInt(1); }
  LLVM_ABI_FOR_TEST KnownBits getKnownBits() const { return Known; }
  LLVM_ABI_FOR_TEST void unionKnownWith(const KnownBits &NewKnown) {
    Known = Known.unionWith(NewKnown);
  }

protected:
  SemilatticeNode() : ValHasKnownBits(nullptr, 0) {}
  SemilatticeNode(Value *V, const DataLayout &DL);
  SemilatticeNode(const SemilatticeNode &) = delete;
  SemilatticeNode &operator=(const SemilatticeNode &) = delete;

  void resetKnownBits() { Known.resetAll(); }
  SemilatticeNode *addParent(SemilatticeNode *N) {
    if (!hasParent(N))
      Parents.push_back(N);
    return this;
  }
  SemilatticeNode *addChild(SemilatticeNode *N) {
    if (!is_contained(Children, N))
      Children.push_back(N);
    return this;
  }
  SemilatticeNode *rauw(Value *NewV) {
    ValHasKnownBits.setPointer(NewV);
    return this;
  }
  bool hasParent(SemilatticeNode *N) const { return is_contained(Parents, N); }
  void eraseParent(SemilatticeNode *P) {
    auto *It = find(Parents, P);
    if (It != Parents.end())
      Parents.erase(It);
  }
};

class Semilattice {
  using NodeT = SemilatticeNode;

  static constexpr size_t SlabSize = 8 * sizeof(NodeT);
  BumpPtrAllocatorImpl<MallocAllocator, SlabSize, /*SizeThreshold=*/SlabSize,
                       /*GrowthDelay=*/2>
      NodeAllocator;

  // The SentinelRoot is a sentinel value to allow for graph traverals to work
  // smoothly. Typically, to traverse the entire semilattice, a
  // drop_begin(depth_first(Lat->getSentinelRoot())) is used.
  NodeT *SentinelRoot;
  DenseMap<Value *, NodeT *> NodeMap;
  const DataLayout &DL;
  NodeT *create() { return new (NodeAllocator) NodeT(); }
  NodeT *create(Value *V) { return new (NodeAllocator) NodeT(V, DL); }
  NodeT *getOrCreate(Value *V) { return NodeMap.lookup_or(V, create(V)); }
  NodeT *insert(Value *V, NodeT *Parent = nullptr);
  SmallVector<NodeT *, 4> insert_range(NodeT *Parent, // NOLINT
                                       ArrayRef<User *> R);
  void recurseInsertChildren(ArrayRef<NodeT *>);
  SmallVector<NodeT *> invalidateKnownBits(NodeT *N);

  // The roots (excluding the sentinel value) are the arguments of the function,
  // and PHI nodes and Instructions like fptosi in each Basic Block, excluding
  // values whose types are not either integer or pointer, or a vector of those.
  void initialize(ArrayRef<Value *> Roots);
  void initialize(Function &F);

public:
  LLVM_ABI_FOR_TEST Semilattice(Function &F);
  LLVM_ABI_FOR_TEST Semilattice(const Semilattice &) = delete;
  LLVM_ABI_FOR_TEST Semilattice &operator=(const Semilattice &) = delete;

  LLVM_ABI_FOR_TEST NodeT *getSentinelRoot() const { return SentinelRoot; }
  LLVM_ABI_FOR_TEST bool empty() const { return SentinelRoot->isLeaf(); }
  LLVM_ABI_FOR_TEST bool contains(const Value *V) const {
    return NodeMap.contains(V);
  }
  LLVM_ABI_FOR_TEST NodeT *lookup(const Value *V) const {
    return NodeMap.lookup_or(V, SentinelRoot);
  }
  LLVM_ABI_FOR_TEST size_t size() const { return NodeMap.size(); }

  LLVM_ABI_FOR_TEST bool hasKnownBits(const Value *V) const {
    return lookup(V)->hasKnownBits();
  }
  LLVM_ABI_FOR_TEST KnownBits getKnownBits(const Value *V) const {
    return lookup(V)->getKnownBits();
  }
  LLVM_ABI_FOR_TEST void updateKnownBits(const Value *V,
                                         const KnownBits &Known) const {
    if (!contains(V))
      return;
    NodeT *LookupV = lookup(V);
    LookupV->unionKnownWith(Known);
    LookupV->setHasKnownBits();
  }

  // Theser functions return a reverse-breadth-first of the invalidated
  // subgraph.
  LLVM_ABI_FOR_TEST SmallVector<NodeT *> invalidateKnownBits(Value *V);
  LLVM_ABI_FOR_TEST SmallVector<NodeT *> rauw(Value *OldV, Value *NewV);

  LLVM_ABI_FOR_TEST void print(raw_ostream &OS) const;
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const;
#endif
};

template <typename NodeRef> struct NodeGraphTraitsBase {
  using ChildIteratorType = SemilatticeNode::iterator;
  using nodes_iterator = df_iterator<NodeRef, df_iterator_default_set<NodeRef>>;

  static NodeRef getEntryNode(NodeRef N) { return N; }
  static ChildIteratorType child_begin(NodeRef N) { // NOLINT
    return N->child_begin();
  }
  static ChildIteratorType child_end(NodeRef N) { // NOLINT
    return N->child_end();
  }
};

template <>
struct GraphTraits<SemilatticeNode *>
    : public NodeGraphTraitsBase<SemilatticeNode *> {
  using NodeRef = SemilatticeNode *;
};
template <>
struct GraphTraits<const SemilatticeNode *>
    : public NodeGraphTraitsBase<const SemilatticeNode *> {
  using NodeRef = const SemilatticeNode *;
};
} // end namespace llvm

#endif // LLVM_ANALYSIS_SEMILATTICE_H
