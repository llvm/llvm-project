
//===- Semilattice.h - A semilattice of KnownBits -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_SEMILATTICE_H
#define LLVM_ANALYSIS_SEMILATTICE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/KnownBits.h"

namespace llvm {
class Semilattice;
class Function;

class SemilatticeNode {
  friend class Semilattice;
  PointerIntPair<Value *, 1> ValHasKnownBits;
  KnownBits Known;
  SmallVector<SemilatticeNode *, 4> Parents;
  SmallVector<SemilatticeNode *, 4> Children;

public:
  SemilatticeNode() : ValHasKnownBits(nullptr, 0) {}
  SemilatticeNode(Value *V)
      : ValHasKnownBits(V, 0),
        Known(V->getType()->getScalarType()->getIntegerBitWidth()) {
    assert(V && "Attempting to create a node with an empty Value");
  }
  SemilatticeNode(const SemilatticeNode &) = delete;
  SemilatticeNode &operator=(const SemilatticeNode &) = delete;

public:
  using iterator = SmallVectorImpl<SemilatticeNode *>::iterator;
  iterator child_begin() { return Children.begin(); } // NOLINT
  iterator child_end() { return Children.end(); }     // NOLINT
  iterator_range<iterator> children() {
    return make_range(child_begin(), child_end());
  }
  bool isLeaf() const { return Children.empty(); }
  bool isRoot() const { return Parents.empty(); }

  bool hasParent(SemilatticeNode *N) const { return is_contained(Parents, N); }
  Value *getValue() const { return ValHasKnownBits.getPointer(); }
  bool hasKnownBits() const { return ValHasKnownBits.getInt(); }
  void setHasKnownBits() { ValHasKnownBits.setInt(1); }
  KnownBits getKnownBits() const { return Known; }
  void unionKnownWith(const KnownBits &NewKnown) {
    Known = Known.unionWith(NewKnown);
  }
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
};

class Semilattice {
  using NodeT = SemilatticeNode;

  static constexpr size_t SlabSize = 8 * sizeof(NodeT);
  BumpPtrAllocatorImpl<MallocAllocator, SlabSize, /*SizeThreshold=*/SlabSize,
                       /*GrowthDelay=*/2>
      NodeAllocator;
  NodeT *RootNode;
  DenseMap<Value *, NodeT *> NodeMap;
  NodeT *create() { return new (NodeAllocator) NodeT(); }
  NodeT *create(Value *V) { return new (NodeAllocator) NodeT(V); }
  NodeT *getOrCreate(Value *V) { return NodeMap.lookup_or(V, create(V)); }
  NodeT *insert(Value *V, NodeT *Parent = nullptr);
  SmallVector<NodeT *, 4> insert_range(NodeT *Parent, // NOLINT
                                       ArrayRef<User *> R);
  void recurseInsertChildren(ArrayRef<NodeT *>);
  void initialize(ArrayRef<Value *> Roots);
  void initialize(Function &F);

public:
  Semilattice(Function &F);
  Semilattice(const Semilattice &) = delete;
  Semilattice &operator=(const Semilattice &) = delete;
  Semilattice(Semilattice &&) = default;
  Semilattice &operator=(Semilattice &&) = default;

  NodeT *getRootNode() const { return RootNode; }
  bool empty() const { return RootNode->isLeaf(); }
  bool contains(const Value *V) const { return NodeMap.contains(V); }
  NodeT *lookup(const Value *V) const { return NodeMap.lookup_or(V, RootNode); }
  size_t size() const { return NodeMap.size(); }

  bool hasKnownBits(const Value *V) const { return lookup(V)->hasKnownBits(); }
  KnownBits getKnownBits(const Value *V) const {
    return lookup(V)->getKnownBits();
  }
  void updateKnownBits(const Value *V, const KnownBits &Known) const {
    if (!contains(V))
      return;
    NodeT *LookupV = lookup(V);
    LookupV->unionKnownWith(Known);
    LookupV->setHasKnownBits();
  }
  SmallVector<NodeT *> invalidateKnownBits(Value *V);
  SmallVector<NodeT *> rauw(Value *OldV, Value *NewV);

  void reset(Function &F);
  void print(raw_ostream &OS) const;
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
  static ChildIteratorType child_end(NodeRef N) {
    return N->child_end();
  } // NOLINT
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
