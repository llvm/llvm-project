//===- VPlanCFG.h - GraphTraits for VP blocks -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// Specializations of GraphTraits that allow VPBlockBase graphs to be
/// treated as proper graphs for generic algorithms;
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_VPLANCFG_H
#define LLVM_TRANSFORMS_VECTORIZE_VPLANCFG_H

#include "VPlan.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {

//===----------------------------------------------------------------------===//
// GraphTraits specializations for VPlan Hierarchical Control-Flow Graphs     //
//===----------------------------------------------------------------------===//

// The following set of template specializations implement GraphTraits to treat
// any VPBlockBase as a node in a graph of VPBlockBases. It's important to note
// that VPBlockBase traits don't recurse into VPRegioBlocks, i.e., if the
// VPBlockBase is a VPRegionBlock, this specialization provides access to its
// successors/predecessors but not to the blocks inside the region.

template <> struct GraphTraits<VPBlockBase *> {
  using NodeRef = VPBlockBase *;
  using ChildIteratorType = SmallVectorImpl<VPBlockBase *>::iterator;

  static NodeRef getEntryNode(NodeRef N) { return N; }

  static inline ChildIteratorType child_begin(NodeRef N) {
    return N->getSuccessors().begin();
  }

  static inline ChildIteratorType child_end(NodeRef N) {
    return N->getSuccessors().end();
  }
};

template <> struct GraphTraits<const VPBlockBase *> {
  using NodeRef = const VPBlockBase *;
  using ChildIteratorType = SmallVectorImpl<VPBlockBase *>::const_iterator;

  static NodeRef getEntryNode(NodeRef N) { return N; }

  static inline ChildIteratorType child_begin(NodeRef N) {
    return N->getSuccessors().begin();
  }

  static inline ChildIteratorType child_end(NodeRef N) {
    return N->getSuccessors().end();
  }
};

// Inverse order specialization for VPBasicBlocks. Predecessors are used instead
// of successors for the inverse traversal.
template <> struct GraphTraits<Inverse<VPBlockBase *>> {
  using NodeRef = VPBlockBase *;
  using ChildIteratorType = SmallVectorImpl<VPBlockBase *>::iterator;

  static NodeRef getEntryNode(Inverse<NodeRef> B) { return B.Graph; }

  static inline ChildIteratorType child_begin(NodeRef N) {
    return N->getPredecessors().begin();
  }

  static inline ChildIteratorType child_end(NodeRef N) {
    return N->getPredecessors().end();
  }
};

// The following set of template specializations implement GraphTraits to
// treat VPRegionBlock as a graph and recurse inside its nodes. It's important
// to note that the blocks inside the VPRegionBlock are treated as VPBlockBases
// (i.e., no dyn_cast is performed, VPBlockBases specialization is used), so
// there won't be automatic recursion into other VPBlockBases that turn to be
// VPRegionBlocks.

template <>
struct GraphTraits<VPRegionBlock *> : public GraphTraits<VPBlockBase *> {
  using GraphRef = VPRegionBlock *;
  using nodes_iterator = df_iterator<NodeRef>;

  static NodeRef getEntryNode(GraphRef N) { return N->getEntry(); }

  static nodes_iterator nodes_begin(GraphRef N) {
    return nodes_iterator::begin(N->getEntry());
  }

  static nodes_iterator nodes_end(GraphRef N) {
    // df_iterator::end() returns an empty iterator so the node used doesn't
    // matter.
    return nodes_iterator::end(N);
  }
};

template <>
struct GraphTraits<const VPRegionBlock *>
    : public GraphTraits<const VPBlockBase *> {
  using GraphRef = const VPRegionBlock *;
  using nodes_iterator = df_iterator<NodeRef>;

  static NodeRef getEntryNode(GraphRef N) { return N->getEntry(); }

  static nodes_iterator nodes_begin(GraphRef N) {
    return nodes_iterator::begin(N->getEntry());
  }

  static nodes_iterator nodes_end(GraphRef N) {
    // df_iterator::end() returns an empty iterator so the node used doesn't
    // matter.
    return nodes_iterator::end(N);
  }
};

template <>
struct GraphTraits<Inverse<VPRegionBlock *>>
    : public GraphTraits<Inverse<VPBlockBase *>> {
  using GraphRef = VPRegionBlock *;
  using nodes_iterator = df_iterator<NodeRef>;

  static NodeRef getEntryNode(Inverse<GraphRef> N) {
    return N.Graph->getExiting();
  }

  static nodes_iterator nodes_begin(GraphRef N) {
    return nodes_iterator::begin(N->getExiting());
  }

  static nodes_iterator nodes_end(GraphRef N) {
    // df_iterator::end() returns an empty iterator so the node used doesn't
    // matter.
    return nodes_iterator::end(N);
  }
};

/// Iterator to traverse all successors of a VPBlockBase node. This includes the
/// entry node of VPRegionBlocks. Exit blocks of a region implicitly have their
/// parent region's successors. This ensures all blocks in a region are visited
/// before any blocks in a successor region when doing a reverse post-order
// traversal of the graph.
template <typename BlockPtrTy>
class VPAllSuccessorsIterator
    : public iterator_facade_base<VPAllSuccessorsIterator<BlockPtrTy>,
                                  std::forward_iterator_tag, VPBlockBase> {
  BlockPtrTy Block;
  /// Index of the current successor. For VPBasicBlock nodes, this simply is the
  /// index for the successor array. For VPRegionBlock, SuccessorIdx == 0 is
  /// used for the region's entry block, and SuccessorIdx - 1 are the indices
  /// for the successor array.
  size_t SuccessorIdx;

  static BlockPtrTy getBlockWithSuccs(BlockPtrTy Current) {
    while (Current && Current->getNumSuccessors() == 0)
      Current = Current->getParent();
    return Current;
  }

  /// Templated helper to dereference successor \p SuccIdx of \p Block. Used by
  /// both the const and non-const operator* implementations.
  template <typename T1> static T1 deref(T1 Block, unsigned SuccIdx) {
    if (auto *R = dyn_cast<VPRegionBlock>(Block)) {
      if (SuccIdx == 0)
        return R->getEntry();
      SuccIdx--;
    }

    // For exit blocks, use the next parent region with successors.
    return getBlockWithSuccs(Block)->getSuccessors()[SuccIdx];
  }

public:
  VPAllSuccessorsIterator(BlockPtrTy Block, size_t Idx = 0)
      : Block(Block), SuccessorIdx(Idx) {}
  VPAllSuccessorsIterator(const VPAllSuccessorsIterator &Other)
      : Block(Other.Block), SuccessorIdx(Other.SuccessorIdx) {}

  VPAllSuccessorsIterator &operator=(const VPAllSuccessorsIterator &R) {
    Block = R.Block;
    SuccessorIdx = R.SuccessorIdx;
    return *this;
  }

  static VPAllSuccessorsIterator end(BlockPtrTy Block) {
    BlockPtrTy ParentWithSuccs = getBlockWithSuccs(Block);
    unsigned NumSuccessors =
        ParentWithSuccs ? ParentWithSuccs->getNumSuccessors() : 0;

    if (auto *R = dyn_cast<VPRegionBlock>(Block))
      return {R, NumSuccessors + 1};
    return {Block, NumSuccessors};
  }

  bool operator==(const VPAllSuccessorsIterator &R) const {
    return Block == R.Block && SuccessorIdx == R.SuccessorIdx;
  }

  const VPBlockBase *operator*() const { return deref(Block, SuccessorIdx); }

  BlockPtrTy operator*() { return deref(Block, SuccessorIdx); }

  VPAllSuccessorsIterator &operator++() {
    SuccessorIdx++;
    return *this;
  }

  VPAllSuccessorsIterator operator++(int X) {
    VPAllSuccessorsIterator Orig = *this;
    SuccessorIdx++;
    return Orig;
  }
};

/// Helper for GraphTraits specialization that traverses through VPRegionBlocks.
template <typename BlockTy> class VPBlockRecursiveTraversalWrapper {
  BlockTy Entry;

public:
  VPBlockRecursiveTraversalWrapper(BlockTy Entry) : Entry(Entry) {}
  BlockTy getEntry() { return Entry; }
};

/// GraphTraits specialization to recursively traverse VPBlockBase nodes,
/// including traversing through VPRegionBlocks.  Exit blocks of a region
/// implicitly have their parent region's successors. This ensures all blocks in
/// a region are visited before any blocks in a successor region when doing a
/// reverse post-order traversal of the graph.
template <>
struct GraphTraits<VPBlockRecursiveTraversalWrapper<VPBlockBase *>> {
  using NodeRef = VPBlockBase *;
  using ChildIteratorType = VPAllSuccessorsIterator<VPBlockBase *>;

  static NodeRef
  getEntryNode(VPBlockRecursiveTraversalWrapper<VPBlockBase *> N) {
    return N.getEntry();
  }

  static inline ChildIteratorType child_begin(NodeRef N) {
    return ChildIteratorType(N);
  }

  static inline ChildIteratorType child_end(NodeRef N) {
    return ChildIteratorType::end(N);
  }
};

template <>
struct GraphTraits<VPBlockRecursiveTraversalWrapper<const VPBlockBase *>> {
  using NodeRef = const VPBlockBase *;
  using ChildIteratorType = VPAllSuccessorsIterator<const VPBlockBase *>;

  static NodeRef
  getEntryNode(VPBlockRecursiveTraversalWrapper<const VPBlockBase *> N) {
    return N.getEntry();
  }

  static inline ChildIteratorType child_begin(NodeRef N) {
    return ChildIteratorType(N);
  }

  static inline ChildIteratorType child_end(NodeRef N) {
    return ChildIteratorType::end(N);
  }
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_VPLANCFG_H
