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
#include "VPlanUtils.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {

//===----------------------------------------------------------------------===//
// GraphTraits specializations for VPlan Hierarchical Control-Flow Graphs     //
//===----------------------------------------------------------------------===//

/// Iterator to traverse all successors/predecessors of a VPBlockBase node,
/// including its hierarchical successors/predecessors:
///
///     A
///     |
///  +-----+ <- Region R
///  |  b  |
///  |     |
///  | ... |
///  |     |
///  |  e  |
///  +-----+
///     |
///     B
///
///  Forward == true:
///    Region blocks themselves traverse only their entries directly.
///    Region's successor is implictly traversed when processing its exiting
///    block.
///    children(A) == {R}
///    children(R) == {b}
///    children(e) == {B}
///
///  Forward == false:
///    Region blocks themselves traverse only their exiting blocks directly.
///    Region's predecessor is implicitly traversed when processing its entry
///    block.
///    children(B) == {R}
///    children(R) == {e}
///    children(b) == {A}
///
/// The scheme described above ensures that all blocks of the region are visited
/// before continuing traversal outside the region when doing a reverse
/// post-order traversal of the VPlan.
template <typename BlockPtrTy, bool Forward = true>
class VPHierarchicalChildrenIterator
    : public iterator_facade_base<
          VPHierarchicalChildrenIterator<BlockPtrTy, Forward>,
          std::bidirectional_iterator_tag, VPBlockBase> {
  BlockPtrTy Block;
  /// Index of the current successor/predecessor. For VPBasicBlock nodes, this
  /// simply is the index for the successors/predecessors array. For
  /// VPRegionBlock, EdgeIdx == 0 is used for the region's entry/exiting block,
  /// and EdgeIdx - 1 are the indices for the successors/predecessors array.
  size_t EdgeIdx;

  static size_t getNumOutgoingEdges(BlockPtrTy Current) {
    if constexpr (Forward)
      return Current->getNumSuccessors();
    else
      return Current->getNumPredecessors();
  }

  static ArrayRef<BlockPtrTy> getOutgoingEdges(BlockPtrTy Current) {
    if constexpr (Forward)
      return Current->getSuccessors();
    else
      return Current->getPredecessors();
  }

  static BlockPtrTy getBlockWithOutgoingEdges(BlockPtrTy Current) {
    while (Current && getNumOutgoingEdges(Current) == 0)
      Current = Current->getParent();
    return Current;
  }

  /// Templated helper to dereference successor/predecessor \p EdgeIdx of \p
  /// Block. Used by both the const and non-const operator* implementations.
  template <typename T1> static T1 deref(T1 Block, unsigned EdgeIdx) {
    if (auto *R = dyn_cast<VPRegionBlock>(Block)) {
      assert(EdgeIdx == 0);
      if constexpr (Forward)
        return R->getEntry();
      else
        return R->getExiting();
    }

    // For exit blocks, use the next parent region with successors.
    return getOutgoingEdges(getBlockWithOutgoingEdges(Block))[EdgeIdx];
  }

public:
  /// Used by iterator_facade_base with bidirectional_iterator_tag.
  using reference = BlockPtrTy;

  VPHierarchicalChildrenIterator(BlockPtrTy Block, size_t Idx = 0)
      : Block(Block), EdgeIdx(Idx) {}
  VPHierarchicalChildrenIterator(const VPHierarchicalChildrenIterator &Other)
      : Block(Other.Block), EdgeIdx(Other.EdgeIdx) {}

  VPHierarchicalChildrenIterator &
  operator=(const VPHierarchicalChildrenIterator &R) {
    Block = R.Block;
    EdgeIdx = R.EdgeIdx;
    return *this;
  }

  static VPHierarchicalChildrenIterator end(BlockPtrTy Block) {
    if (auto *R = dyn_cast<VPRegionBlock>(Block)) {
      // Traverse through the region's entry/exiting (based on Forward) node.
      return {R, 1};
    }
    BlockPtrTy ParentWithOutgoingEdges = getBlockWithOutgoingEdges(Block);
    unsigned NumOutgoingEdges =
        ParentWithOutgoingEdges ? getNumOutgoingEdges(ParentWithOutgoingEdges)
                                : 0;
    return {Block, NumOutgoingEdges};
  }

  bool operator==(const VPHierarchicalChildrenIterator &R) const {
    return Block == R.Block && EdgeIdx == R.EdgeIdx;
  }

  const VPBlockBase *operator*() const { return deref(Block, EdgeIdx); }

  BlockPtrTy operator*() { return deref(Block, EdgeIdx); }

  VPHierarchicalChildrenIterator &operator++() {
    EdgeIdx++;
    return *this;
  }

  VPHierarchicalChildrenIterator &operator--() {
    EdgeIdx--;
    return *this;
  }

  VPHierarchicalChildrenIterator operator++(int X) {
    VPHierarchicalChildrenIterator Orig = *this;
    EdgeIdx++;
    return Orig;
  }
};

/// Helper for GraphTraits specialization that traverses through VPRegionBlocks.
template <typename BlockTy> class VPBlockDeepTraversalWrapper {
  BlockTy Entry;

public:
  VPBlockDeepTraversalWrapper(BlockTy Entry) : Entry(Entry) {}
  BlockTy getEntry() { return Entry; }
};

/// GraphTraits specialization to recursively traverse VPBlockBase nodes,
/// including traversing through VPRegionBlocks.  Exit blocks of a region
/// implicitly have their parent region's successors. This ensures all blocks in
/// a region are visited before any blocks in a successor region when doing a
/// reverse post-order traversal of the graph.
template <> struct GraphTraits<VPBlockDeepTraversalWrapper<VPBlockBase *>> {
  using NodeRef = VPBlockBase *;
  using ChildIteratorType = VPHierarchicalChildrenIterator<VPBlockBase *>;

  static NodeRef getEntryNode(VPBlockDeepTraversalWrapper<VPBlockBase *> N) {
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
struct GraphTraits<VPBlockDeepTraversalWrapper<const VPBlockBase *>> {
  using NodeRef = const VPBlockBase *;
  using ChildIteratorType = VPHierarchicalChildrenIterator<const VPBlockBase *>;

  static NodeRef
  getEntryNode(VPBlockDeepTraversalWrapper<const VPBlockBase *> N) {
    return N.getEntry();
  }

  static inline ChildIteratorType child_begin(NodeRef N) {
    return ChildIteratorType(N);
  }

  static inline ChildIteratorType child_end(NodeRef N) {
    return ChildIteratorType::end(N);
  }
};

/// Helper for GraphTraits specialization that does not traverses through
/// VPRegionBlocks.
template <typename BlockTy> class VPBlockShallowTraversalWrapper {
  BlockTy Entry;

public:
  VPBlockShallowTraversalWrapper(BlockTy Entry) : Entry(Entry) {}
  BlockTy getEntry() { return Entry; }
};

template <> struct GraphTraits<VPBlockShallowTraversalWrapper<VPBlockBase *>> {
  using NodeRef = VPBlockBase *;
  using ChildIteratorType = SmallVectorImpl<VPBlockBase *>::iterator;

  static NodeRef getEntryNode(VPBlockShallowTraversalWrapper<VPBlockBase *> N) {
    return N.getEntry();
  }

  static inline ChildIteratorType child_begin(NodeRef N) {
    return N->getSuccessors().begin();
  }

  static inline ChildIteratorType child_end(NodeRef N) {
    return N->getSuccessors().end();
  }
};

template <>
struct GraphTraits<VPBlockShallowTraversalWrapper<const VPBlockBase *>> {
  using NodeRef = const VPBlockBase *;
  using ChildIteratorType = SmallVectorImpl<VPBlockBase *>::const_iterator;

  static NodeRef
  getEntryNode(VPBlockShallowTraversalWrapper<const VPBlockBase *> N) {
    return N.getEntry();
  }

  static inline ChildIteratorType child_begin(NodeRef N) {
    return N->getSuccessors().begin();
  }

  static inline ChildIteratorType child_end(NodeRef N) {
    return N->getSuccessors().end();
  }
};

/// Returns an iterator range to traverse the graph starting at \p G in
/// depth-first order. The iterator won't traverse through region blocks.
inline iterator_range<
    df_iterator<VPBlockShallowTraversalWrapper<VPBlockBase *>>>
vp_depth_first_shallow(VPBlockBase *G) {
  return depth_first(VPBlockShallowTraversalWrapper<VPBlockBase *>(G));
}
inline iterator_range<
    df_iterator<VPBlockShallowTraversalWrapper<const VPBlockBase *>>>
vp_depth_first_shallow(const VPBlockBase *G) {
  return depth_first(VPBlockShallowTraversalWrapper<const VPBlockBase *>(G));
}

/// Returns an iterator range to traverse the graph starting at \p G in
/// post order. The iterator won't traverse through region blocks.
inline iterator_range<
    po_iterator<VPBlockShallowTraversalWrapper<VPBlockBase *>>>
vp_post_order_shallow(VPBlockBase *G) {
  return post_order(VPBlockShallowTraversalWrapper<VPBlockBase *>(G));
}

/// Returns an iterator range to traverse the graph starting at \p G in
/// post order while traversing through region blocks.
inline iterator_range<po_iterator<VPBlockDeepTraversalWrapper<VPBlockBase *>>>
vp_post_order_deep(VPBlockBase *G) {
  return post_order(VPBlockDeepTraversalWrapper<VPBlockBase *>(G));
}

/// Returns an iterator range to traverse the graph starting at \p G in
/// depth-first order while traversing through region blocks.
inline iterator_range<df_iterator<VPBlockDeepTraversalWrapper<VPBlockBase *>>>
vp_depth_first_deep(VPBlockBase *G) {
  return depth_first(VPBlockDeepTraversalWrapper<VPBlockBase *>(G));
}
inline iterator_range<
    df_iterator<VPBlockDeepTraversalWrapper<const VPBlockBase *>>>
vp_depth_first_deep(const VPBlockBase *G) {
  return depth_first(VPBlockDeepTraversalWrapper<const VPBlockBase *>(G));
}

// The following set of template specializations implement GraphTraits to treat
// any VPBlockBase as a node in a graph of VPBlockBases. It's important to note
// that VPBlockBase traits don't recurse into VPRegioBlocks, i.e., if the
// VPBlockBase is a VPRegionBlock, this specialization provides access to its
// successors/predecessors but not to the blocks inside the region.

template <> struct GraphTraits<VPBlockBase *> {
  using NodeRef = VPBlockBase *;
  using ChildIteratorType = VPHierarchicalChildrenIterator<VPBlockBase *>;

  static NodeRef getEntryNode(NodeRef N) { return N; }

  static inline ChildIteratorType child_begin(NodeRef N) {
    return ChildIteratorType(N);
  }

  static inline ChildIteratorType child_end(NodeRef N) {
    return ChildIteratorType::end(N);
  }
};

template <> struct GraphTraits<const VPBlockBase *> {
  using NodeRef = const VPBlockBase *;
  using ChildIteratorType = VPHierarchicalChildrenIterator<const VPBlockBase *>;

  static NodeRef getEntryNode(NodeRef N) { return N; }

  static inline ChildIteratorType child_begin(NodeRef N) {
    return ChildIteratorType(N);
  }

  static inline ChildIteratorType child_end(NodeRef N) {
    return ChildIteratorType::end(N);
  }
};

template <> struct GraphTraits<Inverse<VPBlockBase *>> {
  using NodeRef = VPBlockBase *;
  using ChildIteratorType =
      VPHierarchicalChildrenIterator<VPBlockBase *, /*Forward=*/false>;

  static NodeRef getEntryNode(Inverse<NodeRef> B) { return B.Graph; }

  static inline ChildIteratorType child_begin(NodeRef N) {
    return ChildIteratorType(N);
  }

  static inline ChildIteratorType child_end(NodeRef N) {
    return ChildIteratorType::end(N);
  }
};

template <> struct GraphTraits<VPlan *> {
  using GraphRef = VPlan *;
  using NodeRef = VPBlockBase *;
  using nodes_iterator = df_iterator<NodeRef>;

  static NodeRef getEntryNode(GraphRef N) { return N->getEntry(); }

  static nodes_iterator nodes_begin(GraphRef N) {
    return nodes_iterator::begin(N->getEntry());
  }

  static nodes_iterator nodes_end(GraphRef N) {
    // df_iterator::end() returns an empty iterator so the node used doesn't
    // matter.
    return nodes_iterator::end(N->getEntry());
  }
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_VPLANCFG_H
