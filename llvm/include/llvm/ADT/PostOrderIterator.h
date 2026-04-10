//===- llvm/ADT/PostOrderIterator.h - PostOrder iterator --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file builds on the ADT/GraphTraits.h file to build a generic graph
/// post order iterator.  This should work over any graph type that has a
/// GraphTraits specialization.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_POSTORDERITERATOR_H
#define LLVM_ADT_POSTORDERITERATOR_H

#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include <iterator>
#include <optional>
#include <type_traits>
#include <utility>

namespace llvm {
namespace po_detail {

template <typename NodeRef> class NumberSet {
  SmallVector<bool> Data;

public:
  void reserve(size_t Size) {
    if (Size < Data.size())
      Data.resize(Size, false);
  }

  std::pair<std::nullopt_t, bool> insert(NodeRef Node) {
    unsigned Idx = GraphTraits<NodeRef>::getNumber(Node);
    if (Idx >= Data.size())
      Data.resize(Idx + 1);
    bool Inserted = !Data[Idx];
    Data[Idx] = true;
    return {std::nullopt, Inserted};
  }
};

template <typename GraphT>
using DefaultSet =
    std::conditional_t<GraphHasNodeNumbers<GraphT>,
                       NumberSet<typename GraphTraits<GraphT>::NodeRef>,
                       SmallPtrSet<typename GraphTraits<GraphT>::NodeRef, 8>>;

} // namespace po_detail

/// CRTP base class for post-order traversal. Storage for visited nodes must be
/// provided by the sub-class, which must implement insertEdge(). Due to CRTP
/// limitations, the sub-class must call init() with the start node before
/// traversing; not calling init results in an empty iterator.
///
/// Sub-classes can observe the post-order traversal with finishPostorder(),
/// which is called before the iterator moves to the next node, and also the
/// pre-order traversal with insertEdge().
///
/// Unwanted graph nodes (e.g. from a previous traversal) can be skipped by
/// returning false from insertEdge().
///
/// This class only supports a single traversal of the graph.
template <typename DerivedT, typename GraphTraits>
class PostOrderTraversalBase {
  using NodeRef = typename GraphTraits::NodeRef;
  using ChildItTy = typename GraphTraits::ChildIteratorType;

  /// Used to maintain the ordering.
  /// First element is basic block pointer, second is iterator for the next
  /// child to visit, third is the end iterator.
  SmallVector<std::tuple<NodeRef, ChildItTy, ChildItTy>, 8> VisitStack;

public:
  class iterator {
    friend class PostOrderTraversalBase;

  public:
    using iterator_category = std::input_iterator_tag;
    using value_type = NodeRef;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type *;
    using reference = NodeRef;

  private:
    DerivedT *POT = nullptr;
    NodeRef V = nullptr;

  public:
    iterator() = default;

  private:
    iterator(DerivedT &POT, value_type V) : POT(&POT), V(V) {}

  public:
    bool operator==(const iterator &X) const { return V == X.V; }
    bool operator!=(const iterator &X) const { return !(*this == X); }

    reference operator*() const { return V; }
    pointer operator->() const { return &V; }

    iterator &operator++() { // Preincrement
      V = POT->next();
      return *this;
    }

    iterator operator++(int) { // Postincrement
      iterator tmp = *this;
      ++*this;
      return tmp;
    }
  };

protected:
  PostOrderTraversalBase() = default;

  DerivedT *derived() { return static_cast<DerivedT *>(this); }

  /// Initialize post-order traversal at given start node.
  void init(NodeRef Start) {
    if (derived()->insertEdge(std::optional<NodeRef>(), Start)) {
      VisitStack.emplace_back(Start, GraphTraits::child_begin(Start),
                              GraphTraits::child_end(Start));
      traverseChild();
    }
  }

private:
  void traverseChild() {
    while (true) {
      auto &Entry = VisitStack.back();
      if (std::get<1>(Entry) == std::get<2>(Entry))
        break;
      NodeRef BB = *std::get<1>(Entry)++;
      if (derived()->insertEdge(std::optional<NodeRef>(std::get<0>(Entry)),
                                BB)) {
        // If the block is not visited...
        VisitStack.emplace_back(BB, GraphTraits::child_begin(BB),
                                GraphTraits::child_end(BB));
      }
    }
  }

  NodeRef next() {
    derived()->finishPostorder(std::get<0>(VisitStack.back()));
    VisitStack.pop_back();
    if (!VisitStack.empty())
      traverseChild();
    return !VisitStack.empty() ? std::get<0>(VisitStack.back()) : nullptr;
  }

public:
  iterator begin() {
    if (VisitStack.empty())
      return iterator(); // We don't even want to see the start node.
    return iterator(*derived(), std::get<0>(VisitStack.back()));
  }
  iterator end() { return iterator(); }

  // Methods that are intended to be overridden by sub-classes.

  /// Add edge and return whether To should be visited. From is nullopt for the
  /// root node.
  bool insertEdge(std::optional<NodeRef> From, NodeRef To);

  /// Callback just before the iterator moves to the next block.
  void finishPostorder(NodeRef) {}
};

/// Post-order traversal of a graph. Note: the traversal state is stored in this
/// class, not in the iterators -- the lifetime of PostOrderTraversal must
/// exceed the lifetime of the iterators. Special care must be taken with
/// range-based for-loops in combination with LLVM ranges:
///
///   // Fine:
///   for (BasicBlock *BB : post_order(F)) { ... }
///
///   // Problematic! Lifetime of PostOrderTraversal ends before the loop is
///   // entered, because make_filter_range only stores the iterators but not
///   // the range object itself.
///   for (BasicBlock *BB : make_filter_range(post_order(F), ...)) { ... }
///   // Fixed:
///   auto POT = post_order(F);
///   for (BasicBlock *BB : make_filter_range(POT, ...)) { ... }
///
/// This class only supports a single traversal of the graph.
template <typename GraphT, typename SetType = po_detail::DefaultSet<GraphT>>
class PostOrderTraversal
    : public PostOrderTraversalBase<PostOrderTraversal<GraphT, SetType>,
                                    GraphTraits<GraphT>> {
  using NodeRef = typename GraphTraits<GraphT>::NodeRef;

  SetType Visited;

public:
  /// Default constructor for an empty traversal.
  PostOrderTraversal() = default;

  /// Post-order traversal of the graph starting at the root node using an
  /// internal storage.
  PostOrderTraversal(const GraphT &G) {
    this->init(GraphTraits<GraphT>::getEntryNode(G));
  }

  bool insertEdge(std::optional<NodeRef> From, NodeRef To) {
    return Visited.insert(To).second;
  }
};

/// Post-order traversal of the graph starting at the root node using an
/// external storage. This can be used to keep track of visited nodes after the
/// traversal and to skip nodes that are already contained in the set. See
/// PostOrderTraversal for usage restrictions.
template <typename GraphT, typename SetType>
class PostOrderExtTraversal
    : public PostOrderTraversalBase<PostOrderExtTraversal<GraphT, SetType>,
                                    GraphTraits<GraphT>> {
  using NodeRef = typename GraphTraits<GraphT>::NodeRef;

  SetType &Visited;

public:
  PostOrderExtTraversal(const GraphT &G, SetType &S) : Visited(S) {
    this->init(GraphTraits<GraphT>::getEntryNode(G));
  }

  bool insertEdge(std::optional<NodeRef> From, NodeRef To) {
    return Visited.insert(To).second;
  }
};

// Provide global constructors that automatically figure out correct types...
//
/// Post-order traversal of a graph. Note: this returns a PostOrderTraversal,
/// not an iterator range; \see PostOrderTraversal.
template <class T> auto post_order(const T &G) {
  return PostOrderTraversal<T>(G);
}
template <class T, class SetType> auto post_order_ext(const T &G, SetType &S) {
  return PostOrderExtTraversal<T, SetType>(G, S);
}
template <class T, class SetType>
auto inverse_post_order_ext(const T &G, SetType &S) {
  return PostOrderExtTraversal<Inverse<T>, SetType>(G, S);
}

//===--------------------------------------------------------------------===//
// Reverse Post Order CFG iterator code
//===--------------------------------------------------------------------===//
//
// This is used to visit basic blocks in a method in reverse post order.  This
// class is awkward to use because I don't know a good incremental algorithm to
// computer RPO from a graph.  Because of this, the construction of the
// ReversePostOrderTraversal object is expensive (it must walk the entire graph
// with a postorder iterator to build the data structures).  The moral of this
// story is: Don't create more ReversePostOrderTraversal classes than necessary.
//
// Because it does the traversal in its constructor, it won't invalidate when
// BasicBlocks are removed, *but* it may contain erased blocks. Some places
// rely on this behavior (i.e. GVN).
//
// This class should be used like this:
// {
//   ReversePostOrderTraversal<Function*> RPOT(FuncPtr); // Expensive to create
//   for (rpo_iterator I = RPOT.begin(); I != RPOT.end(); ++I) {
//      ...
//   }
//   for (rpo_iterator I = RPOT.begin(); I != RPOT.end(); ++I) {
//      ...
//   }
// }
//

template<class GraphT, class GT = GraphTraits<GraphT>>
class ReversePostOrderTraversal {
  using NodeRef = typename GT::NodeRef;

  using VecTy = SmallVector<NodeRef, 8>;
  VecTy Blocks; // Block list in normal PO order

  void Initialize(const GraphT &G) {
    llvm::copy(post_order(G), std::back_inserter(Blocks));
  }

public:
  using rpo_iterator = typename VecTy::reverse_iterator;
  using const_rpo_iterator = typename VecTy::const_reverse_iterator;

  ReversePostOrderTraversal(const GraphT &G) { Initialize(G); }

  // Because we want a reverse post order, use reverse iterators from the vector
  rpo_iterator begin() { return Blocks.rbegin(); }
  const_rpo_iterator begin() const { return Blocks.rbegin(); }
  rpo_iterator end() { return Blocks.rend(); }
  const_rpo_iterator end() const { return Blocks.rend(); }
};

} // end namespace llvm

#endif // LLVM_ADT_POSTORDERITERATOR_H
