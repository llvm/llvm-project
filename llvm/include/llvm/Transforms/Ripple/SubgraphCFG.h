//===- SubgraphCFG.h - CFG Subgraph for Dom/PDom computation ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines SubgraphCFG, a utility for computing dominator and
// post-dominator trees on a subgraph view of a CFG without modifying the IR.
// It provides a graph abstraction that skips specified basic blocks while
// maintaining proper graph traversal semantics.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_RIPPLE_SUBGRAPHCFG_H
#define LLVM_TRANSFORMS_RIPPLE_SUBGRAPHCFG_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/GenericDomTree.h"
#include "llvm/Support/GenericDomTreeConstruction.h"

namespace llvm {
namespace subgraphcfg {

// Forward-declare the graph owner so nodes can store a back-pointer.
class SubgraphCFG;

/// A wrapper node that carries a BasicBlock pointer and a back-pointer to the
/// subgraph. Successors and predecessors are computed on-the-fly using
/// the SubgraphCFG's IgnoreSet.
struct SubgraphBB {
  BasicBlock *BB = nullptr;
  const SubgraphCFG *G = nullptr;

  /// GenericDomTree requires NodeRef->getParent() to return a pointer to the
  /// parent type. We return our SubgraphCFG*, so the builder enumerates nodes
  /// via GraphTraits<SubgraphCFG*>.
  SubgraphCFG *getParent() const { return const_cast<SubgraphCFG *>(G); }

  /// Required for printing the dominator tree.
  void printAsOperand(raw_ostream &OS, bool PrintType) const {
    if (BB)
      BB->printAsOperand(OS, PrintType);
    else
      OS << "<<null>>";
  }
};

/// Graph owner: holds a map from BasicBlock* to SubgraphBB.
class SubgraphCFG {
public:
  using IgnoreSet = SmallPtrSet<BasicBlock *, 0>;
  using MapType = DenseMap<BasicBlock *, SubgraphBB>;

  /// Iterator adapter that extracts SubgraphBB* from the map's value_type.
  class node_iterator {
    MapType::iterator It;

  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = SubgraphBB *;
    using difference_type = std::ptrdiff_t;
    using pointer = SubgraphBB **;
    using reference = SubgraphBB *;

    node_iterator() = default;
    explicit node_iterator(MapType::iterator It) : It(It) {}

    SubgraphBB *operator*() const { return &It->second; }
    SubgraphBB *operator->() const { return &It->second; }

    node_iterator &operator++() {
      ++It;
      return *this;
    }

    node_iterator operator++(int) {
      node_iterator Tmp = *this;
      ++It;
      return Tmp;
    }

    bool operator==(const node_iterator &RHS) const { return It == RHS.It; }
    bool operator!=(const node_iterator &RHS) const { return It != RHS.It; }
  };

  SubgraphCFG(Function &F, const IgnoreSet &Ignore) : F(F) {
    // The entry block must not be in the ignored set.
    if (Ignore.count(&F.getEntryBlock()))
      llvm_unreachable("Entry basic block cannot be part of the ignored set");

    // Create all SubgraphBB nodes for non-ignored basic blocks.
    for (BasicBlock &B : F) {
      if (Ignore.count(&B))
        continue;
      Map[&B] = SubgraphBB{&B, this};
    }
  }

  SubgraphCFG(const SubgraphCFG &) = delete;
  SubgraphCFG &operator=(const SubgraphCFG &) = delete;

  SubgraphBB *get(BasicBlock *B) {
    auto It = Map.find(B);
    return (It == Map.end()) ? nullptr : &It->second;
  }

  const SubgraphBB *get(BasicBlock *B) const {
    auto It = Map.find(B);
    return (It == Map.end()) ? nullptr : &It->second;
  }

  /// Get the entry node of the CFG subgraph.
  SubgraphBB *getEntry() { return get(&F.getEntryBlock()); }

  const SubgraphBB *getEntry() const { return get(&F.getEntryBlock()); }

  /// Iteration over filtered nodes.
  node_iterator nodes_begin() { return node_iterator(Map.begin()); }
  node_iterator nodes_end() { return node_iterator(Map.end()); }
  size_t size() const { return Map.size(); }

  /// Populate the given vector with BasicBlocks that are ignored (excluded)
  /// from this SubgraphCFG.
  void getIgnoredBBs(SmallVectorImpl<BasicBlock *> &IgnoredBBs) const {
    for (BasicBlock &B : F) {
      if (!Map.count(&B))
        IgnoredBBs.push_back(&B);
    }
  }

private:
  Function &F;
  MapType Map;
};

/// Iterator implementation that wraps succ_iterator or pred_iterator and skips
/// ignored blocks.
///
/// This template is parameterized by:
/// - BaseIterator: The underlying CFG iterator (succ_iterator or pred_iterator)
/// - IteratorCategory: The iterator category (bidirectional or forward)
/// - GraphPtr: Pointer type to SubgraphCFG (const or non-const)
/// - ReturnType: The type returned by operator* (const or non-const
/// SubgraphBB*)
///
/// For succ_iterator (random access), we provide bidirectional iteration.
/// For pred_iterator (forward only), we provide forward iteration only.
template <typename BaseIterator, typename IteratorCategory, typename GraphPtr,
          typename ReturnType>
class FilteredEdgeIteratorImpl
    : public iterator_facade_base<
          FilteredEdgeIteratorImpl<BaseIterator, IteratorCategory, GraphPtr,
                                   ReturnType>,
          IteratorCategory, ReturnType, std::ptrdiff_t, ReturnType,
          ReturnType> {
  BaseIterator It;
  BaseIterator Begin;
  BaseIterator End;
  GraphPtr Graph;

  /// Skip forward over ignored blocks.
  void skipIgnoredForward() {
    while (It != End && Graph->get(*It) == nullptr)
      ++It;
  }

  /// Skip backward over ignored blocks.
  void skipIgnoredBackward() {
    while (It != Begin && Graph->get(*It) == nullptr)
      --It;
  }

public:
  FilteredEdgeIteratorImpl() : Graph(nullptr) {}

  FilteredEdgeIteratorImpl(BaseIterator Begin, BaseIterator End, GraphPtr Graph)
      : It(Begin), Begin(Begin), End(End), Graph(Graph) {
    skipIgnoredForward();
  }

  bool operator==(const FilteredEdgeIteratorImpl &RHS) const {
    return It == RHS.It;
  }

  ReturnType operator*() const { return Graph->get(*It); }

  FilteredEdgeIteratorImpl &operator++() {
    ++It;
    skipIgnoredForward();
    return *this;
  }

  /// Decrement operator - only enabled for bidirectional iterators.
  template <typename T = IteratorCategory>
  typename std::enable_if<
      std::is_base_of<std::bidirectional_iterator_tag, T>::value,
      FilteredEdgeIteratorImpl &>::type
  operator--() {
    --It;
    skipIgnoredBackward();
    return *this;
  }
};

/// Non-const edge iterator type alias.
template <typename BaseIterator, typename IteratorCategory>
using SubgraphEdgeIterator =
    FilteredEdgeIteratorImpl<BaseIterator, IteratorCategory, SubgraphCFG *,
                             SubgraphBB *>;

/// Const edge iterator type alias.
template <typename BaseIterator, typename IteratorCategory>
using ConstSubgraphEdgeIterator =
    FilteredEdgeIteratorImpl<BaseIterator, IteratorCategory,
                             const SubgraphCFG *, const SubgraphBB *>;

template <typename InstTy>
std::enable_if_t<std::is_base_of_v<Instruction, InstTy>, SubgraphCFG::IgnoreSet>
getAllBBsLeadingTo(Function &F) {
  SubgraphCFG::IgnoreSet IgnoreBBs;
  for (auto &BB : F)
    if (isa<InstTy>(BB.getTerminator())) {
      SmallVector<BasicBlock *, 8> ToProcess;
      ToProcess.push_back(&BB);
      while (!ToProcess.empty()) {
        auto *CurrentBB = ToProcess.back();
        ToProcess.pop_back();
        IgnoreBBs.insert(CurrentBB);
        // We add the predecessors that can only lead to the specified
        // terminator instruction type
        for (auto *Pred : predecessors(CurrentBB))
          if (Pred->getUniqueSuccessor() ||
              all_of(successors(Pred),
                     [&](auto *BB) { return IgnoreBBs.contains(BB); }))
            ToProcess.push_back(Pred);
      }
    }

  return IgnoreBBs;
}

} // namespace subgraphcfg

// GraphTraits specializations must be in the llvm namespace for ADL
/// GraphTraits specialization for forward CFG traversal.
/// Children are successors.
template <> struct GraphTraits<subgraphcfg::SubgraphBB *> {
  using NodeRef = subgraphcfg::SubgraphBB *;
  using ChildIteratorType =
      subgraphcfg::SubgraphEdgeIterator<succ_iterator,
                                        std::bidirectional_iterator_tag>;

  static NodeRef getEntryNode(NodeRef N) { return N; }
  static ChildIteratorType child_begin(NodeRef N) {
    return ChildIteratorType(succ_begin(N->BB), succ_end(N->BB),
                             N->getParent());
  }
  static ChildIteratorType child_end(NodeRef N) {
    return ChildIteratorType(succ_end(N->BB), succ_end(N->BB), N->getParent());
  }
};

/// GraphTraits specialization for inverse CFG traversal (for PDT).
/// Children are predecessors.
template <> struct GraphTraits<Inverse<subgraphcfg::SubgraphBB *>> {
  using NodeRef = subgraphcfg::SubgraphBB *;
  using ChildIteratorType =
      subgraphcfg::SubgraphEdgeIterator<pred_iterator,
                                        std::forward_iterator_tag>;

  static NodeRef getEntryNode(Inverse<NodeRef> N) { return N.Graph; }
  static ChildIteratorType child_begin(Inverse<NodeRef> N) {
    return ChildIteratorType(pred_begin(N.Graph->BB), pred_end(N.Graph->BB),
                             N.Graph->getParent());
  }
  static ChildIteratorType child_end(Inverse<NodeRef> N) {
    return ChildIteratorType(pred_end(N.Graph->BB), pred_end(N.Graph->BB),
                             N.Graph->getParent());
  }
};

/// Graph-level traits for node enumeration over the subgraph view.
/// The dominator builder uses NodeRef->getParent()'s GraphTraits to enumerate
/// nodes.
template <>
struct GraphTraits<subgraphcfg::SubgraphCFG *>
    : public GraphTraits<subgraphcfg::SubgraphBB *> {
  using nodes_iterator = subgraphcfg::SubgraphCFG::node_iterator;
  static subgraphcfg::SubgraphBB *getEntryNode(subgraphcfg::SubgraphCFG *G) {
    return G->getEntry();
  }
  static nodes_iterator nodes_begin(subgraphcfg::SubgraphCFG *G) {
    return G->nodes_begin();
  }
  static nodes_iterator nodes_end(subgraphcfg::SubgraphCFG *G) {
    return G->nodes_end();
  }
  static size_t size(subgraphcfg::SubgraphCFG *G) { return G->size(); }
};

/// Inverse graph-level traits for PostDominatorTree construction.
template <>
struct GraphTraits<Inverse<subgraphcfg::SubgraphCFG *>>
    : public GraphTraits<Inverse<subgraphcfg::SubgraphBB *>> {
  using nodes_iterator = subgraphcfg::SubgraphCFG::node_iterator;
  static subgraphcfg::SubgraphBB *
  getEntryNode(Inverse<subgraphcfg::SubgraphCFG *> G) {
    return G.Graph->getEntry();
  }
  static nodes_iterator nodes_begin(Inverse<subgraphcfg::SubgraphCFG *> G) {
    return G.Graph->nodes_begin();
  }
  static nodes_iterator nodes_end(Inverse<subgraphcfg::SubgraphCFG *> G) {
    return G.Graph->nodes_end();
  }
  static size_t size(Inverse<subgraphcfg::SubgraphCFG *> G) {
    return G.Graph->size();
  }
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_RIPPLE_SUBGRAPHCFG_H
