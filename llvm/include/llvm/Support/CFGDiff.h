//===- CFGDiff.h - Define a CFG snapshot. -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines specializations of GraphTraits that allows generic
// algorithms to see a different snapshot of a CFG.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_CFGDIFF_H
#define LLVM_SUPPORT_CFGDIFF_H

#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/CFGUpdate.h"
#include "llvm/Support/type_traits.h"
#include <cassert>
#include <cstddef>
#include <iterator>

// Two booleans are used to define orders in graphs:
// InverseGraph defines when we need to reverse the whole graph and is as such
// also equivalent to applying updates in reverse.
// InverseEdge defines whether we want to change the edges direction. E.g., for
// a non-inversed graph, the children are naturally the successors when
// InverseEdge is false and the predecessors when InverseEdge is true.

namespace llvm {

namespace detail {
template <typename Range>
auto reverse_if_helper(Range &&R, std::integral_constant<bool, false>) {
  return std::forward<Range>(R);
}

template <typename Range>
auto reverse_if_helper(Range &&R, std::integral_constant<bool, true>) {
  return llvm::reverse(std::forward<Range>(R));
}

template <bool B, typename Range> auto reverse_if(Range &&R) {
  return reverse_if_helper(std::forward<Range>(R),
                           std::integral_constant<bool, B>{});
}
} // namespace detail

// GraphDiff defines a CFG snapshot: given a set of Update<NodePtr>, provides
// a getChildren method to get a Node's children based on the additional updates
// in the snapshot. The current diff treats the CFG as a graph rather than a
// multigraph. Added edges are pruned to be unique, and deleted edges will
// remove all existing edges between two blocks.
template <typename NodePtr, bool InverseGraph = false> class GraphDiff {
  using UpdateMapType = SmallDenseMap<NodePtr, SmallVector<NodePtr, 2>>;
  struct EdgesInsertedDeleted {
    UpdateMapType Succ;
    UpdateMapType Pred;
  };
  // Store Deleted edges on position 0, and Inserted edges on position 1.
  EdgesInsertedDeleted Edges[2];
  // By default, it is assumed that, given a CFG and a set of updates, we wish
  // to apply these updates as given. If UpdatedAreReverseApplied is set, the
  // updates will be applied in reverse: deleted edges are considered re-added
  // and inserted edges are considered deleted when returning children.
  bool UpdatedAreReverseApplied;
  // Using a singleton empty vector for all node requests with no
  // children.
  SmallVector<NodePtr, 0> Empty;

  // Keep the list of legalized updates for a deterministic order of updates
  // when using a GraphDiff for incremental updates in the DominatorTree.
  // The list is kept in reverse to allow popping from end.
  SmallVector<cfg::Update<NodePtr>, 4> LegalizedUpdates;

  void printMap(raw_ostream &OS, const UpdateMapType &M) const {
    for (auto Pair : M)
      for (auto Child : Pair.second) {
        OS << "(";
        Pair.first->printAsOperand(OS, false);
        OS << ", ";
        Child->printAsOperand(OS, false);
        OS << ") ";
      }
    OS << "\n";
  }

public:
  GraphDiff() : UpdatedAreReverseApplied(false) {}
  GraphDiff(ArrayRef<cfg::Update<NodePtr>> Updates,
            bool ReverseApplyUpdates = false) {
    cfg::LegalizeUpdates<NodePtr>(Updates, LegalizedUpdates, InverseGraph);
    // The legalized updates are stored in reverse so we can pop_back when doing
    // incremental updates.
    for (auto U : LegalizedUpdates) {
      unsigned IsInsert =
          (U.getKind() == cfg::UpdateKind::Insert) == !ReverseApplyUpdates;
      Edges[IsInsert].Succ[U.getFrom()].push_back(U.getTo());
      Edges[IsInsert].Pred[U.getTo()].push_back(U.getFrom());
    }
    UpdatedAreReverseApplied = ReverseApplyUpdates;
  }

  auto getLegalizedUpdates() const {
    return make_range(LegalizedUpdates.begin(), LegalizedUpdates.end());
  }

  unsigned getNumLegalizedUpdates() const { return LegalizedUpdates.size(); }

  cfg::Update<NodePtr> popUpdateForIncrementalUpdates() {
    assert(!LegalizedUpdates.empty() && "No updates to apply!");
    auto U = LegalizedUpdates.pop_back_val();
    unsigned IsInsert =
        (U.getKind() == cfg::UpdateKind::Insert) == !UpdatedAreReverseApplied;
    auto &SuccList = Edges[IsInsert].Succ[U.getFrom()];
    assert(SuccList.back() == U.getTo());
    SuccList.pop_back();
    if (SuccList.empty())
      Edges[IsInsert].Succ.erase(U.getFrom());

    auto &PredList = Edges[IsInsert].Pred[U.getTo()];
    assert(PredList.back() == U.getFrom());
    PredList.pop_back();
    if (PredList.empty())
      Edges[IsInsert].Pred.erase(U.getTo());
    return U;
  }

  bool ignoreChild(const NodePtr BB, NodePtr EdgeEnd, bool InverseEdge) const {
    // Used to filter nullptr in clang.
    if (EdgeEnd == nullptr)
      return true;
    auto &DeleteChildren =
        (InverseEdge != InverseGraph) ? Edges[0].Pred : Edges[0].Succ;
    auto It = DeleteChildren.find(BB);
    if (It == DeleteChildren.end())
      return false;
    auto &EdgesForBB = It->second;
    return llvm::find(EdgesForBB, EdgeEnd) != EdgesForBB.end();
  }

  iterator_range<typename SmallVectorImpl<NodePtr>::const_iterator>
  getAddedChildren(const NodePtr BB, bool InverseEdge) const {
    auto &InsertChildren =
        (InverseEdge != InverseGraph) ? Edges[1].Pred : Edges[1].Succ;
    auto It = InsertChildren.find(BB);
    if (It == InsertChildren.end())
      return make_range(Empty.begin(), Empty.end());
    return make_range(It->second.begin(), It->second.end());
  }

  using VectRet = SmallVector<NodePtr, 8>;
  template <bool InverseEdge> VectRet getChildren(NodePtr N) const {
    using DirectedNodeT =
        std::conditional_t<InverseEdge, Inverse<NodePtr>, NodePtr>;
    auto R = children<DirectedNodeT>(N);
    auto CurrentCFGChildren = detail::reverse_if<!InverseEdge>(R);

    VectRet UpdatedCFGChildren;
    for (auto Child : CurrentCFGChildren)
      if (Child && !ignoreChild(N, Child, InverseEdge))
        UpdatedCFGChildren.push_back(Child);

    auto AddedCFGChildren = getAddedChildren(N, InverseEdge);
    UpdatedCFGChildren.insert(UpdatedCFGChildren.end(),
                              AddedCFGChildren.begin(), AddedCFGChildren.end());
    return UpdatedCFGChildren;
  }

  void print(raw_ostream &OS) const {
    OS << "===== GraphDiff: CFG edge changes to create a CFG snapshot. \n"
          "===== (Note: notion of children/inverse_children depends on "
          "the direction of edges and the graph.)\n";
    OS << "Children to insert:\n\t";
    printMap(OS, Edges[1].Succ);
    OS << "Children to delete:\n\t";
    printMap(OS, Edges[0].Succ);
    OS << "Inverse_children to insert:\n\t";
    printMap(OS, Edges[1].Pred);
    OS << "Inverse_children to delete:\n\t";
    printMap(OS, Edges[0].Pred);
    OS << "\n";
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const { print(dbgs()); }
#endif
};
} // end namespace llvm

#endif // LLVM_SUPPORT_CFGDIFF_H
