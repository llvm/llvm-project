//===- GIMatchDag.h - Represent a DAG to be matched -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UTILS_TABLEGEN_GIMATCHDAG_H
#define LLVM_UTILS_TABLEGEN_GIMATCHDAG_H

#include "GIMatchDagEdge.h"
#include "GIMatchDagInstr.h"
#include "GIMatchDagOperands.h"
#include "GIMatchDagPredicate.h"
#include "GIMatchDagPredicateDependencyEdge.h"

namespace llvm {
class GIMatchDag;

/// This class manages lifetimes for data associated with the GIMatchDag object.
class GIMatchDagContext {
  GIMatchDagOperandListContext OperandListCtx;

public:
  const GIMatchDagOperandList &makeEmptyOperandList() {
    return OperandListCtx.makeEmptyOperandList();
  }

  const GIMatchDagOperandList &makeOperandList(const CodeGenInstruction &I) {
    return OperandListCtx.makeOperandList(I);
  }

  const GIMatchDagOperandList &makeMIPredicateOperandList() {
    return OperandListCtx.makeMIPredicateOperandList();
  }


  const GIMatchDagOperandList &makeTwoMOPredicateOperandList() {
    return OperandListCtx.makeTwoMOPredicateOperandList();
  }

  void print(raw_ostream &OS) const {
    OperandListCtx.print(OS);
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const { print(errs()); }
#endif // if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
};

class GIMatchDag {
public:
  using InstrNodesVec = std::vector<std::unique_ptr<GIMatchDagInstr>>;
  using EdgesVec = std::vector<std::unique_ptr<GIMatchDagEdge>>;
  using edge_iterator = raw_pointer_iterator<EdgesVec::iterator>;
  using const_edge_iterator = raw_pointer_iterator<EdgesVec::const_iterator>;

  using PredicateNodesVec = std::vector<std::unique_ptr<GIMatchDagPredicate>>;

  using PredicateDependencyEdgesVec =
      std::vector<std::unique_ptr<GIMatchDagPredicateDependencyEdge>>;

protected:
  GIMatchDagContext &Ctx;
  InstrNodesVec InstrNodes;
  PredicateNodesVec PredicateNodes;
  EdgesVec Edges;
  PredicateDependencyEdgesVec PredicateDependencies;
  std::vector<GIMatchDagInstr *> MatchRoots;

public:
  GIMatchDag(GIMatchDagContext &Ctx)
      : Ctx(Ctx), InstrNodes(), PredicateNodes(), Edges(),
        PredicateDependencies() {}
  GIMatchDag(const GIMatchDag &) = delete;

  GIMatchDagContext &getContext() const { return Ctx; }
  edge_iterator edges_begin() {
    return raw_pointer_iterator<EdgesVec::iterator>(Edges.begin());
  }
  edge_iterator edges_end() {
    return raw_pointer_iterator<EdgesVec::iterator>(Edges.end());
  }
  const_edge_iterator edges_begin() const {
    return raw_pointer_iterator<EdgesVec::const_iterator>(Edges.begin());
  }
  const_edge_iterator edges_end() const {
    return raw_pointer_iterator<EdgesVec::const_iterator>(Edges.end());
  }
  iterator_range<edge_iterator> edges() {
    return make_range(edges_begin(), edges_end());
  }
  iterator_range<const_edge_iterator> edges() const {
    return make_range(edges_begin(), edges_end());
  }
  iterator_range<std::vector<GIMatchDagInstr *>::iterator> roots() {
    return make_range(MatchRoots.begin(), MatchRoots.end());
  }
  iterator_range<std::vector<GIMatchDagInstr *>::const_iterator> roots() const {
    return make_range(MatchRoots.begin(), MatchRoots.end());
  }

  template <class... Args> GIMatchDagInstr *addInstrNode(Args &&... args) {
    auto Obj =
        std::make_unique<GIMatchDagInstr>(*this, std::forward<Args>(args)...);
    auto ObjRaw = Obj.get();
    InstrNodes.push_back(std::move(Obj));
    return ObjRaw;
  }

  template <class T, class... Args>
  T *addPredicateNode(Args &&... args) {
    auto Obj = std::make_unique<T>(getContext(), std::forward<Args>(args)...);
    auto ObjRaw = Obj.get();
    PredicateNodes.push_back(std::move(Obj));
    return ObjRaw;
  }

  template <class... Args> GIMatchDagEdge *addEdge(Args &&... args) {
    auto Obj = std::make_unique<GIMatchDagEdge>(std::forward<Args>(args)...);
    auto ObjRaw = Obj.get();
    Edges.push_back(std::move(Obj));
    return ObjRaw;
  }

  template <class... Args>
  GIMatchDagPredicateDependencyEdge *addPredicateDependency(Args &&... args) {
    auto Obj = std::make_unique<GIMatchDagPredicateDependencyEdge>(
        std::forward<Args>(args)...);
    auto ObjRaw = Obj.get();
    PredicateDependencies.push_back(std::move(Obj));
    return ObjRaw;
  }

  void addMatchRoot(GIMatchDagInstr *N) { MatchRoots.push_back(N); }

  LLVM_DUMP_METHOD void print(raw_ostream &OS) const;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const { print(errs()); }
#endif // if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)

  void writeDOTGraph(raw_ostream &OS, StringRef ID) const;
};

raw_ostream &operator<<(raw_ostream &OS, const GIMatchDag &G);

} // end namespace llvm

#endif // ifndef LLVM_UTILS_TABLEGEN_GIMATCHDAG_H
