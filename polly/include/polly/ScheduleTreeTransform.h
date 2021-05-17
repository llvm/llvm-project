//===- polly/ScheduleTreeTransform.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Make changes to isl's schedule tree data structure.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_SCHEDULETREETRANSFORM_H
#define POLLY_SCHEDULETREETRANSFORM_H

#include "llvm/Support/ErrorHandling.h"
#include "isl/isl-noexceptions.h"
#include <cassert>

namespace polly {
struct BandAttr;

/// This class defines a simple visitor class that may be used for
/// various schedule tree analysis purposes.
template <typename Derived, typename RetTy = void, typename... Args>
struct ScheduleTreeVisitor {
  Derived &getDerived() { return *static_cast<Derived *>(this); }
  const Derived &getDerived() const {
    return *static_cast<const Derived *>(this);
  }

  RetTy visit(const isl::schedule_node &Node, Args... args) {
    assert(!Node.is_null());
    switch (isl_schedule_node_get_type(Node.get())) {
    case isl_schedule_node_domain:
      assert(isl_schedule_node_n_children(Node.get()) == 1);
      return getDerived().visitDomain(Node, std::forward<Args>(args)...);
    case isl_schedule_node_band:
      assert(isl_schedule_node_n_children(Node.get()) == 1);
      return getDerived().visitBand(Node, std::forward<Args>(args)...);
    case isl_schedule_node_sequence:
      assert(isl_schedule_node_n_children(Node.get()) >= 2);
      return getDerived().visitSequence(Node, std::forward<Args>(args)...);
    case isl_schedule_node_set:
      return getDerived().visitSet(Node, std::forward<Args>(args)...);
      assert(isl_schedule_node_n_children(Node.get()) >= 2);
    case isl_schedule_node_leaf:
      assert(isl_schedule_node_n_children(Node.get()) == 0);
      return getDerived().visitLeaf(Node, std::forward<Args>(args)...);
    case isl_schedule_node_mark:
      assert(isl_schedule_node_n_children(Node.get()) == 1);
      return getDerived().visitMark(Node, std::forward<Args>(args)...);
    case isl_schedule_node_extension:
      assert(isl_schedule_node_n_children(Node.get()) == 1);
      return getDerived().visitExtension(Node, std::forward<Args>(args)...);
    case isl_schedule_node_filter:
      assert(isl_schedule_node_n_children(Node.get()) == 1);
      return getDerived().visitFilter(Node, std::forward<Args>(args)...);
    default:
      llvm_unreachable("unimplemented schedule node type");
    }
  }

  RetTy visitDomain(const isl::schedule_node &Domain, Args... args) {
    return getDerived().visitSingleChild(Domain, std::forward<Args>(args)...);
  }

  RetTy visitBand(const isl::schedule_node &Band, Args... args) {
    return getDerived().visitSingleChild(Band, std::forward<Args>(args)...);
  }

  RetTy visitSequence(const isl::schedule_node &Sequence, Args... args) {
    return getDerived().visitMultiChild(Sequence, std::forward<Args>(args)...);
  }

  RetTy visitSet(const isl::schedule_node &Set, Args... args) {
    return getDerived().visitMultiChild(Set, std::forward<Args>(args)...);
  }

  RetTy visitLeaf(const isl::schedule_node &Leaf, Args... args) {
    return getDerived().visitNode(Leaf, std::forward<Args>(args)...);
  }

  RetTy visitMark(const isl::schedule_node &Mark, Args... args) {
    return getDerived().visitSingleChild(Mark, std::forward<Args>(args)...);
  }

  RetTy visitExtension(const isl::schedule_node &Extension, Args... args) {
    return getDerived().visitSingleChild(Extension,
                                         std::forward<Args>(args)...);
  }

  RetTy visitFilter(const isl::schedule_node &Extension, Args... args) {
    return getDerived().visitSingleChild(Extension,
                                         std::forward<Args>(args)...);
  }

  RetTy visitSingleChild(const isl::schedule_node &Node, Args... args) {
    return getDerived().visitNode(Node, std::forward<Args>(args)...);
  }

  RetTy visitMultiChild(const isl::schedule_node &Node, Args... args) {
    return getDerived().visitNode(Node, std::forward<Args>(args)...);
  }

  RetTy visitNode(const isl::schedule_node &Node, Args... args) {
    llvm_unreachable("Unimplemented other");
  }
};

/// Recursively visit all nodes of a schedule tree.
template <typename Derived, typename RetTy = void, typename... Args>
struct RecursiveScheduleTreeVisitor
    : public ScheduleTreeVisitor<Derived, RetTy, Args...> {
  using BaseTy = ScheduleTreeVisitor<Derived, RetTy, Args...>;
  BaseTy &getBase() { return *this; }
  const BaseTy &getBase() const { return *this; }
  Derived &getDerived() { return *static_cast<Derived *>(this); }
  const Derived &getDerived() const {
    return *static_cast<const Derived *>(this);
  }

  /// When visiting an entire schedule tree, start at its root node.
  RetTy visit(const isl::schedule &Schedule, Args... args) {
    return getDerived().visit(Schedule.get_root(), std::forward<Args>(args)...);
  }

  // Necessary to allow overload resolution with the added visit(isl::schedule)
  // overload.
  RetTy visit(const isl::schedule_node &Node, Args... args) {
    return getBase().visit(Node, std::forward<Args>(args)...);
  }

  /// By default, recursively visit the child nodes.
  RetTy visitNode(const isl::schedule_node &Node, Args... args) {
    isl_size NumChildren = Node.n_children();
    for (isl_size i = 0; i < NumChildren; i += 1)
      getDerived().visit(Node.child(i), std::forward<Args>(args)...);
    return RetTy();
  }
};

/// Is this node the marker for its parent band?
bool isBandMark(const isl::schedule_node &Node);

/// Extract the BandAttr from a band's wrapping marker. Can also pass the band
/// itself and this methods will try to find its wrapping mark. Returns nullptr
/// if the band has not BandAttr.
BandAttr *getBandAttr(isl::schedule_node MarkOrBand);

/// Hoist all domains from extension into the root domain node, such that there
/// are no more extension nodes (which isl does not support for some
/// operations). This assumes that domains added by to extension nodes do not
/// overlap.
isl::schedule hoistExtensionNodes(isl::schedule Sched);

/// Replace the AST band @p BandToUnroll by a sequence of all its iterations.
///
/// The implementation enumerates all points in the partial schedule and creates
/// an ISL sequence node for each point. The number of iterations must be a
/// constant.
isl::schedule applyFullUnroll(isl::schedule_node BandToUnroll);

/// Replace the AST band @p BandToUnroll by a partially unrolled equivalent.
isl::schedule applyPartialUnroll(isl::schedule_node BandToUnroll, int Factor);

} // namespace polly

#endif // POLLY_SCHEDULETREETRANSFORM_H
