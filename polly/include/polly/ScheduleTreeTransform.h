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

#include "llvm/ADT/ArrayRef.h"
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

/// Build the desired set of partial tile prefixes.
///
/// We build a set of partial tile prefixes, which are prefixes of the vector
/// loop that have exactly VectorWidth iterations.
///
/// 1. Drop all constraints involving the dimension that represents the
///    vector loop.
/// 2. Constrain the last dimension to get a set, which has exactly VectorWidth
///    iterations.
/// 3. Subtract loop domain from it, project out the vector loop dimension and
///    get a set that contains prefixes, which do not have exactly VectorWidth
///    iterations.
/// 4. Project out the vector loop dimension of the set that was build on the
///    first step and subtract the set built on the previous step to get the
///    desired set of prefixes.
///
/// @param ScheduleRange A range of a map, which describes a prefix schedule
///                      relation.
isl::set getPartialTilePrefixes(isl::set ScheduleRange, int VectorWidth);

/// Create an isl::union_set, which describes the isolate option based on
/// IsolateDomain.
///
/// @param IsolateDomain An isl::set whose @p OutDimsNum last dimensions should
///                      belong to the current band node.
/// @param OutDimsNum    A number of dimensions that should belong to
///                      the current band node.
isl::union_set getIsolateOptions(isl::set IsolateDomain, isl_size OutDimsNum);

/// Create an isl::union_set, which describes the specified option for the
/// dimension of the current node.
///
/// @param Ctx    An isl::ctx, which is used to create the isl::union_set.
/// @param Option The name of the option.
isl::union_set getDimOptions(isl::ctx Ctx, const char *Option);

/// Tile a schedule node.
///
/// @param Node            The node to tile.
/// @param Identifier      An name that identifies this kind of tiling and
///                        that is used to mark the tiled loops in the
///                        generated AST.
/// @param TileSizes       A vector of tile sizes that should be used for
///                        tiling.
/// @param DefaultTileSize A default tile size that is used for dimensions
///                        that are not covered by the TileSizes vector.
isl::schedule_node tileNode(isl::schedule_node Node, const char *Identifier,
                            llvm::ArrayRef<int> TileSizes, int DefaultTileSize);

/// Tile a schedule node and unroll point loops.
///
/// @param Node            The node to register tile.
/// @param TileSizes       A vector of tile sizes that should be used for
///                        tiling.
/// @param DefaultTileSize A default tile size that is used for dimensions
isl::schedule_node applyRegisterTiling(isl::schedule_node Node,
                                       llvm::ArrayRef<int> TileSizes,
                                       int DefaultTileSize);

} // namespace polly

#endif // POLLY_SCHEDULETREETRANSFORM_H
