//===- polly/ScheduleTreeTransform.cpp --------------------------*- C++ -*-===//
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

#include "polly/ScheduleTreeTransform.h"
#include "polly/Support/ISLTools.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

using namespace polly;

namespace {

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

  RetTy visitNode(const isl::schedule_node &Node, Args... args) {
    int NumChildren = isl_schedule_node_n_children(Node.get());
    for (int i = 0; i < NumChildren; i += 1)
      getDerived().visit(Node.child(i), std::forward<Args>(args)...);
    return RetTy();
  }
};

/// Recursively visit all nodes of a schedule tree while allowing changes.
///
/// The visit methods return an isl::schedule_node that is used to continue
/// visiting the tree. Structural changes such as returning a different node
/// will confuse the visitor.
template <typename Derived, typename... Args>
struct ScheduleNodeRewriter
    : public RecursiveScheduleTreeVisitor<Derived, isl::schedule_node,
                                          Args...> {
  Derived &getDerived() { return *static_cast<Derived *>(this); }
  const Derived &getDerived() const {
    return *static_cast<const Derived *>(this);
  }

  isl::schedule_node visitNode(const isl::schedule_node &Node, Args... args) {
    if (!Node.has_children())
      return Node;

    isl::schedule_node It = Node.first_child();
    while (true) {
      It = getDerived().visit(It, std::forward<Args>(args)...);
      if (!It.has_next_sibling())
        break;
      It = It.next_sibling();
    }
    return It.parent();
  }
};

/// Rewrite a schedule tree by reconstructing it bottom-up.
///
/// By default, the original schedule tree is reconstructed. To build a
/// different tree, redefine visitor methods in a derived class (CRTP).
///
/// Note that AST build options are not applied; Setting the isolate[] option
/// makes the schedule tree 'anchored' and cannot be modified afterwards. Hence,
/// AST build options must be set after the tree has been constructed.
template <typename Derived, typename... Args>
struct ScheduleTreeRewriter
    : public RecursiveScheduleTreeVisitor<Derived, isl::schedule, Args...> {
  Derived &getDerived() { return *static_cast<Derived *>(this); }
  const Derived &getDerived() const {
    return *static_cast<const Derived *>(this);
  }

  isl::schedule visitDomain(const isl::schedule_node &Node, Args... args) {
    // Every schedule_tree already has a domain node, no need to add one.
    return getDerived().visit(Node.first_child(), std::forward<Args>(args)...);
  }

  isl::schedule visitBand(const isl::schedule_node &Band, Args... args) {
    isl::multi_union_pw_aff PartialSched =
        isl::manage(isl_schedule_node_band_get_partial_schedule(Band.get()));
    isl::schedule NewChild =
        getDerived().visit(Band.child(0), std::forward<Args>(args)...);
    isl::schedule_node NewNode =
        NewChild.insert_partial_schedule(PartialSched).get_root().get_child(0);

    // Reapply permutability and coincidence attributes.
    NewNode = isl::manage(isl_schedule_node_band_set_permutable(
        NewNode.release(), isl_schedule_node_band_get_permutable(Band.get())));
    unsigned BandDims = isl_schedule_node_band_n_member(Band.get());
    for (unsigned i = 0; i < BandDims; i += 1)
      NewNode = isl::manage(isl_schedule_node_band_member_set_coincident(
          NewNode.release(), i,
          isl_schedule_node_band_member_get_coincident(Band.get(), i)));

    return NewNode.get_schedule();
  }

  isl::schedule visitSequence(const isl::schedule_node &Sequence,
                              Args... args) {
    int NumChildren = isl_schedule_node_n_children(Sequence.get());
    isl::schedule Result =
        getDerived().visit(Sequence.child(0), std::forward<Args>(args)...);
    for (int i = 1; i < NumChildren; i += 1)
      Result = Result.sequence(
          getDerived().visit(Sequence.child(i), std::forward<Args>(args)...));
    return Result;
  }

  isl::schedule visitSet(const isl::schedule_node &Set, Args... args) {
    int NumChildren = isl_schedule_node_n_children(Set.get());
    isl::schedule Result =
        getDerived().visit(Set.child(0), std::forward<Args>(args)...);
    for (int i = 1; i < NumChildren; i += 1)
      Result = isl::manage(
          isl_schedule_set(Result.release(),
                           getDerived()
                               .visit(Set.child(i), std::forward<Args>(args)...)
                               .release()));
    return Result;
  }

  isl::schedule visitLeaf(const isl::schedule_node &Leaf, Args... args) {
    return isl::schedule::from_domain(Leaf.get_domain());
  }

  isl::schedule visitMark(const isl::schedule_node &Mark, Args... args) {
    isl::id TheMark = Mark.mark_get_id();
    isl::schedule_node NewChild =
        getDerived()
            .visit(Mark.first_child(), std::forward<Args>(args)...)
            .get_root()
            .first_child();
    return NewChild.insert_mark(TheMark).get_schedule();
  }

  isl::schedule visitExtension(const isl::schedule_node &Extension,
                               Args... args) {
    isl::union_map TheExtension = Extension.extension_get_extension();
    isl::schedule_node NewChild = getDerived()
                                      .visit(Extension.child(0), args...)
                                      .get_root()
                                      .first_child();
    isl::schedule_node NewExtension =
        isl::schedule_node::from_extension(TheExtension);
    return NewChild.graft_before(NewExtension).get_schedule();
  }

  isl::schedule visitFilter(const isl::schedule_node &Filter, Args... args) {
    isl::union_set FilterDomain = Filter.filter_get_filter();
    isl::schedule NewSchedule =
        getDerived().visit(Filter.child(0), std::forward<Args>(args)...);
    return NewSchedule.intersect_domain(FilterDomain);
  }

  isl::schedule visitNode(const isl::schedule_node &Node, Args... args) {
    llvm_unreachable("Not implemented");
  }
};

/// Rewrite a schedule tree to an equivalent one without extension nodes.
///
/// Each visit method takes two additional arguments:
///
///  * The new domain the node, which is the inherited domain plus any domains
///    added by extension nodes.
///
///  * A map of extension domains of all children is returned; it is required by
///    band nodes to schedule the additional domains at the same position as the
///    extension node would.
///
struct ExtensionNodeRewriter
    : public ScheduleTreeRewriter<ExtensionNodeRewriter, const isl::union_set &,
                                  isl::union_map &> {
  using BaseTy = ScheduleTreeRewriter<ExtensionNodeRewriter,
                                      const isl::union_set &, isl::union_map &>;
  BaseTy &getBase() { return *this; }
  const BaseTy &getBase() const { return *this; }

  isl::schedule visitSchedule(const isl::schedule &Schedule) {
    isl::union_map Extensions;
    isl::schedule Result =
        visit(Schedule.get_root(), Schedule.get_domain(), Extensions);
    assert(Extensions && Extensions.is_empty());
    return Result;
  }

  isl::schedule visitSequence(const isl::schedule_node &Sequence,
                              const isl::union_set &Domain,
                              isl::union_map &Extensions) {
    int NumChildren = isl_schedule_node_n_children(Sequence.get());
    isl::schedule NewNode = visit(Sequence.first_child(), Domain, Extensions);
    for (int i = 1; i < NumChildren; i += 1) {
      isl::schedule_node OldChild = Sequence.child(i);
      isl::union_map NewChildExtensions;
      isl::schedule NewChildNode = visit(OldChild, Domain, NewChildExtensions);
      NewNode = NewNode.sequence(NewChildNode);
      Extensions = Extensions.unite(NewChildExtensions);
    }
    return NewNode;
  }

  isl::schedule visitSet(const isl::schedule_node &Set,
                         const isl::union_set &Domain,
                         isl::union_map &Extensions) {
    int NumChildren = isl_schedule_node_n_children(Set.get());
    isl::schedule NewNode = visit(Set.first_child(), Domain, Extensions);
    for (int i = 1; i < NumChildren; i += 1) {
      isl::schedule_node OldChild = Set.child(i);
      isl::union_map NewChildExtensions;
      isl::schedule NewChildNode = visit(OldChild, Domain, NewChildExtensions);
      NewNode = isl::manage(
          isl_schedule_set(NewNode.release(), NewChildNode.release()));
      Extensions = Extensions.unite(NewChildExtensions);
    }
    return NewNode;
  }

  isl::schedule visitLeaf(const isl::schedule_node &Leaf,
                          const isl::union_set &Domain,
                          isl::union_map &Extensions) {
    isl::ctx Ctx = Leaf.get_ctx();
    Extensions = isl::union_map::empty(isl::space::params_alloc(Ctx, 0));
    return isl::schedule::from_domain(Domain);
  }

  isl::schedule visitBand(const isl::schedule_node &OldNode,
                          const isl::union_set &Domain,
                          isl::union_map &OuterExtensions) {
    isl::schedule_node OldChild = OldNode.first_child();
    isl::multi_union_pw_aff PartialSched =
        isl::manage(isl_schedule_node_band_get_partial_schedule(OldNode.get()));

    isl::union_map NewChildExtensions;
    isl::schedule NewChild = visit(OldChild, Domain, NewChildExtensions);

    // Add the extensions to the partial schedule.
    OuterExtensions = isl::union_map::empty(NewChildExtensions.get_space());
    isl::union_map NewPartialSchedMap = isl::union_map::from(PartialSched);
    unsigned BandDims = isl_schedule_node_band_n_member(OldNode.get());
    for (isl::map Ext : NewChildExtensions.get_map_list()) {
      unsigned ExtDims = Ext.dim(isl::dim::in);
      assert(ExtDims >= BandDims);
      unsigned OuterDims = ExtDims - BandDims;

      isl::map BandSched =
          Ext.project_out(isl::dim::in, 0, OuterDims).reverse();
      NewPartialSchedMap = NewPartialSchedMap.unite(BandSched);

      // There might be more outer bands that have to schedule the extensions.
      if (OuterDims > 0) {
        isl::map OuterSched =
            Ext.project_out(isl::dim::in, OuterDims, BandDims);
        OuterExtensions = OuterExtensions.add_map(OuterSched);
      }
    }
    isl::multi_union_pw_aff NewPartialSchedAsAsMultiUnionPwAff =
        isl::multi_union_pw_aff::from_union_map(NewPartialSchedMap);
    isl::schedule_node NewNode =
        NewChild.insert_partial_schedule(NewPartialSchedAsAsMultiUnionPwAff)
            .get_root()
            .get_child(0);

    // Reapply permutability and coincidence attributes.
    NewNode = isl::manage(isl_schedule_node_band_set_permutable(
        NewNode.release(),
        isl_schedule_node_band_get_permutable(OldNode.get())));
    for (unsigned i = 0; i < BandDims; i += 1) {
      NewNode = isl::manage(isl_schedule_node_band_member_set_coincident(
          NewNode.release(), i,
          isl_schedule_node_band_member_get_coincident(OldNode.get(), i)));
    }

    return NewNode.get_schedule();
  }

  isl::schedule visitFilter(const isl::schedule_node &Filter,
                            const isl::union_set &Domain,
                            isl::union_map &Extensions) {
    isl::union_set FilterDomain = Filter.filter_get_filter();
    isl::union_set NewDomain = Domain.intersect(FilterDomain);

    // A filter is added implicitly if necessary when joining schedule trees.
    return visit(Filter.first_child(), NewDomain, Extensions);
  }

  isl::schedule visitExtension(const isl::schedule_node &Extension,
                               const isl::union_set &Domain,
                               isl::union_map &Extensions) {
    isl::union_map ExtDomain = Extension.extension_get_extension();
    isl::union_set NewDomain = Domain.unite(ExtDomain.range());
    isl::union_map ChildExtensions;
    isl::schedule NewChild =
        visit(Extension.first_child(), NewDomain, ChildExtensions);
    Extensions = ChildExtensions.unite(ExtDomain);
    return NewChild;
  }
};

/// Collect all AST build options in any schedule tree band.
///
/// ScheduleTreeRewriter cannot apply the schedule tree options. This class
/// collects these options to apply them later.
struct CollectASTBuildOptions
    : public RecursiveScheduleTreeVisitor<CollectASTBuildOptions> {
  using BaseTy = RecursiveScheduleTreeVisitor<CollectASTBuildOptions>;
  BaseTy &getBase() { return *this; }
  const BaseTy &getBase() const { return *this; }

  llvm::SmallVector<isl::union_set, 8> ASTBuildOptions;

  void visitBand(const isl::schedule_node &Band) {
    ASTBuildOptions.push_back(
        isl::manage(isl_schedule_node_band_get_ast_build_options(Band.get())));
    return getBase().visitBand(Band);
  }
};

/// Apply AST build options to the bands in a schedule tree.
///
/// This rewrites a schedule tree with the AST build options applied. We assume
/// that the band nodes are visited in the same order as they were when the
/// build options were collected, typically by CollectASTBuildOptions.
struct ApplyASTBuildOptions
    : public ScheduleNodeRewriter<ApplyASTBuildOptions> {
  using BaseTy = ScheduleNodeRewriter<ApplyASTBuildOptions>;
  BaseTy &getBase() { return *this; }
  const BaseTy &getBase() const { return *this; }

  size_t Pos;
  llvm::ArrayRef<isl::union_set> ASTBuildOptions;

  ApplyASTBuildOptions(llvm::ArrayRef<isl::union_set> ASTBuildOptions)
      : ASTBuildOptions(ASTBuildOptions) {}

  isl::schedule visitSchedule(const isl::schedule &Schedule) {
    Pos = 0;
    isl::schedule Result = visit(Schedule).get_schedule();
    assert(Pos == ASTBuildOptions.size() &&
           "AST build options must match to band nodes");
    return Result;
  }

  isl::schedule_node visitBand(const isl::schedule_node &Band) {
    isl::schedule_node Result =
        Band.band_set_ast_build_options(ASTBuildOptions[Pos]);
    Pos += 1;
    return getBase().visitBand(Result);
  }
};

} // namespace

/// Return whether the schedule contains an extension node.
static bool containsExtensionNode(isl::schedule Schedule) {
  assert(!Schedule.is_null());

  auto Callback = [](__isl_keep isl_schedule_node *Node,
                     void *User) -> isl_bool {
    if (isl_schedule_node_get_type(Node) == isl_schedule_node_extension) {
      // Stop walking the schedule tree.
      return isl_bool_error;
    }

    // Continue searching the subtree.
    return isl_bool_true;
  };
  isl_stat RetVal = isl_schedule_foreach_schedule_node_top_down(
      Schedule.get(), Callback, nullptr);

  // We assume that the traversal itself does not fail, i.e. the only reason to
  // return isl_stat_error is that an extension node was found.
  return RetVal == isl_stat_error;
}

isl::schedule polly::hoistExtensionNodes(isl::schedule Sched) {
  // If there is no extension node in the first place, return the original
  // schedule tree.
  if (!containsExtensionNode(Sched))
    return Sched;

  // Build options can anchor schedule nodes, such that the schedule tree cannot
  // be modified anymore. Therefore, apply build options after the tree has been
  // created.
  CollectASTBuildOptions Collector;
  Collector.visit(Sched);

  // Rewrite the schedule tree without extension nodes.
  ExtensionNodeRewriter Rewriter;
  isl::schedule NewSched = Rewriter.visitSchedule(Sched);

  // Reapply the AST build options. The rewriter must not change the iteration
  // order of bands. Any other node type is ignored.
  ApplyASTBuildOptions Applicator(Collector.ASTBuildOptions);
  NewSched = Applicator.visitSchedule(NewSched);

  return NewSched;
}
