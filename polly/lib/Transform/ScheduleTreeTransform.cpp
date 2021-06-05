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
#include "polly/Support/ScopHelper.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"

using namespace polly;
using namespace llvm;

namespace {
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

/// Find a named MDNode property in a LoopID.
static MDNode *findOptionalNodeOperand(MDNode *LoopMD, StringRef Name) {
  return dyn_cast_or_null<MDNode>(
      findMetadataOperand(LoopMD, Name).getValueOr(nullptr));
}

/// Is this node of type mark?
static bool isMark(const isl::schedule_node &Node) {
  return isl_schedule_node_get_type(Node.get()) == isl_schedule_node_mark;
}

#ifndef NDEBUG
/// Is this node of type band?
static bool isBand(const isl::schedule_node &Node) {
  return isl_schedule_node_get_type(Node.get()) == isl_schedule_node_band;
}

/// Is this node a band of a single dimension (i.e. could represent a loop)?
static bool isBandWithSingleLoop(const isl::schedule_node &Node) {

  return isBand(Node) && isl_schedule_node_band_n_member(Node.get()) == 1;
}
#endif

/// Create an isl::id representing the output loop after a transformation.
static isl::id createGeneratedLoopAttr(isl::ctx Ctx, MDNode *FollowupLoopMD) {
  // Don't need to id the followup.
  // TODO: Append llvm.loop.disable_heustistics metadata unless overridden by
  //       user followup-MD
  if (!FollowupLoopMD)
    return {};

  BandAttr *Attr = new BandAttr();
  Attr->Metadata = FollowupLoopMD;
  return getIslLoopAttr(Ctx, Attr);
}

/// A loop consists of a band and an optional marker that wraps it. Return the
/// outermost of the two.

/// That is, either the mark or, if there is not mark, the loop itself. Can
/// start with either the mark or the band.
static isl::schedule_node moveToBandMark(isl::schedule_node BandOrMark) {
  if (isBandMark(BandOrMark)) {
    assert(isBandWithSingleLoop(BandOrMark.get_child(0)));
    return BandOrMark;
  }
  assert(isBandWithSingleLoop(BandOrMark));

  isl::schedule_node Mark = BandOrMark.parent();
  if (isBandMark(Mark))
    return Mark;

  // Band has no loop marker.
  return BandOrMark;
}

static isl::schedule_node removeMark(isl::schedule_node MarkOrBand,
                                     BandAttr *&Attr) {
  MarkOrBand = moveToBandMark(MarkOrBand);

  isl::schedule_node Band;
  if (isMark(MarkOrBand)) {
    Attr = getLoopAttr(MarkOrBand.mark_get_id());
    Band = isl::manage(isl_schedule_node_delete(MarkOrBand.release()));
  } else {
    Attr = nullptr;
    Band = MarkOrBand;
  }

  assert(isBandWithSingleLoop(Band));
  return Band;
}

/// Remove the mark that wraps a loop. Return the band representing the loop.
static isl::schedule_node removeMark(isl::schedule_node MarkOrBand) {
  BandAttr *Attr;
  return removeMark(MarkOrBand, Attr);
}

static isl::schedule_node insertMark(isl::schedule_node Band, isl::id Mark) {
  assert(isBand(Band));
  assert(moveToBandMark(Band).is_equal(Band) &&
         "Don't add a two marks for a band");

  return Band.insert_mark(Mark).get_child(0);
}

/// Return the (one-dimensional) set of numbers that are divisible by @p Factor
/// with remainder @p Offset.
///
///  isDivisibleBySet(Ctx, 4, 0) = { [i] : floord(i,4) = 0 }
///  isDivisibleBySet(Ctx, 4, 1) = { [i] : floord(i,4) = 1 }
///
static isl::basic_set isDivisibleBySet(isl::ctx &Ctx, long Factor,
                                       long Offset) {
  isl::val ValFactor{Ctx, Factor};
  isl::val ValOffset{Ctx, Offset};

  isl::space Unispace{Ctx, 0, 1};
  isl::local_space LUnispace{Unispace};
  isl::aff AffFactor{LUnispace, ValFactor};
  isl::aff AffOffset{LUnispace, ValOffset};

  isl::aff Id = isl::aff::var_on_domain(LUnispace, isl::dim::out, 0);
  isl::aff DivMul = Id.mod(ValFactor);
  isl::basic_map Divisible = isl::basic_map::from_aff(DivMul);
  isl::basic_map Modulo = Divisible.fix_val(isl::dim::out, 0, ValOffset);
  return Modulo.domain();
}

/// Make the last dimension of Set to take values from 0 to VectorWidth - 1.
///
/// @param Set         A set, which should be modified.
/// @param VectorWidth A parameter, which determines the constraint.
static isl::set addExtentConstraints(isl::set Set, int VectorWidth) {
  unsigned Dims = Set.dim(isl::dim::set);
  isl::space Space = Set.get_space();
  isl::local_space LocalSpace = isl::local_space(Space);
  isl::constraint ExtConstr = isl::constraint::alloc_inequality(LocalSpace);
  ExtConstr = ExtConstr.set_constant_si(0);
  ExtConstr = ExtConstr.set_coefficient_si(isl::dim::set, Dims - 1, 1);
  Set = Set.add_constraint(ExtConstr);
  ExtConstr = isl::constraint::alloc_inequality(LocalSpace);
  ExtConstr = ExtConstr.set_constant_si(VectorWidth - 1);
  ExtConstr = ExtConstr.set_coefficient_si(isl::dim::set, Dims - 1, -1);
  return Set.add_constraint(ExtConstr);
}
} // namespace

bool polly::isBandMark(const isl::schedule_node &Node) {
  return isMark(Node) && isLoopAttr(Node.mark_get_id());
}

BandAttr *polly::getBandAttr(isl::schedule_node MarkOrBand) {
  MarkOrBand = moveToBandMark(MarkOrBand);
  if (!isMark(MarkOrBand))
    return nullptr;

  return getLoopAttr(MarkOrBand.mark_get_id());
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

isl::schedule polly::applyFullUnroll(isl::schedule_node BandToUnroll) {
  isl::ctx Ctx = BandToUnroll.get_ctx();

  // Remove the loop's mark, the loop will disappear anyway.
  BandToUnroll = removeMark(BandToUnroll);
  assert(isBandWithSingleLoop(BandToUnroll));

  isl::multi_union_pw_aff PartialSched = isl::manage(
      isl_schedule_node_band_get_partial_schedule(BandToUnroll.get()));
  assert(PartialSched.dim(isl::dim::out) == 1 &&
         "Can only unroll a single dimension");
  isl::union_pw_aff PartialSchedUAff = PartialSched.get_union_pw_aff(0);

  isl::union_set Domain = BandToUnroll.get_domain();
  PartialSchedUAff = PartialSchedUAff.intersect_domain(Domain);
  isl::union_map PartialSchedUMap = isl::union_map(PartialSchedUAff);

  // Enumerator only the scatter elements.
  isl::union_set ScatterList = PartialSchedUMap.range();

  // Enumerate all loop iterations.
  // TODO: Diagnose if not enumerable or depends on a parameter.
  SmallVector<isl::point, 16> Elts;
  ScatterList.foreach_point([&Elts](isl::point P) -> isl::stat {
    Elts.push_back(P);
    return isl::stat::ok();
  });

  // Don't assume that foreach_point returns in execution order.
  llvm::sort(Elts, [](isl::point P1, isl::point P2) -> bool {
    isl::val C1 = P1.get_coordinate_val(isl::dim::set, 0);
    isl::val C2 = P2.get_coordinate_val(isl::dim::set, 0);
    return C1.lt(C2);
  });

  // Convert the points to a sequence of filters.
  isl::union_set_list List = isl::union_set_list::alloc(Ctx, Elts.size());
  for (isl::point P : Elts) {
    // Determine the domains that map this scatter element.
    isl::union_set DomainFilter = PartialSchedUMap.intersect_range(P).domain();

    List = List.add(DomainFilter);
  }

  // Replace original band with unrolled sequence.
  isl::schedule_node Body =
      isl::manage(isl_schedule_node_delete(BandToUnroll.release()));
  Body = Body.insert_sequence(List);
  return Body.get_schedule();
}

isl::schedule polly::applyPartialUnroll(isl::schedule_node BandToUnroll,
                                        int Factor) {
  assert(Factor > 0 && "Positive unroll factor required");
  isl::ctx Ctx = BandToUnroll.get_ctx();

  // Remove the mark, save the attribute for later use.
  BandAttr *Attr;
  BandToUnroll = removeMark(BandToUnroll, Attr);
  assert(isBandWithSingleLoop(BandToUnroll));

  isl::multi_union_pw_aff PartialSched = isl::manage(
      isl_schedule_node_band_get_partial_schedule(BandToUnroll.get()));

  // { Stmt[] -> [x] }
  isl::union_pw_aff PartialSchedUAff = PartialSched.get_union_pw_aff(0);

  // Here we assume the schedule stride is one and starts with 0, which is not
  // necessarily the case.
  isl::union_pw_aff StridedPartialSchedUAff =
      isl::union_pw_aff::empty(PartialSchedUAff.get_space());
  isl::val ValFactor{Ctx, Factor};
  PartialSchedUAff.foreach_pw_aff([&StridedPartialSchedUAff,
                                   &ValFactor](isl::pw_aff PwAff) -> isl::stat {
    isl::space Space = PwAff.get_space();
    isl::set Universe = isl::set::universe(Space.domain());
    isl::pw_aff AffFactor{Universe, ValFactor};
    isl::pw_aff DivSchedAff = PwAff.div(AffFactor).floor().mul(AffFactor);
    StridedPartialSchedUAff = StridedPartialSchedUAff.union_add(DivSchedAff);
    return isl::stat::ok();
  });

  isl::union_set_list List = isl::union_set_list::alloc(Ctx, Factor);
  for (auto i : seq<int>(0, Factor)) {
    // { Stmt[] -> [x] }
    isl::union_map UMap{PartialSchedUAff};

    // { [x] }
    isl::basic_set Divisible = isDivisibleBySet(Ctx, Factor, i);

    // { Stmt[] }
    isl::union_set UnrolledDomain = UMap.intersect_range(Divisible).domain();

    List = List.add(UnrolledDomain);
  }

  isl::schedule_node Body =
      isl::manage(isl_schedule_node_delete(BandToUnroll.copy()));
  Body = Body.insert_sequence(List);
  isl::schedule_node NewLoop =
      Body.insert_partial_schedule(StridedPartialSchedUAff);

  MDNode *FollowupMD = nullptr;
  if (Attr && Attr->Metadata)
    FollowupMD =
        findOptionalNodeOperand(Attr->Metadata, LLVMLoopUnrollFollowupUnrolled);

  isl::id NewBandId = createGeneratedLoopAttr(Ctx, FollowupMD);
  if (NewBandId)
    NewLoop = insertMark(NewLoop, NewBandId);

  return NewLoop.get_schedule();
}

isl::set polly::getPartialTilePrefixes(isl::set ScheduleRange,
                                       int VectorWidth) {
  isl_size Dims = ScheduleRange.dim(isl::dim::set);
  isl::set LoopPrefixes =
      ScheduleRange.drop_constraints_involving_dims(isl::dim::set, Dims - 1, 1);
  auto ExtentPrefixes = addExtentConstraints(LoopPrefixes, VectorWidth);
  isl::set BadPrefixes = ExtentPrefixes.subtract(ScheduleRange);
  BadPrefixes = BadPrefixes.project_out(isl::dim::set, Dims - 1, 1);
  LoopPrefixes = LoopPrefixes.project_out(isl::dim::set, Dims - 1, 1);
  return LoopPrefixes.subtract(BadPrefixes);
}

isl::union_set polly::getIsolateOptions(isl::set IsolateDomain,
                                        isl_size OutDimsNum) {
  isl_size Dims = IsolateDomain.dim(isl::dim::set);
  assert(OutDimsNum <= Dims &&
         "The isl::set IsolateDomain is used to describe the range of schedule "
         "dimensions values, which should be isolated. Consequently, the "
         "number of its dimensions should be greater than or equal to the "
         "number of the schedule dimensions.");
  isl::map IsolateRelation = isl::map::from_domain(IsolateDomain);
  IsolateRelation = IsolateRelation.move_dims(isl::dim::out, 0, isl::dim::in,
                                              Dims - OutDimsNum, OutDimsNum);
  isl::set IsolateOption = IsolateRelation.wrap();
  isl::id Id = isl::id::alloc(IsolateOption.get_ctx(), "isolate", nullptr);
  IsolateOption = IsolateOption.set_tuple_id(Id);
  return isl::union_set(IsolateOption);
}

isl::union_set polly::getDimOptions(isl::ctx Ctx, const char *Option) {
  isl::space Space(Ctx, 0, 1);
  auto DimOption = isl::set::universe(Space);
  auto Id = isl::id::alloc(Ctx, Option, nullptr);
  DimOption = DimOption.set_tuple_id(Id);
  return isl::union_set(DimOption);
}

isl::schedule_node polly::tileNode(isl::schedule_node Node,
                                   const char *Identifier,
                                   ArrayRef<int> TileSizes,
                                   int DefaultTileSize) {
  auto Space = isl::manage(isl_schedule_node_band_get_space(Node.get()));
  auto Dims = Space.dim(isl::dim::set);
  auto Sizes = isl::multi_val::zero(Space);
  std::string IdentifierString(Identifier);
  for (auto i : seq<isl_size>(0, Dims)) {
    auto tileSize =
        i < (isl_size)TileSizes.size() ? TileSizes[i] : DefaultTileSize;
    Sizes = Sizes.set_val(i, isl::val(Node.get_ctx(), tileSize));
  }
  auto TileLoopMarkerStr = IdentifierString + " - Tiles";
  auto TileLoopMarker =
      isl::id::alloc(Node.get_ctx(), TileLoopMarkerStr, nullptr);
  Node = Node.insert_mark(TileLoopMarker);
  Node = Node.child(0);
  Node =
      isl::manage(isl_schedule_node_band_tile(Node.release(), Sizes.release()));
  Node = Node.child(0);
  auto PointLoopMarkerStr = IdentifierString + " - Points";
  auto PointLoopMarker =
      isl::id::alloc(Node.get_ctx(), PointLoopMarkerStr, nullptr);
  Node = Node.insert_mark(PointLoopMarker);
  return Node.child(0);
}

isl::schedule_node polly::applyRegisterTiling(isl::schedule_node Node,
                                              ArrayRef<int> TileSizes,
                                              int DefaultTileSize) {
  Node = tileNode(Node, "Register tiling", TileSizes, DefaultTileSize);
  auto Ctx = Node.get_ctx();
  return Node.band_set_ast_build_options(isl::union_set(Ctx, "{unroll[x]}"));
}
