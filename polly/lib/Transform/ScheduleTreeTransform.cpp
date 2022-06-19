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
#include "polly/Support/GICHelper.h"
#include "polly/Support/ISLTools.h"
#include "polly/Support/ScopHelper.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"

#define DEBUG_TYPE "polly-opt-isl"

using namespace polly;
using namespace llvm;

namespace {

/// Copy the band member attributes (coincidence, loop type, isolate ast loop
/// type) from one band to another.
static isl::schedule_node_band
applyBandMemberAttributes(isl::schedule_node_band Target, int TargetIdx,
                          const isl::schedule_node_band &Source,
                          int SourceIdx) {
  bool Coincident = Source.member_get_coincident(SourceIdx).release();
  Target = Target.member_set_coincident(TargetIdx, Coincident);

  isl_ast_loop_type LoopType =
      isl_schedule_node_band_member_get_ast_loop_type(Source.get(), SourceIdx);
  Target = isl::manage(isl_schedule_node_band_member_set_ast_loop_type(
                           Target.release(), TargetIdx, LoopType))
               .as<isl::schedule_node_band>();

  isl_ast_loop_type IsolateType =
      isl_schedule_node_band_member_get_isolate_ast_loop_type(Source.get(),
                                                              SourceIdx);
  Target = isl::manage(isl_schedule_node_band_member_set_isolate_ast_loop_type(
                           Target.release(), TargetIdx, IsolateType))
               .as<isl::schedule_node_band>();

  return Target;
}

/// Create a new band by copying members from another @p Band. @p IncludeCb
/// decides which band indices are copied to the result.
template <typename CbTy>
static isl::schedule rebuildBand(isl::schedule_node_band OldBand,
                                 isl::schedule Body, CbTy IncludeCb) {
  int NumBandDims = unsignedFromIslSize(OldBand.n_member());

  bool ExcludeAny = false;
  bool IncludeAny = false;
  for (auto OldIdx : seq<int>(0, NumBandDims)) {
    if (IncludeCb(OldIdx))
      IncludeAny = true;
    else
      ExcludeAny = true;
  }

  // Instead of creating a zero-member band, don't create a band at all.
  if (!IncludeAny)
    return Body;

  isl::multi_union_pw_aff PartialSched = OldBand.get_partial_schedule();
  isl::multi_union_pw_aff NewPartialSched;
  if (ExcludeAny) {
    // Select the included partial scatter functions.
    isl::union_pw_aff_list List = PartialSched.list();
    int NewIdx = 0;
    for (auto OldIdx : seq<int>(0, NumBandDims)) {
      if (IncludeCb(OldIdx))
        NewIdx += 1;
      else
        List = List.drop(NewIdx, 1);
    }
    isl::space ParamSpace = PartialSched.get_space().params();
    isl::space NewScatterSpace = ParamSpace.add_unnamed_tuple(NewIdx);
    NewPartialSched = isl::multi_union_pw_aff(NewScatterSpace, List);
  } else {
    // Just reuse original scatter function of copying all of them.
    NewPartialSched = PartialSched;
  }

  // Create the new band node.
  isl::schedule_node_band NewBand =
      Body.insert_partial_schedule(NewPartialSched)
          .get_root()
          .child(0)
          .as<isl::schedule_node_band>();

  // If OldBand was permutable, so is the new one, even if some dimensions are
  // missing.
  bool IsPermutable = OldBand.permutable().release();
  NewBand = NewBand.set_permutable(IsPermutable);

  // Reapply member attributes.
  int NewIdx = 0;
  for (auto OldIdx : seq<int>(0, NumBandDims)) {
    if (!IncludeCb(OldIdx))
      continue;
    NewBand =
        applyBandMemberAttributes(std::move(NewBand), NewIdx, OldBand, OldIdx);
    NewIdx += 1;
  }

  return NewBand.get_schedule();
}

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
    : RecursiveScheduleTreeVisitor<Derived, isl::schedule, Args...> {
  Derived &getDerived() { return *static_cast<Derived *>(this); }
  const Derived &getDerived() const {
    return *static_cast<const Derived *>(this);
  }

  isl::schedule visitDomain(isl::schedule_node_domain Node, Args... args) {
    // Every schedule_tree already has a domain node, no need to add one.
    return getDerived().visit(Node.first_child(), std::forward<Args>(args)...);
  }

  isl::schedule visitBand(isl::schedule_node_band Band, Args... args) {
    isl::schedule NewChild =
        getDerived().visit(Band.child(0), std::forward<Args>(args)...);
    return rebuildBand(Band, NewChild, [](int) { return true; });
  }

  isl::schedule visitSequence(isl::schedule_node_sequence Sequence,
                              Args... args) {
    int NumChildren = isl_schedule_node_n_children(Sequence.get());
    isl::schedule Result =
        getDerived().visit(Sequence.child(0), std::forward<Args>(args)...);
    for (int i = 1; i < NumChildren; i += 1)
      Result = Result.sequence(
          getDerived().visit(Sequence.child(i), std::forward<Args>(args)...));
    return Result;
  }

  isl::schedule visitSet(isl::schedule_node_set Set, Args... args) {
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

  isl::schedule visitLeaf(isl::schedule_node_leaf Leaf, Args... args) {
    return isl::schedule::from_domain(Leaf.get_domain());
  }

  isl::schedule visitMark(const isl::schedule_node &Mark, Args... args) {

    isl::id TheMark = Mark.as<isl::schedule_node_mark>().get_id();
    isl::schedule_node NewChild =
        getDerived()
            .visit(Mark.first_child(), std::forward<Args>(args)...)
            .get_root()
            .first_child();
    return NewChild.insert_mark(TheMark).get_schedule();
  }

  isl::schedule visitExtension(isl::schedule_node_extension Extension,
                               Args... args) {
    isl::union_map TheExtension =
        Extension.as<isl::schedule_node_extension>().get_extension();
    isl::schedule_node NewChild = getDerived()
                                      .visit(Extension.child(0), args...)
                                      .get_root()
                                      .first_child();
    isl::schedule_node NewExtension =
        isl::schedule_node::from_extension(TheExtension);
    return NewChild.graft_before(NewExtension).get_schedule();
  }

  isl::schedule visitFilter(isl::schedule_node_filter Filter, Args... args) {
    isl::union_set FilterDomain =
        Filter.as<isl::schedule_node_filter>().get_filter();
    isl::schedule NewSchedule =
        getDerived().visit(Filter.child(0), std::forward<Args>(args)...);
    return NewSchedule.intersect_domain(FilterDomain);
  }

  isl::schedule visitNode(isl::schedule_node Node, Args... args) {
    llvm_unreachable("Not implemented");
  }
};

/// Rewrite the schedule tree without any changes. Useful to copy a subtree into
/// a new schedule, discarding everything but.
struct IdentityRewriter : ScheduleTreeRewriter<IdentityRewriter> {};

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
struct ExtensionNodeRewriter final
    : ScheduleTreeRewriter<ExtensionNodeRewriter, const isl::union_set &,
                           isl::union_map &> {
  using BaseTy = ScheduleTreeRewriter<ExtensionNodeRewriter,
                                      const isl::union_set &, isl::union_map &>;
  BaseTy &getBase() { return *this; }
  const BaseTy &getBase() const { return *this; }

  isl::schedule visitSchedule(isl::schedule Schedule) {
    isl::union_map Extensions;
    isl::schedule Result =
        visit(Schedule.get_root(), Schedule.get_domain(), Extensions);
    assert(!Extensions.is_null() && Extensions.is_empty());
    return Result;
  }

  isl::schedule visitSequence(isl::schedule_node_sequence Sequence,
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

  isl::schedule visitSet(isl::schedule_node_set Set,
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

  isl::schedule visitLeaf(isl::schedule_node_leaf Leaf,
                          const isl::union_set &Domain,
                          isl::union_map &Extensions) {
    Extensions = isl::union_map::empty(Leaf.ctx());
    return isl::schedule::from_domain(Domain);
  }

  isl::schedule visitBand(isl::schedule_node_band OldNode,
                          const isl::union_set &Domain,
                          isl::union_map &OuterExtensions) {
    isl::schedule_node OldChild = OldNode.first_child();
    isl::multi_union_pw_aff PartialSched =
        isl::manage(isl_schedule_node_band_get_partial_schedule(OldNode.get()));

    isl::union_map NewChildExtensions;
    isl::schedule NewChild = visit(OldChild, Domain, NewChildExtensions);

    // Add the extensions to the partial schedule.
    OuterExtensions = isl::union_map::empty(NewChildExtensions.ctx());
    isl::union_map NewPartialSchedMap = isl::union_map::from(PartialSched);
    unsigned BandDims = isl_schedule_node_band_n_member(OldNode.get());
    for (isl::map Ext : NewChildExtensions.get_map_list()) {
      unsigned ExtDims = unsignedFromIslSize(Ext.domain_tuple_dim());
      assert(ExtDims >= BandDims);
      unsigned OuterDims = ExtDims - BandDims;

      isl::map BandSched =
          Ext.project_out(isl::dim::in, 0, OuterDims).reverse();
      NewPartialSchedMap = NewPartialSchedMap.unite(BandSched);

      // There might be more outer bands that have to schedule the extensions.
      if (OuterDims > 0) {
        isl::map OuterSched =
            Ext.project_out(isl::dim::in, OuterDims, BandDims);
        OuterExtensions = OuterExtensions.unite(OuterSched);
      }
    }
    isl::multi_union_pw_aff NewPartialSchedAsAsMultiUnionPwAff =
        isl::multi_union_pw_aff::from_union_map(NewPartialSchedMap);
    isl::schedule_node NewNode =
        NewChild.insert_partial_schedule(NewPartialSchedAsAsMultiUnionPwAff)
            .get_root()
            .child(0);

    // Reapply permutability and coincidence attributes.
    NewNode = isl::manage(isl_schedule_node_band_set_permutable(
        NewNode.release(),
        isl_schedule_node_band_get_permutable(OldNode.get())));
    for (unsigned i = 0; i < BandDims; i += 1)
      NewNode = applyBandMemberAttributes(NewNode.as<isl::schedule_node_band>(),
                                          i, OldNode, i);

    return NewNode.get_schedule();
  }

  isl::schedule visitFilter(isl::schedule_node_filter Filter,
                            const isl::union_set &Domain,
                            isl::union_map &Extensions) {
    isl::union_set FilterDomain =
        Filter.as<isl::schedule_node_filter>().get_filter();
    isl::union_set NewDomain = Domain.intersect(FilterDomain);

    // A filter is added implicitly if necessary when joining schedule trees.
    return visit(Filter.first_child(), NewDomain, Extensions);
  }

  isl::schedule visitExtension(isl::schedule_node_extension Extension,
                               const isl::union_set &Domain,
                               isl::union_map &Extensions) {
    isl::union_map ExtDomain =
        Extension.as<isl::schedule_node_extension>().get_extension();
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
struct CollectASTBuildOptions final
    : RecursiveScheduleTreeVisitor<CollectASTBuildOptions> {
  using BaseTy = RecursiveScheduleTreeVisitor<CollectASTBuildOptions>;
  BaseTy &getBase() { return *this; }
  const BaseTy &getBase() const { return *this; }

  llvm::SmallVector<isl::union_set, 8> ASTBuildOptions;

  void visitBand(isl::schedule_node_band Band) {
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
struct ApplyASTBuildOptions final : ScheduleNodeRewriter<ApplyASTBuildOptions> {
  using BaseTy = ScheduleNodeRewriter<ApplyASTBuildOptions>;
  BaseTy &getBase() { return *this; }
  const BaseTy &getBase() const { return *this; }

  size_t Pos;
  llvm::ArrayRef<isl::union_set> ASTBuildOptions;

  ApplyASTBuildOptions(llvm::ArrayRef<isl::union_set> ASTBuildOptions)
      : ASTBuildOptions(ASTBuildOptions) {}

  isl::schedule visitSchedule(isl::schedule Schedule) {
    Pos = 0;
    isl::schedule Result = visit(Schedule).get_schedule();
    assert(Pos == ASTBuildOptions.size() &&
           "AST build options must match to band nodes");
    return Result;
  }

  isl::schedule_node visitBand(isl::schedule_node_band Band) {
    isl::schedule_node_band Result =
        Band.set_ast_build_options(ASTBuildOptions[Pos]);
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

/// Is this node of type band?
static bool isBand(const isl::schedule_node &Node) {
  return isl_schedule_node_get_type(Node.get()) == isl_schedule_node_band;
}

#ifndef NDEBUG
/// Is this node a band of a single dimension (i.e. could represent a loop)?
static bool isBandWithSingleLoop(const isl::schedule_node &Node) {
  return isBand(Node) && isl_schedule_node_band_n_member(Node.get()) == 1;
}
#endif

static bool isLeaf(const isl::schedule_node &Node) {
  return isl_schedule_node_get_type(Node.get()) == isl_schedule_node_leaf;
}

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
    assert(isBandWithSingleLoop(BandOrMark.child(0)));
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
    Attr = getLoopAttr(MarkOrBand.as<isl::schedule_node_mark>().get_id());
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

  return Band.insert_mark(Mark).child(0);
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
  unsigned Dims = unsignedFromIslSize(Set.tuple_dim());
  assert(Dims >= 1);
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

/// Collapse perfectly nested bands into a single band.
class BandCollapseRewriter final
    : public ScheduleTreeRewriter<BandCollapseRewriter> {
private:
  using BaseTy = ScheduleTreeRewriter<BandCollapseRewriter>;
  BaseTy &getBase() { return *this; }
  const BaseTy &getBase() const { return *this; }

public:
  isl::schedule visitBand(isl::schedule_node_band RootBand) {
    isl::schedule_node_band Band = RootBand;
    isl::ctx Ctx = Band.ctx();

    // Do not merge permutable band to avoid loosing the permutability property.
    // Cannot collapse even two permutable loops, they might be permutable
    // individually, but not necassarily accross.
    if (unsignedFromIslSize(Band.n_member()) > 1u && Band.permutable())
      return getBase().visitBand(Band);

    // Find collapsable bands.
    SmallVector<isl::schedule_node_band> Nest;
    int NumTotalLoops = 0;
    isl::schedule_node Body;
    while (true) {
      Nest.push_back(Band);
      NumTotalLoops += unsignedFromIslSize(Band.n_member());
      Body = Band.first_child();
      if (!Body.isa<isl::schedule_node_band>())
        break;
      Band = Body.as<isl::schedule_node_band>();

      // Do not include next band if it is permutable to not lose its
      // permutability property.
      if (unsignedFromIslSize(Band.n_member()) > 1u && Band.permutable())
        break;
    }

    // Nothing to collapse, preserve permutability.
    if (Nest.size() <= 1)
      return getBase().visitBand(Band);

    LLVM_DEBUG({
      dbgs() << "Found loops to collapse between\n";
      dumpIslObj(RootBand, dbgs());
      dbgs() << "and\n";
      dumpIslObj(Body, dbgs());
      dbgs() << "\n";
    });

    isl::schedule NewBody = visit(Body);

    // Collect partial schedules from all members.
    isl::union_pw_aff_list PartScheds{Ctx, NumTotalLoops};
    for (isl::schedule_node_band Band : Nest) {
      int NumLoops = unsignedFromIslSize(Band.n_member());
      isl::multi_union_pw_aff BandScheds = Band.get_partial_schedule();
      for (auto j : seq<int>(0, NumLoops))
        PartScheds = PartScheds.add(BandScheds.at(j));
    }
    isl::space ScatterSpace = isl::space(Ctx, 0, NumTotalLoops);
    isl::multi_union_pw_aff PartSchedsMulti{ScatterSpace, PartScheds};

    isl::schedule_node_band CollapsedBand =
        NewBody.insert_partial_schedule(PartSchedsMulti)
            .get_root()
            .first_child()
            .as<isl::schedule_node_band>();

    // Copy over loop attributes form original bands.
    int LoopIdx = 0;
    for (isl::schedule_node_band Band : Nest) {
      int NumLoops = unsignedFromIslSize(Band.n_member());
      for (int i : seq<int>(0, NumLoops)) {
        CollapsedBand = applyBandMemberAttributes(std::move(CollapsedBand),
                                                  LoopIdx, Band, i);
        LoopIdx += 1;
      }
    }
    assert(LoopIdx == NumTotalLoops &&
           "Expect the same number of loops to add up again");

    return CollapsedBand.get_schedule();
  }
};

static isl::schedule collapseBands(isl::schedule Sched) {
  LLVM_DEBUG(dbgs() << "Collapse bands in schedule\n");
  BandCollapseRewriter Rewriter;
  return Rewriter.visit(Sched);
}

/// Collect sequentially executed bands (or anything else), even if nested in a
/// mark or other nodes whose child is executed just once. If we can
/// successfully fuse the bands, we allow them to be removed.
static void collectPotentiallyFusableBands(
    isl::schedule_node Node,
    SmallVectorImpl<std::pair<isl::schedule_node, isl::schedule_node>>
        &ScheduleBands,
    const isl::schedule_node &DirectChild) {
  switch (isl_schedule_node_get_type(Node.get())) {
  case isl_schedule_node_sequence:
  case isl_schedule_node_set:
  case isl_schedule_node_mark:
  case isl_schedule_node_domain:
  case isl_schedule_node_filter:
    if (Node.has_children()) {
      isl::schedule_node C = Node.first_child();
      while (true) {
        collectPotentiallyFusableBands(C, ScheduleBands, DirectChild);
        if (!C.has_next_sibling())
          break;
        C = C.next_sibling();
      }
    }
    break;

  default:
    // Something that does not execute suquentially (e.g. a band)
    ScheduleBands.push_back({Node, DirectChild});
    break;
  }
}

/// Remove dependencies that are resolved by @p PartSched. That is, remove
/// everything that we already know is executed in-order.
static isl::union_map remainingDepsFromPartialSchedule(isl::union_map PartSched,
                                                       isl::union_map Deps) {
  unsigned NumDims = getNumScatterDims(PartSched);
  auto ParamSpace = PartSched.get_space().params();

  // { Scatter[] }
  isl::space ScatterSpace =
      ParamSpace.set_from_params().add_dims(isl::dim::set, NumDims);

  // { Scatter[] -> Domain[] }
  isl::union_map PartSchedRev = PartSched.reverse();

  // { Scatter[] -> Scatter[] }
  isl::map MaybeBefore = isl::map::lex_le(ScatterSpace);

  // { Domain[] -> Domain[] }
  isl::union_map DomMaybeBefore =
      MaybeBefore.apply_domain(PartSchedRev).apply_range(PartSchedRev);

  // { Domain[] -> Domain[] }
  isl::union_map ChildRemainingDeps = Deps.intersect(DomMaybeBefore);

  return ChildRemainingDeps;
}

/// Remove dependencies that are resolved by executing them in the order
/// specified by @p Domains;
static isl::union_map remainigDepsFromSequence(ArrayRef<isl::union_set> Domains,
                                               isl::union_map Deps) {
  isl::ctx Ctx = Deps.ctx();
  isl::space ParamSpace = Deps.get_space().params();

  // Create a partial schedule mapping to constants that reflect the execution
  // order.
  isl::union_map PartialSchedules = isl::union_map::empty(Ctx);
  for (auto P : enumerate(Domains)) {
    isl::val ExecTime = isl::val(Ctx, P.index());
    isl::union_pw_aff DomSched{P.value(), ExecTime};
    PartialSchedules = PartialSchedules.unite(DomSched.as_union_map());
  }

  return remainingDepsFromPartialSchedule(PartialSchedules, Deps);
}

/// Determine whether the outermost loop of to bands can be fused while
/// respecting validity dependencies.
static bool canFuseOutermost(const isl::schedule_node_band &LHS,
                             const isl::schedule_node_band &RHS,
                             const isl::union_map &Deps) {
  // { LHSDomain[] -> Scatter[] }
  isl::union_map LHSPartSched =
      LHS.get_partial_schedule().get_at(0).as_union_map();

  // { Domain[] -> Scatter[] }
  isl::union_map RHSPartSched =
      RHS.get_partial_schedule().get_at(0).as_union_map();

  // Dependencies that are already resolved because LHS executes before RHS, but
  // will not be anymore after fusion. { DefDomain[] -> UseDomain[] }
  isl::union_map OrderedBySequence =
      Deps.intersect_domain(LHSPartSched.domain())
          .intersect_range(RHSPartSched.domain());

  isl::space ParamSpace = OrderedBySequence.get_space().params();
  isl::space NewScatterSpace = ParamSpace.add_unnamed_tuple(1);

  // { Scatter[] -> Scatter[] }
  isl::map After = isl::map::lex_gt(NewScatterSpace);

  // After fusion, instances with smaller (or equal, which means they will be
  // executed in the same iteration, but the LHS instance is still sequenced
  // before RHS) scatter value will still be executed before. This are the
  // orderings where this is not necessarily the case.
  // { LHSDomain[] -> RHSDomain[] }
  isl::union_map MightBeAfterDoms = After.apply_domain(LHSPartSched.reverse())
                                        .apply_range(RHSPartSched.reverse());

  // Dependencies that are not resolved by the new execution order.
  isl::union_map WithBefore = OrderedBySequence.intersect(MightBeAfterDoms);

  return WithBefore.is_empty();
}

/// Fuse @p LHS and @p RHS if possible while preserving validity dependenvies.
static isl::schedule tryGreedyFuse(isl::schedule_node_band LHS,
                                   isl::schedule_node_band RHS,
                                   const isl::union_map &Deps) {
  if (!canFuseOutermost(LHS, RHS, Deps))
    return {};

  LLVM_DEBUG({
    dbgs() << "Found loops for greedy fusion:\n";
    dumpIslObj(LHS, dbgs());
    dbgs() << "and\n";
    dumpIslObj(RHS, dbgs());
    dbgs() << "\n";
  });

  // The partial schedule of the bands outermost loop that we need to combine
  // for the fusion.
  isl::union_pw_aff LHSPartOuterSched = LHS.get_partial_schedule().get_at(0);
  isl::union_pw_aff RHSPartOuterSched = RHS.get_partial_schedule().get_at(0);

  // Isolate band bodies as roots of their own schedule trees.
  IdentityRewriter Rewriter;
  isl::schedule LHSBody = Rewriter.visit(LHS.first_child());
  isl::schedule RHSBody = Rewriter.visit(RHS.first_child());

  // Reconstruct the non-outermost (not going to be fused) loops from both
  // bands.
  // TODO: Maybe it is possibly to transfer the 'permutability' property from
  // LHS+RHS. At minimum we need merge multiple band members at once, otherwise
  // permutability has no meaning.
  isl::schedule LHSNewBody =
      rebuildBand(LHS, LHSBody, [](int i) { return i > 0; });
  isl::schedule RHSNewBody =
      rebuildBand(RHS, RHSBody, [](int i) { return i > 0; });

  // The loop body of the fused loop.
  isl::schedule NewCommonBody = LHSNewBody.sequence(RHSNewBody);

  // Combine the partial schedules of both loops to a new one. Instances with
  // the same scatter value are put together.
  isl::union_map NewCommonPartialSched =
      LHSPartOuterSched.as_union_map().unite(RHSPartOuterSched.as_union_map());
  isl::schedule NewCommonSchedule = NewCommonBody.insert_partial_schedule(
      NewCommonPartialSched.as_multi_union_pw_aff());

  return NewCommonSchedule;
}

static isl::schedule tryGreedyFuse(isl::schedule_node LHS,
                                   isl::schedule_node RHS,
                                   const isl::union_map &Deps) {
  // TODO: Non-bands could be interpreted as a band with just as single
  // iteration. However, this is only useful if both ends of a fused loop were
  // originally loops themselves.
  if (!LHS.isa<isl::schedule_node_band>())
    return {};
  if (!RHS.isa<isl::schedule_node_band>())
    return {};

  return tryGreedyFuse(LHS.as<isl::schedule_node_band>(),
                       RHS.as<isl::schedule_node_band>(), Deps);
}

/// Fuse all fusable loop top-down in a schedule tree.
///
/// The isl::union_map parameters is the set of validity dependencies that have
/// not been resolved/carried by a parent schedule node.
class GreedyFusionRewriter final
    : public ScheduleTreeRewriter<GreedyFusionRewriter, isl::union_map> {
private:
  using BaseTy = ScheduleTreeRewriter<GreedyFusionRewriter, isl::union_map>;
  BaseTy &getBase() { return *this; }
  const BaseTy &getBase() const { return *this; }

public:
  /// Is set to true if anything has been fused.
  bool AnyChange = false;

  isl::schedule visitBand(isl::schedule_node_band Band, isl::union_map Deps) {
    // { Domain[] -> Scatter[] }
    isl::union_map PartSched =
        isl::union_map::from(Band.get_partial_schedule());
    assert(getNumScatterDims(PartSched) ==
           unsignedFromIslSize(Band.n_member()));
    isl::space ParamSpace = PartSched.get_space().params();

    // { Scatter[] -> Domain[] }
    isl::union_map PartSchedRev = PartSched.reverse();

    // Possible within the same iteration. Dependencies with smaller scatter
    // value are carried by this loop and therefore have been resolved by the
    // in-order execution if the loop iteration. A dependency with small scatter
    // value would be a dependency violation that we assume did not happen. {
    // Domain[] -> Domain[] }
    isl::union_map Unsequenced = PartSchedRev.apply_domain(PartSchedRev);

    // Actual dependencies within the same iteration.
    // { DefDomain[] -> UseDomain[] }
    isl::union_map RemDeps = Deps.intersect(Unsequenced);

    return getBase().visitBand(Band, RemDeps);
  }

  isl::schedule visitSequence(isl::schedule_node_sequence Sequence,
                              isl::union_map Deps) {
    int NumChildren = isl_schedule_node_n_children(Sequence.get());

    // List of fusion candidates. The first element is the fusion candidate, the
    // second is candidate's ancestor that is the sequence's direct child. It is
    // preferable to use the direct child if not if its non-direct children is
    // fused to preserve its structure such as mark nodes.
    SmallVector<std::pair<isl::schedule_node, isl::schedule_node>> Bands;
    for (auto i : seq<int>(0, NumChildren)) {
      isl::schedule_node Child = Sequence.child(i);
      collectPotentiallyFusableBands(Child, Bands, Child);
    }

    // Direct children that had at least one of its decendants fused.
    SmallDenseSet<isl_schedule_node *, 4> ChangedDirectChildren;

    // Fuse neigboring bands until reaching the end of candidates.
    int i = 0;
    while (i + 1 < (int)Bands.size()) {
      isl::schedule Fused =
          tryGreedyFuse(Bands[i].first, Bands[i + 1].first, Deps);
      if (Fused.is_null()) {
        // Cannot merge this node with the next; look at next pair.
        i += 1;
        continue;
      }

      // Mark the direct children as (partially) fused.
      if (!Bands[i].second.is_null())
        ChangedDirectChildren.insert(Bands[i].second.get());
      if (!Bands[i + 1].second.is_null())
        ChangedDirectChildren.insert(Bands[i + 1].second.get());

      // Collapse the neigbros to a single new candidate that could be fused
      // with the next candidate.
      Bands[i] = {Fused.get_root(), {}};
      Bands.erase(Bands.begin() + i + 1);

      AnyChange = true;
    }

    // By construction equal if done with collectPotentiallyFusableBands's
    // output.
    SmallVector<isl::union_set> SubDomains;
    SubDomains.reserve(NumChildren);
    for (int i = 0; i < NumChildren; i += 1)
      SubDomains.push_back(Sequence.child(i).domain());
    auto SubRemainingDeps = remainigDepsFromSequence(SubDomains, Deps);

    // We may iterate over direct children multiple times, be sure to add each
    // at most once.
    SmallDenseSet<isl_schedule_node *, 4> AlreadyAdded;

    isl::schedule Result;
    for (auto &P : Bands) {
      isl::schedule_node MaybeFused = P.first;
      isl::schedule_node DirectChild = P.second;

      // If not modified, use the direct child.
      if (!DirectChild.is_null() &&
          !ChangedDirectChildren.count(DirectChild.get())) {
        if (AlreadyAdded.count(DirectChild.get()))
          continue;
        AlreadyAdded.insert(DirectChild.get());
        MaybeFused = DirectChild;
      } else {
        assert(AnyChange &&
               "Need changed flag for be consistent with actual change");
      }

      // Top-down recursion: If the outermost loop has been fused, their nested
      // bands might be fusable now as well.
      isl::schedule InnerFused = visit(MaybeFused, SubRemainingDeps);

      // Reconstruct the sequence, with some of the children fused.
      if (Result.is_null())
        Result = InnerFused;
      else
        Result = Result.sequence(InnerFused);
    }

    return Result;
  }
};

} // namespace

bool polly::isBandMark(const isl::schedule_node &Node) {
  return isMark(Node) &&
         isLoopAttr(Node.as<isl::schedule_node_mark>().get_id());
}

BandAttr *polly::getBandAttr(isl::schedule_node MarkOrBand) {
  MarkOrBand = moveToBandMark(MarkOrBand);
  if (!isMark(MarkOrBand))
    return nullptr;

  return getLoopAttr(MarkOrBand.as<isl::schedule_node_mark>().get_id());
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
  isl::ctx Ctx = BandToUnroll.ctx();

  // Remove the loop's mark, the loop will disappear anyway.
  BandToUnroll = removeMark(BandToUnroll);
  assert(isBandWithSingleLoop(BandToUnroll));

  isl::multi_union_pw_aff PartialSched = isl::manage(
      isl_schedule_node_band_get_partial_schedule(BandToUnroll.get()));
  assert(unsignedFromIslSize(PartialSched.dim(isl::dim::out)) == 1u &&
         "Can only unroll a single dimension");
  isl::union_pw_aff PartialSchedUAff = PartialSched.at(0);

  isl::union_set Domain = BandToUnroll.get_domain();
  PartialSchedUAff = PartialSchedUAff.intersect_domain(Domain);
  isl::union_map PartialSchedUMap =
      isl::union_map::from(isl::union_pw_multi_aff(PartialSchedUAff));

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
  isl::union_set_list List = isl::union_set_list(Ctx, Elts.size());
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
  isl::ctx Ctx = BandToUnroll.ctx();

  // Remove the mark, save the attribute for later use.
  BandAttr *Attr;
  BandToUnroll = removeMark(BandToUnroll, Attr);
  assert(isBandWithSingleLoop(BandToUnroll));

  isl::multi_union_pw_aff PartialSched = isl::manage(
      isl_schedule_node_band_get_partial_schedule(BandToUnroll.get()));

  // { Stmt[] -> [x] }
  isl::union_pw_aff PartialSchedUAff = PartialSched.at(0);

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

  isl::union_set_list List = isl::union_set_list(Ctx, Factor);
  for (auto i : seq<int>(0, Factor)) {
    // { Stmt[] -> [x] }
    isl::union_map UMap =
        isl::union_map::from(isl::union_pw_multi_aff(PartialSchedUAff));

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
  if (!NewBandId.is_null())
    NewLoop = insertMark(NewLoop, NewBandId);

  return NewLoop.get_schedule();
}

isl::set polly::getPartialTilePrefixes(isl::set ScheduleRange,
                                       int VectorWidth) {
  unsigned Dims = unsignedFromIslSize(ScheduleRange.tuple_dim());
  assert(Dims >= 1);
  isl::set LoopPrefixes =
      ScheduleRange.drop_constraints_involving_dims(isl::dim::set, Dims - 1, 1);
  auto ExtentPrefixes = addExtentConstraints(LoopPrefixes, VectorWidth);
  isl::set BadPrefixes = ExtentPrefixes.subtract(ScheduleRange);
  BadPrefixes = BadPrefixes.project_out(isl::dim::set, Dims - 1, 1);
  LoopPrefixes = LoopPrefixes.project_out(isl::dim::set, Dims - 1, 1);
  return LoopPrefixes.subtract(BadPrefixes);
}

isl::union_set polly::getIsolateOptions(isl::set IsolateDomain,
                                        unsigned OutDimsNum) {
  unsigned Dims = unsignedFromIslSize(IsolateDomain.tuple_dim());
  assert(OutDimsNum <= Dims &&
         "The isl::set IsolateDomain is used to describe the range of schedule "
         "dimensions values, which should be isolated. Consequently, the "
         "number of its dimensions should be greater than or equal to the "
         "number of the schedule dimensions.");
  isl::map IsolateRelation = isl::map::from_domain(IsolateDomain);
  IsolateRelation = IsolateRelation.move_dims(isl::dim::out, 0, isl::dim::in,
                                              Dims - OutDimsNum, OutDimsNum);
  isl::set IsolateOption = IsolateRelation.wrap();
  isl::id Id = isl::id::alloc(IsolateOption.ctx(), "isolate", nullptr);
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
  for (unsigned i : rangeIslSize(0, Dims)) {
    unsigned tileSize = i < TileSizes.size() ? TileSizes[i] : DefaultTileSize;
    Sizes = Sizes.set_val(i, isl::val(Node.ctx(), tileSize));
  }
  auto TileLoopMarkerStr = IdentifierString + " - Tiles";
  auto TileLoopMarker = isl::id::alloc(Node.ctx(), TileLoopMarkerStr, nullptr);
  Node = Node.insert_mark(TileLoopMarker);
  Node = Node.child(0);
  Node =
      isl::manage(isl_schedule_node_band_tile(Node.release(), Sizes.release()));
  Node = Node.child(0);
  auto PointLoopMarkerStr = IdentifierString + " - Points";
  auto PointLoopMarker =
      isl::id::alloc(Node.ctx(), PointLoopMarkerStr, nullptr);
  Node = Node.insert_mark(PointLoopMarker);
  return Node.child(0);
}

isl::schedule_node polly::applyRegisterTiling(isl::schedule_node Node,
                                              ArrayRef<int> TileSizes,
                                              int DefaultTileSize) {
  Node = tileNode(Node, "Register tiling", TileSizes, DefaultTileSize);
  auto Ctx = Node.ctx();
  return Node.as<isl::schedule_node_band>().set_ast_build_options(
      isl::union_set(Ctx, "{unroll[x]}"));
}

/// Find statements and sub-loops in (possibly nested) sequences.
static void
collectFissionableStmts(isl::schedule_node Node,
                        SmallVectorImpl<isl::schedule_node> &ScheduleStmts) {
  if (isBand(Node) || isLeaf(Node)) {
    ScheduleStmts.push_back(Node);
    return;
  }

  if (Node.has_children()) {
    isl::schedule_node C = Node.first_child();
    while (true) {
      collectFissionableStmts(C, ScheduleStmts);
      if (!C.has_next_sibling())
        break;
      C = C.next_sibling();
    }
  }
}

isl::schedule polly::applyMaxFission(isl::schedule_node BandToFission) {
  isl::ctx Ctx = BandToFission.ctx();
  BandToFission = removeMark(BandToFission);
  isl::schedule_node BandBody = BandToFission.child(0);

  SmallVector<isl::schedule_node> FissionableStmts;
  collectFissionableStmts(BandBody, FissionableStmts);
  size_t N = FissionableStmts.size();

  // Collect the domain for each of the statements that will get their own loop.
  isl::union_set_list DomList = isl::union_set_list(Ctx, N);
  for (size_t i = 0; i < N; ++i) {
    isl::schedule_node BodyPart = FissionableStmts[i];
    DomList = DomList.add(BodyPart.get_domain());
  }

  // Apply the fission by copying the entire loop, but inserting a filter for
  // the statement domains for each fissioned loop.
  isl::schedule_node Fissioned = BandToFission.insert_sequence(DomList);

  return Fissioned.get_schedule();
}

isl::schedule polly::applyGreedyFusion(isl::schedule Sched,
                                       const isl::union_map &Deps) {
  LLVM_DEBUG(dbgs() << "Greedy loop fusion\n");

  GreedyFusionRewriter Rewriter;
  isl::schedule Result = Rewriter.visit(Sched, Deps);
  if (!Rewriter.AnyChange) {
    LLVM_DEBUG(dbgs() << "Found nothing to fuse\n");
    return Sched;
  }

  // GreedyFusionRewriter due to working loop-by-loop, bands with multiple loops
  // may have been split into multiple bands.
  return collapseBands(Result);
}
