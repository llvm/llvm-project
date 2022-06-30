//===- ScheduleOptimizer.cpp - Calculate an optimized schedule ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass generates an entirely new schedule tree from the data dependences
// and iteration domains. The new schedule tree is computed in two steps:
//
// 1) The isl scheduling optimizer is run
//
// The isl scheduling optimizer creates a new schedule tree that maximizes
// parallelism and tileability and minimizes data-dependence distances. The
// algorithm used is a modified version of the ``Pluto'' algorithm:
//
//   U. Bondhugula, A. Hartono, J. Ramanujam, and P. Sadayappan.
//   A Practical Automatic Polyhedral Parallelizer and Locality Optimizer.
//   In Proceedings of the 2008 ACM SIGPLAN Conference On Programming Language
//   Design and Implementation, PLDI ’08, pages 101–113. ACM, 2008.
//
// 2) A set of post-scheduling transformations is applied on the schedule tree.
//
// These optimizations include:
//
//  - Tiling of the innermost tilable bands
//  - Prevectorization - The choice of a possible outer loop that is strip-mined
//                       to the innermost level to enable inner-loop
//                       vectorization.
//  - Some optimizations for spatial locality are also planned.
//
// For a detailed description of the schedule tree itself please see section 6
// of:
//
// Polyhedral AST generation is more than scanning polyhedra
// Tobias Grosser, Sven Verdoolaege, Albert Cohen
// ACM Transactions on Programming Languages and Systems (TOPLAS),
// 37(4), July 2015
// http://www.grosser.es/#pub-polyhedral-AST-generation
//
// This publication also contains a detailed discussion of the different options
// for polyhedral loop unrolling, full/partial tile separation and other uses
// of the schedule tree.
//
//===----------------------------------------------------------------------===//

#include "polly/ScheduleOptimizer.h"
#include "polly/CodeGen/CodeGeneration.h"
#include "polly/DependenceInfo.h"
#include "polly/ManualOptimizer.h"
#include "polly/MatmulOptimizer.h"
#include "polly/Options.h"
#include "polly/ScheduleTreeTransform.h"
#include "polly/Support/ISLOStream.h"
#include "polly/Support/ISLTools.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "isl/options.h"

using namespace llvm;
using namespace polly;

namespace llvm {
class Loop;
class Module;
} // namespace llvm

#define DEBUG_TYPE "polly-opt-isl"

static cl::opt<std::string>
    OptimizeDeps("polly-opt-optimize-only",
                 cl::desc("Only a certain kind of dependences (all/raw)"),
                 cl::Hidden, cl::init("all"), cl::cat(PollyCategory));

static cl::opt<std::string>
    SimplifyDeps("polly-opt-simplify-deps",
                 cl::desc("Dependences should be simplified (yes/no)"),
                 cl::Hidden, cl::init("yes"), cl::cat(PollyCategory));

static cl::opt<int> MaxConstantTerm(
    "polly-opt-max-constant-term",
    cl::desc("The maximal constant term allowed (-1 is unlimited)"), cl::Hidden,
    cl::init(20), cl::cat(PollyCategory));

static cl::opt<int> MaxCoefficient(
    "polly-opt-max-coefficient",
    cl::desc("The maximal coefficient allowed (-1 is unlimited)"), cl::Hidden,
    cl::init(20), cl::cat(PollyCategory));

static cl::opt<std::string>
    MaximizeBandDepth("polly-opt-maximize-bands",
                      cl::desc("Maximize the band depth (yes/no)"), cl::Hidden,
                      cl::init("yes"), cl::cat(PollyCategory));

static cl::opt<bool>
    GreedyFusion("polly-loopfusion-greedy",
                 cl::desc("Aggressively try to fuse everything"), cl::Hidden,
                 cl::cat(PollyCategory));

static cl::opt<std::string> OuterCoincidence(
    "polly-opt-outer-coincidence",
    cl::desc("Try to construct schedules where the outer member of each band "
             "satisfies the coincidence constraints (yes/no)"),
    cl::Hidden, cl::init("no"), cl::cat(PollyCategory));

static cl::opt<int> PrevectorWidth(
    "polly-prevect-width",
    cl::desc(
        "The number of loop iterations to strip-mine for pre-vectorization"),
    cl::Hidden, cl::init(4), cl::cat(PollyCategory));

static cl::opt<bool> FirstLevelTiling("polly-tiling",
                                      cl::desc("Enable loop tiling"),
                                      cl::init(true), cl::cat(PollyCategory));

static cl::opt<int> FirstLevelDefaultTileSize(
    "polly-default-tile-size",
    cl::desc("The default tile size (if not enough were provided by"
             " --polly-tile-sizes)"),
    cl::Hidden, cl::init(32), cl::cat(PollyCategory));

static cl::list<int>
    FirstLevelTileSizes("polly-tile-sizes",
                        cl::desc("A tile size for each loop dimension, filled "
                                 "with --polly-default-tile-size"),
                        cl::Hidden, cl::CommaSeparated, cl::cat(PollyCategory));

static cl::opt<bool>
    SecondLevelTiling("polly-2nd-level-tiling",
                      cl::desc("Enable a 2nd level loop of loop tiling"),
                      cl::cat(PollyCategory));

static cl::opt<int> SecondLevelDefaultTileSize(
    "polly-2nd-level-default-tile-size",
    cl::desc("The default 2nd-level tile size (if not enough were provided by"
             " --polly-2nd-level-tile-sizes)"),
    cl::Hidden, cl::init(16), cl::cat(PollyCategory));

static cl::list<int>
    SecondLevelTileSizes("polly-2nd-level-tile-sizes",
                         cl::desc("A tile size for each loop dimension, filled "
                                  "with --polly-default-tile-size"),
                         cl::Hidden, cl::CommaSeparated,
                         cl::cat(PollyCategory));

static cl::opt<bool> RegisterTiling("polly-register-tiling",
                                    cl::desc("Enable register tiling"),
                                    cl::cat(PollyCategory));

static cl::opt<int> RegisterDefaultTileSize(
    "polly-register-tiling-default-tile-size",
    cl::desc("The default register tile size (if not enough were provided by"
             " --polly-register-tile-sizes)"),
    cl::Hidden, cl::init(2), cl::cat(PollyCategory));

static cl::list<int>
    RegisterTileSizes("polly-register-tile-sizes",
                      cl::desc("A tile size for each loop dimension, filled "
                               "with --polly-register-tile-size"),
                      cl::Hidden, cl::CommaSeparated, cl::cat(PollyCategory));

static cl::opt<bool> PragmaBasedOpts(
    "polly-pragma-based-opts",
    cl::desc("Apply user-directed transformation from metadata"),
    cl::init(true), cl::cat(PollyCategory));

static cl::opt<bool> EnableReschedule("polly-reschedule",
                                      cl::desc("Optimize SCoPs using ISL"),
                                      cl::init(true), cl::cat(PollyCategory));

static cl::opt<bool>
    PMBasedOpts("polly-pattern-matching-based-opts",
                cl::desc("Perform optimizations based on pattern matching"),
                cl::init(true), cl::cat(PollyCategory));

static cl::opt<bool>
    EnablePostopts("polly-postopts",
                   cl::desc("Apply post-rescheduling optimizations such as "
                            "tiling (requires -polly-reschedule)"),
                   cl::init(true), cl::cat(PollyCategory));

static cl::opt<bool> OptimizedScops(
    "polly-optimized-scops",
    cl::desc("Polly - Dump polyhedral description of Scops optimized with "
             "the isl scheduling optimizer and the set of post-scheduling "
             "transformations is applied on the schedule tree"),
    cl::cat(PollyCategory));

STATISTIC(ScopsProcessed, "Number of scops processed");
STATISTIC(ScopsRescheduled, "Number of scops rescheduled");
STATISTIC(ScopsOptimized, "Number of scops optimized");

STATISTIC(NumAffineLoopsOptimized, "Number of affine loops optimized");
STATISTIC(NumBoxedLoopsOptimized, "Number of boxed loops optimized");

#define THREE_STATISTICS(VARNAME, DESC)                                        \
  static Statistic VARNAME[3] = {                                              \
      {DEBUG_TYPE, #VARNAME "0", DESC " (original)"},                          \
      {DEBUG_TYPE, #VARNAME "1", DESC " (after scheduler)"},                   \
      {DEBUG_TYPE, #VARNAME "2", DESC " (after optimizer)"}}

THREE_STATISTICS(NumBands, "Number of bands");
THREE_STATISTICS(NumBandMembers, "Number of band members");
THREE_STATISTICS(NumCoincident, "Number of coincident band members");
THREE_STATISTICS(NumPermutable, "Number of permutable bands");
THREE_STATISTICS(NumFilters, "Number of filter nodes");
THREE_STATISTICS(NumExtension, "Number of extension nodes");

STATISTIC(FirstLevelTileOpts, "Number of first level tiling applied");
STATISTIC(SecondLevelTileOpts, "Number of second level tiling applied");
STATISTIC(RegisterTileOpts, "Number of register tiling applied");
STATISTIC(PrevectOpts, "Number of strip-mining for prevectorization applied");
STATISTIC(MatMulOpts,
          "Number of matrix multiplication patterns detected and optimized");

namespace {
/// Additional parameters of the schedule optimizer.
///
/// Target Transform Info and the SCoP dependencies used by the schedule
/// optimizer.
struct OptimizerAdditionalInfoTy {
  const llvm::TargetTransformInfo *TTI;
  const Dependences *D;
  bool PatternOpts;
  bool Postopts;
  bool Prevect;
  bool &DepsChanged;
};

class ScheduleTreeOptimizer final {
public:
  /// Apply schedule tree transformations.
  ///
  /// This function takes an (possibly already optimized) schedule tree and
  /// applies a set of additional optimizations on the schedule tree. The
  /// transformations applied include:
  ///
  ///   - Pattern-based optimizations
  ///   - Tiling
  ///   - Prevectorization
  ///
  /// @param Schedule The schedule object the transformations will be applied
  ///                 to.
  /// @param OAI      Target Transform Info and the SCoP dependencies.
  /// @returns        The transformed schedule.
  static isl::schedule
  optimizeSchedule(isl::schedule Schedule,
                   const OptimizerAdditionalInfoTy *OAI = nullptr);

  /// Apply schedule tree transformations.
  ///
  /// This function takes a node in an (possibly already optimized) schedule
  /// tree and applies a set of additional optimizations on this schedule tree
  /// node and its descendants. The transformations applied include:
  ///
  ///   - Pattern-based optimizations
  ///   - Tiling
  ///   - Prevectorization
  ///
  /// @param Node The schedule object post-transformations will be applied to.
  /// @param OAI  Target Transform Info and the SCoP dependencies.
  /// @returns    The transformed schedule.
  static isl::schedule_node
  optimizeScheduleNode(isl::schedule_node Node,
                       const OptimizerAdditionalInfoTy *OAI = nullptr);

  /// Decide if the @p NewSchedule is profitable for @p S.
  ///
  /// @param S           The SCoP we optimize.
  /// @param NewSchedule The new schedule we computed.
  ///
  /// @return True, if we believe @p NewSchedule is an improvement for @p S.
  static bool isProfitableSchedule(polly::Scop &S, isl::schedule NewSchedule);

  /// Isolate a set of partial tile prefixes.
  ///
  /// This set should ensure that it contains only partial tile prefixes that
  /// have exactly VectorWidth iterations.
  ///
  /// @param Node A schedule node band, which is a parent of a band node,
  ///             that contains a vector loop.
  /// @return Modified isl_schedule_node.
  static isl::schedule_node isolateFullPartialTiles(isl::schedule_node Node,
                                                    int VectorWidth);

private:
  /// Check if this node is a band node we want to tile.
  ///
  /// We look for innermost band nodes where individual dimensions are marked as
  /// permutable.
  ///
  /// @param Node The node to check.
  static bool isTileableBandNode(isl::schedule_node Node);

  /// Pre-vectorizes one scheduling dimension of a schedule band.
  ///
  /// prevectSchedBand splits out the dimension DimToVectorize, tiles it and
  /// sinks the resulting point loop.
  ///
  /// Example (DimToVectorize=0, VectorWidth=4):
  ///
  /// | Before transformation:
  /// |
  /// | A[i,j] -> [i,j]
  /// |
  /// | for (i = 0; i < 128; i++)
  /// |    for (j = 0; j < 128; j++)
  /// |      A(i,j);
  ///
  /// | After transformation:
  /// |
  /// | for (it = 0; it < 32; it+=1)
  /// |    for (j = 0; j < 128; j++)
  /// |      for (ip = 0; ip <= 3; ip++)
  /// |        A(4 * it + ip,j);
  ///
  /// The goal of this transformation is to create a trivially vectorizable
  /// loop.  This means a parallel loop at the innermost level that has a
  /// constant number of iterations corresponding to the target vector width.
  ///
  /// This transformation creates a loop at the innermost level. The loop has
  /// a constant number of iterations, if the number of loop iterations at
  /// DimToVectorize can be divided by VectorWidth. The default VectorWidth is
  /// currently constant and not yet target specific. This function does not
  /// reason about parallelism.
  static isl::schedule_node prevectSchedBand(isl::schedule_node Node,
                                             unsigned DimToVectorize,
                                             int VectorWidth);

  /// Apply additional optimizations on the bands in the schedule tree.
  ///
  /// We are looking for an innermost band node and apply the following
  /// transformations:
  ///
  ///  - Tile the band
  ///      - if the band is tileable
  ///      - if the band has more than one loop dimension
  ///
  ///  - Prevectorize the schedule of the band (or the point loop in case of
  ///    tiling).
  ///      - if vectorization is enabled
  ///
  /// @param Node The schedule node to (possibly) optimize.
  /// @param User A pointer to forward some use information
  ///        (currently unused).
  static isl_schedule_node *optimizeBand(isl_schedule_node *Node, void *User);

  /// Apply tiling optimizations on the bands in the schedule tree.
  ///
  /// @param Node The schedule node to (possibly) optimize.
  static isl::schedule_node applyTileBandOpt(isl::schedule_node Node);

  /// Apply prevectorization on the bands in the schedule tree.
  ///
  /// @param Node The schedule node to (possibly) prevectorize.
  static isl::schedule_node applyPrevectBandOpt(isl::schedule_node Node);
};

isl::schedule_node
ScheduleTreeOptimizer::isolateFullPartialTiles(isl::schedule_node Node,
                                               int VectorWidth) {
  assert(isl_schedule_node_get_type(Node.get()) == isl_schedule_node_band);
  Node = Node.child(0).child(0);
  isl::union_map SchedRelUMap = Node.get_prefix_schedule_relation();
  isl::union_set ScheduleRangeUSet = SchedRelUMap.range();
  isl::set ScheduleRange{ScheduleRangeUSet};
  isl::set IsolateDomain = getPartialTilePrefixes(ScheduleRange, VectorWidth);
  auto AtomicOption = getDimOptions(IsolateDomain.ctx(), "atomic");
  isl::union_set IsolateOption = getIsolateOptions(IsolateDomain, 1);
  Node = Node.parent().parent();
  isl::union_set Options = IsolateOption.unite(AtomicOption);
  isl::schedule_node_band Result =
      Node.as<isl::schedule_node_band>().set_ast_build_options(Options);
  return Result;
}

struct InsertSimdMarkers final : ScheduleNodeRewriter<InsertSimdMarkers> {
  isl::schedule_node visitBand(isl::schedule_node_band Band) {
    isl::schedule_node Node = visitChildren(Band);

    // Only add SIMD markers to innermost bands.
    if (!Node.first_child().isa<isl::schedule_node_leaf>())
      return Node;

    isl::id LoopMarker = isl::id::alloc(Band.ctx(), "SIMD", nullptr);
    return Band.insert_mark(LoopMarker);
  }
};

isl::schedule_node ScheduleTreeOptimizer::prevectSchedBand(
    isl::schedule_node Node, unsigned DimToVectorize, int VectorWidth) {
  assert(isl_schedule_node_get_type(Node.get()) == isl_schedule_node_band);

  auto Space = isl::manage(isl_schedule_node_band_get_space(Node.get()));
  unsigned ScheduleDimensions = unsignedFromIslSize(Space.dim(isl::dim::set));
  assert(DimToVectorize < ScheduleDimensions);

  if (DimToVectorize > 0) {
    Node = isl::manage(
        isl_schedule_node_band_split(Node.release(), DimToVectorize));
    Node = Node.child(0);
  }
  if (DimToVectorize < ScheduleDimensions - 1)
    Node = isl::manage(isl_schedule_node_band_split(Node.release(), 1));
  Space = isl::manage(isl_schedule_node_band_get_space(Node.get()));
  auto Sizes = isl::multi_val::zero(Space);
  Sizes = Sizes.set_val(0, isl::val(Node.ctx(), VectorWidth));
  Node =
      isl::manage(isl_schedule_node_band_tile(Node.release(), Sizes.release()));
  Node = isolateFullPartialTiles(Node, VectorWidth);
  Node = Node.child(0);
  // Make sure the "trivially vectorizable loop" is not unrolled. Otherwise,
  // we will have troubles to match it in the backend.
  Node = Node.as<isl::schedule_node_band>().set_ast_build_options(
      isl::union_set(Node.ctx(), "{ unroll[x]: 1 = 0 }"));

  // Sink the inner loop into the smallest possible statements to make them
  // represent a single vector instruction if possible.
  Node = isl::manage(isl_schedule_node_band_sink(Node.release()));

  // Add SIMD markers to those vector statements.
  InsertSimdMarkers SimdMarkerInserter;
  Node = SimdMarkerInserter.visit(Node);

  PrevectOpts++;
  return Node.parent();
}

static bool isSimpleInnermostBand(const isl::schedule_node &Node) {
  assert(isl_schedule_node_get_type(Node.get()) == isl_schedule_node_band);
  assert(isl_schedule_node_n_children(Node.get()) == 1);

  auto ChildType = isl_schedule_node_get_type(Node.child(0).get());

  if (ChildType == isl_schedule_node_leaf)
    return true;

  if (ChildType != isl_schedule_node_sequence)
    return false;

  auto Sequence = Node.child(0);

  for (int c = 0, nc = isl_schedule_node_n_children(Sequence.get()); c < nc;
       ++c) {
    auto Child = Sequence.child(c);
    if (isl_schedule_node_get_type(Child.get()) != isl_schedule_node_filter)
      return false;
    if (isl_schedule_node_get_type(Child.child(0).get()) !=
        isl_schedule_node_leaf)
      return false;
  }
  return true;
}

bool ScheduleTreeOptimizer::isTileableBandNode(isl::schedule_node Node) {
  if (isl_schedule_node_get_type(Node.get()) != isl_schedule_node_band)
    return false;

  if (isl_schedule_node_n_children(Node.get()) != 1)
    return false;

  if (!isl_schedule_node_band_get_permutable(Node.get()))
    return false;

  auto Space = isl::manage(isl_schedule_node_band_get_space(Node.get()));

  if (unsignedFromIslSize(Space.dim(isl::dim::set)) <= 1u)
    return false;

  return isSimpleInnermostBand(Node);
}

__isl_give isl::schedule_node
ScheduleTreeOptimizer::applyTileBandOpt(isl::schedule_node Node) {
  if (FirstLevelTiling) {
    Node = tileNode(Node, "1st level tiling", FirstLevelTileSizes,
                    FirstLevelDefaultTileSize);
    FirstLevelTileOpts++;
  }

  if (SecondLevelTiling) {
    Node = tileNode(Node, "2nd level tiling", SecondLevelTileSizes,
                    SecondLevelDefaultTileSize);
    SecondLevelTileOpts++;
  }

  if (RegisterTiling) {
    Node =
        applyRegisterTiling(Node, RegisterTileSizes, RegisterDefaultTileSize);
    RegisterTileOpts++;
  }

  return Node;
}

isl::schedule_node
ScheduleTreeOptimizer::applyPrevectBandOpt(isl::schedule_node Node) {
  auto Space = isl::manage(isl_schedule_node_band_get_space(Node.get()));
  int Dims = unsignedFromIslSize(Space.dim(isl::dim::set));

  for (int i = Dims - 1; i >= 0; i--)
    if (Node.as<isl::schedule_node_band>().member_get_coincident(i)) {
      Node = prevectSchedBand(Node, i, PrevectorWidth);
      break;
    }

  return Node;
}

__isl_give isl_schedule_node *
ScheduleTreeOptimizer::optimizeBand(__isl_take isl_schedule_node *NodeArg,
                                    void *User) {
  const OptimizerAdditionalInfoTy *OAI =
      static_cast<const OptimizerAdditionalInfoTy *>(User);
  assert(OAI && "Expecting optimization options");

  isl::schedule_node Node = isl::manage(NodeArg);
  if (!isTileableBandNode(Node))
    return Node.release();

  if (OAI->PatternOpts) {
    isl::schedule_node PatternOptimizedSchedule =
        tryOptimizeMatMulPattern(Node, OAI->TTI, OAI->D);
    if (!PatternOptimizedSchedule.is_null()) {
      MatMulOpts++;
      OAI->DepsChanged = true;
      return PatternOptimizedSchedule.release();
    }
  }

  if (OAI->Postopts)
    Node = applyTileBandOpt(Node);

  if (OAI->Prevect) {
    // FIXME: Prevectorization requirements are different from those checked by
    // isTileableBandNode.
    Node = applyPrevectBandOpt(Node);
  }

  return Node.release();
}

isl::schedule
ScheduleTreeOptimizer::optimizeSchedule(isl::schedule Schedule,
                                        const OptimizerAdditionalInfoTy *OAI) {
  auto Root = Schedule.get_root();
  Root = optimizeScheduleNode(Root, OAI);
  return Root.get_schedule();
}

isl::schedule_node ScheduleTreeOptimizer::optimizeScheduleNode(
    isl::schedule_node Node, const OptimizerAdditionalInfoTy *OAI) {
  Node = isl::manage(isl_schedule_node_map_descendant_bottom_up(
      Node.release(), optimizeBand,
      const_cast<void *>(static_cast<const void *>(OAI))));
  return Node;
}

bool ScheduleTreeOptimizer::isProfitableSchedule(Scop &S,
                                                 isl::schedule NewSchedule) {
  // To understand if the schedule has been optimized we check if the schedule
  // has changed at all.
  // TODO: We can improve this by tracking if any necessarily beneficial
  // transformations have been performed. This can e.g. be tiling, loop
  // interchange, or ...) We can track this either at the place where the
  // transformation has been performed or, in case of automatic ILP based
  // optimizations, by comparing (yet to be defined) performance metrics
  // before/after the scheduling optimizer
  // (e.g., #stride-one accesses)
  // FIXME: A schedule tree whose union_map-conversion is identical to the
  // original schedule map may still allow for parallelization, i.e. can still
  // be profitable.
  auto NewScheduleMap = NewSchedule.get_map();
  auto OldSchedule = S.getSchedule();
  assert(!OldSchedule.is_null() &&
         "Only IslScheduleOptimizer can insert extension nodes "
         "that make Scop::getSchedule() return nullptr.");
  bool changed = !OldSchedule.is_equal(NewScheduleMap);
  return changed;
}

class IslScheduleOptimizerWrapperPass final : public ScopPass {
public:
  static char ID;

  explicit IslScheduleOptimizerWrapperPass() : ScopPass(ID) {}

  /// Optimize the schedule of the SCoP @p S.
  bool runOnScop(Scop &S) override;

  /// Print the new schedule for the SCoP @p S.
  void printScop(raw_ostream &OS, Scop &S) const override;

  /// Register all analyses and transformation required.
  void getAnalysisUsage(AnalysisUsage &AU) const override;

  /// Release the internal memory.
  void releaseMemory() override {
    LastSchedule = {};
    IslCtx.reset();
  }

private:
  std::shared_ptr<isl_ctx> IslCtx;
  isl::schedule LastSchedule;
};

char IslScheduleOptimizerWrapperPass::ID = 0;

#ifndef NDEBUG
static void printSchedule(llvm::raw_ostream &OS, const isl::schedule &Schedule,
                          StringRef Desc) {
  isl::ctx Ctx = Schedule.ctx();
  isl_printer *P = isl_printer_to_str(Ctx.get());
  P = isl_printer_set_yaml_style(P, ISL_YAML_STYLE_BLOCK);
  P = isl_printer_print_schedule(P, Schedule.get());
  char *Str = isl_printer_get_str(P);
  OS << Desc << ": \n" << Str << "\n";
  free(Str);
  isl_printer_free(P);
}
#endif

/// Collect statistics for the schedule tree.
///
/// @param Schedule The schedule tree to analyze. If not a schedule tree it is
/// ignored.
/// @param Version  The version of the schedule tree that is analyzed.
///                 0 for the original schedule tree before any transformation.
///                 1 for the schedule tree after isl's rescheduling.
///                 2 for the schedule tree after optimizations are applied
///                 (tiling, pattern matching)
static void walkScheduleTreeForStatistics(isl::schedule Schedule, int Version) {
  auto Root = Schedule.get_root();
  if (Root.is_null())
    return;

  isl_schedule_node_foreach_descendant_top_down(
      Root.get(),
      [](__isl_keep isl_schedule_node *nodeptr, void *user) -> isl_bool {
        isl::schedule_node Node = isl::manage_copy(nodeptr);
        int Version = *static_cast<int *>(user);

        switch (isl_schedule_node_get_type(Node.get())) {
        case isl_schedule_node_band: {
          NumBands[Version]++;
          if (isl_schedule_node_band_get_permutable(Node.get()) ==
              isl_bool_true)
            NumPermutable[Version]++;

          int CountMembers = isl_schedule_node_band_n_member(Node.get());
          NumBandMembers[Version] += CountMembers;
          for (int i = 0; i < CountMembers; i += 1) {
            if (Node.as<isl::schedule_node_band>().member_get_coincident(i))
              NumCoincident[Version]++;
          }
          break;
        }

        case isl_schedule_node_filter:
          NumFilters[Version]++;
          break;

        case isl_schedule_node_extension:
          NumExtension[Version]++;
          break;

        default:
          break;
        }

        return isl_bool_true;
      },
      &Version);
}

static void runIslScheduleOptimizer(
    Scop &S,
    function_ref<const Dependences &(Dependences::AnalysisLevel)> GetDeps,
    TargetTransformInfo *TTI, OptimizationRemarkEmitter *ORE,
    isl::schedule &LastSchedule, bool &DepsChanged) {

  // Skip SCoPs in case they're already optimised by PPCGCodeGeneration
  if (S.isToBeSkipped())
    return;

  // Skip empty SCoPs but still allow code generation as it will delete the
  // loops present but not needed.
  if (S.getSize() == 0) {
    S.markAsOptimized();
    return;
  }

  ScopsProcessed++;

  // Schedule without optimizations.
  isl::schedule Schedule = S.getScheduleTree();
  walkScheduleTreeForStatistics(S.getScheduleTree(), 0);
  LLVM_DEBUG(printSchedule(dbgs(), Schedule, "Original schedule tree"));

  bool HasUserTransformation = false;
  if (PragmaBasedOpts) {
    isl::schedule ManuallyTransformed = applyManualTransformations(
        &S, Schedule, GetDeps(Dependences::AL_Statement), ORE);
    if (ManuallyTransformed.is_null()) {
      LLVM_DEBUG(dbgs() << "Error during manual optimization\n");
      return;
    }

    if (ManuallyTransformed.get() != Schedule.get()) {
      // User transformations have precedence over other transformations.
      HasUserTransformation = true;
      Schedule = std::move(ManuallyTransformed);
      LLVM_DEBUG(
          printSchedule(dbgs(), Schedule, "After manual transformations"));
    }
  }

  // Only continue if either manual transformations have been applied or we are
  // allowed to apply heuristics.
  // TODO: Detect disabled heuristics and no user-directed transformation
  // metadata earlier in ScopDetection.
  if (!HasUserTransformation && S.hasDisableHeuristicsHint()) {
    LLVM_DEBUG(dbgs() << "Heuristic optimizations disabled by metadata\n");
    return;
  }

  // Get dependency analysis.
  const Dependences &D = GetDeps(Dependences::AL_Statement);
  if (D.getSharedIslCtx() != S.getSharedIslCtx()) {
    LLVM_DEBUG(dbgs() << "DependenceInfo for another SCoP/isl_ctx\n");
    return;
  }
  if (!D.hasValidDependences()) {
    LLVM_DEBUG(dbgs() << "Dependency information not available\n");
    return;
  }

  // Apply ISL's algorithm only if not overriden by the user. Note that
  // post-rescheduling optimizations (tiling, pattern-based, prevectorization)
  // rely on the coincidence/permutable annotations on schedule tree bands that
  // are added by the rescheduling analyzer. Therefore, disabling the
  // rescheduler implicitly also disables these optimizations.
  if (!EnableReschedule) {
    LLVM_DEBUG(dbgs() << "Skipping rescheduling due to command line option\n");
  } else if (HasUserTransformation) {
    LLVM_DEBUG(
        dbgs() << "Skipping rescheduling due to manual transformation\n");
  } else {
    // Build input data.
    int ValidityKinds =
        Dependences::TYPE_RAW | Dependences::TYPE_WAR | Dependences::TYPE_WAW;
    int ProximityKinds;

    if (OptimizeDeps == "all")
      ProximityKinds =
          Dependences::TYPE_RAW | Dependences::TYPE_WAR | Dependences::TYPE_WAW;
    else if (OptimizeDeps == "raw")
      ProximityKinds = Dependences::TYPE_RAW;
    else {
      errs() << "Do not know how to optimize for '" << OptimizeDeps << "'"
             << " Falling back to optimizing all dependences.\n";
      ProximityKinds =
          Dependences::TYPE_RAW | Dependences::TYPE_WAR | Dependences::TYPE_WAW;
    }

    isl::union_set Domain = S.getDomains();

    if (Domain.is_null())
      return;

    isl::union_map Validity = D.getDependences(ValidityKinds);
    isl::union_map Proximity = D.getDependences(ProximityKinds);

    // Simplify the dependences by removing the constraints introduced by the
    // domains. This can speed up the scheduling time significantly, as large
    // constant coefficients will be removed from the dependences. The
    // introduction of some additional dependences reduces the possible
    // transformations, but in most cases, such transformation do not seem to be
    // interesting anyway. In some cases this option may stop the scheduler to
    // find any schedule.
    if (SimplifyDeps == "yes") {
      Validity = Validity.gist_domain(Domain);
      Validity = Validity.gist_range(Domain);
      Proximity = Proximity.gist_domain(Domain);
      Proximity = Proximity.gist_range(Domain);
    } else if (SimplifyDeps != "no") {
      errs()
          << "warning: Option -polly-opt-simplify-deps should either be 'yes' "
             "or 'no'. Falling back to default: 'yes'\n";
    }

    LLVM_DEBUG(dbgs() << "\n\nCompute schedule from: ");
    LLVM_DEBUG(dbgs() << "Domain := " << Domain << ";\n");
    LLVM_DEBUG(dbgs() << "Proximity := " << Proximity << ";\n");
    LLVM_DEBUG(dbgs() << "Validity := " << Validity << ";\n");

    int IslMaximizeBands;
    if (MaximizeBandDepth == "yes") {
      IslMaximizeBands = 1;
    } else if (MaximizeBandDepth == "no") {
      IslMaximizeBands = 0;
    } else {
      errs()
          << "warning: Option -polly-opt-maximize-bands should either be 'yes'"
             " or 'no'. Falling back to default: 'yes'\n";
      IslMaximizeBands = 1;
    }

    int IslOuterCoincidence;
    if (OuterCoincidence == "yes") {
      IslOuterCoincidence = 1;
    } else if (OuterCoincidence == "no") {
      IslOuterCoincidence = 0;
    } else {
      errs() << "warning: Option -polly-opt-outer-coincidence should either be "
                "'yes' or 'no'. Falling back to default: 'no'\n";
      IslOuterCoincidence = 0;
    }

    isl_ctx *Ctx = S.getIslCtx().get();

    isl_options_set_schedule_outer_coincidence(Ctx, IslOuterCoincidence);
    isl_options_set_schedule_maximize_band_depth(Ctx, IslMaximizeBands);
    isl_options_set_schedule_max_constant_term(Ctx, MaxConstantTerm);
    isl_options_set_schedule_max_coefficient(Ctx, MaxCoefficient);
    isl_options_set_tile_scale_tile_loops(Ctx, 0);

    auto OnErrorStatus = isl_options_get_on_error(Ctx);
    isl_options_set_on_error(Ctx, ISL_ON_ERROR_CONTINUE);

    auto SC = isl::schedule_constraints::on_domain(Domain);
    SC = SC.set_proximity(Proximity);
    SC = SC.set_validity(Validity);
    SC = SC.set_coincidence(Validity);
    Schedule = SC.compute_schedule();
    isl_options_set_on_error(Ctx, OnErrorStatus);

    ScopsRescheduled++;
    LLVM_DEBUG(printSchedule(dbgs(), Schedule, "After rescheduling"));
  }

  walkScheduleTreeForStatistics(Schedule, 1);

  // In cases the scheduler is not able to optimize the code, we just do not
  // touch the schedule.
  if (Schedule.is_null())
    return;

  if (GreedyFusion) {
    isl::union_map Validity = D.getDependences(
        Dependences::TYPE_RAW | Dependences::TYPE_WAR | Dependences::TYPE_WAW);
    Schedule = applyGreedyFusion(Schedule, Validity);
    assert(!Schedule.is_null());
  }

  // Apply post-rescheduling optimizations (if enabled) and/or prevectorization.
  const OptimizerAdditionalInfoTy OAI = {
      TTI,
      const_cast<Dependences *>(&D),
      /*PatternOpts=*/!HasUserTransformation && PMBasedOpts,
      /*Postopts=*/!HasUserTransformation && EnablePostopts,
      /*Prevect=*/PollyVectorizerChoice != VECTORIZER_NONE,
      DepsChanged};
  if (OAI.PatternOpts || OAI.Postopts || OAI.Prevect) {
    Schedule = ScheduleTreeOptimizer::optimizeSchedule(Schedule, &OAI);
    Schedule = hoistExtensionNodes(Schedule);
    LLVM_DEBUG(printSchedule(dbgs(), Schedule, "After post-optimizations"));
    walkScheduleTreeForStatistics(Schedule, 2);
  }

  // Skip profitability check if user transformation(s) have been applied.
  if (!HasUserTransformation &&
      !ScheduleTreeOptimizer::isProfitableSchedule(S, Schedule))
    return;

  auto ScopStats = S.getStatistics();
  ScopsOptimized++;
  NumAffineLoopsOptimized += ScopStats.NumAffineLoops;
  NumBoxedLoopsOptimized += ScopStats.NumBoxedLoops;
  LastSchedule = Schedule;

  S.setScheduleTree(Schedule);
  S.markAsOptimized();

  if (OptimizedScops)
    errs() << S;
}

bool IslScheduleOptimizerWrapperPass::runOnScop(Scop &S) {
  releaseMemory();

  Function &F = S.getFunction();
  IslCtx = S.getSharedIslCtx();

  auto getDependences =
      [this](Dependences::AnalysisLevel) -> const Dependences & {
    return getAnalysis<DependenceInfo>().getDependences(
        Dependences::AL_Statement);
  };
  OptimizationRemarkEmitter &ORE =
      getAnalysis<OptimizationRemarkEmitterWrapperPass>().getORE();
  TargetTransformInfo *TTI =
      &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);

  bool DepsChanged = false;
  runIslScheduleOptimizer(S, getDependences, TTI, &ORE, LastSchedule,
                          DepsChanged);
  if (DepsChanged)
    getAnalysis<DependenceInfo>().abandonDependences();
  return false;
}

static void runScheduleOptimizerPrinter(raw_ostream &OS,
                                        isl::schedule LastSchedule) {
  isl_printer *p;
  char *ScheduleStr;

  OS << "Calculated schedule:\n";

  if (LastSchedule.is_null()) {
    OS << "n/a\n";
    return;
  }

  p = isl_printer_to_str(LastSchedule.ctx().get());
  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
  p = isl_printer_print_schedule(p, LastSchedule.get());
  ScheduleStr = isl_printer_get_str(p);
  isl_printer_free(p);

  OS << ScheduleStr << "\n";

  free(ScheduleStr);
}

void IslScheduleOptimizerWrapperPass::printScop(raw_ostream &OS, Scop &) const {
  runScheduleOptimizerPrinter(OS, LastSchedule);
}

void IslScheduleOptimizerWrapperPass::getAnalysisUsage(
    AnalysisUsage &AU) const {
  ScopPass::getAnalysisUsage(AU);
  AU.addRequired<DependenceInfo>();
  AU.addRequired<TargetTransformInfoWrapperPass>();
  AU.addRequired<OptimizationRemarkEmitterWrapperPass>();

  AU.addPreserved<DependenceInfo>();
  AU.addPreserved<OptimizationRemarkEmitterWrapperPass>();
}

} // namespace

Pass *polly::createIslScheduleOptimizerWrapperPass() {
  return new IslScheduleOptimizerWrapperPass();
}

INITIALIZE_PASS_BEGIN(IslScheduleOptimizerWrapperPass, "polly-opt-isl",
                      "Polly - Optimize schedule of SCoP", false, false);
INITIALIZE_PASS_DEPENDENCY(DependenceInfo);
INITIALIZE_PASS_DEPENDENCY(ScopInfoRegionPass);
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass);
INITIALIZE_PASS_DEPENDENCY(OptimizationRemarkEmitterWrapperPass);
INITIALIZE_PASS_END(IslScheduleOptimizerWrapperPass, "polly-opt-isl",
                    "Polly - Optimize schedule of SCoP", false, false)

static llvm::PreservedAnalyses
runIslScheduleOptimizerUsingNPM(Scop &S, ScopAnalysisManager &SAM,
                                ScopStandardAnalysisResults &SAR, SPMUpdater &U,
                                raw_ostream *OS) {
  DependenceAnalysis::Result &Deps = SAM.getResult<DependenceAnalysis>(S, SAR);
  auto GetDeps = [&Deps](Dependences::AnalysisLevel) -> const Dependences & {
    return Deps.getDependences(Dependences::AL_Statement);
  };
  OptimizationRemarkEmitter ORE(&S.getFunction());
  TargetTransformInfo *TTI = &SAR.TTI;
  isl::schedule LastSchedule;
  bool DepsChanged = false;
  runIslScheduleOptimizer(S, GetDeps, TTI, &ORE, LastSchedule, DepsChanged);
  if (DepsChanged)
    Deps.abandonDependences();

  if (OS) {
    *OS << "Printing analysis 'Polly - Optimize schedule of SCoP' for region: '"
        << S.getName() << "' in function '" << S.getFunction().getName()
        << "':\n";
    runScheduleOptimizerPrinter(*OS, LastSchedule);
  }
  return PreservedAnalyses::all();
}

llvm::PreservedAnalyses
IslScheduleOptimizerPass::run(Scop &S, ScopAnalysisManager &SAM,
                              ScopStandardAnalysisResults &SAR, SPMUpdater &U) {
  return runIslScheduleOptimizerUsingNPM(S, SAM, SAR, U, nullptr);
}

llvm::PreservedAnalyses
IslScheduleOptimizerPrinterPass::run(Scop &S, ScopAnalysisManager &SAM,
                                     ScopStandardAnalysisResults &SAR,
                                     SPMUpdater &U) {
  return runIslScheduleOptimizerUsingNPM(S, SAM, SAR, U, &OS);
}

//===----------------------------------------------------------------------===//

namespace {
/// Print result from IslScheduleOptimizerWrapperPass.
class IslScheduleOptimizerPrinterLegacyPass final : public ScopPass {
public:
  static char ID;

  IslScheduleOptimizerPrinterLegacyPass()
      : IslScheduleOptimizerPrinterLegacyPass(outs()) {}
  explicit IslScheduleOptimizerPrinterLegacyPass(llvm::raw_ostream &OS)
      : ScopPass(ID), OS(OS) {}

  bool runOnScop(Scop &S) override {
    IslScheduleOptimizerWrapperPass &P =
        getAnalysis<IslScheduleOptimizerWrapperPass>();

    OS << "Printing analysis '" << P.getPassName() << "' for region: '"
       << S.getRegion().getNameStr() << "' in function '"
       << S.getFunction().getName() << "':\n";
    P.printScop(OS, S);

    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    ScopPass::getAnalysisUsage(AU);
    AU.addRequired<IslScheduleOptimizerWrapperPass>();
    AU.setPreservesAll();
  }

private:
  llvm::raw_ostream &OS;
};

char IslScheduleOptimizerPrinterLegacyPass::ID = 0;
} // namespace

Pass *polly::createIslScheduleOptimizerPrinterLegacyPass(raw_ostream &OS) {
  return new IslScheduleOptimizerPrinterLegacyPass(OS);
}

INITIALIZE_PASS_BEGIN(IslScheduleOptimizerPrinterLegacyPass,
                      "polly-print-opt-isl",
                      "Polly - Print optimizer schedule of SCoP", false, false);
INITIALIZE_PASS_DEPENDENCY(IslScheduleOptimizerWrapperPass)
INITIALIZE_PASS_END(IslScheduleOptimizerPrinterLegacyPass,
                    "polly-print-opt-isl",
                    "Polly - Print optimizer schedule of SCoP", false, false)
