//===- Schedule.cpp - Calculate an optimized schedule ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "polly/LinkAllPasses.h"
#include "polly/Options.h"
#include "polly/ScopInfo.h"
#include "polly/Support/GICHelper.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Support/Debug.h"
#include "isl/aff.h"
#include "isl/band.h"
#include "isl/constraint.h"
#include "isl/map.h"
#include "isl/options.h"
#include "isl/printer.h"
#include "isl/schedule.h"
#include "isl/schedule_node.h"
#include "isl/space.h"
#include "isl/union_map.h"
#include "isl/union_set.h"

using namespace llvm;
using namespace polly;

#define DEBUG_TYPE "polly-opt-isl"

static cl::opt<std::string>
    OptimizeDeps("polly-opt-optimize-only",
                 cl::desc("Only a certain kind of dependences (all/raw)"),
                 cl::Hidden, cl::init("all"), cl::ZeroOrMore,
                 cl::cat(PollyCategory));

static cl::opt<std::string>
    SimplifyDeps("polly-opt-simplify-deps",
                 cl::desc("Dependences should be simplified (yes/no)"),
                 cl::Hidden, cl::init("yes"), cl::ZeroOrMore,
                 cl::cat(PollyCategory));

static cl::opt<int> MaxConstantTerm(
    "polly-opt-max-constant-term",
    cl::desc("The maximal constant term allowed (-1 is unlimited)"), cl::Hidden,
    cl::init(20), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> MaxCoefficient(
    "polly-opt-max-coefficient",
    cl::desc("The maximal coefficient allowed (-1 is unlimited)"), cl::Hidden,
    cl::init(20), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<std::string> FusionStrategy(
    "polly-opt-fusion", cl::desc("The fusion strategy to choose (min/max)"),
    cl::Hidden, cl::init("min"), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<std::string>
    MaximizeBandDepth("polly-opt-maximize-bands",
                      cl::desc("Maximize the band depth (yes/no)"), cl::Hidden,
                      cl::init("yes"), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<std::string> OuterCoincidence(
    "polly-opt-outer-coincidence",
    cl::desc("Try to construct schedules where the outer member of each band "
             "satisfies the coincidence constraints (yes/no)"),
    cl::Hidden, cl::init("no"), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> PrevectorWidth(
    "polly-prevect-width",
    cl::desc(
        "The number of loop iterations to strip-mine for pre-vectorization"),
    cl::Hidden, cl::init(4), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<bool> FirstLevelTiling("polly-tiling",
                                      cl::desc("Enable loop tiling"),
                                      cl::init(true), cl::ZeroOrMore,
                                      cl::cat(PollyCategory));

static cl::opt<int> LatencyVectorFma(
    "polly-target-latency-vector-fma",
    cl::desc("The minimal number of cycles between issuing two "
             "dependent consecutive vector fused multiply-add "
             "instructions."),
    cl::Hidden, cl::init(8), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> ThroughputVectorFma(
    "polly-target-throughput-vector-fma",
    cl::desc("A throughput of the processor floating-point arithmetic units "
             "expressed in the number of vector fused multiply-add "
             "instructions per clock cycle."),
    cl::Hidden, cl::init(1), cl::ZeroOrMore, cl::cat(PollyCategory));

// This option, along with --polly-target-2nd-cache-level-associativity,
// --polly-target-1st-cache-level-size, and --polly-target-2st-cache-level-size
// represent the parameters of the target cache, which do not have typical
// values that can be used by default. However, to apply the pattern matching
// optimizations, we use the values of the parameters of Intel Core i7-3820
// SandyBridge in case the parameters are not specified. Such an approach helps
// also to attain the high-performance on IBM POWER System S822 and IBM Power
// 730 Express server.
static cl::opt<int> FirstCacheLevelAssociativity(
    "polly-target-1st-cache-level-associativity",
    cl::desc("The associativity of the first cache level."), cl::Hidden,
    cl::init(8), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> SecondCacheLevelAssociativity(
    "polly-target-2nd-cache-level-associativity",
    cl::desc("The associativity of the second cache level."), cl::Hidden,
    cl::init(8), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> FirstCacheLevelSize(
    "polly-target-1st-cache-level-size",
    cl::desc("The size of the first cache level specified in bytes."),
    cl::Hidden, cl::init(32768), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> SecondCacheLevelSize(
    "polly-target-2nd-cache-level-size",
    cl::desc("The size of the second level specified in bytes."), cl::Hidden,
    cl::init(262144), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> VectorRegisterBitwidth(
    "polly-target-vector-register-bitwidth",
    cl::desc("The size in bits of a vector register (if not set, this "
             "information is taken from LLVM's target information."),
    cl::Hidden, cl::init(-1), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> FirstLevelDefaultTileSize(
    "polly-default-tile-size",
    cl::desc("The default tile size (if not enough were provided by"
             " --polly-tile-sizes)"),
    cl::Hidden, cl::init(32), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::list<int>
    FirstLevelTileSizes("polly-tile-sizes",
                        cl::desc("A tile size for each loop dimension, filled "
                                 "with --polly-default-tile-size"),
                        cl::Hidden, cl::ZeroOrMore, cl::CommaSeparated,
                        cl::cat(PollyCategory));

static cl::opt<bool>
    SecondLevelTiling("polly-2nd-level-tiling",
                      cl::desc("Enable a 2nd level loop of loop tiling"),
                      cl::init(false), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> SecondLevelDefaultTileSize(
    "polly-2nd-level-default-tile-size",
    cl::desc("The default 2nd-level tile size (if not enough were provided by"
             " --polly-2nd-level-tile-sizes)"),
    cl::Hidden, cl::init(16), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::list<int>
    SecondLevelTileSizes("polly-2nd-level-tile-sizes",
                         cl::desc("A tile size for each loop dimension, filled "
                                  "with --polly-default-tile-size"),
                         cl::Hidden, cl::ZeroOrMore, cl::CommaSeparated,
                         cl::cat(PollyCategory));

static cl::opt<bool> RegisterTiling("polly-register-tiling",
                                    cl::desc("Enable register tiling"),
                                    cl::init(false), cl::ZeroOrMore,
                                    cl::cat(PollyCategory));

static cl::opt<int> RegisterDefaultTileSize(
    "polly-register-tiling-default-tile-size",
    cl::desc("The default register tile size (if not enough were provided by"
             " --polly-register-tile-sizes)"),
    cl::Hidden, cl::init(2), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> PollyPatternMatchingNcQuotient(
    "polly-pattern-matching-nc-quotient",
    cl::desc("Quotient that is obtained by dividing Nc, the parameter of the"
             "macro-kernel, by Nr, the parameter of the micro-kernel"),
    cl::Hidden, cl::init(256), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::list<int>
    RegisterTileSizes("polly-register-tile-sizes",
                      cl::desc("A tile size for each loop dimension, filled "
                               "with --polly-register-tile-size"),
                      cl::Hidden, cl::ZeroOrMore, cl::CommaSeparated,
                      cl::cat(PollyCategory));

static cl::opt<bool>
    PMBasedOpts("polly-pattern-matching-based-opts",
                cl::desc("Perform optimizations based on pattern matching"),
                cl::init(true), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<bool> OptimizedScops(
    "polly-optimized-scops",
    cl::desc("Polly - Dump polyhedral description of Scops optimized with "
             "the isl scheduling optimizer and the set of post-scheduling "
             "transformations is applied on the schedule tree"),
    cl::init(false), cl::ZeroOrMore, cl::cat(PollyCategory));

/// Create an isl_union_set, which describes the isolate option based on
/// IsoalteDomain.
///
/// @param IsolateDomain An isl_set whose @p OutDimsNum last dimensions should
///                      belong to the current band node.
/// @param OutDimsNum    A number of dimensions that should belong to
///                      the current band node.
static __isl_give isl_union_set *
getIsolateOptions(__isl_take isl_set *IsolateDomain, unsigned OutDimsNum) {
  auto Dims = isl_set_dim(IsolateDomain, isl_dim_set);
  assert(OutDimsNum <= Dims &&
         "The isl_set IsolateDomain is used to describe the range of schedule "
         "dimensions values, which should be isolated. Consequently, the "
         "number of its dimensions should be greater than or equal to the "
         "number of the schedule dimensions.");
  auto *IsolateRelation = isl_map_from_domain(IsolateDomain);
  IsolateRelation =
      isl_map_move_dims(IsolateRelation, isl_dim_out, 0, isl_dim_in,
                        Dims - OutDimsNum, OutDimsNum);
  auto *IsolateOption = isl_map_wrap(IsolateRelation);
  auto *Id = isl_id_alloc(isl_set_get_ctx(IsolateOption), "isolate", nullptr);
  return isl_union_set_from_set(isl_set_set_tuple_id(IsolateOption, Id));
}

/// Create an isl_union_set, which describes the atomic option for the dimension
/// of the current node.
///
/// It may help to reduce the size of generated code.
///
/// @param Ctx An isl_ctx, which is used to create the isl_union_set.
static __isl_give isl_union_set *getAtomicOptions(isl_ctx *Ctx) {
  auto *Space = isl_space_set_alloc(Ctx, 0, 1);
  auto *AtomicOption = isl_set_universe(Space);
  auto *Id = isl_id_alloc(Ctx, "atomic", nullptr);
  return isl_union_set_from_set(isl_set_set_tuple_id(AtomicOption, Id));
}

/// Create an isl_union_set, which describes the option of the form
/// [isolate[] -> unroll[x]].
///
/// @param Ctx An isl_ctx, which is used to create the isl_union_set.
static __isl_give isl_union_set *getUnrollIsolatedSetOptions(isl_ctx *Ctx) {
  auto *Space = isl_space_alloc(Ctx, 0, 0, 1);
  auto *UnrollIsolatedSetOption = isl_map_universe(Space);
  auto *DimInId = isl_id_alloc(Ctx, "isolate", nullptr);
  auto *DimOutId = isl_id_alloc(Ctx, "unroll", nullptr);
  UnrollIsolatedSetOption =
      isl_map_set_tuple_id(UnrollIsolatedSetOption, isl_dim_in, DimInId);
  UnrollIsolatedSetOption =
      isl_map_set_tuple_id(UnrollIsolatedSetOption, isl_dim_out, DimOutId);
  return isl_union_set_from_set(isl_map_wrap(UnrollIsolatedSetOption));
}

/// Make the last dimension of Set to take values from 0 to VectorWidth - 1.
///
/// @param Set         A set, which should be modified.
/// @param VectorWidth A parameter, which determines the constraint.
static __isl_give isl_set *addExtentConstraints(__isl_take isl_set *Set,
                                                int VectorWidth) {
  auto Dims = isl_set_dim(Set, isl_dim_set);
  auto Space = isl_set_get_space(Set);
  auto *LocalSpace = isl_local_space_from_space(Space);
  auto *ExtConstr =
      isl_constraint_alloc_inequality(isl_local_space_copy(LocalSpace));
  ExtConstr = isl_constraint_set_constant_si(ExtConstr, 0);
  ExtConstr =
      isl_constraint_set_coefficient_si(ExtConstr, isl_dim_set, Dims - 1, 1);
  Set = isl_set_add_constraint(Set, ExtConstr);
  ExtConstr = isl_constraint_alloc_inequality(LocalSpace);
  ExtConstr = isl_constraint_set_constant_si(ExtConstr, VectorWidth - 1);
  ExtConstr =
      isl_constraint_set_coefficient_si(ExtConstr, isl_dim_set, Dims - 1, -1);
  return isl_set_add_constraint(Set, ExtConstr);
}

/// Build the desired set of partial tile prefixes.
///
/// We build a set of partial tile prefixes, which are prefixes of the vector
/// loop that have exactly VectorWidth iterations.
///
/// 1. Get all prefixes of the vector loop.
/// 2. Extend it to a set, which has exactly VectorWidth iterations for
///    any prefix from the set that was built on the previous step.
/// 3. Subtract loop domain from it, project out the vector loop dimension and
///    get a set of prefixes, which don't have exactly VectorWidth iterations.
/// 4. Subtract it from all prefixes of the vector loop and get the desired
///    set.
///
/// @param ScheduleRange A range of a map, which describes a prefix schedule
///                      relation.
static __isl_give isl_set *
getPartialTilePrefixes(__isl_take isl_set *ScheduleRange, int VectorWidth) {
  auto Dims = isl_set_dim(ScheduleRange, isl_dim_set);
  auto *LoopPrefixes = isl_set_project_out(isl_set_copy(ScheduleRange),
                                           isl_dim_set, Dims - 1, 1);
  auto *ExtentPrefixes =
      isl_set_add_dims(isl_set_copy(LoopPrefixes), isl_dim_set, 1);
  ExtentPrefixes = addExtentConstraints(ExtentPrefixes, VectorWidth);
  auto *BadPrefixes = isl_set_subtract(ExtentPrefixes, ScheduleRange);
  BadPrefixes = isl_set_project_out(BadPrefixes, isl_dim_set, Dims - 1, 1);
  return isl_set_subtract(LoopPrefixes, BadPrefixes);
}

__isl_give isl_schedule_node *ScheduleTreeOptimizer::isolateFullPartialTiles(
    __isl_take isl_schedule_node *Node, int VectorWidth) {
  assert(isl_schedule_node_get_type(Node) == isl_schedule_node_band);
  Node = isl_schedule_node_child(Node, 0);
  Node = isl_schedule_node_child(Node, 0);
  auto *SchedRelUMap = isl_schedule_node_get_prefix_schedule_relation(Node);
  auto *ScheduleRelation = isl_map_from_union_map(SchedRelUMap);
  auto *ScheduleRange = isl_map_range(ScheduleRelation);
  auto *IsolateDomain = getPartialTilePrefixes(ScheduleRange, VectorWidth);
  auto *AtomicOption = getAtomicOptions(isl_set_get_ctx(IsolateDomain));
  auto *IsolateOption = getIsolateOptions(IsolateDomain, 1);
  Node = isl_schedule_node_parent(Node);
  Node = isl_schedule_node_parent(Node);
  auto *Options = isl_union_set_union(IsolateOption, AtomicOption);
  Node = isl_schedule_node_band_set_ast_build_options(Node, Options);
  return Node;
}

__isl_give isl_schedule_node *
ScheduleTreeOptimizer::prevectSchedBand(__isl_take isl_schedule_node *Node,
                                        unsigned DimToVectorize,
                                        int VectorWidth) {
  assert(isl_schedule_node_get_type(Node) == isl_schedule_node_band);

  auto Space = isl_schedule_node_band_get_space(Node);
  auto ScheduleDimensions = isl_space_dim(Space, isl_dim_set);
  isl_space_free(Space);
  assert(DimToVectorize < ScheduleDimensions);

  if (DimToVectorize > 0) {
    Node = isl_schedule_node_band_split(Node, DimToVectorize);
    Node = isl_schedule_node_child(Node, 0);
  }
  if (DimToVectorize < ScheduleDimensions - 1)
    Node = isl_schedule_node_band_split(Node, 1);
  Space = isl_schedule_node_band_get_space(Node);
  auto Sizes = isl_multi_val_zero(Space);
  auto Ctx = isl_schedule_node_get_ctx(Node);
  Sizes =
      isl_multi_val_set_val(Sizes, 0, isl_val_int_from_si(Ctx, VectorWidth));
  Node = isl_schedule_node_band_tile(Node, Sizes);
  Node = isolateFullPartialTiles(Node, VectorWidth);
  Node = isl_schedule_node_child(Node, 0);
  // Make sure the "trivially vectorizable loop" is not unrolled. Otherwise,
  // we will have troubles to match it in the backend.
  Node = isl_schedule_node_band_set_ast_build_options(
      Node, isl_union_set_read_from_str(Ctx, "{ unroll[x]: 1 = 0 }"));
  Node = isl_schedule_node_band_sink(Node);
  Node = isl_schedule_node_child(Node, 0);
  if (isl_schedule_node_get_type(Node) == isl_schedule_node_leaf)
    Node = isl_schedule_node_parent(Node);
  isl_id *LoopMarker = isl_id_alloc(Ctx, "SIMD", nullptr);
  Node = isl_schedule_node_insert_mark(Node, LoopMarker);
  return Node;
}

__isl_give isl_schedule_node *
ScheduleTreeOptimizer::tileNode(__isl_take isl_schedule_node *Node,
                                const char *Identifier, ArrayRef<int> TileSizes,
                                int DefaultTileSize) {
  auto Ctx = isl_schedule_node_get_ctx(Node);
  auto Space = isl_schedule_node_band_get_space(Node);
  auto Dims = isl_space_dim(Space, isl_dim_set);
  auto Sizes = isl_multi_val_zero(Space);
  std::string IdentifierString(Identifier);
  for (unsigned i = 0; i < Dims; i++) {
    auto tileSize = i < TileSizes.size() ? TileSizes[i] : DefaultTileSize;
    Sizes = isl_multi_val_set_val(Sizes, i, isl_val_int_from_si(Ctx, tileSize));
  }
  auto TileLoopMarkerStr = IdentifierString + " - Tiles";
  isl_id *TileLoopMarker =
      isl_id_alloc(Ctx, TileLoopMarkerStr.c_str(), nullptr);
  Node = isl_schedule_node_insert_mark(Node, TileLoopMarker);
  Node = isl_schedule_node_child(Node, 0);
  Node = isl_schedule_node_band_tile(Node, Sizes);
  Node = isl_schedule_node_child(Node, 0);
  auto PointLoopMarkerStr = IdentifierString + " - Points";
  isl_id *PointLoopMarker =
      isl_id_alloc(Ctx, PointLoopMarkerStr.c_str(), nullptr);
  Node = isl_schedule_node_insert_mark(Node, PointLoopMarker);
  Node = isl_schedule_node_child(Node, 0);
  return Node;
}

__isl_give isl_schedule_node *
ScheduleTreeOptimizer::applyRegisterTiling(__isl_take isl_schedule_node *Node,
                                           llvm::ArrayRef<int> TileSizes,
                                           int DefaultTileSize) {
  auto *Ctx = isl_schedule_node_get_ctx(Node);
  Node = tileNode(Node, "Register tiling", TileSizes, DefaultTileSize);
  Node = isl_schedule_node_band_set_ast_build_options(
      Node, isl_union_set_read_from_str(Ctx, "{unroll[x]}"));
  return Node;
}

namespace {
bool isSimpleInnermostBand(const isl::schedule_node &Node) {
  assert(isl_schedule_node_get_type(Node.keep()) == isl_schedule_node_band);
  assert(isl_schedule_node_n_children(Node.keep()) == 1);

  auto ChildType = isl_schedule_node_get_type(Node.child(0).keep());

  if (ChildType == isl_schedule_node_leaf)
    return true;

  if (ChildType != isl_schedule_node_sequence)
    return false;

  auto Sequence = Node.child(0);

  for (int c = 0, nc = isl_schedule_node_n_children(Sequence.keep()); c < nc;
       ++c) {
    auto Child = Sequence.child(c);
    if (isl_schedule_node_get_type(Child.keep()) != isl_schedule_node_filter)
      return false;
    if (isl_schedule_node_get_type(Child.child(0).keep()) !=
        isl_schedule_node_leaf)
      return false;
  }
  return true;
}
} // namespace

bool ScheduleTreeOptimizer::isTileableBandNode(
    __isl_keep isl_schedule_node *Node) {
  if (isl_schedule_node_get_type(Node) != isl_schedule_node_band)
    return false;

  if (isl_schedule_node_n_children(Node) != 1)
    return false;

  if (!isl_schedule_node_band_get_permutable(Node))
    return false;

  auto Space = isl_schedule_node_band_get_space(Node);
  auto Dims = isl_space_dim(Space, isl_dim_set);
  isl_space_free(Space);

  if (Dims <= 1)
    return false;

  auto ManagedNode = isl::manage(isl_schedule_node_copy(Node));
  return isSimpleInnermostBand(ManagedNode);
}

__isl_give isl_schedule_node *
ScheduleTreeOptimizer::standardBandOpts(__isl_take isl_schedule_node *Node,
                                        void *User) {
  if (FirstLevelTiling)
    Node = tileNode(Node, "1st level tiling", FirstLevelTileSizes,
                    FirstLevelDefaultTileSize);

  if (SecondLevelTiling)
    Node = tileNode(Node, "2nd level tiling", SecondLevelTileSizes,
                    SecondLevelDefaultTileSize);

  if (RegisterTiling)
    Node =
        applyRegisterTiling(Node, RegisterTileSizes, RegisterDefaultTileSize);

  if (PollyVectorizerChoice == VECTORIZER_NONE)
    return Node;

  auto Space = isl_schedule_node_band_get_space(Node);
  auto Dims = isl_space_dim(Space, isl_dim_set);
  isl_space_free(Space);

  for (int i = Dims - 1; i >= 0; i--)
    if (isl_schedule_node_band_member_get_coincident(Node, i)) {
      Node = prevectSchedBand(Node, i, PrevectorWidth);
      break;
    }

  return Node;
}

/// Get the position of a dimension with a non-zero coefficient.
///
/// Check that isl constraint @p Constraint has only one non-zero
/// coefficient for dimensions that have type @p DimType. If this is true,
/// return the position of the dimension corresponding to the non-zero
/// coefficient and negative value, otherwise.
///
/// @param Constraint The isl constraint to be checked.
/// @param DimType    The type of the dimensions.
/// @return           The position of the dimension in case the isl
///                   constraint satisfies the requirements, a negative
///                   value, otherwise.
static int getMatMulConstraintDim(__isl_keep isl_constraint *Constraint,
                                  enum isl_dim_type DimType) {
  int DimPos = -1;
  auto *LocalSpace = isl_constraint_get_local_space(Constraint);
  int LocalSpaceDimNum = isl_local_space_dim(LocalSpace, DimType);
  for (int i = 0; i < LocalSpaceDimNum; i++) {
    auto *Val = isl_constraint_get_coefficient_val(Constraint, DimType, i);
    if (isl_val_is_zero(Val)) {
      isl_val_free(Val);
      continue;
    }
    if (DimPos >= 0 || (DimType == isl_dim_out && !isl_val_is_one(Val)) ||
        (DimType == isl_dim_in && !isl_val_is_negone(Val))) {
      isl_val_free(Val);
      isl_local_space_free(LocalSpace);
      return -1;
    }
    DimPos = i;
    isl_val_free(Val);
  }
  isl_local_space_free(LocalSpace);
  return DimPos;
}

/// Check the form of the isl constraint.
///
/// Check that the @p DimInPos input dimension of the isl constraint
/// @p Constraint has a coefficient that is equal to negative one, the @p
/// DimOutPos has a coefficient that is equal to one and others
/// have coefficients equal to zero.
///
/// @param Constraint The isl constraint to be checked.
/// @param DimInPos   The input dimension of the isl constraint.
/// @param DimOutPos  The output dimension of the isl constraint.
/// @return           isl_stat_ok in case the isl constraint satisfies
///                   the requirements, isl_stat_error otherwise.
static isl_stat isMatMulOperandConstraint(__isl_keep isl_constraint *Constraint,
                                          int &DimInPos, int &DimOutPos) {
  auto *Val = isl_constraint_get_constant_val(Constraint);
  if (!isl_constraint_is_equality(Constraint) || !isl_val_is_zero(Val)) {
    isl_val_free(Val);
    return isl_stat_error;
  }
  isl_val_free(Val);
  DimInPos = getMatMulConstraintDim(Constraint, isl_dim_in);
  if (DimInPos < 0)
    return isl_stat_error;
  DimOutPos = getMatMulConstraintDim(Constraint, isl_dim_out);
  if (DimOutPos < 0)
    return isl_stat_error;
  return isl_stat_ok;
}

/// Check that the access relation corresponds to a non-constant operand
/// of the matrix multiplication.
///
/// Access relations that correspond to non-constant operands of the matrix
/// multiplication depend only on two input dimensions and have two output
/// dimensions. The function checks that the isl basic map @p bmap satisfies
/// the requirements. The two input dimensions can be specified via @p user
/// array.
///
/// @param bmap The isl basic map to be checked.
/// @param user The input dimensions of @p bmap.
/// @return     isl_stat_ok in case isl basic map satisfies the requirements,
///             isl_stat_error otherwise.
static isl_stat isMatMulOperandBasicMap(__isl_take isl_basic_map *bmap,
                                        void *user) {
  auto *Constraints = isl_basic_map_get_constraint_list(bmap);
  isl_basic_map_free(bmap);
  if (isl_constraint_list_n_constraint(Constraints) != 2) {
    isl_constraint_list_free(Constraints);
    return isl_stat_error;
  }
  int InPosPair[] = {-1, -1};
  auto DimInPos = user ? static_cast<int *>(user) : InPosPair;
  for (int i = 0; i < 2; i++) {
    auto *Constraint = isl_constraint_list_get_constraint(Constraints, i);
    int InPos, OutPos;
    if (isMatMulOperandConstraint(Constraint, InPos, OutPos) ==
            isl_stat_error ||
        OutPos > 1 || (DimInPos[OutPos] >= 0 && DimInPos[OutPos] != InPos)) {
      isl_constraint_free(Constraint);
      isl_constraint_list_free(Constraints);
      return isl_stat_error;
    }
    DimInPos[OutPos] = InPos;
    isl_constraint_free(Constraint);
  }
  isl_constraint_list_free(Constraints);
  return isl_stat_ok;
}

/// Permute the two dimensions of the isl map.
///
/// Permute @p DstPos and @p SrcPos dimensions of the isl map @p Map that
/// have type @p DimType.
///
/// @param Map     The isl map to be modified.
/// @param DimType The type of the dimensions.
/// @param DstPos  The first dimension.
/// @param SrcPos  The second dimension.
/// @return        The modified map.
__isl_give isl_map *permuteDimensions(__isl_take isl_map *Map,
                                      enum isl_dim_type DimType,
                                      unsigned DstPos, unsigned SrcPos) {
  assert(DstPos < isl_map_dim(Map, DimType) &&
         SrcPos < isl_map_dim(Map, DimType));
  if (DstPos == SrcPos)
    return Map;
  isl_id *DimId = nullptr;
  if (isl_map_has_tuple_id(Map, DimType))
    DimId = isl_map_get_tuple_id(Map, DimType);
  auto FreeDim = DimType == isl_dim_in ? isl_dim_out : isl_dim_in;
  isl_id *FreeDimId = nullptr;
  if (isl_map_has_tuple_id(Map, FreeDim))
    FreeDimId = isl_map_get_tuple_id(Map, FreeDim);
  auto MaxDim = std::max(DstPos, SrcPos);
  auto MinDim = std::min(DstPos, SrcPos);
  Map = isl_map_move_dims(Map, FreeDim, 0, DimType, MaxDim, 1);
  Map = isl_map_move_dims(Map, FreeDim, 0, DimType, MinDim, 1);
  Map = isl_map_move_dims(Map, DimType, MinDim, FreeDim, 1, 1);
  Map = isl_map_move_dims(Map, DimType, MaxDim, FreeDim, 0, 1);
  if (DimId)
    Map = isl_map_set_tuple_id(Map, DimType, DimId);
  if (FreeDimId)
    Map = isl_map_set_tuple_id(Map, FreeDim, FreeDimId);
  return Map;
}

/// Check the form of the access relation.
///
/// Check that the access relation @p AccMap has the form M[i][j], where i
/// is a @p FirstPos and j is a @p SecondPos.
///
/// @param AccMap    The access relation to be checked.
/// @param FirstPos  The index of the input dimension that is mapped to
///                  the first output dimension.
/// @param SecondPos The index of the input dimension that is mapped to the
///                  second output dimension.
/// @return          True in case @p AccMap has the expected form and false,
///                  otherwise.
static bool isMatMulOperandAcc(__isl_keep isl_map *AccMap, int &FirstPos,
                               int &SecondPos) {
  int DimInPos[] = {FirstPos, SecondPos};
  if (isl_map_foreach_basic_map(AccMap, isMatMulOperandBasicMap,
                                static_cast<void *>(DimInPos)) != isl_stat_ok ||
      DimInPos[0] < 0 || DimInPos[1] < 0)
    return false;
  FirstPos = DimInPos[0];
  SecondPos = DimInPos[1];
  return true;
}

/// Does the memory access represent a non-scalar operand of the matrix
/// multiplication.
///
/// Check that the memory access @p MemAccess is the read access to a non-scalar
/// operand of the matrix multiplication or its result.
///
/// @param MemAccess The memory access to be checked.
/// @param MMI       Parameters of the matrix multiplication operands.
/// @return          True in case the memory access represents the read access
///                  to a non-scalar operand of the matrix multiplication and
///                  false, otherwise.
static bool isMatMulNonScalarReadAccess(MemoryAccess *MemAccess,
                                        MatMulInfoTy &MMI) {
  if (!MemAccess->isArrayKind() || !MemAccess->isRead())
    return false;
  isl_map *AccMap = MemAccess->getAccessRelation();
  if (isMatMulOperandAcc(AccMap, MMI.i, MMI.j) && !MMI.ReadFromC &&
      isl_map_n_basic_map(AccMap) == 1) {
    MMI.ReadFromC = MemAccess;
    isl_map_free(AccMap);
    return true;
  }
  if (isMatMulOperandAcc(AccMap, MMI.i, MMI.k) && !MMI.A &&
      isl_map_n_basic_map(AccMap) == 1) {
    MMI.A = MemAccess;
    isl_map_free(AccMap);
    return true;
  }
  if (isMatMulOperandAcc(AccMap, MMI.k, MMI.j) && !MMI.B &&
      isl_map_n_basic_map(AccMap) == 1) {
    MMI.B = MemAccess;
    isl_map_free(AccMap);
    return true;
  }
  isl_map_free(AccMap);
  return false;
}

/// Check accesses to operands of the matrix multiplication.
///
/// Check that accesses of the SCoP statement, which corresponds to
/// the partial schedule @p PartialSchedule, are scalar in terms of loops
/// containing the matrix multiplication, in case they do not represent
/// accesses to the non-scalar operands of the matrix multiplication or
/// its result.
///
/// @param  PartialSchedule The partial schedule of the SCoP statement.
/// @param  MMI             Parameters of the matrix multiplication operands.
/// @return                 True in case the corresponding SCoP statement
///                         represents matrix multiplication and false,
///                         otherwise.
static bool containsOnlyMatrMultAcc(__isl_keep isl_map *PartialSchedule,
                                    MatMulInfoTy &MMI) {
  auto *InputDimId = isl_map_get_tuple_id(PartialSchedule, isl_dim_in);
  auto *Stmt = static_cast<ScopStmt *>(isl_id_get_user(InputDimId));
  isl_id_free(InputDimId);
  unsigned OutDimNum = isl_map_dim(PartialSchedule, isl_dim_out);
  assert(OutDimNum > 2 && "In case of the matrix multiplication the loop nest "
                          "and, consequently, the corresponding scheduling "
                          "functions have at least three dimensions.");
  auto *MapI = permuteDimensions(isl_map_copy(PartialSchedule), isl_dim_out,
                                 MMI.i, OutDimNum - 1);
  auto *MapJ = permuteDimensions(isl_map_copy(PartialSchedule), isl_dim_out,
                                 MMI.j, OutDimNum - 1);
  auto *MapK = permuteDimensions(isl_map_copy(PartialSchedule), isl_dim_out,
                                 MMI.k, OutDimNum - 1);
  for (auto *MemA = Stmt->begin(); MemA != Stmt->end() - 1; MemA++) {
    auto *MemAccessPtr = *MemA;
    if (MemAccessPtr->isArrayKind() && MemAccessPtr != MMI.WriteToC &&
        !isMatMulNonScalarReadAccess(MemAccessPtr, MMI) &&
        !(MemAccessPtr->isStrideZero(isl_map_copy(MapI)) &&
          MemAccessPtr->isStrideZero(isl_map_copy(MapJ)) &&
          MemAccessPtr->isStrideZero(isl_map_copy(MapK)))) {
      isl_map_free(MapI);
      isl_map_free(MapJ);
      isl_map_free(MapK);
      return false;
    }
  }
  isl_map_free(MapI);
  isl_map_free(MapJ);
  isl_map_free(MapK);
  return true;
}

/// Check for dependencies corresponding to the matrix multiplication.
///
/// Check that there is only true dependence of the form
/// S(..., k, ...) -> S(..., k + 1, …), where S is the SCoP statement
/// represented by @p Schedule and k is @p Pos. Such a dependence corresponds
/// to the dependency produced by the matrix multiplication.
///
/// @param  Schedule The schedule of the SCoP statement.
/// @param  D The SCoP dependencies.
/// @param  Pos The parameter to desribe an acceptable true dependence.
///             In case it has a negative value, try to determine its
///             acceptable value.
/// @return True in case dependencies correspond to the matrix multiplication
///         and false, otherwise.
static bool containsOnlyMatMulDep(__isl_keep isl_map *Schedule,
                                  const Dependences *D, int &Pos) {
  auto *WAR = D->getDependences(Dependences::TYPE_WAR);
  if (!isl_union_map_is_empty(WAR)) {
    isl_union_map_free(WAR);
    return false;
  }
  isl_union_map_free(WAR);
  auto *Dep = D->getDependences(Dependences::TYPE_RAW);
  auto *Red = D->getDependences(Dependences::TYPE_RED);
  if (Red)
    Dep = isl_union_map_union(Dep, Red);
  auto *DomainSpace = isl_space_domain(isl_map_get_space(Schedule));
  auto *Space = isl_space_map_from_domain_and_range(isl_space_copy(DomainSpace),
                                                    DomainSpace);
  auto *Deltas = isl_map_deltas(isl_union_map_extract_map(Dep, Space));
  isl_union_map_free(Dep);
  int DeltasDimNum = isl_set_dim(Deltas, isl_dim_set);
  for (int i = 0; i < DeltasDimNum; i++) {
    auto *Val = isl_set_plain_get_val_if_fixed(Deltas, isl_dim_set, i);
    Pos = Pos < 0 && isl_val_is_one(Val) ? i : Pos;
    if (isl_val_is_nan(Val) ||
        !(isl_val_is_zero(Val) || (i == Pos && isl_val_is_one(Val)))) {
      isl_val_free(Val);
      isl_set_free(Deltas);
      return false;
    }
    isl_val_free(Val);
  }
  isl_set_free(Deltas);
  if (DeltasDimNum == 0 || Pos < 0)
    return false;
  return true;
}

/// Check if the SCoP statement could probably be optimized with analytical
/// modeling.
///
/// containsMatrMult tries to determine whether the following conditions
/// are true:
/// 1. The last memory access modeling an array, MA1, represents writing to
///    memory and has the form S(..., i1, ..., i2, ...) -> M(i1, i2) or
///    S(..., i2, ..., i1, ...) -> M(i1, i2), where S is the SCoP statement
///    under consideration.
/// 2. There is only one loop-carried true dependency, and it has the
///    form S(..., i3, ...) -> S(..., i3 + 1, ...), and there are no
///    loop-carried or anti dependencies.
/// 3. SCoP contains three access relations, MA2, MA3, and MA4 that represent
///    reading from memory and have the form S(..., i3, ...) -> M(i1, i3),
///    S(..., i3, ...) -> M(i3, i2), S(...) -> M(i1, i2), respectively,
///    and all memory accesses of the SCoP that are different from MA1, MA2,
///    MA3, and MA4 have stride 0, if the innermost loop is exchanged with any
///    of loops i1, i2 and i3.
///
/// @param PartialSchedule The PartialSchedule that contains a SCoP statement
///        to check.
/// @D     The SCoP dependencies.
/// @MMI   Parameters of the matrix multiplication operands.
static bool containsMatrMult(__isl_keep isl_map *PartialSchedule,
                             const Dependences *D, MatMulInfoTy &MMI) {
  auto *InputDimsId = isl_map_get_tuple_id(PartialSchedule, isl_dim_in);
  auto *Stmt = static_cast<ScopStmt *>(isl_id_get_user(InputDimsId));
  isl_id_free(InputDimsId);
  if (Stmt->size() <= 1)
    return false;
  for (auto *MemA = Stmt->end() - 1; MemA != Stmt->begin(); MemA--) {
    auto *MemAccessPtr = *MemA;
    if (!MemAccessPtr->isArrayKind())
      continue;
    if (!MemAccessPtr->isWrite())
      return false;
    auto *AccMap = MemAccessPtr->getAccessRelation();
    if (isl_map_n_basic_map(AccMap) != 1 ||
        !isMatMulOperandAcc(AccMap, MMI.i, MMI.j)) {
      isl_map_free(AccMap);
      return false;
    }
    isl_map_free(AccMap);
    MMI.WriteToC = MemAccessPtr;
    break;
  }

  if (!containsOnlyMatMulDep(PartialSchedule, D, MMI.k))
    return false;

  if (!MMI.WriteToC || !containsOnlyMatrMultAcc(PartialSchedule, MMI))
    return false;

  if (!MMI.A || !MMI.B || !MMI.ReadFromC)
    return false;
  return true;
}

/// Permute two dimensions of the band node.
///
/// Permute FirstDim and SecondDim dimensions of the Node.
///
/// @param Node The band node to be modified.
/// @param FirstDim The first dimension to be permuted.
/// @param SecondDim The second dimension to be permuted.
static __isl_give isl_schedule_node *
permuteBandNodeDimensions(__isl_take isl_schedule_node *Node, unsigned FirstDim,
                          unsigned SecondDim) {
  assert(isl_schedule_node_get_type(Node) == isl_schedule_node_band &&
         isl_schedule_node_band_n_member(Node) > std::max(FirstDim, SecondDim));
  auto PartialSchedule = isl_schedule_node_band_get_partial_schedule(Node);
  auto PartialScheduleFirstDim =
      isl_multi_union_pw_aff_get_union_pw_aff(PartialSchedule, FirstDim);
  auto PartialScheduleSecondDim =
      isl_multi_union_pw_aff_get_union_pw_aff(PartialSchedule, SecondDim);
  PartialSchedule = isl_multi_union_pw_aff_set_union_pw_aff(
      PartialSchedule, SecondDim, PartialScheduleFirstDim);
  PartialSchedule = isl_multi_union_pw_aff_set_union_pw_aff(
      PartialSchedule, FirstDim, PartialScheduleSecondDim);
  Node = isl_schedule_node_delete(Node);
  Node = isl_schedule_node_insert_partial_schedule(Node, PartialSchedule);
  return Node;
}

__isl_give isl_schedule_node *ScheduleTreeOptimizer::createMicroKernel(
    __isl_take isl_schedule_node *Node, MicroKernelParamsTy MicroKernelParams) {
  applyRegisterTiling(Node, {MicroKernelParams.Mr, MicroKernelParams.Nr}, 1);
  Node = isl_schedule_node_parent(isl_schedule_node_parent(Node));
  Node = permuteBandNodeDimensions(Node, 0, 1);
  return isl_schedule_node_child(isl_schedule_node_child(Node, 0), 0);
}

__isl_give isl_schedule_node *ScheduleTreeOptimizer::createMacroKernel(
    __isl_take isl_schedule_node *Node, MacroKernelParamsTy MacroKernelParams) {
  assert(isl_schedule_node_get_type(Node) == isl_schedule_node_band);
  if (MacroKernelParams.Mc == 1 && MacroKernelParams.Nc == 1 &&
      MacroKernelParams.Kc == 1)
    return Node;
  int DimOutNum = isl_schedule_node_band_n_member(Node);
  std::vector<int> TileSizes(DimOutNum, 1);
  TileSizes[DimOutNum - 3] = MacroKernelParams.Mc;
  TileSizes[DimOutNum - 2] = MacroKernelParams.Nc;
  TileSizes[DimOutNum - 1] = MacroKernelParams.Kc;
  Node = tileNode(Node, "1st level tiling", TileSizes, 1);
  Node = isl_schedule_node_parent(isl_schedule_node_parent(Node));
  Node = permuteBandNodeDimensions(Node, DimOutNum - 2, DimOutNum - 1);
  Node = permuteBandNodeDimensions(Node, DimOutNum - 3, DimOutNum - 1);
  return isl_schedule_node_child(isl_schedule_node_child(Node, 0), 0);
}

/// Get the size of the widest type of the matrix multiplication operands
/// in bytes, including alignment padding.
///
/// @param MMI Parameters of the matrix multiplication operands.
/// @return The size of the widest type of the matrix multiplication operands
///         in bytes, including alignment padding.
static uint64_t getMatMulAlignTypeSize(MatMulInfoTy MMI) {
  auto *S = MMI.A->getStatement()->getParent();
  auto &DL = S->getFunction().getParent()->getDataLayout();
  auto ElementSizeA = DL.getTypeAllocSize(MMI.A->getElementType());
  auto ElementSizeB = DL.getTypeAllocSize(MMI.B->getElementType());
  auto ElementSizeC = DL.getTypeAllocSize(MMI.WriteToC->getElementType());
  return std::max({ElementSizeA, ElementSizeB, ElementSizeC});
}

/// Get the size of the widest type of the matrix multiplication operands
/// in bits.
///
/// @param MMI Parameters of the matrix multiplication operands.
/// @return The size of the widest type of the matrix multiplication operands
///         in bits.
static uint64_t getMatMulTypeSize(MatMulInfoTy MMI) {
  auto *S = MMI.A->getStatement()->getParent();
  auto &DL = S->getFunction().getParent()->getDataLayout();
  auto ElementSizeA = DL.getTypeSizeInBits(MMI.A->getElementType());
  auto ElementSizeB = DL.getTypeSizeInBits(MMI.B->getElementType());
  auto ElementSizeC = DL.getTypeSizeInBits(MMI.WriteToC->getElementType());
  return std::max({ElementSizeA, ElementSizeB, ElementSizeC});
}

/// Get parameters of the BLIS micro kernel.
///
/// We choose the Mr and Nr parameters of the micro kernel to be large enough
/// such that no stalls caused by the combination of latencies and dependencies
/// are introduced during the updates of the resulting matrix of the matrix
/// multiplication. However, they should also be as small as possible to
/// release more registers for entries of multiplied matrices.
///
/// @param TTI Target Transform Info.
/// @param MMI Parameters of the matrix multiplication operands.
/// @return The structure of type MicroKernelParamsTy.
/// @see MicroKernelParamsTy
static struct MicroKernelParamsTy
getMicroKernelParams(const llvm::TargetTransformInfo *TTI, MatMulInfoTy MMI) {
  assert(TTI && "The target transform info should be provided.");

  // Nvec - Number of double-precision floating-point numbers that can be hold
  // by a vector register. Use 2 by default.
  long RegisterBitwidth = VectorRegisterBitwidth;

  if (RegisterBitwidth == -1)
    RegisterBitwidth = TTI->getRegisterBitWidth(true);
  auto ElementSize = getMatMulTypeSize(MMI);
  assert(ElementSize > 0 && "The element size of the matrix multiplication "
                            "operands should be greater than zero.");
  auto Nvec = RegisterBitwidth / ElementSize;
  if (Nvec == 0)
    Nvec = 2;
  int Nr =
      ceil(sqrt(Nvec * LatencyVectorFma * ThroughputVectorFma) / Nvec) * Nvec;
  int Mr = ceil(Nvec * LatencyVectorFma * ThroughputVectorFma / Nr);
  return {Mr, Nr};
}

/// Get parameters of the BLIS macro kernel.
///
/// During the computation of matrix multiplication, blocks of partitioned
/// matrices are mapped to different layers of the memory hierarchy.
/// To optimize data reuse, blocks should be ideally kept in cache between
/// iterations. Since parameters of the macro kernel determine sizes of these
/// blocks, there are upper and lower bounds on these parameters.
///
/// @param MicroKernelParams Parameters of the micro-kernel
///                          to be taken into account.
/// @param MMI Parameters of the matrix multiplication operands.
/// @return The structure of type MacroKernelParamsTy.
/// @see MacroKernelParamsTy
/// @see MicroKernelParamsTy
static struct MacroKernelParamsTy
getMacroKernelParams(const MicroKernelParamsTy &MicroKernelParams,
                     MatMulInfoTy MMI) {
  // According to www.cs.utexas.edu/users/flame/pubs/TOMS-BLIS-Analytical.pdf,
  // it requires information about the first two levels of a cache to determine
  // all the parameters of a macro-kernel. It also checks that an associativity
  // degree of a cache level is greater than two. Otherwise, another algorithm
  // for determination of the parameters should be used.
  if (!(MicroKernelParams.Mr > 0 && MicroKernelParams.Nr > 0 &&
        FirstCacheLevelSize > 0 && SecondCacheLevelSize > 0 &&
        FirstCacheLevelAssociativity > 2 && SecondCacheLevelAssociativity > 2))
    return {1, 1, 1};
  // The quotient should be greater than zero.
  if (PollyPatternMatchingNcQuotient <= 0)
    return {1, 1, 1};
  int Car = floor(
      (FirstCacheLevelAssociativity - 1) /
      (1 + static_cast<double>(MicroKernelParams.Nr) / MicroKernelParams.Mr));
  auto ElementSize = getMatMulAlignTypeSize(MMI);
  assert(ElementSize > 0 && "The element size of the matrix multiplication "
                            "operands should be greater than zero.");
  int Kc = (Car * FirstCacheLevelSize) /
           (MicroKernelParams.Mr * FirstCacheLevelAssociativity * ElementSize);
  double Cac =
      static_cast<double>(Kc * ElementSize * SecondCacheLevelAssociativity) /
      SecondCacheLevelSize;
  int Mc = floor((SecondCacheLevelAssociativity - 2) / Cac);
  int Nc = PollyPatternMatchingNcQuotient * MicroKernelParams.Nr;
  return {Mc, Nc, Kc};
}

/// Create an access relation that is specific to
///        the matrix multiplication pattern.
///
/// Create an access relation of the following form:
/// [O0, O1, O2, O3, O4, O5, O6, O7, O8] -> [OI, O5, OJ]
/// where I is @p FirstDim, J is @p SecondDim.
///
/// It can be used, for example, to create relations that helps to consequently
/// access elements of operands of a matrix multiplication after creation of
/// the BLIS micro and macro kernels.
///
/// @see ScheduleTreeOptimizer::createMicroKernel
/// @see ScheduleTreeOptimizer::createMacroKernel
///
/// Subsequently, the described access relation is applied to the range of
/// @p MapOldIndVar, that is used to map original induction variables to
/// the ones, which are produced by schedule transformations. It helps to
/// define relations using a new space and, at the same time, keep them
/// in the original one.
///
/// @param MapOldIndVar The relation, which maps original induction variables
///                     to the ones, which are produced by schedule
///                     transformations.
/// @param FirstDim, SecondDim The input dimensions that are used to define
///        the specified access relation.
/// @return The specified access relation.
__isl_give isl_map *getMatMulAccRel(__isl_take isl_map *MapOldIndVar,
                                    unsigned FirstDim, unsigned SecondDim) {
  auto *Ctx = isl_map_get_ctx(MapOldIndVar);
  auto *AccessRelSpace = isl_space_alloc(Ctx, 0, 9, 3);
  auto *AccessRel = isl_map_universe(AccessRelSpace);
  AccessRel = isl_map_equate(AccessRel, isl_dim_in, FirstDim, isl_dim_out, 0);
  AccessRel = isl_map_equate(AccessRel, isl_dim_in, 5, isl_dim_out, 1);
  AccessRel = isl_map_equate(AccessRel, isl_dim_in, SecondDim, isl_dim_out, 2);
  return isl_map_apply_range(MapOldIndVar, AccessRel);
}

__isl_give isl_schedule_node *
createExtensionNode(__isl_take isl_schedule_node *Node,
                    __isl_take isl_map *ExtensionMap) {
  auto *Extension = isl_union_map_from_map(ExtensionMap);
  auto *NewNode = isl_schedule_node_from_extension(Extension);
  return isl_schedule_node_graft_before(Node, NewNode);
}

/// Apply the packing transformation.
///
/// The packing transformation can be described as a data-layout
/// transformation that requires to introduce a new array, copy data
/// to the array, and change memory access locations to reference the array.
/// It can be used to ensure that elements of the new array are read in-stride
/// access, aligned to cache lines boundaries, and preloaded into certain cache
/// levels.
///
/// As an example let us consider the packing of the array A that would help
/// to read its elements with in-stride access. An access to the array A
/// is represented by an access relation that has the form
/// S[i, j, k] -> A[i, k]. The scheduling function of the SCoP statement S has
/// the form S[i,j, k] -> [floor((j mod Nc) / Nr), floor((i mod Mc) / Mr),
/// k mod Kc, j mod Nr, i mod Mr].
///
/// To ensure that elements of the array A are read in-stride access, we add
/// a new array Packed_A[Mc/Mr][Kc][Mr] to the SCoP, using
/// Scop::createScopArrayInfo, change the access relation
/// S[i, j, k] -> A[i, k] to
/// S[i, j, k] -> Packed_A[floor((i mod Mc) / Mr), k mod Kc, i mod Mr], using
/// MemoryAccess::setNewAccessRelation, and copy the data to the array, using
/// the copy statement created by Scop::addScopStmt.
///
/// @param Node The schedule node to be optimized.
/// @param MapOldIndVar The relation, which maps original induction variables
///                     to the ones, which are produced by schedule
///                     transformations.
/// @param MicroParams, MacroParams Parameters of the BLIS kernel
///                                 to be taken into account.
/// @param MMI Parameters of the matrix multiplication operands.
/// @return The optimized schedule node.
static __isl_give isl_schedule_node *optimizeDataLayoutMatrMulPattern(
    __isl_take isl_schedule_node *Node, __isl_take isl_map *MapOldIndVar,
    MicroKernelParamsTy MicroParams, MacroKernelParamsTy MacroParams,
    MatMulInfoTy &MMI) {
  auto InputDimsId = isl_map_get_tuple_id(MapOldIndVar, isl_dim_in);
  auto *Stmt = static_cast<ScopStmt *>(isl_id_get_user(InputDimsId));
  isl_id_free(InputDimsId);

  // Create a copy statement that corresponds to the memory access to the
  // matrix B, the second operand of the matrix multiplication.
  Node = isl_schedule_node_parent(isl_schedule_node_parent(Node));
  Node = isl_schedule_node_parent(isl_schedule_node_parent(Node));
  Node = isl_schedule_node_parent(Node);
  Node = isl_schedule_node_child(isl_schedule_node_band_split(Node, 2), 0);
  auto *AccRel = getMatMulAccRel(isl_map_copy(MapOldIndVar), 3, 7);
  unsigned FirstDimSize = MacroParams.Nc / MicroParams.Nr;
  unsigned SecondDimSize = MacroParams.Kc;
  unsigned ThirdDimSize = MicroParams.Nr;
  auto *SAI = Stmt->getParent()->createScopArrayInfo(
      MMI.B->getElementType(), "Packed_B",
      {FirstDimSize, SecondDimSize, ThirdDimSize});
  AccRel = isl_map_set_tuple_id(AccRel, isl_dim_out, SAI->getBasePtrId());
  auto *OldAcc = MMI.B->getAccessRelation();
  MMI.B->setNewAccessRelation(AccRel);
  auto *ExtMap =
      isl_map_project_out(isl_map_copy(MapOldIndVar), isl_dim_out, 2,
                          isl_map_dim(MapOldIndVar, isl_dim_out) - 2);
  ExtMap = isl_map_reverse(ExtMap);
  ExtMap = isl_map_fix_si(ExtMap, isl_dim_out, MMI.i, 0);
  auto *Domain = Stmt->getDomain();

  // Restrict the domains of the copy statements to only execute when also its
  // originating statement is executed.
  auto *DomainId = isl_set_get_tuple_id(Domain);
  auto *NewStmt = Stmt->getParent()->addScopStmt(
      OldAcc, MMI.B->getAccessRelation(), isl_set_copy(Domain));
  ExtMap = isl_map_set_tuple_id(ExtMap, isl_dim_out, isl_id_copy(DomainId));
  ExtMap = isl_map_intersect_range(ExtMap, isl_set_copy(Domain));
  ExtMap = isl_map_set_tuple_id(ExtMap, isl_dim_out, NewStmt->getDomainId());
  Node = createExtensionNode(Node, ExtMap);

  // Create a copy statement that corresponds to the memory access
  // to the matrix A, the first operand of the matrix multiplication.
  Node = isl_schedule_node_child(Node, 0);
  AccRel = getMatMulAccRel(isl_map_copy(MapOldIndVar), 4, 6);
  FirstDimSize = MacroParams.Mc / MicroParams.Mr;
  ThirdDimSize = MicroParams.Mr;
  SAI = Stmt->getParent()->createScopArrayInfo(
      MMI.A->getElementType(), "Packed_A",
      {FirstDimSize, SecondDimSize, ThirdDimSize});
  AccRel = isl_map_set_tuple_id(AccRel, isl_dim_out, SAI->getBasePtrId());
  OldAcc = MMI.A->getAccessRelation();
  MMI.A->setNewAccessRelation(AccRel);
  ExtMap = isl_map_project_out(MapOldIndVar, isl_dim_out, 3,
                               isl_map_dim(MapOldIndVar, isl_dim_out) - 3);
  ExtMap = isl_map_reverse(ExtMap);
  ExtMap = isl_map_fix_si(ExtMap, isl_dim_out, MMI.j, 0);
  NewStmt = Stmt->getParent()->addScopStmt(OldAcc, MMI.A->getAccessRelation(),
                                           isl_set_copy(Domain));

  // Restrict the domains of the copy statements to only execute when also its
  // originating statement is executed.
  ExtMap = isl_map_set_tuple_id(ExtMap, isl_dim_out, DomainId);
  ExtMap = isl_map_intersect_range(ExtMap, Domain);
  ExtMap = isl_map_set_tuple_id(ExtMap, isl_dim_out, NewStmt->getDomainId());
  Node = createExtensionNode(Node, ExtMap);
  Node = isl_schedule_node_child(isl_schedule_node_child(Node, 0), 0);
  return isl_schedule_node_child(isl_schedule_node_child(Node, 0), 0);
}

/// Get a relation mapping induction variables produced by schedule
/// transformations to the original ones.
///
/// @param Node The schedule node produced as the result of creation
///        of the BLIS kernels.
/// @param MicroKernelParams, MacroKernelParams Parameters of the BLIS kernel
///                                             to be taken into account.
/// @return  The relation mapping original induction variables to the ones
///          produced by schedule transformation.
/// @see ScheduleTreeOptimizer::createMicroKernel
/// @see ScheduleTreeOptimizer::createMacroKernel
/// @see getMacroKernelParams
__isl_give isl_map *
getInductionVariablesSubstitution(__isl_take isl_schedule_node *Node,
                                  MicroKernelParamsTy MicroKernelParams,
                                  MacroKernelParamsTy MacroKernelParams) {
  auto *Child = isl_schedule_node_get_child(Node, 0);
  auto *UnMapOldIndVar = isl_schedule_node_get_prefix_schedule_union_map(Child);
  isl_schedule_node_free(Child);
  auto *MapOldIndVar = isl_map_from_union_map(UnMapOldIndVar);
  if (isl_map_dim(MapOldIndVar, isl_dim_out) > 9)
    MapOldIndVar =
        isl_map_project_out(MapOldIndVar, isl_dim_out, 0,
                            isl_map_dim(MapOldIndVar, isl_dim_out) - 9);
  return MapOldIndVar;
}

/// Isolate a set of partial tile prefixes and unroll the isolated part.
///
/// The set should ensure that it contains only partial tile prefixes that have
/// exactly Mr x Nr iterations of the two innermost loops produced by
/// the optimization of the matrix multiplication. Mr and Nr are parameters of
/// the micro-kernel.
///
/// In case of parametric bounds, this helps to auto-vectorize the unrolled
/// innermost loops, using the SLP vectorizer.
///
/// @param Node              The schedule node to be modified.
/// @param MicroKernelParams Parameters of the micro-kernel
///                          to be taken into account.
/// @return The modified isl_schedule_node.
static __isl_give isl_schedule_node *
isolateAndUnrollMatMulInnerLoops(__isl_take isl_schedule_node *Node,
                                 struct MicroKernelParamsTy MicroKernelParams) {
  auto *Child = isl_schedule_node_get_child(Node, 0);
  auto *UnMapOldIndVar = isl_schedule_node_get_prefix_schedule_relation(Child);
  isl_schedule_node_free(Child);
  auto *Prefix = isl_map_range(isl_map_from_union_map(UnMapOldIndVar));
  auto Dims = isl_set_dim(Prefix, isl_dim_set);
  Prefix = isl_set_project_out(Prefix, isl_dim_set, Dims - 1, 1);
  Prefix = getPartialTilePrefixes(Prefix, MicroKernelParams.Nr);
  Prefix = getPartialTilePrefixes(Prefix, MicroKernelParams.Mr);
  auto *IsolateOption = getIsolateOptions(
      isl_set_add_dims(isl_set_copy(Prefix), isl_dim_set, 3), 3);
  auto *Ctx = isl_schedule_node_get_ctx(Node);
  auto *AtomicOption = getAtomicOptions(Ctx);
  auto *Options =
      isl_union_set_union(IsolateOption, isl_union_set_copy(AtomicOption));
  Options = isl_union_set_union(Options, getUnrollIsolatedSetOptions(Ctx));
  Node = isl_schedule_node_band_set_ast_build_options(Node, Options);
  Node = isl_schedule_node_parent(isl_schedule_node_parent(Node));
  IsolateOption = getIsolateOptions(Prefix, 3);
  Options = isl_union_set_union(IsolateOption, AtomicOption);
  Node = isl_schedule_node_band_set_ast_build_options(Node, Options);
  Node = isl_schedule_node_child(isl_schedule_node_child(Node, 0), 0);
  return Node;
}

/// Mark @p BasePtr with "Inter iteration alias-free" mark node.
///
/// @param Node The child of the mark node to be inserted.
/// @param BasePtr The pointer to be marked.
/// @return The modified isl_schedule_node.
static isl_schedule_node *markInterIterationAliasFree(isl_schedule_node *Node,
                                                      llvm::Value *BasePtr) {
  if (!BasePtr)
    return Node;

  auto *Ctx = isl_schedule_node_get_ctx(Node);
  auto *Id = isl_id_alloc(Ctx, "Inter iteration alias-free", BasePtr);
  return isl_schedule_node_child(isl_schedule_node_insert_mark(Node, Id), 0);
}

__isl_give isl_schedule_node *ScheduleTreeOptimizer::optimizeMatMulPattern(
    __isl_take isl_schedule_node *Node, const llvm::TargetTransformInfo *TTI,
    MatMulInfoTy &MMI) {
  assert(TTI && "The target transform info should be provided.");
  Node = markInterIterationAliasFree(Node, MMI.WriteToC->getLatestBaseAddr());
  int DimOutNum = isl_schedule_node_band_n_member(Node);
  assert(DimOutNum > 2 && "In case of the matrix multiplication the loop nest "
                          "and, consequently, the corresponding scheduling "
                          "functions have at least three dimensions.");
  Node = permuteBandNodeDimensions(Node, MMI.i, DimOutNum - 3);
  int NewJ = MMI.j == DimOutNum - 3 ? MMI.i : MMI.j;
  int NewK = MMI.k == DimOutNum - 3 ? MMI.i : MMI.k;
  Node = permuteBandNodeDimensions(Node, NewJ, DimOutNum - 2);
  NewK = MMI.k == DimOutNum - 2 ? MMI.j : MMI.k;
  Node = permuteBandNodeDimensions(Node, NewK, DimOutNum - 1);
  auto MicroKernelParams = getMicroKernelParams(TTI, MMI);
  auto MacroKernelParams = getMacroKernelParams(MicroKernelParams, MMI);
  Node = createMacroKernel(Node, MacroKernelParams);
  Node = createMicroKernel(Node, MicroKernelParams);
  if (MacroKernelParams.Mc == 1 || MacroKernelParams.Nc == 1 ||
      MacroKernelParams.Kc == 1)
    return Node;
  auto *MapOldIndVar = getInductionVariablesSubstitution(
      Node, MicroKernelParams, MacroKernelParams);
  if (!MapOldIndVar)
    return Node;
  Node = isolateAndUnrollMatMulInnerLoops(Node, MicroKernelParams);
  return optimizeDataLayoutMatrMulPattern(Node, MapOldIndVar, MicroKernelParams,
                                          MacroKernelParams, MMI);
}

bool ScheduleTreeOptimizer::isMatrMultPattern(
    __isl_keep isl_schedule_node *Node, const Dependences *D,
    MatMulInfoTy &MMI) {
  auto *PartialSchedule =
      isl_schedule_node_band_get_partial_schedule_union_map(Node);
  if (isl_schedule_node_band_n_member(Node) < 3 ||
      isl_union_map_n_map(PartialSchedule) != 1) {
    isl_union_map_free(PartialSchedule);
    return false;
  }
  auto *NewPartialSchedule = isl_map_from_union_map(PartialSchedule);
  if (containsMatrMult(NewPartialSchedule, D, MMI)) {
    isl_map_free(NewPartialSchedule);
    return true;
  }
  isl_map_free(NewPartialSchedule);
  return false;
}

__isl_give isl_schedule_node *
ScheduleTreeOptimizer::optimizeBand(__isl_take isl_schedule_node *Node,
                                    void *User) {
  if (!isTileableBandNode(Node))
    return Node;

  const OptimizerAdditionalInfoTy *OAI =
      static_cast<const OptimizerAdditionalInfoTy *>(User);

  MatMulInfoTy MMI;
  if (PMBasedOpts && User && isMatrMultPattern(Node, OAI->D, MMI)) {
    DEBUG(dbgs() << "The matrix multiplication pattern was detected\n");
    return optimizeMatMulPattern(Node, OAI->TTI, MMI);
  }

  return standardBandOpts(Node, User);
}

__isl_give isl_schedule *
ScheduleTreeOptimizer::optimizeSchedule(__isl_take isl_schedule *Schedule,
                                        const OptimizerAdditionalInfoTy *OAI) {
  isl_schedule_node *Root = isl_schedule_get_root(Schedule);
  Root = optimizeScheduleNode(Root, OAI);
  isl_schedule_free(Schedule);
  auto S = isl_schedule_node_get_schedule(Root);
  isl_schedule_node_free(Root);
  return S;
}

__isl_give isl_schedule_node *ScheduleTreeOptimizer::optimizeScheduleNode(
    __isl_take isl_schedule_node *Node, const OptimizerAdditionalInfoTy *OAI) {
  Node = isl_schedule_node_map_descendant_bottom_up(
      Node, optimizeBand, const_cast<void *>(static_cast<const void *>(OAI)));
  return Node;
}

bool ScheduleTreeOptimizer::isProfitableSchedule(
    Scop &S, __isl_keep isl_schedule *NewSchedule) {
  // To understand if the schedule has been optimized we check if the schedule
  // has changed at all.
  // TODO: We can improve this by tracking if any necessarily beneficial
  // transformations have been performed. This can e.g. be tiling, loop
  // interchange, or ...) We can track this either at the place where the
  // transformation has been performed or, in case of automatic ILP based
  // optimizations, by comparing (yet to be defined) performance metrics
  // before/after the scheduling optimizer
  // (e.g., #stride-one accesses)
  if (S.containsExtensionNode(NewSchedule))
    return true;
  auto *NewScheduleMap = isl_schedule_get_map(NewSchedule);
  isl_union_map *OldSchedule = S.getSchedule();
  assert(OldSchedule && "Only IslScheduleOptimizer can insert extension nodes "
                        "that make Scop::getSchedule() return nullptr.");
  bool changed = !isl_union_map_is_equal(OldSchedule, NewScheduleMap);
  isl_union_map_free(OldSchedule);
  isl_union_map_free(NewScheduleMap);
  return changed;
}

namespace {
class IslScheduleOptimizer : public ScopPass {
public:
  static char ID;
  explicit IslScheduleOptimizer() : ScopPass(ID) { LastSchedule = nullptr; }

  ~IslScheduleOptimizer() { isl_schedule_free(LastSchedule); }

  /// Optimize the schedule of the SCoP @p S.
  bool runOnScop(Scop &S) override;

  /// Print the new schedule for the SCoP @p S.
  void printScop(raw_ostream &OS, Scop &S) const override;

  /// Register all analyses and transformation required.
  void getAnalysisUsage(AnalysisUsage &AU) const override;

  /// Release the internal memory.
  void releaseMemory() override {
    isl_schedule_free(LastSchedule);
    LastSchedule = nullptr;
  }

private:
  isl_schedule *LastSchedule;
};
} // namespace

char IslScheduleOptimizer::ID = 0;

bool IslScheduleOptimizer::runOnScop(Scop &S) {

  // Skip empty SCoPs but still allow code generation as it will delete the
  // loops present but not needed.
  if (S.getSize() == 0) {
    S.markAsOptimized();
    return false;
  }

  const Dependences &D =
      getAnalysis<DependenceInfo>().getDependences(Dependences::AL_Statement);

  if (!D.hasValidDependences())
    return false;

  isl_schedule_free(LastSchedule);
  LastSchedule = nullptr;

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

  isl_union_set *Domain = S.getDomains();

  if (!Domain)
    return false;

  isl_union_map *Validity = D.getDependences(ValidityKinds);
  isl_union_map *Proximity = D.getDependences(ProximityKinds);

  // Simplify the dependences by removing the constraints introduced by the
  // domains. This can speed up the scheduling time significantly, as large
  // constant coefficients will be removed from the dependences. The
  // introduction of some additional dependences reduces the possible
  // transformations, but in most cases, such transformation do not seem to be
  // interesting anyway. In some cases this option may stop the scheduler to
  // find any schedule.
  if (SimplifyDeps == "yes") {
    Validity = isl_union_map_gist_domain(Validity, isl_union_set_copy(Domain));
    Validity = isl_union_map_gist_range(Validity, isl_union_set_copy(Domain));
    Proximity =
        isl_union_map_gist_domain(Proximity, isl_union_set_copy(Domain));
    Proximity = isl_union_map_gist_range(Proximity, isl_union_set_copy(Domain));
  } else if (SimplifyDeps != "no") {
    errs() << "warning: Option -polly-opt-simplify-deps should either be 'yes' "
              "or 'no'. Falling back to default: 'yes'\n";
  }

  DEBUG(dbgs() << "\n\nCompute schedule from: ");
  DEBUG(dbgs() << "Domain := " << stringFromIslObj(Domain) << ";\n");
  DEBUG(dbgs() << "Proximity := " << stringFromIslObj(Proximity) << ";\n");
  DEBUG(dbgs() << "Validity := " << stringFromIslObj(Validity) << ";\n");

  unsigned IslSerializeSCCs;

  if (FusionStrategy == "max") {
    IslSerializeSCCs = 0;
  } else if (FusionStrategy == "min") {
    IslSerializeSCCs = 1;
  } else {
    errs() << "warning: Unknown fusion strategy. Falling back to maximal "
              "fusion.\n";
    IslSerializeSCCs = 0;
  }

  int IslMaximizeBands;

  if (MaximizeBandDepth == "yes") {
    IslMaximizeBands = 1;
  } else if (MaximizeBandDepth == "no") {
    IslMaximizeBands = 0;
  } else {
    errs() << "warning: Option -polly-opt-maximize-bands should either be 'yes'"
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

  isl_ctx *Ctx = S.getIslCtx();

  isl_options_set_schedule_outer_coincidence(Ctx, IslOuterCoincidence);
  isl_options_set_schedule_serialize_sccs(Ctx, IslSerializeSCCs);
  isl_options_set_schedule_maximize_band_depth(Ctx, IslMaximizeBands);
  isl_options_set_schedule_max_constant_term(Ctx, MaxConstantTerm);
  isl_options_set_schedule_max_coefficient(Ctx, MaxCoefficient);
  isl_options_set_tile_scale_tile_loops(Ctx, 0);

  auto OnErrorStatus = isl_options_get_on_error(Ctx);
  isl_options_set_on_error(Ctx, ISL_ON_ERROR_CONTINUE);

  isl_schedule_constraints *ScheduleConstraints;
  ScheduleConstraints = isl_schedule_constraints_on_domain(Domain);
  ScheduleConstraints =
      isl_schedule_constraints_set_proximity(ScheduleConstraints, Proximity);
  ScheduleConstraints = isl_schedule_constraints_set_validity(
      ScheduleConstraints, isl_union_map_copy(Validity));
  ScheduleConstraints =
      isl_schedule_constraints_set_coincidence(ScheduleConstraints, Validity);
  isl_schedule *Schedule;
  Schedule = isl_schedule_constraints_compute_schedule(ScheduleConstraints);
  isl_options_set_on_error(Ctx, OnErrorStatus);

  // In cases the scheduler is not able to optimize the code, we just do not
  // touch the schedule.
  if (!Schedule)
    return false;

  DEBUG({
    auto *P = isl_printer_to_str(Ctx);
    P = isl_printer_set_yaml_style(P, ISL_YAML_STYLE_BLOCK);
    P = isl_printer_print_schedule(P, Schedule);
    auto *str = isl_printer_get_str(P);
    dbgs() << "NewScheduleTree: \n" << str << "\n";
    free(str);
    isl_printer_free(P);
  });

  Function &F = S.getFunction();
  auto *TTI = &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
  const OptimizerAdditionalInfoTy OAI = {TTI, const_cast<Dependences *>(&D)};
  isl_schedule *NewSchedule =
      ScheduleTreeOptimizer::optimizeSchedule(Schedule, &OAI);

  if (!ScheduleTreeOptimizer::isProfitableSchedule(S, NewSchedule)) {
    isl_schedule_free(NewSchedule);
    return false;
  }

  S.setScheduleTree(NewSchedule);
  S.markAsOptimized();

  if (OptimizedScops)
    S.dump();

  return false;
}

void IslScheduleOptimizer::printScop(raw_ostream &OS, Scop &) const {
  isl_printer *p;
  char *ScheduleStr;

  OS << "Calculated schedule:\n";

  if (!LastSchedule) {
    OS << "n/a\n";
    return;
  }

  p = isl_printer_to_str(isl_schedule_get_ctx(LastSchedule));
  p = isl_printer_print_schedule(p, LastSchedule);
  ScheduleStr = isl_printer_get_str(p);
  isl_printer_free(p);

  OS << ScheduleStr << "\n";
}

void IslScheduleOptimizer::getAnalysisUsage(AnalysisUsage &AU) const {
  ScopPass::getAnalysisUsage(AU);
  AU.addRequired<DependenceInfo>();
  AU.addRequired<TargetTransformInfoWrapperPass>();
}

Pass *polly::createIslScheduleOptimizerPass() {
  return new IslScheduleOptimizer();
}

INITIALIZE_PASS_BEGIN(IslScheduleOptimizer, "polly-opt-isl",
                      "Polly - Optimize schedule of SCoP", false, false);
INITIALIZE_PASS_DEPENDENCY(DependenceInfo);
INITIALIZE_PASS_DEPENDENCY(ScopInfoRegionPass);
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass);
INITIALIZE_PASS_END(IslScheduleOptimizer, "polly-opt-isl",
                    "Polly - Optimize schedule of SCoP", false, false)
