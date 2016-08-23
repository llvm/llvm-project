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
//  - Prevectorization - The coice of a possible outer loop that is strip-mined
//                       to the innermost level to enable inner-loop
//                       vectorization.
//  - Some optimizations for spatial locality are also planned.
//
// For a detailed description of the schedule tree itself please see section 6
// of:
//
// Polyhedral AST generation is more than scanning polyhedra
// Tobias Grosser, Sven Verdoolaege, Albert Cohen
// ACM Transations on Programming Languages and Systems (TOPLAS),
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

static cl::opt<int> ThrougputVectorFma(
    "polly-target-througput-vector-fma",
    cl::desc("A throughput of the processor floating-point arithmetic units "
             "expressed in the number of vector fused multiply-add "
             "instructions per clock cycle."),
    cl::Hidden, cl::init(1), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::list<int>
    CacheLevelAssociativity("polly-target-cache-level-associativity",
                            cl::desc("The associativity of each cache level."),
                            cl::Hidden, cl::ZeroOrMore, cl::CommaSeparated,
                            cl::cat(PollyCategory));

static cl::list<int> CacheLevelSizes(
    "polly-target-cache-level-sizes",
    cl::desc("The size of each cache level specified in bytes."), cl::Hidden,
    cl::ZeroOrMore, cl::CommaSeparated, cl::cat(PollyCategory));

static cl::opt<int> FirstLevelDefaultTileSize(
    "polly-default-tile-size",
    cl::desc("The default tile size (if not enough were provided by"
             " --polly-tile-sizes)"),
    cl::Hidden, cl::init(32), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::list<int> FirstLevelTileSizes(
    "polly-tile-sizes", cl::desc("A tile size for each loop dimension, filled "
                                 "with --polly-default-tile-size"),
    cl::Hidden, cl::ZeroOrMore, cl::CommaSeparated, cl::cat(PollyCategory));

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

static cl::list<int>
    RegisterTileSizes("polly-register-tile-sizes",
                      cl::desc("A tile size for each loop dimension, filled "
                               "with --polly-register-tile-size"),
                      cl::Hidden, cl::ZeroOrMore, cl::CommaSeparated,
                      cl::cat(PollyCategory));

static cl::opt<bool>
    PMBasedOpts("polly-pattern-matching-based-opts",
                cl::desc("Perform optimizations based on pattern matching"),
                cl::init(false), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<bool> OptimizedScops(
    "polly-optimized-scops",
    cl::desc("Polly - Dump polyhedral description of Scops optimized with "
             "the isl scheduling optimizer and the set of post-scheduling "
             "transformations is applied on the schedule tree"),
    cl::init(false), cl::ZeroOrMore, cl::cat(PollyCategory));

/// @brief Create an isl_union_set, which describes the isolate option based
///        on IsoalteDomain.
///
/// @param IsolateDomain An isl_set whose last dimension is the only one that
///                      should belong to the current band node.
static __isl_give isl_union_set *
getIsolateOptions(__isl_take isl_set *IsolateDomain) {
  auto Dims = isl_set_dim(IsolateDomain, isl_dim_set);
  auto *IsolateRelation = isl_map_from_domain(IsolateDomain);
  IsolateRelation = isl_map_move_dims(IsolateRelation, isl_dim_out, 0,
                                      isl_dim_in, Dims - 1, 1);
  auto *IsolateOption = isl_map_wrap(IsolateRelation);
  auto *Id = isl_id_alloc(isl_set_get_ctx(IsolateOption), "isolate", nullptr);
  return isl_union_set_from_set(isl_set_set_tuple_id(IsolateOption, Id));
}

/// @brief Create an isl_union_set, which describes the atomic option for the
///        dimension of the current node.
///
/// It may help to reduce the size of generated code.
///
/// @param Ctx An isl_ctx, which is used to create the isl_union_set.
static __isl_give isl_union_set *getAtomicOptions(__isl_take isl_ctx *Ctx) {
  auto *Space = isl_space_set_alloc(Ctx, 0, 1);
  auto *AtomicOption = isl_set_universe(Space);
  auto *Id = isl_id_alloc(Ctx, "atomic", nullptr);
  return isl_union_set_from_set(isl_set_set_tuple_id(AtomicOption, Id));
}

/// @brief Make the last dimension of Set to take values
///        from 0 to VectorWidth - 1.
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

/// @brief Build the desired set of partial tile prefixes.
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
  auto *IsolateOption = getIsolateOptions(IsolateDomain);
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

  auto Child = isl_schedule_node_get_child(Node, 0);
  auto Type = isl_schedule_node_get_type(Child);
  isl_schedule_node_free(Child);

  if (Type != isl_schedule_node_leaf)
    return false;

  return true;
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

/// @brief Check whether output dimensions of the map rely on the specified
///        input dimension.
///
/// @param IslMap The isl map to be considered.
/// @param DimNum The number of an input dimension to be checked.
static bool isInputDimUsed(__isl_take isl_map *IslMap, unsigned DimNum) {
  auto *CheckedAccessRelation =
      isl_map_project_out(isl_map_copy(IslMap), isl_dim_in, DimNum, 1);
  CheckedAccessRelation =
      isl_map_insert_dims(CheckedAccessRelation, isl_dim_in, DimNum, 1);
  auto *InputDimsId = isl_map_get_tuple_id(IslMap, isl_dim_in);
  CheckedAccessRelation =
      isl_map_set_tuple_id(CheckedAccessRelation, isl_dim_in, InputDimsId);
  InputDimsId = isl_map_get_tuple_id(IslMap, isl_dim_out);
  CheckedAccessRelation =
      isl_map_set_tuple_id(CheckedAccessRelation, isl_dim_out, InputDimsId);
  auto res = !isl_map_is_equal(CheckedAccessRelation, IslMap);
  isl_map_free(CheckedAccessRelation);
  isl_map_free(IslMap);
  return res;
}

/// @brief Check if the SCoP statement could probably be optimized with
///        analytical modeling.
///
/// containsMatrMult tries to determine whether the following conditions
/// are true:
/// 1. all memory accesses of the statement will have stride 0 or 1,
///    if we interchange loops (switch the variable used in the inner
///    loop to the outer loop).
/// 2. all memory accesses of the statement except from the last one, are
///    read memory access and the last one is write memory access.
/// 3. all subscripts of the last memory access of the statement don't contain
///    the variable used in the inner loop.
///
/// @param PartialSchedule The PartialSchedule that contains a SCoP statement
///        to check.
static bool containsMatrMult(__isl_keep isl_map *PartialSchedule) {
  auto InputDimsId = isl_map_get_tuple_id(PartialSchedule, isl_dim_in);
  auto *ScpStmt = static_cast<ScopStmt *>(isl_id_get_user(InputDimsId));
  isl_id_free(InputDimsId);
  if (ScpStmt->size() <= 1)
    return false;
  auto MemA = ScpStmt->begin();
  for (unsigned i = 0; i < ScpStmt->size() - 2 && MemA != ScpStmt->end();
       i++, MemA++)
    if (!(*MemA)->isRead() ||
        ((*MemA)->isArrayKind() &&
         !((*MemA)->isStrideOne(isl_map_copy(PartialSchedule)) ||
           (*MemA)->isStrideZero(isl_map_copy(PartialSchedule)))))
      return false;
  MemA++;
  if (!(*MemA)->isWrite() || !(*MemA)->isArrayKind() ||
      !((*MemA)->isStrideOne(isl_map_copy(PartialSchedule)) ||
        (*MemA)->isStrideZero(isl_map_copy(PartialSchedule))))
    return false;
  auto DimNum = isl_map_dim(PartialSchedule, isl_dim_in);
  return !isInputDimUsed((*MemA)->getAccessRelation(), DimNum - 1);
}

/// @brief Circular shift of output dimensions of the integer map.
///
/// @param IslMap The isl map to be modified.
static __isl_give isl_map *circularShiftOutputDims(__isl_take isl_map *IslMap) {
  auto DimNum = isl_map_dim(IslMap, isl_dim_out);
  if (DimNum == 0)
    return IslMap;
  auto InputDimsId = isl_map_get_tuple_id(IslMap, isl_dim_in);
  IslMap = isl_map_move_dims(IslMap, isl_dim_in, 0, isl_dim_out, DimNum - 1, 1);
  IslMap = isl_map_move_dims(IslMap, isl_dim_out, 0, isl_dim_in, 0, 1);
  return isl_map_set_tuple_id(IslMap, isl_dim_in, InputDimsId);
}

/// @brief Permute two dimensions of the band node.
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
  return applyRegisterTiling(Node, {MicroKernelParams.Mr, MicroKernelParams.Nr},
                             1);
}

__isl_give isl_schedule_node *ScheduleTreeOptimizer::createMacroKernel(
    __isl_take isl_schedule_node *Node, MacroKernelParamsTy MacroKernelParams) {
  assert(isl_schedule_node_get_type(Node) == isl_schedule_node_band);
  if (MacroKernelParams.Mc == 1 && MacroKernelParams.Nc == 1 &&
      MacroKernelParams.Kc == 1)
    return Node;
  Node = tileNode(
      Node, "1st level tiling",
      {MacroKernelParams.Mc, MacroKernelParams.Nc, MacroKernelParams.Kc}, 1);
  Node = isl_schedule_node_parent(isl_schedule_node_parent(Node));
  Node = permuteBandNodeDimensions(Node, 1, 2);
  return isl_schedule_node_child(isl_schedule_node_child(Node, 0), 0);
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
/// @return The structure of type MicroKernelParamsTy.
/// @see MicroKernelParamsTy
static struct MicroKernelParamsTy
getMicroKernelParams(const llvm::TargetTransformInfo *TTI) {
  assert(TTI && "The target transform info should be provided.");

  // Nvec - Number of double-precision floating-point numbers that can be hold
  // by a vector register. Use 2 by default.
  auto Nvec = TTI->getRegisterBitWidth(true) / 64;
  if (Nvec == 0)
    Nvec = 2;
  int Nr =
      ceil(sqrt(Nvec * LatencyVectorFma * ThrougputVectorFma) / Nvec) * Nvec;
  int Mr = ceil(Nvec * LatencyVectorFma * ThrougputVectorFma / Nr);
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
/// @return The structure of type MacroKernelParamsTy.
/// @see MacroKernelParamsTy
/// @see MicroKernelParamsTy
static struct MacroKernelParamsTy
getMacroKernelParams(const MicroKernelParamsTy &MicroKernelParams) {
  // According to www.cs.utexas.edu/users/flame/pubs/TOMS-BLIS-Analytical.pdf,
  // it requires information about the first two levels of a cache to determine
  // all the parameters of a macro-kernel. It also checks that an associativity
  // degree of a cache level is greater than two. Otherwise, another algorithm
  // for determination of the parameters should be used.
  if (!(MicroKernelParams.Mr > 0 && MicroKernelParams.Nr > 0 &&
        CacheLevelSizes.size() >= 2 && CacheLevelAssociativity.size() >= 2 &&
        CacheLevelSizes[0] > 0 && CacheLevelSizes[1] > 0 &&
        CacheLevelAssociativity[0] > 2 && CacheLevelAssociativity[1] > 2))
    return {1, 1, 1};
  int Cbr = floor(
      (CacheLevelAssociativity[0] - 1) /
      (1 + static_cast<double>(MicroKernelParams.Mr) / MicroKernelParams.Nr));
  int Kc = (Cbr * CacheLevelSizes[0]) /
           (MicroKernelParams.Nr * CacheLevelAssociativity[0] * 8);
  double Cac = static_cast<double>(MicroKernelParams.Mr * Kc * 8 *
                                   CacheLevelAssociativity[1]) /
               CacheLevelSizes[1];
  double Cbc = static_cast<double>(MicroKernelParams.Nr * Kc * 8 *
                                   CacheLevelAssociativity[1]) /
               CacheLevelSizes[1];
  int Mc = floor(MicroKernelParams.Mr / Cac);
  int Nc =
      floor((MicroKernelParams.Nr * (CacheLevelAssociativity[1] - 2)) / Cbc);
  return {Mc, Nc, Kc};
}

/// @brief Identify a memory access through the shape of its memory access
///        relation.
///
/// Identify the unique memory access in @p Stmt, that has an access relation
/// equal to @p ExpectedAccessRelation.
///
/// @param Stmt The SCoP statement that contains the memory accesses under
///             consideration.
/// @param ExpectedAccessRelation The access relation that identifies
///                               the memory access.
/// @return  The memory access of @p Stmt whose memory access relation is equal
///          to @p ExpectedAccessRelation. nullptr in case there is no or more
///          than one such access.
MemoryAccess *
identifyAccessByAccessRelation(ScopStmt *Stmt,
                               __isl_take isl_map *ExpectedAccessRelation) {
  if (isl_map_has_tuple_id(ExpectedAccessRelation, isl_dim_out))
    ExpectedAccessRelation =
        isl_map_reset_tuple_id(ExpectedAccessRelation, isl_dim_out);
  MemoryAccess *IdentifiedAccess = nullptr;
  for (auto *Access : *Stmt) {
    auto *AccessRelation = Access->getAccessRelation();
    AccessRelation = isl_map_reset_tuple_id(AccessRelation, isl_dim_out);
    if (isl_map_is_equal(ExpectedAccessRelation, AccessRelation)) {
      if (IdentifiedAccess) {
        isl_map_free(AccessRelation);
        isl_map_free(ExpectedAccessRelation);
        return nullptr;
      }
      IdentifiedAccess = Access;
    }
    isl_map_free(AccessRelation);
  }
  isl_map_free(ExpectedAccessRelation);
  return IdentifiedAccess;
}

/// @brief Create an access relation that is specific to the matrix
///        multiplication pattern.
///
/// Create an access relation of the following form:
/// Stmt[O0, O1, O2]->[OI, OJ],
/// where I is @p I, J is @J
///
/// @param Stmt The SCoP statement for which to generate the access relation.
/// @param I The index of the input dimension that is mapped to the first output
///          dimension.
/// @param J The index of the input dimension that is mapped to the second
///          output dimension.
/// @return The specified access relation.
__isl_give isl_map *
getMatMulPatternOriginalAccessRelation(ScopStmt *Stmt, unsigned I, unsigned J) {
  auto *AccessRelSpace = isl_space_alloc(Stmt->getIslCtx(), 0, 3, 2);
  auto *AccessRel = isl_map_universe(AccessRelSpace);
  AccessRel = isl_map_equate(AccessRel, isl_dim_in, I, isl_dim_out, 0);
  AccessRel = isl_map_equate(AccessRel, isl_dim_in, J, isl_dim_out, 1);
  AccessRel = isl_map_set_tuple_id(AccessRel, isl_dim_in, Stmt->getDomainId());
  return AccessRel;
}

/// @brief Identify the memory access that corresponds to the access
///        to the second operand of the matrix multiplication.
///
/// Identify the memory access that corresponds to the access
/// to the matrix B of the matrix multiplication C = A x B.
///
/// @param Stmt The SCoP statement that contains the memory accesses
///             under consideration.
/// @return The memory access of @p Stmt that corresponds to the access
///         to the second operand of the matrix multiplication.
MemoryAccess *identifyAccessA(ScopStmt *Stmt) {
  auto *OriginalRel = getMatMulPatternOriginalAccessRelation(Stmt, 0, 2);
  return identifyAccessByAccessRelation(Stmt, OriginalRel);
}

/// @brief Identify the memory access that corresponds to the access
///        to the first operand of the matrix multiplication.
///
/// Identify the memory access that corresponds to the access
/// to the matrix A of the matrix multiplication C = A x B.
///
/// @param Stmt The SCoP statement that contains the memory accesses
///             under consideration.
/// @return The memory access of @p Stmt that corresponds to the access
///         to the first operand of the matrix multiplication.
MemoryAccess *identifyAccessB(ScopStmt *Stmt) {
  auto *OriginalRel = getMatMulPatternOriginalAccessRelation(Stmt, 2, 1);
  return identifyAccessByAccessRelation(Stmt, OriginalRel);
}

/// @brief Create an access relation that is specific to
///        the matrix multiplication pattern.
///
/// Create an access relation of the following form:
/// [O0, O1, O2, O3, O4, O5, O6, O7, O8] -> [0, O5 + K * OI, OJ],
/// where K is @p Coeff, I is @p FirstDim, J is @p SecondDim.
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
/// @param Coeff The coefficient that is used to define the specified access
///              relation.
/// @param FirstDim, SecondDim The input dimensions that are used to define
///        the specified access relation.
/// @return The specified access relation.
__isl_give isl_map *getMatMulAccRel(__isl_take isl_map *MapOldIndVar,
                                    unsigned Coeff, unsigned FirstDim,
                                    unsigned SecondDim) {
  auto *Ctx = isl_map_get_ctx(MapOldIndVar);
  auto *AccessRelSpace = isl_space_alloc(Ctx, 0, 9, 3);
  auto *AccessRel = isl_map_universe(isl_space_copy(AccessRelSpace));
  auto *ConstrSpace = isl_local_space_from_space(AccessRelSpace);
  auto *Constr = isl_constraint_alloc_equality(ConstrSpace);
  Constr = isl_constraint_set_coefficient_si(Constr, isl_dim_out, 1, -1);
  Constr = isl_constraint_set_coefficient_si(Constr, isl_dim_in, 5, 1);
  Constr =
      isl_constraint_set_coefficient_si(Constr, isl_dim_in, FirstDim, Coeff);
  AccessRel = isl_map_add_constraint(AccessRel, Constr);
  AccessRel = isl_map_fix_si(AccessRel, isl_dim_out, 0, 0);
  AccessRel = isl_map_equate(AccessRel, isl_dim_in, SecondDim, isl_dim_out, 2);
  return isl_map_apply_range(MapOldIndVar, AccessRel);
}

/// @brief Apply the packing transformation.
///
/// The packing transformation can be described as a data-layout
/// transformation that requires to introduce a new array, copy data
/// to the array, and change memory access locations of the compute kernel
/// to reference the array.
///
/// @param Node The schedule node to be optimized.
/// @param MapOldIndVar The relation, which maps original induction variables
///                     to the ones, which are produced by schedule
///                     transformations.
/// @param MicroParams, MacroParams Parameters of the BLIS kernel
///                                 to be taken into account.
/// @return The optimized schedule node.
static void optimizeDataLayoutMatrMulPattern(__isl_take isl_map *MapOldIndVar,
                                             MicroKernelParamsTy MicroParams,
                                             MacroKernelParamsTy MacroParams) {
  auto InputDimsId = isl_map_get_tuple_id(MapOldIndVar, isl_dim_in);
  auto *Stmt = static_cast<ScopStmt *>(isl_id_get_user(InputDimsId));
  isl_id_free(InputDimsId);
  MemoryAccess *MemAccessA = identifyAccessA(Stmt);
  MemoryAccess *MemAccessB = identifyAccessB(Stmt);
  if (!MemAccessA || !MemAccessB) {
    isl_map_free(MapOldIndVar);
    return;
  }
  auto *AccRel =
      getMatMulAccRel(isl_map_copy(MapOldIndVar), MacroParams.Kc, 3, 6);
  unsigned FirstDimSize = MacroParams.Mc * MacroParams.Kc / MicroParams.Mr;
  unsigned SecondDimSize = MicroParams.Mr;
  auto *SAI = Stmt->getParent()->createScopArrayInfo(
      MemAccessA->getElementType(), "Packed_A", {FirstDimSize, SecondDimSize});
  AccRel = isl_map_set_tuple_id(AccRel, isl_dim_out, SAI->getBasePtrId());
  MemAccessA->setNewAccessRelation(AccRel);
  AccRel = getMatMulAccRel(MapOldIndVar, MacroParams.Kc, 4, 7);
  FirstDimSize = MacroParams.Nc * MacroParams.Kc / MicroParams.Nr;
  SecondDimSize = MicroParams.Nr;
  SAI = Stmt->getParent()->createScopArrayInfo(
      MemAccessB->getElementType(), "Packed_B", {FirstDimSize, SecondDimSize});
  AccRel = isl_map_set_tuple_id(AccRel, isl_dim_out, SAI->getBasePtrId());
  MemAccessB->setNewAccessRelation(AccRel);
}

/// @brief Get a relation mapping induction variables produced by schedule
///        transformations to the original ones.
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

__isl_give isl_schedule_node *ScheduleTreeOptimizer::optimizeMatMulPattern(
    __isl_take isl_schedule_node *Node, const llvm::TargetTransformInfo *TTI) {
  assert(TTI && "The target transform info should be provided.");
  auto MicroKernelParams = getMicroKernelParams(TTI);
  auto MacroKernelParams = getMacroKernelParams(MicroKernelParams);
  Node = createMacroKernel(Node, MacroKernelParams);
  Node = createMicroKernel(Node, MicroKernelParams);
  if (MacroKernelParams.Mc == 1 || MacroKernelParams.Nc == 1 ||
      MacroKernelParams.Kc == 1)
    return Node;
  auto *MapOldIndVar = getInductionVariablesSubstitution(
      Node, MicroKernelParams, MacroKernelParams);
  if (!MapOldIndVar)
    return Node;
  optimizeDataLayoutMatrMulPattern(MapOldIndVar, MicroKernelParams,
                                   MacroKernelParams);
  return Node;
}

bool ScheduleTreeOptimizer::isMatrMultPattern(
    __isl_keep isl_schedule_node *Node) {
  auto *PartialSchedule =
      isl_schedule_node_band_get_partial_schedule_union_map(Node);
  if (isl_schedule_node_band_n_member(Node) != 3 ||
      isl_union_map_n_map(PartialSchedule) != 1) {
    isl_union_map_free(PartialSchedule);
    return false;
  }
  auto *NewPartialSchedule = isl_map_from_union_map(PartialSchedule);
  NewPartialSchedule = circularShiftOutputDims(NewPartialSchedule);
  if (containsMatrMult(NewPartialSchedule)) {
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

  if (PMBasedOpts && User && isMatrMultPattern(Node)) {
    DEBUG(dbgs() << "The matrix multiplication pattern was detected\n");
    const llvm::TargetTransformInfo *TTI;
    TTI = static_cast<const llvm::TargetTransformInfo *>(User);
    Node = optimizeMatMulPattern(Node, TTI);
  }

  return standardBandOpts(Node, User);
}

__isl_give isl_schedule *
ScheduleTreeOptimizer::optimizeSchedule(__isl_take isl_schedule *Schedule,
                                        const llvm::TargetTransformInfo *TTI) {
  isl_schedule_node *Root = isl_schedule_get_root(Schedule);
  Root = optimizeScheduleNode(Root, TTI);
  isl_schedule_free(Schedule);
  auto S = isl_schedule_node_get_schedule(Root);
  isl_schedule_node_free(Root);
  return S;
}

__isl_give isl_schedule_node *ScheduleTreeOptimizer::optimizeScheduleNode(
    __isl_take isl_schedule_node *Node, const llvm::TargetTransformInfo *TTI) {
  Node = isl_schedule_node_map_descendant_bottom_up(
      Node, optimizeBand, const_cast<void *>(static_cast<const void *>(TTI)));
  return Node;
}

bool ScheduleTreeOptimizer::isProfitableSchedule(
    Scop &S, __isl_keep isl_union_map *NewSchedule) {
  // To understand if the schedule has been optimized we check if the schedule
  // has changed at all.
  // TODO: We can improve this by tracking if any necessarily beneficial
  // transformations have been performed. This can e.g. be tiling, loop
  // interchange, or ...) We can track this either at the place where the
  // transformation has been performed or, in case of automatic ILP based
  // optimizations, by comparing (yet to be defined) performance metrics
  // before/after the scheduling optimizer
  // (e.g., #stride-one accesses)
  isl_union_map *OldSchedule = S.getSchedule();
  bool changed = !isl_union_map_is_equal(OldSchedule, NewSchedule);
  isl_union_map_free(OldSchedule);
  return changed;
}

namespace {
class IslScheduleOptimizer : public ScopPass {
public:
  static char ID;
  explicit IslScheduleOptimizer() : ScopPass(ID) { LastSchedule = nullptr; }

  ~IslScheduleOptimizer() { isl_schedule_free(LastSchedule); }

  /// @brief Optimize the schedule of the SCoP @p S.
  bool runOnScop(Scop &S) override;

  /// @brief Print the new schedule for the SCoP @p S.
  void printScop(raw_ostream &OS, Scop &S) const override;

  /// @brief Register all analyses and transformation required.
  void getAnalysisUsage(AnalysisUsage &AU) const override;

  /// @brief Release the internal memory.
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
    dbgs() << "NewScheduleTree: \n" << isl_printer_get_str(P) << "\n";
    isl_printer_free(P);
  });

  Function &F = S.getFunction();
  auto *TTI = &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
  isl_schedule *NewSchedule =
      ScheduleTreeOptimizer::optimizeSchedule(Schedule, TTI);
  isl_union_map *NewScheduleMap = isl_schedule_get_map(NewSchedule);

  if (!ScheduleTreeOptimizer::isProfitableSchedule(S, NewScheduleMap)) {
    isl_union_map_free(NewScheduleMap);
    isl_schedule_free(NewSchedule);
    return false;
  }

  S.setScheduleTree(NewSchedule);
  S.markAsOptimized();

  if (OptimizedScops)
    S.dump();

  isl_union_map_free(NewScheduleMap);
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
