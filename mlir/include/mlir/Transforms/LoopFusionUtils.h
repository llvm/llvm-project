//===- LoopFusionUtils.h - Loop fusion utilities ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for various loop fusion utility
// methods: these are not passes by themselves but are used either by passes,
// optimization sequences, or in turn by other transformation utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_LOOP_FUSION_UTILS_H
#define MLIR_TRANSFORMS_LOOP_FUSION_UTILS_H

#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class AffineForOp;
struct ComputationSliceState;
class Operation;

// TODO: Extend this module to include utility functions for querying fusion
// cost/storage reduction, and for performing the loop fusion transformation.

struct FusionResult {
  enum ResultEnum {
    Success,
    FailPrecondition,     // Failed precondition for fusion. (e.g. same block).
    FailBlockDependence,  // Fusion would violate another dependence in block.
    FailFusionDependence, // Fusion would reverse dependences between loops.
    FailComputationSlice, // Unable to compute src loop computation slice.
  } value;
  FusionResult(ResultEnum v) : value(v) {}
};

/// Describes the fusion strategy to be used in the Affine loop fusion
/// utilities. Currently, it is used to specialized the loop fusion utilities
/// with the assumptions made in the AffineLoopFusion pass for producer-consumer
/// and sibling fusion, while sharing a single implementation. The latter
/// strategies are also limited to scenarios where a single memref is involved
/// in the producer-consume or sibling relationship between the candidate
/// loops. We use 'memref' to keep track of such a memref.
// TODO: Remove 'memref' when we support more generic scenarios.
// TODO: Generalize utilities so that producer-consumer and sibling fusion
// strategies can be used without the assumptions made in the AffineLoopFusion
// pass.
struct FusionStrategy {
  enum StrategyEnum {
    // Generic loop fusion: Arbitrary loops are considered for fusion. No
    // assumptions about a specific fusion strategy from AffineLoopFusion pass
    // are made.
    // TODO: Generic fusion is not fully implemented by fusion utilities yet.
    // It should only be used for testing.
    Generic,
    // Producer-consumer fusion: Only loops with a producer-consumer
    // memref dependence are considered for fusion. Currently, assumptions from
    // the producer-consumer fusion implementation in AffineLoopFusion pass are
    // made. See pass for specific details.
    ProducerConsumer,
    // Sibling fusion: Only sibling loops with no producer-consumer memref
    // dependences are considered for fusion. Memref reuse is taken into account
    // for profitability. Currently, assumptions from the sibling fusion
    // implementation in AffineLoopFusion pass are made. See pass for specific
    // details.
    Sibling
  } strategy;

  // Target memref for this fusion transformation.
  Value memref;

  FusionStrategy(StrategyEnum strategy, Value memref)
      : strategy(strategy), memref(memref) {}
};

/// Checks the feasibility of fusing the loop nest rooted at 'srcForOp' into the
/// loop nest rooted at 'dstForOp' at 'dstLoopDepth'. Returns FusionResult
/// 'Success' if fusion of the src/dst loop nests is feasible (i.e. they are
/// in the same block and dependences would not be violated). Otherwise
/// returns a FusionResult explaining why fusion is not feasible.
/// NOTE: This function is not feature complete and should only be used in
/// testing.
/// TODO: Update comments when this function is fully implemented.
FusionResult canFuseLoops(AffineForOp srcForOp, AffineForOp dstForOp,
                          unsigned dstLoopDepth,
                          ComputationSliceState *srcSlice,
                          FusionStrategy fusionStrategy = {
                              FusionStrategy::Generic, Value()});

/// Fuses 'srcForOp' into 'dstForOp' with destination loop block insertion point
/// and source slice loop bounds specified in 'srcSlice'.
void fuseLoops(AffineForOp srcForOp, AffineForOp dstForOp,
               const ComputationSliceState &srcSlice);

/// LoopNestStats aggregates various per-loop statistics (eg. loop trip count
/// and operation count) for a loop nest up until (and including) the innermost
/// loop body.
struct LoopNestStats {
  /// Map from AffineForOp to immediate child AffineForOps in its loop body.
  DenseMap<Operation *, SmallVector<AffineForOp, 2>> loopMap;
  /// Map from AffineForOp to count of operations in its loop body.
  DenseMap<Operation *, uint64_t> opCountMap;
  /// Map from AffineForOp to its constant trip count.
  DenseMap<Operation *, uint64_t> tripCountMap;
};

/// Collect loop nest statistics (eg. loop trip count and operation count)
/// in 'stats' for loop nest rooted at 'forOp'. Returns true on success,
/// returns false otherwise.
// TODO: Consider moving this to LoopUtils.
bool getLoopNestStats(AffineForOp forOp, LoopNestStats *stats);

/// Computes the total cost of the loop nest rooted at 'forOp' using 'stats'.
/// Currently, the total cost is computed by counting the total operation
/// instance count (i.e. total number of operations in the loop body * loop
/// trip count) for the entire loop nest.
// TODO: Improve this cost model.
int64_t getComputeCost(AffineForOp forOp, LoopNestStats &stats);

/// Computes and returns in 'computeCost', the total compute cost of fusing the
/// 'slice' of the loop nest rooted at 'srcForOp' into 'dstForOp'. Currently,
/// the total cost is computed by counting the total operation instance count
/// (i.e. total number of operations in the loop body * loop trip count) for
/// the entire loop nest.
/// Returns true on success, failure otherwise (e.g. non-constant trip counts).
// TODO: Improve this cost model.
bool getFusionComputeCost(AffineForOp srcForOp, LoopNestStats &srcStats,
                          AffineForOp dstForOp, LoopNestStats &dstStats,
                          const ComputationSliceState &slice,
                          int64_t *computeCost);

} // end namespace mlir

#endif // MLIR_TRANSFORMS_LOOP_FUSION_UTILS_H
