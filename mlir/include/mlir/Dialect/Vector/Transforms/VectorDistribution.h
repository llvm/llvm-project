//===- VectorDistribution.h - Vector distribution patterns --*- C++------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_VECTOR_TRANSFORMS_VECTORDISTRIBUTION_H_
#define MLIR_DIALECT_VECTOR_TRANSFORMS_VECTORDISTRIBUTION_H_

#include "mlir/Dialect/Vector/IR/VectorOps.h"

namespace mlir {
class RewritePatternSet;
namespace vector {

struct WarpExecuteOnLane0LoweringOptions {
  /// Lamdba function to let users allocate memory needed for the lowering of
  /// WarpExecuteOnLane0Op.
  /// The function needs to return an allocation that the lowering can use as
  /// temporary memory. The allocation needs to match the shape of the type (the
  /// type may be VectorType or a scalar) and be availble for the current warp.
  /// If there are several warps running in parallel the allocation needs to be
  /// split so that each warp has its own allocation.
  using WarpAllocationFn =
      std::function<Value(Location, OpBuilder &, WarpExecuteOnLane0Op, Type)>;
  WarpAllocationFn warpAllocationFn = nullptr;

  /// Lamdba function to let user emit operation to syncronize all the thread
  /// within a warp. After this operation all the threads can see any memory
  /// written before the operation.
  using WarpSyncronizationFn =
      std::function<void(Location, OpBuilder &, WarpExecuteOnLane0Op)>;
  WarpSyncronizationFn warpSyncronizationFn = nullptr;
};

void populateWarpExecuteOnLane0OpToScfForPattern(
    RewritePatternSet &patterns,
    const WarpExecuteOnLane0LoweringOptions &options);

using DistributionMapFn = std::function<AffineMap(vector::TransferWriteOp)>;

/// Distribute transfer_write ops based on the affine map returned by
/// `distributionMapFn`.
/// Example:
/// ```
/// %0 = vector.warp_execute_on_lane_0(%id){
///   ...
///   vector.transfer_write %v, %A[%c0] : vector<32xf32>, memref<128xf32>
///   vector.yield
/// }
/// ```
/// To
/// ```
/// %r:3 = vector.warp_execute_on_lane_0(%id) -> (vector<1xf32>) {
///   ...
///   vector.yield %v : vector<32xf32>
/// }
/// vector.transfer_write %v, %A[%id] : vector<1xf32>, memref<128xf32>
void populateDistributeTransferWriteOpPatterns(
    RewritePatternSet &patterns, DistributionMapFn distributionMapFn);

/// Move scalar operations with no dependency on the warp op outside of the
/// region.
void moveScalarUniformCode(WarpExecuteOnLane0Op op);

/// Collect patterns to propagate warp distribution.
void populatePropagateWarpVectorDistributionPatterns(
    RewritePatternSet &pattern);

} // namespace vector
} // namespace mlir
#endif // MLIR_DIALECT_VECTOR_TRANSFORMS_VECTORDISTRIBUTION_H_
