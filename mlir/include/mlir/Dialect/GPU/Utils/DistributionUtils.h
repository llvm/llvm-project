//===- VectorDistributionUtils.h - Distribution Utilities -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_GPU_TRANSFORMS_DISTRIBUTIONUTILS_H_
#define MLIR_DIALECT_GPU_TRANSFORMS_DISTRIBITIONUTILS_H_

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/PatternMatch.h"

#include <utility>

namespace mlir {
namespace gpu {
/// Return a value yielded by `warpOp` which statifies the filter lamdba
/// condition and is not dead.
OpOperand *getWarpResult(WarpExecuteOnLane0Op warpOp,
                         const std::function<bool(Operation *)> &fn);

/// Helper to create a new WarpExecuteOnLane0Op with different signature.
WarpExecuteOnLane0Op moveRegionToNewWarpOpAndReplaceReturns(
    RewriterBase &rewriter, WarpExecuteOnLane0Op warpOp,
    ValueRange newYieldedValues, TypeRange newReturnTypes);

/// Helper to create a new WarpExecuteOnLane0Op region with extra outputs.
/// `indices` return the index of each new output.
WarpExecuteOnLane0Op moveRegionToNewWarpOpAndAppendReturns(
    RewriterBase &rewriter, WarpExecuteOnLane0Op warpOp,
    ValueRange newYieldedValues, TypeRange newReturnTypes,
    llvm::SmallVector<size_t> &indices);

/// Helper to know if an op can be hoisted out of the region.
bool canBeHoisted(Operation *op, function_ref<bool(Value)> definedOutside);

/// Return a value yielded by `warpOp` which statifies the filter lamdba
/// condition and is not dead.
OpOperand *getWarpResult(WarpExecuteOnLane0Op warpOp,
                         const std::function<bool(Operation *)> &fn);

/// Delinearize the given `laneId` into multiple dimensions, where each
/// dimension's size is determined by `originalShape` and `distributedShape`
/// together. This function expects the total numbers of threads needed for
/// distribution is equal to `warpSize`. Returns true and updates
/// `delinearizedIds` if so.
bool delinearizeLaneId(OpBuilder &builder, Location loc,
                       ArrayRef<int64_t> originalShape,
                       ArrayRef<int64_t> distributedShape, int64_t warpSize,
                       Value laneId, SmallVectorImpl<Value> &delinearizedIds);

} // namespace gpu
} // namespace mlir

#endif // MLIR_DIALECT_GPU_TRANSFORMS_DISTRIBUTIONUTILS_H_
