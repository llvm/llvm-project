//===- VectorDistributionUtils.h - Distribution Utilities -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_VECTOR_DISTRIBUTION_UTILS_VECTORUTILS_H_
#define MLIR_DIALECT_VECTOR_DISTRIBITION_UTILS_VECTORUTILS_H_

#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include <utility>

namespace mlir {
namespace vector {
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
} // namespace vector
} // namespace mlir

#endif // MLIR_DIALECT_VECTOR_DISTRIBITION_UTILS_VECTORUTILS_H_
