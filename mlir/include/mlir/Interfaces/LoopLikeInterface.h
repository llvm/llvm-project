//===- LoopLikeInterface.h - Loop-like operations interface ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the operation interface for loop like operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_LOOPLIKEINTERFACE_H_
#define MLIR_INTERFACES_LOOPLIKEINTERFACE_H_

#include "mlir/IR/OpDefinition.h"

namespace mlir {
class RewriterBase;

/// A function that returns the additional yielded values during
/// `replaceWithAdditionalYields`. `newBbArgs` are the newly added region
/// iter_args. This function should return as many values as there are block
/// arguments in `newBbArgs`.
using NewYieldValuesFn = llvm::function_ref<SmallVector<Value>(
    OpBuilder &b, Location loc, ArrayRef<BlockArgument> newBbArgs)>;

/// A function that allows returning additional yielded values during
/// `yieldTiledValuesAndReplace`.
/// - `ivs` induction variable for the loop.
/// - `newBbArgs` basic block arguments corresponding to newly added iter_args.
/// - `tiledValues` the tiled values to return. Must be of same size as
///   `newbbArgs`, each element of this array is inserted into the corresponding
///   element in `newbbArgs`.
/// - `resultOffsets` is of the same size as `tiledValues` and represents
///   the offsets to use when inserting corresponding element from `tiledValues`
///   into the element from `newBbArgs`.
/// - `resultSizes` is of the same size as `tiledValues` and represents
///   the size of the corresponding element from `tiledValues` inserted into
///   the element from `newBbArgs`.
/// - `resultStrides` is of the same size as `tiledValues` and represents
///   the strides to use when inserting corresponding element from `tiledValues`
///   into the element from `newBbArgs`.
using YieldTiledValuesFn = llvm::function_ref<LogicalResult(
    RewriterBase &rewriter, Location loc, ValueRange ivs, ValueRange newBbArgs,
    SmallVector<Value> &tiledValues,
    SmallVector<SmallVector<OpFoldResult>> &resultOffsets,
    SmallVector<SmallVector<OpFoldResult>> &resultSizes)>;

namespace detail {
/// Verify invariants of the LoopLikeOpInterface.
LogicalResult verifyLoopLikeOpInterface(Operation *op);
} // namespace detail
} // namespace mlir

/// Include the generated interface declarations.
#include "mlir/Interfaces/LoopLikeInterface.h.inc"

#endif // MLIR_INTERFACES_LOOPLIKEINTERFACE_H_
