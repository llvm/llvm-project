//===- LoopInvariantCodeMotionUtils.h - LICM Utils --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_LOOPINVARIANTCODEMOTIONUTILS_H
#define MLIR_TRANSFORMS_LOOPINVARIANTCODEMOTIONUTILS_H

#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir {

class LoopLikeOpInterface;
class Operation;
class Region;
class Value;

/// Given a list of regions, perform loop-invariant code motion. An operation is
/// loop-invariant if it depends only of values defined outside of the loop.
/// LICM moves these operations out of the loop body so that they are not
/// computed more than once.
///
/// Example:
///
/// ```mlir
/// affine.for %arg0 = 0 to 10 {
///   affine.for %arg1 = 0 to 10 {
///     %v0 = arith.addi %arg0, %arg0 : i32
///     %v1 = arith.addi %v0, %arg1 : i32
///   }
/// }
/// ```
///
/// After LICM:
///
/// ```mlir
/// affine.for %arg0 = 0 to 10 {
///   %v0 = arith.addi %arg0, %arg0 : i32
///   affine.for %arg1 = 0 to 10 {
///     %v1 = arith.addi %v0, %arg1 : i32
///   }
/// }
/// ```
///
/// Users must supply three callbacks.
///
/// - `isDefinedOutsideRegion` returns true if the given value is invariant with
///   respect to the given region. A common implementation might be:
///   `value.getParentRegion()->isProperAncestor(region)`.
/// - `shouldMoveOutOfRegion` returns true if the provided operation can be
///   moved of the given region, e.g. if it is side-effect free.
/// - `moveOutOfRegion` moves the operation out of the given region. A common
///   implementation might be: `op->moveBefore(region->getParentOp())`.
///
/// An operation is moved if all of its operands satisfy
/// `isDefinedOutsideRegion` and it satisfies `shouldMoveOutOfRegion`.
///
/// Returns the number of operations moved.
size_t moveLoopInvariantCode(
    ArrayRef<Region *> regions,
    function_ref<bool(Value, Region *)> isDefinedOutsideRegion,
    function_ref<bool(Operation *, Region *)> shouldMoveOutOfRegion,
    function_ref<void(Operation *, Region *)> moveOutOfRegion);

/// Move side-effect free loop invariant code out of a loop-like op using
/// methods provided by the interface.
size_t moveLoopInvariantCode(LoopLikeOpInterface loopLike);

} // end namespace mlir

#endif // MLIR_TRANSFORMS_LOOPINVARIANTCODEMOTIONUTILS_H
