//===- MemRefUtils.h - MemRef transformation utilities ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for various transformation utilities for
// the MemRefOps dialect. These are not passes by themselves but are used
// either by passes, optimization sequences, or in turn by other transformation
// utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MEMREF_UTILS_MEMREFUTILS_H
#define MLIR_DIALECT_MEMREF_UTILS_MEMREFUTILS_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir {

class MemRefType;

namespace memref {

/// Returns true, if the memref type has static shapes and represents a
/// contiguous chunk of memory.
bool isStaticShapeAndContiguousRowMajor(MemRefType type);

/// For a `memref` with `offset`, `sizes` and `strides`, returns the
/// offset and size to use for the linearized `memref`.
/// - If the linearization is done for emulating load/stores of
///   element type with bitwidth `srcBits` using element type with
///   bitwidth `dstBits`, the linearized offset and size are
///   scaled down by `dstBits`/`srcBits`.
/// - If `indices` is provided, it represents the position in the
///   original `memref` being accessed. The method then returns the
///   index to use in the linearized `memref`. The linearized index
///   is also scaled down by `dstBits`/`srcBits`. If `indices` is not provided
///   0, is returned for the linearized index.
struct LinearizedMemRefInfo {
  OpFoldResult linearizedOffset;
  OpFoldResult linearizedSize;
};
std::pair<LinearizedMemRefInfo, OpFoldResult> getLinearizedMemRefOffsetAndSize(
    OpBuilder &builder, Location loc, int srcBits, int dstBits,
    OpFoldResult offset, ArrayRef<OpFoldResult> sizes,
    ArrayRef<OpFoldResult> strides, ArrayRef<OpFoldResult> indices = {});

/// For a `memref` with `offset` and `sizes`, returns the
/// offset and size to use for the linearized `memref`, assuming that
/// the strides are computed from a row-major ordering of the sizes;
/// - If the linearization is done for emulating load/stores of
///   element type with bitwidth `srcBits` using element type with
///   bitwidth `dstBits`, the linearized offset and size are
///   scaled down by `dstBits`/`srcBits`.
LinearizedMemRefInfo
getLinearizedMemRefOffsetAndSize(OpBuilder &builder, Location loc, int srcBits,
                                 int dstBits, OpFoldResult offset,
                                 ArrayRef<OpFoldResult> sizes);

// Track temporary allocations that are never read from. If this is the case
// it means both the allocations and associated stores can be removed.
void eraseDeadAllocAndStores(RewriterBase &rewriter, Operation *parentOp);

} // namespace memref
} // namespace mlir

#endif // MLIR_DIALECT_MEMREF_UTILS_MEMREFUTILS_H
