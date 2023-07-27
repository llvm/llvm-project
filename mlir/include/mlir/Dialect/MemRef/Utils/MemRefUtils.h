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

/// Returns the flattened 1-D memref and linearized offset for narrow type
/// emulation.
///
/// The emulation only works on 1D memref types. To make this work on N-D
/// memref, we need to linearize the offset.
///
/// For example, to emulate i4 to i8, the following op:
///
/// %0 = memref.load %arg0[%v0, %v1] :
///                  memref<?x?xi4, strided<[?, ?], offset: ?>>
///
/// can be replaced with
///
/// %b, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %0
///
/// %linearized_offset = %v0 * %stride#0 + %v1 * %stride#1
/// %linearized_size = %size0 * %size1
/// %scaled_linear_offset = %linearized_offset / 8 * 4
/// %scaled_base_offset = %offset / 8 * 4
///
/// %linearized = memref.reinterpret_cast %b, offset = [%scaled_base_offset],
///                      sizes = [%linearized_size], strides = [%stride#1]
///
/// %new_load = memref.load %linearized[%scaled_linear_offset] :
///                         memref<?xi8, strided<[?], offset: ?>>
std::pair<Value, Value>
getLinearizeMemRefAndOffset(Location loc, MemRefType sourceType, int srcBits,
                            int dstBits, SmallVector<Value> indices,
                            memref::ExtractStridedMetadataOp stridedMetadata,
                            OpBuilder &builder);

} // namespace memref
} // namespace mlir

#endif // MLIR_DIALECT_MEMREF_UTILS_MEMREFUTILS_H
