//===- IndexingUtils.h - Indexing utilities supporting Linalg ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_UTILS_INDEXINGUTILS_H
#define MLIR_DIALECT_LINALG_UTILS_INDEXINGUTILS_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "llvm/ADT/StringSet.h"
#include <optional>

namespace mlir {
namespace linalg {

/// Create one memref::DimOp or tensor::DimOp depending on the type of `val`.
/// This is a polymorphic convenience function to abstract away the rank and
/// concrete type of `val`.
/// Asserts that `val` is a memref or tensor type.
Value createOrFoldDimOp(OpBuilder &b, Location loc, Value val, int64_t dim);

/// Create one memref::DimOp or tensor::DimOp depending on the type of `val`.
/// This is a polymorphic convenience function to abstract away the rank and
/// concrete type of `val`.
/// Asserts that `val` is a memref or tensor type.
OpFoldResult createFoldedDimOp(OpBuilder &b, Location loc, Value val,
                               int64_t dim);

/// Build the list of DimOp for the dynamic dimensions of `val`.
/// Asserts that `val` is a ranked shaped type.
SmallVector<Value> createDynamicDimensions(OpBuilder &b, Location loc,
                                           Value val);

/// Build the list of all dimensions for `val`, mixing static attributes and
/// dynamic values where appropriate.
/// Asserts that `val` is a ranked shaped type.
SmallVector<OpFoldResult> getMixedDimensions(OpBuilder &b, Location loc,
                                             Value val);

} // namespace linalg
} // namespace mlir
#endif // MLIR_DIALECT_LINALG_UTILS_INDEXINGUTILS_H
