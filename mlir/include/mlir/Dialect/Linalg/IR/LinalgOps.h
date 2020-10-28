//===- LinalgOps.h - Linalg Operations --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_LINALGOPS_H_
#define MLIR_DIALECT_LINALG_LINALGOPS_H_

#include "mlir/Dialect/Linalg/IR/LinalgTraits.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/CopyOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace linalg {

class ConvOp;
class PoolingMaxOp;
class PoolingMinOp;
class PoolingSumOp;

using ReassociationIndices = SmallVector<int64_t, 2>;
using ReassociationExprs = SmallVector<AffineExpr, 2>;

/// Returns the name mangled library call name to disambiguate between different
/// overloads at the C level. The name mangling scheme is basic and uses MLIR
/// type names:
///   1. form a string which is the concatenation of the linalg op name with all
///      the operand type names, separate by underscores;
///   2. drop the `linalg.` prefix, and the `<`, `>`, `?` symbols from the type.
/// Assumes `op` is a LinalgOp.
///
/// Examples:
///
/// 1. linalg.fill(%A, %f) : memref<f32>, f32
///   name mangles into `linalg_fill_viewf32_f32_impl`
///
/// 2. linalg.dot %A, %B, %C :
///      (memref<?xf32, stride_specification>,
///       memref<?xf32, stride_specification>, memref<f32>)
///   name mangles into `linalg_dot_viewxf32_viewxf32_viewf32_impl`
///
/// 3. linalg.matmul(...) :
///      memref<?x?xf32, stride_specification>,
///      memref<?x?xf32, stride_specification>,
///      memref<?x?xf32, stride_specification>
///   name mangles into `linalg_matmul_viewxxf32_viewxxf32_viewxxf32_impl`
std::string generateLibraryCallName(Operation *op);

/// Returns `num` AffineDimExpr dimensions at positions
///   [startIdx, startIdx + num) and increments `startIdx` to `startIdx + num`.
SmallVector<AffineExpr, 4> makeAffineDimExprs(unsigned num, unsigned &startIdx,
                                              MLIRContext *context);

/// Builds the indexing expressions for a ConvOp/PoolingOp `op`. Returns the
/// vector of AffineMaps representing:
///   `stride[i] * outputDims[i] + dilation[i] * windowDims[i] - pad_low[i]`
template <typename PoolingOp>
extern SmallVector<AffineExpr, 4>
weightedPoolingInputIndex(PoolingOp op, ArrayRef<AffineExpr> outputDims,
                          ArrayRef<AffineExpr> windowDims);

/// Returns `maybeMap.get()` if `maybeMap` is set, otherwise returns the
/// symbol-less identity map of `rank`.
AffineMap extractOrIdentityMap(Optional<AffineMap> maybeMap, unsigned rank,
                               MLIRContext *context);

/// Return the vector that is the concatenation of `a` and `b`.
SmallVector<AffineExpr, 4> concat(ArrayRef<AffineExpr> a,
                                  ArrayRef<AffineExpr> b);

/// Return the dims that are `iteratorTypeName` loops in the LinalgOp `op`.
/// Assumes `op` is a LinalgOp.
void getDimsOfType(Operation *op, StringRef iteratorTypeName,
                   SmallVectorImpl<AffineExpr> &res);

} // namespace linalg
} // namespace mlir

#include "mlir/Dialect/Linalg/IR/LinalgStructuredOpsInterfaces.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Linalg/IR/LinalgOps.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.h.inc"

#endif // MLIR_DIALECT_LINALG_LINALGOPS_H_
