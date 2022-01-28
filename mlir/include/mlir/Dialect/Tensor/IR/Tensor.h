//===- Tensor.h - Tensor dialect --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TENSOR_IR_TENSOR_H_
#define MLIR_DIALECT_TENSOR_IR_TENSOR_H_

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

//===----------------------------------------------------------------------===//
// Tensor Dialect Helpers
//===----------------------------------------------------------------------===//

namespace mlir {

/// Return the list of Range (i.e. offset, size, stride). Each Range
/// entry contains either the dynamic value or a ConstantIndexOp constructed
/// with `b` at location `loc`.
SmallVector<Range, 8> getOrCreateRanges(OffsetSizeAndStrideOpInterface op,
                                        OpBuilder &b, Location loc);

} // namespace mlir

//===----------------------------------------------------------------------===//
// Tensor Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tensor/IR/TensorOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Tensor Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Tensor/IR/TensorOps.h.inc"

//===----------------------------------------------------------------------===//
// Tensor Dialect Helpers
//===----------------------------------------------------------------------===//

namespace mlir {
namespace tensor {

/// Returns true if `target` is a ranked tensor type that preserves static
/// information available in the `source` ranked tensor type.
bool preservesStaticInformation(Type source, Type target);

/// Determines whether tensor::CastOp casts to a more dynamic version of the
/// source tensor. This is useful to fold a tensor.cast into a consuming op and
/// implement canonicalization patterns for ops in different dialects that may
/// consume the results of tensor.cast operations. Such foldable tensor.cast
/// operations are typically inserted as `extract_slice` ops and are
/// canonicalized, to preserve the type compatibility of their uses.
///
/// Returns true when all conditions are met:
/// 1. source and result are ranked tensors with same element type and rank.
/// 2. the tensor type has more static information than the result
///
/// Example:
/// ```mlir
///   %1 = tensor.cast %0 : tensor<8x16xf32> to tensor<?x?xf32>
///   %2 = consumer %1 ... : tensor<?x?xf32> ...
/// ```
///
/// folds into:
///
/// ```mlir
///   %2 = consumer %0 ... : tensor<8x16xf32> ...
/// ```
bool canFoldIntoConsumerOp(CastOp castOp);

/// Performs folding of any operand of `op` if it comes from a tensor::CastOp
/// that can be folded.
LogicalResult foldTensorCast(Operation *op);

/// Create a rank-reducing ExtractSliceOp @[0 .. 0] with strides [1 .. 1] and
/// appropriate sizes (i.e. `tensor.getSizes()`) to reduce the rank of `tensor`
/// to that of `targetType`.
Value createCanonicalRankReducingExtractSliceOp(OpBuilder &b, Location loc,
                                                Value tensor,
                                                RankedTensorType targetType);

/// Create a rank-reducing InsertSliceOp @[0 .. 0] with strides [1 .. 1] and
/// appropriate sizes (i.e. `dest.getSizes()`). The result is a new tensor with
/// rank increased to that of `dest`, obtained by inserting `tensor` into `dest`
/// at the canonical [0 .. 0] position.
Value createCanonicalRankReducingInsertSliceOp(OpBuilder &b, Location loc,
                                               Value tensor, Value dest);

} // namespace tensor
} // namespace mlir

#endif // MLIR_DIALECT_TENSOR_IR_TENSOR_H_
