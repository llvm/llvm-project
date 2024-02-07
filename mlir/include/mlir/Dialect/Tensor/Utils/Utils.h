//===- Utils.h -  Utilities to support the Tensor dialect -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TENSOR_UTILS_UTILS_H_
#define MLIR_DIALECT_TENSOR_UTILS_UTILS_H_

#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir {
namespace tensor {

// Return a PadOp that pads `source` to `type` size where the static
// sizes are assumed to be greater than the dynamic sizes. If `type` has dynamic
// dimensions the padding width is set to zero. The op performs "high" padding
// (i.e. it adds trailing padding values until the desired size is met).
PadOp createPadHighOp(RankedTensorType type, Value source, Value pad,
                      bool nofold, Location loc, OpBuilder &builder);

// Creates dim ops for each dynamic dimension of the ranked tensor argument and
// returns these as values.
SmallVector<Value> createDynamicDimValues(OpBuilder &b, Location loc,
                                          Value rankedTensor);

/// Returns the transposed `rankedTensorType` if `transposeVector` is non-empty.
/// Fail if `transposeVector` is not a permutation matching the tensor rank.
FailureOr<RankedTensorType>
computeTransposedType(RankedTensorType rankedTensorType,
                      ArrayRef<int64_t> transposeVector);

/// Given a tensor::PackOp, compute the permutation vector to shuffle the
/// packed shape into the shape before any outer or inner permutations have
/// been applied.
/// i.e. for a pack from an ABCD layout to an ABCDba:
/// The packed shape would be ABCDba.
/// The pre-permutation shape would be AaBbCD.
SmallVector<int64_t> getPackInverseDestPermutation(PackOp packOp);

/// A tensor.insert_slice is a cast-like operation if it merely rank-extends the
/// source tensor or inserts the source tensor into a destination tensor with
/// the same shape.
bool isCastLikeInsertSliceOp(InsertSliceOp op);

/// A tensor.extract_slice is a cast-like operation if it merely rank-reduces
/// unit dimensions of the source tensor or extracts the entire source tensor.
bool isCastLikeExtractSliceOp(ExtractSliceOp op);

} // namespace tensor
} // namespace mlir

#endif // MLIR_DIALECT_TENSOR_UTILS_UTILS_H_
