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

// Return a PadOp that pads `source` to `resType` size. The op performs "high"
// padding, i.e. it adds trailing padding values until the desired size is met.
// Output sizes are assumed to be greater than the input sizes. The padding
// width is calculated as: resDim - sourceDim.
//
// Handling static sizes is trivial. Dynamic dimensions are trickier (*):
//  1. dynamic input sizes are extracted from `source`
//  2. for dynamic output dims, there are two options:
//    2.1 all output dynamic dim sizes are specified in `dynOutDim`,
//    2.2 `dynOutDim` is empty and the corresponding padding width is set to 0.
//
// (*) Note that `resType` is just a shape and it only encodes the actual sizes
// for _static_ dimensions.
PadOp createPadHighOp(RankedTensorType resType, Value source, Value pad,
                      bool nofold, Location loc, OpBuilder &builder,
                      SmallVector<Value> dynOutDim = {});

// Creates dim ops for each dynamic dimension of the ranked tensor argument and
// returns these as values.
SmallVector<Value> createDynamicDimValues(OpBuilder &b, Location loc,
                                          Value rankedTensor);

/// Returns the transposed `rankedTensorType` if `transposeVector` is non-empty.
/// Fail if `transposeVector` is not a permutation matching the tensor rank.
FailureOr<RankedTensorType>
computeTransposedType(RankedTensorType rankedTensorType,
                      ArrayRef<int64_t> transposeVector);

/// Shell function to compute the Destination Permutation of PackOp
/// This function uses the helper function `computePackUnPackPerm` to get
/// the permutation vector. Only major difference between UnPack and Pack is
/// that packOp uses destination rank whereas unpack Uses source rank.
SmallVector<int64_t> getPackInverseDestPerm(tensor::PackOp packOp);

/// Shell function to compute the Source Permutation of unPackOp.
/// This function, like the getPackInverseDestPerm uses the helper function
/// computePackUnPackPerm` to get the permutation vector.
/// Only major difference between UnPack and Pack is that packOp uses
/// destination rank whereas unpack Uses source rank.
SmallVector<int64_t> getUnPackInverseSrcPerm(tensor::UnPackOp unpackOp);

/// Shell function to compute the Source rank permutation for unpackOp
/// Unpack requires some packing metadata data information, so created
/// another function where this value is passed by reference.
SmallVector<int64_t> getUnPackInverseSrcPerm(tensor::UnPackOp,
                                             PackingMetadata &metadata);

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
