//===- Transforms.h - Tensor Transformation Patterns ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TENSOR_TRANSFORMS_TRANSFORMS_H
#define MLIR_DIALECT_TENSOR_TRANSFORMS_TRANSFORMS_H

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace tensor {

/// Populates `patterns` with patterns to wrap a tensor.pad op with an scf.if op
/// to separate the cases where we don't need padding (all pad sizes are
/// actually zeros) and where we indeed need padding.
void populateSplitPaddingPatterns(RewritePatternSet &patterns,
                                  PatternBenefit baseBenefit = 1);

/// Pattern to swap an `tensor.extract_slice` with its producer when the
/// producer implements the `TilingInterface`. The pattern itself does not
/// provide a mechanism to control where the application happens. With use of
/// transform dialect that control is done within the transform dialect. Other
/// use cases can inherit from this pattern and add necessary controls.
FailureOr<Value> replaceExtractSliceWithTiledProducer(
    OpBuilder &builder, tensor::ExtractSliceOp sliceOp, OpResult producerOp);

/// Collects patterns to merge consecutive tensor.insert_slice/extract_slice
/// into one. These patterns are in in this separate entry point because the
/// bufferization is sensitive over IR structure, particularly those
/// tensor.extract_slice and tensor.insert_slice ops for creating the slices.
void populateMergeConsecutiveInsertExtractSlicePatterns(
    RewritePatternSet &patterns);

/// Populates `patterns` with patterns that fold `tensor.expand_shape` and
/// `tensor.collapse_shape` into other ops.
void populateReassociativeReshapeFoldingPatterns(RewritePatternSet &patterns);

/// Populates `patterns` with patterns that fold tensor.empty with
/// tensor.[extract_slice|cast|expand_shape|collapse_shape].
void populateFoldTensorEmptyPatterns(RewritePatternSet &patterns);

/// Populates `patterns` with patterns that fold operations like `tensor.pad`
/// and `tensor.extract_slice` into `tensor.pack` and `tensor.unpack` operations
/// respectively.
void populateFoldIntoPackAndUnpackPatterns(RewritePatternSet &patterns);

} // namespace tensor
} // namespace mlir

#endif // MLIR_DIALECT_TENSOR_TRANSFORMS_TRANSFORMS_H
