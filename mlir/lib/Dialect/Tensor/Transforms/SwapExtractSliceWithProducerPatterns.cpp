//===- SwapExtractSliceWithProducerPatterns.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Swap a `tensor.extract_slice` with the producer of the source if the producer
// implements the `TilingInterface`. When used in conjunction with tiling this
// effectively tiles + fuses the producer with its consumer.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/TilingInterface.h"

using namespace mlir;

FailureOr<Value> tensor::replaceExtractSliceWithTiledProducer(
    OpBuilder &builder, tensor::ExtractSliceOp sliceOp, OpResult producer) {
  auto producerOp = dyn_cast<TilingInterface>(producer.getOwner());
  if (!producerOp)
    return failure();

  // `TilingInterface` currently only supports strides being 1.
  if (llvm::any_of(sliceOp.getMixedStrides(), [](OpFoldResult ofr) {
        return !isConstantIntValue(ofr, 1);
      }))
    return failure();

  FailureOr<Value> tiledResult = producerOp.generateResultTileValue(
      builder, producer.getResultNumber(), sliceOp.getMixedOffsets(),
      sliceOp.getMixedSizes());
  if (failed(tiledResult))
    return failure();

  return tiledResult.value();
}
