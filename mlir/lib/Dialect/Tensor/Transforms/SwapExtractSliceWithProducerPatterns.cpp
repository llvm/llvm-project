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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/TilingInterface.h"

using namespace mlir;

FailureOr<TilingResult> tensor::replaceExtractSliceWithTiledProducer(
    OpBuilder &builder, tensor::ExtractSliceOp sliceOp, OpResult producer) {
  auto producerOp = dyn_cast<TilingInterface>(producer.getOwner());
  if (!producerOp)
    return failure();

  // `TilingInterface` currently only supports strides being 1.
  if (!llvm::all_of(sliceOp.getMixedStrides(), isOneInteger))
    return failure();

  FailureOr<TilingResult> tiledResult = producerOp.generateResultTileValue(
      builder, producer.getResultNumber(), sliceOp.getMixedOffsets(),
      sliceOp.getMixedSizes());
  if (failed(tiledResult))
    return failure();

  return *tiledResult;
}

FailureOr<TilingResult> tensor::replaceInsertSliceWithTiledConsumer(
    OpBuilder &builder, OffsetSizeAndStrideOpInterface sliceOp,
    OpOperand &consumer) {
  auto consumerOp = dyn_cast<TilingInterface>(consumer.getOwner());
  if (!consumerOp)
    return failure();

  // `TilingInterface` currently only supports strides being 1.
  if (!llvm::all_of(sliceOp.getMixedStrides(), isOneInteger))
    return failure();

  FailureOr<TilingResult> tiledResult =
      consumerOp.getTiledImplementationFromOperandTile(
          builder, consumer.getOperandNumber(), sliceOp.getMixedOffsets(),
          sliceOp.getMixedSizes());
  if (failed(tiledResult))
    return failure();

  return *tiledResult;
}
