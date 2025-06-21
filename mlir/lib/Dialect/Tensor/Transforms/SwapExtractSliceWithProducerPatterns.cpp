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

FailureOr<TilingResult> tensor::replaceInsertSlicesWithTiledConsumer(
    OpBuilder &builder, ArrayRef<tensor::InsertSliceOp> sliceOps,
    ArrayRef<OpOperand *> consumerOperands) {
  if (sliceOps.empty()) {
    return emitError(builder.getUnknownLoc(),
                     "expected candidate slices list to be non-empty");
  }
  if (sliceOps.size() != consumerOperands.size()) {
    return sliceOps.front()->emitOpError(
        "expected as many operands as the number of slices passed");
  }
  auto consumerOp =
      dyn_cast<TilingInterface>(consumerOperands.front()->getOwner());
  if (!consumerOp)
    return failure();
  for (auto opOperand : consumerOperands.drop_front()) {
    if (opOperand->getOwner() != consumerOp) {
      return consumerOp->emitOpError(
          "expected all consumer operands to be from the same operation");
    }
  }

  auto consumerOperandNums = llvm::map_to_vector(
      consumerOperands, [](OpOperand *opOperand) -> unsigned {
        return opOperand->getOperandNumber();
      });
  SmallVector<SmallVector<OpFoldResult>> allOffsets;
  SmallVector<SmallVector<OpFoldResult>> allSizes;
  for (auto sliceOp : sliceOps) {

    // `TilingInterface` currently only supports strides being 1.
    if (!llvm::all_of(sliceOp.getMixedStrides(), isOneInteger))
      return failure();

    SmallVector<OpFoldResult> offsets = sliceOp.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = sliceOp.getMixedSizes();
    allOffsets.emplace_back(std::move(offsets));
    allSizes.emplace_back(std::move(sizes));
  }
  FailureOr<TilingResult> tiledResult =
      consumerOp.getTiledImplementationFromOperandTiles(
          builder, consumerOperandNums, allOffsets, allSizes);
  if (failed(tiledResult))
    return failure();

  return *tiledResult;
}
