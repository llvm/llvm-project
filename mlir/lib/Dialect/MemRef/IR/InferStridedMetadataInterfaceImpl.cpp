//===- InferStridedMetadataInterfaceImpl.cpp - Impl. of infer strided md --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/InferStridedMetadataInterfaceImpl.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/InferStridedMetadataInterface.h"
#include "mlir/Interfaces/Utils/InferIntRangeCommon.h"

using namespace mlir;
using namespace mlir::memref;

/// Collect the integer range values on a set of op fold results. This function
/// returns failure if any of the int ranges couldn't be collected.
static FailureOr<SmallVector<ConstantIntRanges>>
getIntValueRanges(SmallVector<OpFoldResult> values, GetIntRangeFn getIntRange,
                  int32_t indexBitwidth) {
  SmallVector<ConstantIntRanges> ranges;
  ranges.reserve(values.size());
  for (OpFoldResult ofr : values) {
    if (auto value = dyn_cast<Value>(ofr)) {
      IntegerValueRange range = getIntRange(value);
      // Bail if the range is not available.
      if (range.isUninitialized())
        return failure();
      ranges.push_back(range.getValue());
      continue;
    }

    // Create a constant range.
    auto attr = cast<IntegerAttr>(cast<Attribute>(ofr));
    ranges.emplace_back(ConstantIntRanges::constant(
        attr.getValue().sextOrTrunc(indexBitwidth)));
  }
  return ranges;
}

namespace {
/// Implementation of `InferStridedMetadataOpInterface` for the `memref.subview`
/// operation.
struct SubViewOpInterface
    : public InferStridedMetadataOpInterface::ExternalModel<SubViewOpInterface,
                                                            SubViewOp> {
  void inferStridedMetadataRanges(Operation *op,
                                  ArrayRef<StridedMetadataRange> ranges,
                                  GetIntRangeFn getIntRange,
                                  SetStridedMetadataRangeFn setMetadata,
                                  int32_t indexBitwidth) const {
    auto subViewOp = cast<SubViewOp>(op);

    // Bail early if any of the operands metadata is not ready:
    FailureOr<SmallVector<ConstantIntRanges>> offsetOperands =
        getIntValueRanges(subViewOp.getMixedOffsets(), getIntRange,
                          indexBitwidth);
    if (failed(offsetOperands))
      return;

    FailureOr<SmallVector<ConstantIntRanges>> sizeOperands = getIntValueRanges(
        subViewOp.getMixedSizes(), getIntRange, indexBitwidth);
    if (failed(sizeOperands))
      return;

    FailureOr<SmallVector<ConstantIntRanges>> stridesOperands =
        getIntValueRanges(subViewOp.getMixedStrides(), getIntRange,
                          indexBitwidth);
    if (failed(stridesOperands))
      return;

    StridedMetadataRange sourceRange =
        ranges[subViewOp.getSourceMutable().getOperandNumber()];
    if (sourceRange.isUninitialized())
      return;

    ArrayRef<ConstantIntRanges> srcStrides = sourceRange.getStrides();

    // Get the dropped dims.
    llvm::SmallBitVector droppedDims = subViewOp.getDroppedDims();

    // Compute the new offset, strides and sizes.
    ConstantIntRanges offset = sourceRange.getOffsets()[0];
    SmallVector<ConstantIntRanges> strides, sizes;

    for (size_t i = 0, e = droppedDims.size(); i < e; ++i) {
      bool dropped = droppedDims.test(i);
      // Compute the new offset.
      ConstantIntRanges off =
          intrange::inferMul({(*offsetOperands)[i], srcStrides[i]});
      offset = intrange::inferAdd({offset, off});

      // Skip dropped dimensions.
      if (dropped)
        continue;
      // Multiply the strides.
      strides.push_back(
          intrange::inferMul({(*stridesOperands)[i], srcStrides[i]}));
      // Get the sizes.
      sizes.push_back((*sizeOperands)[i]);
    }

    setMetadata(subViewOp.getResult(),
                StridedMetadataRange::getRanked(
                    SmallVector<ConstantIntRanges>({std::move(offset)}),
                    std::move(sizes), std::move(strides)));
  }
};
} // namespace

void mlir::memref::registerInferStridedMetadataOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, memref::MemRefDialect *dialect) {
    memref::SubViewOp::attachInterface<SubViewOpInterface>(*ctx);
  });
}
