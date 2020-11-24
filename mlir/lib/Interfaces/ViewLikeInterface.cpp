//===- ViewLikeInterface.cpp - View-like operations in MLIR ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/ViewLikeInterface.h"

#include "mlir/IR/StandardTypes.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// ViewLike Interfaces
//===----------------------------------------------------------------------===//

/// Include the definitions of the loop-like interfaces.
#include "mlir/Interfaces/ViewLikeInterface.cpp.inc"

static LogicalResult verifyOpWithOffsetSizesAndStridesPart(
    OffsetSizeAndStrideOpInterface op, StringRef name,
    unsigned expectedNumElements, StringRef attrName, ArrayAttr attr,
    llvm::function_ref<bool(int64_t)> isDynamic, ValueRange values) {
  /// Check static and dynamic offsets/sizes/strides breakdown.
  if (attr.size() != expectedNumElements)
    return op.emitError("expected ")
           << expectedNumElements << " " << name << " values";
  unsigned expectedNumDynamicEntries =
      llvm::count_if(attr.getValue(), [&](Attribute attr) {
        return isDynamic(attr.cast<IntegerAttr>().getInt());
      });
  if (values.size() != expectedNumDynamicEntries)
    return op.emitError("expected ")
           << expectedNumDynamicEntries << " dynamic " << name << " values";
  return success();
}

LogicalResult mlir::verify(OffsetSizeAndStrideOpInterface op) {
  std::array<unsigned, 3> ranks = op.getArrayAttrRanks();
  if (failed(verifyOpWithOffsetSizesAndStridesPart(
          op, "offset", ranks[0],
          OffsetSizeAndStrideOpInterface::getStaticOffsetsAttrName(),
          op.static_offsets(), ShapedType::isDynamicStrideOrOffset,
          op.offsets())))
    return failure();
  if (failed(verifyOpWithOffsetSizesAndStridesPart(
          op, "size", ranks[1],
          OffsetSizeAndStrideOpInterface::getStaticSizesAttrName(),
          op.static_sizes(), ShapedType::isDynamic, op.sizes())))
    return failure();
  if (failed(verifyOpWithOffsetSizesAndStridesPart(
          op, "stride", ranks[2],
          OffsetSizeAndStrideOpInterface::getStaticStridesAttrName(),
          op.static_strides(), ShapedType::isDynamicStrideOrOffset,
          op.strides())))
    return failure();
  return success();
}
