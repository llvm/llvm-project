//===- ViewLikeInterface.cpp - View-like operations in MLIR ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/ViewLikeInterface.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// ViewLike Interfaces
//===----------------------------------------------------------------------===//

/// Include the definitions of the loop-like interfaces.
#include "mlir/Interfaces/ViewLikeInterface.cpp.inc"

LogicalResult mlir::verifyListOfOperandsOrIntegers(
    Operation *op, StringRef name, unsigned numElements, ArrayAttr attr,
    ValueRange values, llvm::function_ref<bool(int64_t)> isDynamic) {
  /// Check static and dynamic offsets/sizes/strides does not overflow type.
  if (attr.size() != numElements)
    return op->emitError("expected ")
           << numElements << " " << name << " values";
  unsigned expectedNumDynamicEntries =
      llvm::count_if(attr.getValue(), [&](Attribute attr) {
        return isDynamic(attr.cast<IntegerAttr>().getInt());
      });
  if (values.size() != expectedNumDynamicEntries)
    return op->emitError("expected ")
           << expectedNumDynamicEntries << " dynamic " << name << " values";
  return success();
}

LogicalResult
mlir::detail::verifyOffsetSizeAndStrideOp(OffsetSizeAndStrideOpInterface op) {
  std::array<unsigned, 3> maxRanks = op.getArrayAttrMaxRanks();
  // Offsets can come in 2 flavors:
  //   1. Either single entry (when maxRanks == 1).
  //   2. Or as an array whose rank must match that of the mixed sizes.
  // So that the result type is well-formed.
  if (!(op.getMixedOffsets().size() == 1 && maxRanks[0] == 1) && // NOLINT
      op.getMixedOffsets().size() != op.getMixedSizes().size())
    return op->emitError(
               "expected mixed offsets rank to match mixed sizes rank (")
           << op.getMixedOffsets().size() << " vs " << op.getMixedSizes().size()
           << ") so the rank of the result type is well-formed.";
  // Ranks of mixed sizes and strides must always match so the result type is
  // well-formed.
  if (op.getMixedSizes().size() != op.getMixedStrides().size())
    return op->emitError(
               "expected mixed sizes rank to match mixed strides rank (")
           << op.getMixedSizes().size() << " vs " << op.getMixedStrides().size()
           << ") so the rank of the result type is well-formed.";

  if (failed(verifyListOfOperandsOrIntegers(
          op, "offset", maxRanks[0], op.static_offsets(), op.offsets(),
          ShapedType::isDynamicStrideOrOffset)))
    return failure();
  if (failed(verifyListOfOperandsOrIntegers(op, "size", maxRanks[1],
                                            op.static_sizes(), op.sizes(),
                                            ShapedType::isDynamic)))
    return failure();
  if (failed(verifyListOfOperandsOrIntegers(
          op, "stride", maxRanks[2], op.static_strides(), op.strides(),
          ShapedType::isDynamicStrideOrOffset)))
    return failure();
  return success();
}

template <int64_t dynVal>
static void printOperandsOrIntegersListImpl(OpAsmPrinter &p, ValueRange values,
                                            ArrayAttr arrayAttr) {
  p << '[';
  if (arrayAttr.empty()) {
    p << "]";
    return;
  }
  unsigned idx = 0;
  llvm::interleaveComma(arrayAttr, p, [&](Attribute a) {
    int64_t val = a.cast<IntegerAttr>().getInt();
    if (val == dynVal)
      p << values[idx++];
    else
      p << val;
  });
  p << ']';
}

void mlir::printOperandsOrIntegersOffsetsOrStridesList(OpAsmPrinter &p,
                                                       Operation *op,
                                                       OperandRange values,
                                                       ArrayAttr integers) {
  return printOperandsOrIntegersListImpl<ShapedType::kDynamicStrideOrOffset>(
      p, values, integers);
}

void mlir::printOperandsOrIntegersSizesList(OpAsmPrinter &p, Operation *op,
                                            OperandRange values,
                                            ArrayAttr integers) {
  return printOperandsOrIntegersListImpl<ShapedType::kDynamicSize>(p, values,
                                                                   integers);
}

template <int64_t dynVal>
static ParseResult parseOperandsOrIntegersImpl(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &values,
    ArrayAttr &integers) {
  if (failed(parser.parseLSquare()))
    return failure();
  // 0-D.
  if (succeeded(parser.parseOptionalRSquare())) {
    integers = parser.getBuilder().getArrayAttr({});
    return success();
  }

  SmallVector<int64_t, 4> attrVals;
  while (true) {
    OpAsmParser::UnresolvedOperand operand;
    auto res = parser.parseOptionalOperand(operand);
    if (res.hasValue() && succeeded(res.getValue())) {
      values.push_back(operand);
      attrVals.push_back(dynVal);
    } else {
      IntegerAttr attr;
      if (failed(parser.parseAttribute<IntegerAttr>(attr)))
        return parser.emitError(parser.getNameLoc())
               << "expected SSA value or integer";
      attrVals.push_back(attr.getInt());
    }

    if (succeeded(parser.parseOptionalComma()))
      continue;
    if (failed(parser.parseRSquare()))
      return failure();
    break;
  }
  integers = parser.getBuilder().getI64ArrayAttr(attrVals);
  return success();
}

ParseResult mlir::parseOperandsOrIntegersOffsetsOrStridesList(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &values,
    ArrayAttr &integers) {
  return parseOperandsOrIntegersImpl<ShapedType::kDynamicStrideOrOffset>(
      parser, values, integers);
}

ParseResult mlir::parseOperandsOrIntegersSizesList(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &values,
    ArrayAttr &integers) {
  return parseOperandsOrIntegersImpl<ShapedType::kDynamicSize>(parser, values,
                                                               integers);
}

bool mlir::detail::sameOffsetsSizesAndStrides(
    OffsetSizeAndStrideOpInterface a, OffsetSizeAndStrideOpInterface b,
    llvm::function_ref<bool(OpFoldResult, OpFoldResult)> cmp) {
  if (a.static_offsets().size() != b.static_offsets().size())
    return false;
  if (a.static_sizes().size() != b.static_sizes().size())
    return false;
  if (a.static_strides().size() != b.static_strides().size())
    return false;
  for (auto it : llvm::zip(a.getMixedOffsets(), b.getMixedOffsets()))
    if (!cmp(std::get<0>(it), std::get<1>(it)))
      return false;
  for (auto it : llvm::zip(a.getMixedSizes(), b.getMixedSizes()))
    if (!cmp(std::get<0>(it), std::get<1>(it)))
      return false;
  for (auto it : llvm::zip(a.getMixedStrides(), b.getMixedStrides()))
    if (!cmp(std::get<0>(it), std::get<1>(it)))
      return false;
  return true;
}

SmallVector<OpFoldResult, 4>
mlir::getMixedOffsets(OffsetSizeAndStrideOpInterface op,
                      ArrayAttr staticOffsets, ValueRange offsets) {
  return getMixedValues(staticOffsets, offsets, op.getDynamicOffsetIndicator());
}

SmallVector<OpFoldResult, 4>
mlir::getMixedSizes(OffsetSizeAndStrideOpInterface op, ArrayAttr staticSizes,
                    ValueRange sizes) {
  return getMixedValues(staticSizes, sizes, op.getDynamicSizeIndicator());
}

SmallVector<OpFoldResult, 4>
mlir::getMixedStrides(OffsetSizeAndStrideOpInterface op,
                      ArrayAttr staticStrides, ValueRange strides) {
  return getMixedValues(staticStrides, strides, op.getDynamicStrideIndicator());
}

std::pair<ArrayAttr, SmallVector<Value>> mlir::decomposeMixedStridesOrOffsets(
    OpBuilder &b, const SmallVectorImpl<OpFoldResult> &mixedValues) {
  return decomposeMixedValues(b, mixedValues,
                              ShapedType::kDynamicStrideOrOffset);
}

std::pair<ArrayAttr, SmallVector<Value>>
mlir::decomposeMixedSizes(OpBuilder &b,
                          const SmallVectorImpl<OpFoldResult> &mixedValues) {
  return decomposeMixedValues(b, mixedValues, ShapedType::kDynamicSize);
}
