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
    Operation *op, StringRef name, unsigned expectedNumElements, ArrayAttr attr,
    ValueRange values, llvm::function_ref<bool(int64_t)> isDynamic) {
  /// Check static and dynamic offsets/sizes/strides breakdown.
  if (attr.size() != expectedNumElements)
    return op->emitError("expected ")
           << expectedNumElements << " " << name << " values";
  unsigned expectedNumDynamicEntries =
      llvm::count_if(attr.getValue(), [&](Attribute attr) {
        return isDynamic(attr.cast<IntegerAttr>().getInt());
      });
  if (values.size() != expectedNumDynamicEntries)
    return op->emitError("expected ")
           << expectedNumDynamicEntries << " dynamic " << name << " values";
  return success();
}

LogicalResult mlir::verify(OffsetSizeAndStrideOpInterface op) {
  std::array<unsigned, 3> ranks = op.getArrayAttrRanks();
  if (failed(verifyListOfOperandsOrIntegers(
          op, "offset", ranks[0], op.static_offsets(), op.offsets(),
          ShapedType::isDynamicStrideOrOffset)))
    return failure();
  if (failed(verifyListOfOperandsOrIntegers(op, "size", ranks[1],
                                            op.static_sizes(), op.sizes(),
                                            ShapedType::isDynamic)))
    return failure();
  if (failed(verifyListOfOperandsOrIntegers(
          op, "stride", ranks[2], op.static_strides(), op.strides(),
          ShapedType::isDynamicStrideOrOffset)))
    return failure();
  return success();
}

void mlir::printListOfOperandsOrIntegers(
    OpAsmPrinter &p, ValueRange values, ArrayAttr arrayAttr,
    llvm::function_ref<bool(int64_t)> isDynamic) {
  p << '[';
  unsigned idx = 0;
  llvm::interleaveComma(arrayAttr, p, [&](Attribute a) {
    int64_t val = a.cast<IntegerAttr>().getInt();
    if (isDynamic(val))
      p << values[idx++];
    else
      p << val;
  });
  p << ']';
}

void mlir::printOffsetsSizesAndStrides(OpAsmPrinter &p,
                                       OffsetSizeAndStrideOpInterface op,
                                       StringRef offsetPrefix,
                                       StringRef sizePrefix,
                                       StringRef stridePrefix,
                                       ArrayRef<StringRef> elidedAttrs) {
  p << offsetPrefix;
  printListOfOperandsOrIntegers(p, op.offsets(), op.static_offsets(),
                                ShapedType::isDynamicStrideOrOffset);
  p << sizePrefix;
  printListOfOperandsOrIntegers(p, op.sizes(), op.static_sizes(),
                                ShapedType::isDynamic);
  p << stridePrefix;
  printListOfOperandsOrIntegers(p, op.strides(), op.static_strides(),
                                ShapedType::isDynamicStrideOrOffset);
  p.printOptionalAttrDict(op.getAttrs(), elidedAttrs);
}

ParseResult mlir::parseListOfOperandsOrIntegers(
    OpAsmParser &parser, OperationState &result, StringRef attrName,
    int64_t dynVal, SmallVectorImpl<OpAsmParser::OperandType> &ssa) {
  if (failed(parser.parseLSquare()))
    return failure();
  // 0-D.
  if (succeeded(parser.parseOptionalRSquare())) {
    result.addAttribute(attrName, parser.getBuilder().getArrayAttr({}));
    return success();
  }

  SmallVector<int64_t, 4> attrVals;
  while (true) {
    OpAsmParser::OperandType operand;
    auto res = parser.parseOptionalOperand(operand);
    if (res.hasValue() && succeeded(res.getValue())) {
      ssa.push_back(operand);
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

  auto arrayAttr = parser.getBuilder().getI64ArrayAttr(attrVals);
  result.addAttribute(attrName, arrayAttr);
  return success();
}

ParseResult mlir::parseOffsetsSizesAndStrides(
    OpAsmParser &parser, OperationState &result, ArrayRef<int> segmentSizes,
    llvm::function_ref<ParseResult(OpAsmParser &)> parseOptionalOffsetPrefix,
    llvm::function_ref<ParseResult(OpAsmParser &)> parseOptionalSizePrefix,
    llvm::function_ref<ParseResult(OpAsmParser &)> parseOptionalStridePrefix) {
  return parseOffsetsSizesAndStrides(
      parser, result, segmentSizes, nullptr, parseOptionalOffsetPrefix,
      parseOptionalSizePrefix, parseOptionalStridePrefix);
}

ParseResult mlir::parseOffsetsSizesAndStrides(
    OpAsmParser &parser, OperationState &result, ArrayRef<int> segmentSizes,
    llvm::function_ref<ParseResult(OpAsmParser &, OperationState &)>
        preResolutionFn,
    llvm::function_ref<ParseResult(OpAsmParser &)> parseOptionalOffsetPrefix,
    llvm::function_ref<ParseResult(OpAsmParser &)> parseOptionalSizePrefix,
    llvm::function_ref<ParseResult(OpAsmParser &)> parseOptionalStridePrefix) {
  SmallVector<OpAsmParser::OperandType, 4> offsetsInfo, sizesInfo, stridesInfo;
  auto indexType = parser.getBuilder().getIndexType();
  if ((parseOptionalOffsetPrefix && parseOptionalOffsetPrefix(parser)) ||
      parseListOfOperandsOrIntegers(
          parser, result,
          OffsetSizeAndStrideOpInterface::getStaticOffsetsAttrName(),
          ShapedType::kDynamicStrideOrOffset, offsetsInfo) ||
      (parseOptionalSizePrefix && parseOptionalSizePrefix(parser)) ||
      parseListOfOperandsOrIntegers(
          parser, result,
          OffsetSizeAndStrideOpInterface::getStaticSizesAttrName(),
          ShapedType::kDynamicSize, sizesInfo) ||
      (parseOptionalStridePrefix && parseOptionalStridePrefix(parser)) ||
      parseListOfOperandsOrIntegers(
          parser, result,
          OffsetSizeAndStrideOpInterface::getStaticStridesAttrName(),
          ShapedType::kDynamicStrideOrOffset, stridesInfo))
    return failure();
  // Add segment sizes to result
  SmallVector<int, 4> segmentSizesFinal(segmentSizes.begin(),
                                        segmentSizes.end());
  segmentSizesFinal.append({static_cast<int>(offsetsInfo.size()),
                            static_cast<int>(sizesInfo.size()),
                            static_cast<int>(stridesInfo.size())});
  result.addAttribute(
      OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr(),
      parser.getBuilder().getI32VectorAttr(segmentSizesFinal));
  return failure(
      (preResolutionFn && preResolutionFn(parser, result)) ||
      parser.resolveOperands(offsetsInfo, indexType, result.operands) ||
      parser.resolveOperands(sizesInfo, indexType, result.operands) ||
      parser.resolveOperands(stridesInfo, indexType, result.operands));
}
