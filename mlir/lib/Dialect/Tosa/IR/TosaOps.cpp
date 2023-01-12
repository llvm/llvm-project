//===- TosaOps.cpp - MLIR Dialect for TOSA --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// This file implements the TOSA Specification:
// https://developer.mlplatform.org/w/tosa/
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
#include "mlir/Dialect/Tosa/Utils/ShapeUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::tosa;

#include "mlir/Dialect/Tosa/IR/TosaOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Tosa dialect interface includes.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaInterfaces.cpp.inc"

namespace {
//===----------------------------------------------------------------------===//
// Dialect Function Inliner Interface.
//===----------------------------------------------------------------------===//
struct TosaInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks.
  //===--------------------------------------------------------------------===//

  /// All operations can be inlined by default.
  bool isLegalToInline(Operation *op, Region *region, bool wouldBeCloned,
                       BlockAndValueMapping &map) const final {
    return true;
  }

  /// All regions with If and While parent operators can be inlined.
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       BlockAndValueMapping &map) const final {
    return (isa<tosa::IfOp>(dest->getParentOp()) ||
            isa<tosa::WhileOp>(dest->getParentOp()));
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// TOSA control flow support.
//===----------------------------------------------------------------------===//

/// Returns the while loop body.
Region &tosa::WhileOp::getLoopBody() { return getBody(); }

//===----------------------------------------------------------------------===//
// Tosa dialect initialization.
//===----------------------------------------------------------------------===//

void TosaDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Tosa/IR/TosaOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/Tosa/IR/TosaAttributes.cpp.inc"
      >();
  addInterfaces<TosaInlinerInterface>();
}

Operation *TosaDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  // Tosa dialect constants only support ElementsAttr unlike standard dialect
  // constant which supports all attributes.
  if (value.isa<ElementsAttr>())
    return builder.create<tosa::ConstOp>(loc, type, value.cast<ElementsAttr>());
  return nullptr;
}

//===----------------------------------------------------------------------===//
// TOSA Operator Verifiers.
//===----------------------------------------------------------------------===//

template <typename T> static LogicalResult verifyConvOp(T op) {
  // All TOSA conv ops have an input() and weight().
  auto inputType =
      op.getInput().getType().template dyn_cast<RankedTensorType>();
  auto weightType =
      op.getWeight().getType().template dyn_cast<RankedTensorType>();

  // Must be ranked tensor types
  if (!inputType) {
    op.emitOpError("expect a ranked tensor for input, got ") << op.getInput();
    return failure();
  }
  if (!weightType) {
    op.emitOpError("expect a ranked tensor for weight, got ") << op.getWeight();
    return failure();
  }

  auto inputEType = inputType.getElementType();
  auto weightEType = weightType.getElementType();

  bool inputIsQuant = !inputEType.template isa<FloatType>();
  bool weightIsQuant = !weightEType.template isa<FloatType>();

  // Either both must be quantized or both unquantized.
  if (inputIsQuant != weightIsQuant) {
    op.emitOpError(
        "expect both input and weight to be float or not together, got ")
        << inputEType << " and " << weightEType;
    return failure();
  }

  // Quantized type must have constructed the quantizationattr, and unquantized
  // types should not have a quantizationattr.
  if ((inputIsQuant && !op.getQuantizationInfo()) ||
      (!inputIsQuant && op.getQuantizationInfo())) {
    op.emitOpError("quantizationattr is required for quantized type, and not "
                   "allowed for float type");
    return failure();
  }

  return success();
}

LogicalResult tosa::AvgPool2dOp::verify() {
  auto inputETy = getInput().getType().cast<ShapedType>().getElementType();
  auto resultETy = getType().cast<ShapedType>().getElementType();

  if (auto quantType = inputETy.dyn_cast<mlir::quant::UniformQuantizedType>())
    inputETy = quantType.getStorageType();

  if (auto quantType = resultETy.dyn_cast<mlir::quant::UniformQuantizedType>())
    resultETy = quantType.getStorageType();

  if (inputETy.isF32() && resultETy.isF32())
    return success();
  if (inputETy.isInteger(8) && resultETy.isInteger(8))
    return success();
  if (inputETy.isInteger(16) && resultETy.isInteger(16))
    return success();

  return emitOpError("input/output element types are incompatible.");
}

//===----------------------------------------------------------------------===//
// TOSA Operator Quantization Builders.
//===----------------------------------------------------------------------===//

/// This builder is called on all convolution operators except TransposeConv,
/// which has specialized output shape semantics. The builder also defines the
/// bitwidth of the output given the bit width of the input & weight content.
static void buildConvOpWithQuantInfo(OpBuilder &builder, OperationState &result,
                                     Type outputType, Value input, Value weight,
                                     Value bias, DenseI64ArrayAttr pad,
                                     DenseI64ArrayAttr stride,
                                     DenseI64ArrayAttr dilation) {

  result.addOperands({input, weight, bias});
  result.addAttribute("pad", pad);
  result.addAttribute("stride", stride);
  result.addAttribute("dilation", dilation);

  auto quantAttr = buildConvOpQuantizationAttr(builder, input, weight);
  if (quantAttr) {
    result.addAttribute("quantization_info", quantAttr);
    result.addTypes(
        buildConvOpResultTypeInfo(builder, outputType, input, weight));
  } else {
    result.addTypes(outputType);
  }
}

/// Handles tosa.transpose_conv2d which has outpad and output shape attributes.
static void buildTransConvOpWithQuantInfo(
    OpBuilder &builder, OperationState &result, Type outputType, Value input,
    Value weight, Value bias, DenseI64ArrayAttr outpad,
    DenseI64ArrayAttr stride, DenseI64ArrayAttr outputShape) {
  result.addOperands({input, weight, bias});
  result.addAttribute("out_pad", outpad);
  result.addAttribute("stride", stride);
  result.addAttribute("out_shape", outputShape);
  auto quantAttr = ::buildConvOpQuantizationAttr(builder, input, weight);

  if (quantAttr) {
    result.addAttribute("quantization_info", quantAttr);
    result.addTypes(
        buildConvOpResultTypeInfo(builder, outputType, input, weight));
  } else {
    result.addTypes(outputType);
  }
}

/// The tosa.fully_connected op has its own builder as it does not have
/// strides/dilation/padding.
static void buildFCOpWithQuantInfo(OpBuilder &builder, OperationState &result,
                                   Type outputType, Value input, Value weight,
                                   Value bias) {

  result.addOperands({input, weight, bias});
  auto quantAttr = ::buildConvOpQuantizationAttr(builder, input, weight);
  if (quantAttr) {
    result.addAttribute("quantization_info", quantAttr);
    result.addTypes(
        buildConvOpResultTypeInfo(builder, outputType, input, weight));
  } else {
    result.addTypes(outputType);
  }
}

/// The tosa.matmul op is also intended to be generated where a fully_connected
/// op must be constructed where the weight is not a constant. In this case,
/// the fully_connected op must be expressed using matmul.
/// TODO: Add link to the leglization document explaining this.
static void buildMatMulOpWithQuantInfo(OpBuilder &builder,
                                       OperationState &result, Type outputType,
                                       Value a, Value b) {
  result.addOperands({a, b});
  auto quantAttr = ::buildMatMulOpQuantizationAttr(builder, a, b);

  if (quantAttr) {
    result.addAttribute("quantization_info", quantAttr);

    auto inputType = a.getType().dyn_cast<ShapedType>();
    assert(inputType && "Input must be a shaped tensor type!");

    auto inputQType = inputType.getElementType()
                          .dyn_cast<mlir::quant::UniformQuantizedType>();
    assert(inputQType && "Tensor must have quantized datatype!");

    unsigned inputBits = inputQType.getStorageTypeIntegralWidth();

    auto outputShapedType = outputType.dyn_cast<ShapedType>();
    assert(outputShapedType && "Output must be a shaped type");

    IntegerType accElementType;
    if (inputBits == 16)
      accElementType = builder.getIntegerType(48);
    else
      accElementType = builder.getI32Type();
    auto accType = outputShapedType.clone(accElementType);
    result.addTypes(accType);
  } else {
    result.addTypes(outputType);
  }
}

/// Both the tosa.avg_pool2d and unary ops use the same UnaruOpQuantizationAttr
/// but avg_pool operator has its own builder as it has additional parameters
/// not part of the unary ops.
static void buildAvgPool2dOpWithQuantInfo(
    OpBuilder &builder, OperationState &result, Type outputType, Value input,
    DenseArrayAttr kernel, DenseArrayAttr stride, DenseArrayAttr pad) {
  result.addOperands(input);
  result.addAttribute("kernel", kernel);
  result.addAttribute("stride", stride);
  result.addAttribute("pad", pad);
  auto quantAttr = buildUnaryOpQuantizationAttr(builder, input, outputType);
  if (quantAttr)
    result.addAttribute("quantization_info", quantAttr);
  result.types.push_back(outputType);
}

/// This builder is called on single-parameter unary operators that have scale
/// relationship between their input and output, expressed by the
/// UnaryOpQuantizationAttr.
static void buildUnaryOpWithQuantInfo(OpBuilder &builder,
                                      OperationState &result, Type outputType,
                                      Value input) {
  result.addOperands(input);
  auto quantAttr = buildUnaryOpQuantizationAttr(builder, input, outputType);
  if (quantAttr)
    result.addAttribute("quantization_info", quantAttr);
  result.types.push_back(outputType);
}

/// This builder is called on TOSA pad operator that needs to create its own
/// OptionalAttr quantization_attr parameter to scale the padding values
/// correctly. No pad_const is interpreted as zero-padding.
static void buildPadOpWithQuantInfo(OpBuilder &builder, OperationState &result,
                                    Type outputType, Value input,
                                    Value paddings) {
  result.addOperands({input, paddings});
  auto quantAttr = buildPadOpQuantizationAttr(builder, input);
  if (quantAttr)
    result.addAttribute("quantization_info", quantAttr);
  result.types.push_back(outputType);
}

/// This builder is called on TOSA pad operator when an explicit pad_const
/// value is passed in. It also optionally constructs quantization_attr.
static void buildExplicitValuePadOpWithQuantInfo(OpBuilder &builder,
                                                 OperationState &result,
                                                 Type outputType, Value input,
                                                 Value paddings,
                                                 Value padConst) {
  result.addOperands({input, paddings, padConst});
  auto quantAttr = buildPadOpQuantizationAttr(builder, input);
  if (quantAttr)
    result.addAttribute("quantization_info", quantAttr);
  result.types.push_back(outputType);
}

//===----------------------------------------------------------------------===//
// TOSA Operator Return Type Inference.
//===----------------------------------------------------------------------===//

static LogicalResult resolveBroadcastShape(const ValueShapeRange &operands,
                                           SmallVector<int64_t> &outShape) {
  int64_t outRank = 0;
  for (int i = 0, e = operands.size(); i != e; ++i) {
    auto shape = operands.getShape(i);
    if (!shape.hasRank()) {
      // TODO(jennik): Update function to have better case handling for invalid
      // operands and for ranked tensors.
      return failure();
    }
    outRank = std::max<int64_t>(outRank, shape.getRank());
  }

  outShape.resize(outRank, 1);

  for (int i = 0, e = operands.size(); i != e; ++i) {
    auto shape = operands.getShape(i);
    auto rankDiff = outShape.size() - shape.getRank();

    for (size_t i = 0, e = shape.getRank(); i < e; ++i) {
      auto dim1 = outShape[i + rankDiff];
      auto dim2 = shape.getDimSize(i);
      auto resolvedDim = dim1;

      if (dim1 == 1) {
        resolvedDim = dim2;
      } else if (dim2 == 1) {
        resolvedDim = dim1;
      } else if (dim1 != dim2) {
        return failure();
      }
      outShape[i + rankDiff] = resolvedDim;
    }
  }

  return success();
}

LogicalResult tosa::ArgMaxOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapeAdaptor inputShape = operands.getShape(0);
  IntegerAttr axis = attributes.get("axis").cast<IntegerAttr>();
  int32_t axisVal = axis.getValue().getSExtValue();

  if (!inputShape.hasRank()) {
    inferredReturnShapes.push_back(ShapedTypeComponents());
    return success();
  }

  SmallVector<int64_t> outShape;
  outShape.reserve(inputShape.getRank() - 1);
  for (int i = 0, s = inputShape.getRank(); i < s; i++) {
    if (i == axisVal)
      continue;
    outShape.push_back(inputShape.getDimSize(i));
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outShape));
  return success();
}

LogicalResult tosa::ConcatOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  // Infer all dimension sizes by reducing based on inputs.
  int32_t axis =
      attributes.get("axis").cast<IntegerAttr>().getValue().getSExtValue();
  llvm::SmallVector<int64_t> outputShape;
  bool hasRankedInput = false;
  for (auto operand : operands) {
    ShapeAdaptor operandShape = operands.getShape(operand);
    if (!operandShape.hasRank())
      continue;

    // Copy the Operand's rank.
    if (!hasRankedInput)
      outputShape.resize(operandShape.getRank(), ShapedType::kDynamic);

    // Copy shapes until the dim is non-dynamic.
    for (int i = 0, s = operandShape.getRank(); i < s; i++) {
      if (i == axis || operandShape.isDynamicDim(i))
        continue;
      if (outputShape[i] == ShapedType::kDynamic)
        outputShape[i] = operandShape.getDimSize(i);
      if (outputShape[i] != operandShape.getDimSize(i))
        return failure();
    }

    hasRankedInput = true;
  }

  if (!hasRankedInput) {
    inferredReturnShapes.push_back(ShapedTypeComponents());
    return success();
  }

  // Determine the dimension size along the concatenation axis.
  int64_t concatDimSize = 0;
  for (auto operand : operands) {
    ShapeAdaptor operandShape = operands.getShape(operand);

    // We need to know the length of the concatenation axis of all inputs to
    // determine the dimension size of the output shape.
    if (!operandShape.hasRank() || operandShape.isDynamicDim(axis)) {
      concatDimSize = ShapedType::kDynamic;
      break;
    }

    concatDimSize += operandShape.getDimSize(axis);
  }

  outputShape[axis] = concatDimSize;

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult tosa::EqualOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<int64_t> outShape;
  if (resolveBroadcastShape(operands, outShape).failed()) {
    inferredReturnShapes.push_back(ShapedTypeComponents());
    return success();
  }

  inferredReturnShapes.push_back(
      ShapedTypeComponents(outShape, IntegerType::get(context, /*width=*/1)));
  return success();
}

bool tosa::EqualOp::isCompatibleReturnTypes(TypeRange l, TypeRange r) {
  if (l.size() != r.size() || l.size() != 1)
    return false;
  return succeeded(verifyCompatibleShape(l[0], r[0]));
}

LogicalResult tosa::FullyConnectedOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapeAdaptor inputShape = operands.getShape(0);
  ShapeAdaptor weightShape = operands.getShape(1);
  ShapeAdaptor biasShape = operands.getShape(2);

  // All shapes are dynamic.
  SmallVector<int64_t> outShape;
  outShape.resize(2, ShapedType::kDynamic);

  if (inputShape.hasRank()) {
    outShape[0] = inputShape.getDimSize(0);
  }

  if (weightShape.hasRank()) {
    outShape[1] = weightShape.getDimSize(0);
  }

  if (biasShape.hasRank()) {
    outShape[1] = outShape[1] == ShapedType::kDynamic ? biasShape.getDimSize(0)
                                                      : outShape[1];
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outShape));
  return success();
}

LogicalResult FullyConnectedOp::verify() { return verifyConvOp(*this); }

LogicalResult tosa::MatMulOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapeAdaptor lhsShape = operands.getShape(0);
  ShapeAdaptor rhsShape = operands.getShape(1);

  // All shapes are dynamic.
  SmallVector<int64_t> outShape;
  outShape.resize(3, ShapedType::kDynamic);

  if (lhsShape.hasRank()) {
    outShape[0] = lhsShape.getDimSize(0);
    outShape[1] = lhsShape.getDimSize(1);
  }

  if (rhsShape.hasRank()) {
    outShape[0] = outShape[0] == ShapedType::kDynamic ? rhsShape.getDimSize(0)
                                                      : outShape[0];
    outShape[2] = rhsShape.getDimSize(2);
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outShape));
  return success();
}

LogicalResult tosa::PadOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapeAdaptor inputShape = operands.getShape(0);
  ShapeAdaptor paddingShape = operands.getShape(1);
  SmallVector<int64_t> outputShape;

  // If both inputs have unknown shape, we cannot determine the shape of the
  // output.
  if (!inputShape.hasRank() && !paddingShape.hasRank()) {
    inferredReturnShapes.push_back(ShapedTypeComponents());
    return success();
  }

  // If the input rank is unknown we can info the output rank using the padding
  // shape's first dim.
  if (!inputShape.hasRank()) {
    if (paddingShape.isDynamicDim(0)) {
      inferredReturnShapes.push_back(ShapedTypeComponents());
      return success();
    }

    outputShape.resize(paddingShape.getDimSize(0), ShapedType::kDynamic);
    inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
    return success();
  }

  DenseIntElementsAttr paddings;
  // If the paddings value is not a constant, all dimensions must be dynamic.
  if (!matchPattern(operands[1], m_Constant(&paddings))) {
    outputShape.resize(inputShape.getRank(), ShapedType::kDynamic);
    inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
    return success();
  }

  SmallVector<int64_t> paddingValues;
  for (auto val : paddings) {
    paddingValues.push_back(val.getSExtValue());
  }

  outputShape.reserve(inputShape.getRank());
  for (int i = 0, s = inputShape.getRank(); i < s; i++) {
    if (inputShape.isDynamicDim(i)) {
      outputShape.push_back(ShapedType::kDynamic);
      continue;
    }

    outputShape.push_back(inputShape.getDimSize(i) + paddingValues[i * 2] +
                          paddingValues[i * 2 + 1]);
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

static SmallVector<int64_t> convertToMlirShape(ArrayRef<int64_t> shape) {
  return to_vector(llvm::map_range(shape, [](int64_t dim) {
    return dim == -1 ? ShapedType::kDynamic : dim;
  }));
}

LogicalResult tosa::SliceOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  inferredReturnShapes.push_back(ShapedTypeComponents(
      convertToMlirShape(SliceOpAdaptor(operands, attributes).getSize())));
  return success();
}

LogicalResult tosa::TableOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapeAdaptor inputShape = operands.getShape(0);

  if (!inputShape.hasRank()) {
    inferredReturnShapes.push_back(ShapedTypeComponents());
    return success();
  }

  inferredReturnShapes.resize(1);
  inputShape.getDims(inferredReturnShapes[0]);
  return success();
}

LogicalResult tosa::TileOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  TileOpAdaptor adaptor(operands, attributes);
  ArrayRef<int64_t> multiples = adaptor.getMultiples();
  ShapeAdaptor inputShape = operands.getShape(0);
  SmallVector<int64_t> outputShape;
  if (!inputShape.hasRank()) {
    outputShape.resize(multiples.size(), ShapedType::kDynamic);
    inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
    return success();
  }

  // Any non dynamic dimension can be multiplied to a known size.
  outputShape.reserve(multiples.size());
  for (int i = 0, s = inputShape.getRank(); i < s; i++) {
    int64_t dim = inputShape.getDimSize(i);
    if (dim != ShapedType::kDynamic)
      dim *= multiples[i];
    outputShape.push_back(dim);
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult tosa::ReshapeOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ReshapeOpAdaptor adaptor(operands, attributes);
  ShapeAdaptor inputShape = operands.getShape(0);
  llvm::SmallVector<int64_t> newShapeValue =
      convertToMlirShape(adaptor.getNewShape());

  // We cannot infer from the total number of elements so we must take the
  // shape attribute as exact.
  if (!inputShape.hasRank() || !inputShape.hasStaticShape()) {
    inferredReturnShapes.push_back(ShapedTypeComponents(newShapeValue));
    return success();
  }

  // Determine the number of elements covered by the slice of all static
  // dimensions. This allows us to infer the length of the remaining dynamic
  // dimension.
  int64_t numElements = inputShape.getNumElements();
  int64_t staticMul = 1;
  for (auto val : newShapeValue) {
    if (!ShapedType::isDynamic(val)) {
      staticMul *= val;
    }
  }

  // Determine the length of the dynamic dimension.
  for (auto &val : newShapeValue) {
    if (ShapedType::isDynamic(val))
      val = numElements / staticMul;
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(newShapeValue));
  return success();
}

mlir::LogicalResult tosa::ReshapeOp::verify() {
  ShapedType inputType = getInput1().getType().cast<ShapedType>();
  ShapedType outputType = getType().cast<ShapedType>();

  if (inputType.hasStaticShape() && outputType.hasStaticShape()) {
    int64_t inputElementsNum = inputType.getNumElements();
    int64_t outputElementsNum = outputType.getNumElements();
    if (inputElementsNum != outputElementsNum) {
      return emitOpError() << "Cannot reshape " << inputElementsNum
                           << " elements into " << outputElementsNum;
    }
  }
  return mlir::success();
}

LogicalResult tosa::TransposeOp::getConstantPerms(SmallVector<int64_t> &perms) {
  // Perms must be constants.
  DenseIntElementsAttr permsAttr;
  if (!matchPattern(getPerms(), m_Constant(&permsAttr)))
    return failure();

  // Transpose is not the identity transpose.
  perms = llvm::to_vector(
      llvm::map_range(permsAttr.getValues<APInt>(),
                      [](const APInt &val) { return val.getSExtValue(); }));

  return success();
}

LogicalResult tosa::TransposeOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapeAdaptor inputShape = operands.getShape(0);
  ShapeAdaptor permsShape = operands.getShape(1);

  // If input rank and permutation length is unknown, the output rank is
  // unknown.
  if (!inputShape.hasRank() || !permsShape.hasRank() ||
      permsShape.isDynamicDim(0)) {
    inferredReturnShapes.push_back(ShapedTypeComponents());
    return success();
  }

  // This would imply the number of permutations does not match the rank of the
  // input which is illegal.
  if (permsShape.getDimSize(0) != inputShape.getRank()) {
    return failure();
  }

  // Without the input dims we cannot determine the output dim sizes but we
  // can determine the output rank.
  SmallVector<int64_t> outputShape;
  if (!inputShape.hasRank()) {
    outputShape.resize(permsShape.getDimSize(0), ShapedType::kDynamic);
    inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
    return success();
  }

  // Rank-0 means no permutations matter.
  if (inputShape.getRank() == 0) {
    inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
    return success();
  }

  // Check whether the input dimensions are all the same.
  bool allTheSame = true;
  for (int i = 1, s = inputShape.getRank(); i < s; i++) {
    if (inputShape.getDimSize(0) != inputShape.getDimSize(i)) {
      allTheSame = false;
      break;
    }
  }

  // If all of the input dimensions are the same we don't care about the
  // permutation.
  if (allTheSame) {
    outputShape.resize(inputShape.getRank(), inputShape.getDimSize(0));
    inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
    return success();
  }

  outputShape.resize(inputShape.getRank(), ShapedType::kDynamic);
  // If the permuations are a constant we can directly determine the output
  // shape.
  if (ShapeAdaptor permShape = operands.getValueAsShape(1)) {
    outputShape.reserve(inputShape.getRank());
    for (int i = 0, s = inputShape.getRank(); i < s; i++) {
      outputShape[i] = inputShape.getDimSize(permShape.getDimSize(i));
    }
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult tosa::GatherOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<int64_t> outputShape;
  outputShape.resize(3, ShapedType::kDynamic);

  ShapeAdaptor valuesShape = operands.getShape(0);
  if (valuesShape.hasRank()) {
    outputShape[0] = valuesShape.getDimSize(0);
    outputShape[2] = valuesShape.getDimSize(2);
  }

  ShapeAdaptor indicesShape = operands.getShape(1);
  if (indicesShape.hasRank()) {
    if (outputShape[0] == ShapedType::kDynamic)
      outputShape[0] = indicesShape.getDimSize(0);
    if (outputShape[1] == ShapedType::kDynamic)
      outputShape[1] = indicesShape.getDimSize(1);
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult tosa::ResizeOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ResizeOpAdaptor adaptor(operands, attributes);
  llvm::SmallVector<int64_t, 4> outputShape;
  outputShape.resize(4, ShapedType::kDynamic);

  ShapeAdaptor inputShape = operands.getShape(adaptor.getInput());
  if (!inputShape.hasRank())
    return failure();

  outputShape[0] = inputShape.getDimSize(0);
  outputShape[3] = inputShape.getDimSize(3);
  int64_t inputHeight = inputShape.getDimSize(1);
  int64_t inputWidth = inputShape.getDimSize(2);

  if ((inputHeight == ShapedType::kDynamic) ||
      (inputWidth == ShapedType::kDynamic))
    return failure();

  llvm::ArrayRef<int64_t> scaleInt = adaptor.getScale();
  llvm::ArrayRef<int64_t> offsetInt = adaptor.getOffset();
  llvm::ArrayRef<int64_t> borderInt = adaptor.getBorder();

  // Compute the output shape based on attributes: scale, offset, and border.
  outputShape[1] =
      (((inputHeight - 1) * scaleInt[0] - offsetInt[0] + borderInt[0]) /
       scaleInt[1]) +
      1;

  outputShape[2] =
      (((inputWidth - 1) * scaleInt[2] - offsetInt[1] + borderInt[1]) /
       scaleInt[3]) +
      1;

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult tosa::ScatterOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<int64_t> outputShape;
  outputShape.resize(3, ShapedType::kDynamic);

  ShapeAdaptor valuesInShape = operands.getShape(0);
  if (valuesInShape.hasRank()) {
    outputShape[0] = valuesInShape.getDimSize(0);
    outputShape[1] = valuesInShape.getDimSize(1);
    outputShape[2] = valuesInShape.getDimSize(2);
  }

  ShapeAdaptor indicesShape = operands.getShape(1);
  if (indicesShape.hasRank()) {
    if (outputShape[0] == ShapedType::kDynamic)
      outputShape[0] = indicesShape.getDimSize(0);
  }

  ShapeAdaptor inputShape = operands.getShape(2);
  if (inputShape.hasRank()) {
    if (outputShape[0] == ShapedType::kDynamic)
      outputShape[0] = inputShape.getDimSize(0);
    if (outputShape[2] == ShapedType::kDynamic)
      outputShape[2] = inputShape.getDimSize(2);
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

static LogicalResult ReduceInferReturnTypes(
    ShapeAdaptor operandShape, IntegerAttr axis,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  if (!operandShape.hasRank()) {
    inferredReturnShapes.push_back(ShapedTypeComponents());
    return success();
  }

  SmallVector<int64_t> outputShape;
  operandShape.getDims(outputShape);
  int64_t axisVal = axis.getValue().getSExtValue();
  outputShape[axisVal] = 1;
  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

#define REDUCE_SHAPE_INFER(OP)                                                 \
  LogicalResult OP::inferReturnTypeComponents(                                 \
      MLIRContext *context, ::std::optional<Location> location,                \
      ValueShapeRange operands, DictionaryAttr attributes,                     \
      RegionRange regions,                                                     \
      SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {           \
    return ReduceInferReturnTypes(operands.getShape(0),                        \
                                  attributes.get("axis").cast<IntegerAttr>(),  \
                                  inferredReturnShapes);                       \
  }

REDUCE_SHAPE_INFER(tosa::ReduceAllOp)
REDUCE_SHAPE_INFER(tosa::ReduceAnyOp)
REDUCE_SHAPE_INFER(tosa::ReduceMaxOp)
REDUCE_SHAPE_INFER(tosa::ReduceMinOp)
REDUCE_SHAPE_INFER(tosa::ReduceProdOp)
REDUCE_SHAPE_INFER(tosa::ReduceSumOp)
#undef REDUCE_SHAPE_INFER

static LogicalResult NAryInferReturnTypes(
    const ValueShapeRange &operands,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<int64_t> outShape;
  if (resolveBroadcastShape(operands, outShape).failed()) {
    inferredReturnShapes.push_back(ShapedTypeComponents());
  } else {
    inferredReturnShapes.push_back(ShapedTypeComponents(outShape));
  }
  return success();
}

#define NARY_SHAPE_INFER(OP)                                                   \
  LogicalResult OP::inferReturnTypeComponents(                                 \
      MLIRContext *context, ::std::optional<Location> location,                \
      ValueShapeRange operands, DictionaryAttr attributes,                     \
      RegionRange regions,                                                     \
      SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {           \
    return NAryInferReturnTypes(operands, inferredReturnShapes);               \
  }

NARY_SHAPE_INFER(tosa::AbsOp)
NARY_SHAPE_INFER(tosa::AddOp)
NARY_SHAPE_INFER(tosa::ArithmeticRightShiftOp)
NARY_SHAPE_INFER(tosa::BitwiseAndOp)
NARY_SHAPE_INFER(tosa::BitwiseOrOp)
NARY_SHAPE_INFER(tosa::BitwiseXorOp)
NARY_SHAPE_INFER(tosa::BitwiseNotOp)
NARY_SHAPE_INFER(tosa::CastOp)
NARY_SHAPE_INFER(tosa::CeilOp)
NARY_SHAPE_INFER(tosa::ClampOp)
NARY_SHAPE_INFER(tosa::ClzOp)
NARY_SHAPE_INFER(tosa::DivOp)
NARY_SHAPE_INFER(tosa::ExpOp)
NARY_SHAPE_INFER(tosa::FloorOp)
NARY_SHAPE_INFER(tosa::GreaterEqualOp)
NARY_SHAPE_INFER(tosa::GreaterOp)
NARY_SHAPE_INFER(tosa::IdentityOp)
NARY_SHAPE_INFER(tosa::LogOp)
NARY_SHAPE_INFER(tosa::LogicalAndOp)
NARY_SHAPE_INFER(tosa::LogicalLeftShiftOp)
NARY_SHAPE_INFER(tosa::LogicalNotOp)
NARY_SHAPE_INFER(tosa::LogicalOrOp)
NARY_SHAPE_INFER(tosa::LogicalRightShiftOp)
NARY_SHAPE_INFER(tosa::LogicalXorOp)
NARY_SHAPE_INFER(tosa::MaximumOp)
NARY_SHAPE_INFER(tosa::MinimumOp)
NARY_SHAPE_INFER(tosa::MulOp)
NARY_SHAPE_INFER(tosa::NegateOp)
NARY_SHAPE_INFER(tosa::PowOp)
NARY_SHAPE_INFER(tosa::ReciprocalOp)
NARY_SHAPE_INFER(tosa::RescaleOp)
NARY_SHAPE_INFER(tosa::ReverseOp)
NARY_SHAPE_INFER(tosa::RsqrtOp)
NARY_SHAPE_INFER(tosa::SelectOp)
NARY_SHAPE_INFER(tosa::SubOp)
NARY_SHAPE_INFER(tosa::TanhOp)
NARY_SHAPE_INFER(tosa::SigmoidOp)
#undef PRED_SHAPE_INFER

static LogicalResult poolingInferReturnTypes(
    const ValueShapeRange &operands, DictionaryAttr attributes,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapeAdaptor inputShape = operands.getShape(0);
  llvm::SmallVector<int64_t> outputShape;
  outputShape.resize(4, ShapedType::kDynamic);

  // We only know the rank if the input type is unranked.
  if (!inputShape) {
    inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
    return success();
  }

  // Batch and number of channels are identical for pooling layer.
  outputShape[0] = inputShape.getDimSize(0);
  outputShape[3] = inputShape.getDimSize(3);

  int64_t height = inputShape.getDimSize(1);
  int64_t width = inputShape.getDimSize(2);

  ArrayRef<int64_t> kernel = attributes.get("kernel").cast<DenseI64ArrayAttr>();
  ArrayRef<int64_t> stride = attributes.get("stride").cast<DenseI64ArrayAttr>();
  ArrayRef<int64_t> pad = attributes.get("pad").cast<DenseI64ArrayAttr>();

  if (!ShapedType::isDynamic(height)) {
    int64_t padded = height + pad[0] + pad[1] - kernel[0];
    outputShape[1] = padded / stride[0] + 1;
  }

  if (!ShapedType::isDynamic(width)) {
    int64_t padded = width + pad[2] + pad[3] - kernel[1];
    outputShape[2] = padded / stride[1] + 1;
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult Conv2DOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<int64_t> outputShape(4, ShapedType::kDynamic);
  Conv2DOp::Adaptor adaptor(operands.getValues(), attributes);

  int64_t inputWidth = ShapedType::kDynamic;
  int64_t inputHeight = ShapedType::kDynamic;
  int64_t weightWidth = ShapedType::kDynamic;
  int64_t weightHeight = ShapedType::kDynamic;

  // Input shape describes input width/height and batch.

  ShapeAdaptor inputShape = operands.getShape(adaptor.getInput());
  if (inputShape.hasRank()) {
    outputShape[0] = inputShape.getDimSize(0);
    inputHeight = inputShape.getDimSize(1);
    inputWidth = inputShape.getDimSize(2);
  }

  // Weight shapes describes the filter width/height and the output channels.
  ShapeAdaptor weightShape = operands.getShape(adaptor.getWeight());
  if (weightShape.hasRank()) {
    outputShape[3] = weightShape.getDimSize(0);
    weightHeight = weightShape.getDimSize(1);
    weightWidth = weightShape.getDimSize(2);
  }

  // Bias shape can describe the output channels.
  ShapeAdaptor biasShape = operands.getShape(adaptor.getBias());
  if (biasShape.hasRank()) {
    outputShape[3] = ShapedType::isDynamic(outputShape[3])
                         ? biasShape.getDimSize(0)
                         : outputShape[3];
  }

  llvm::ArrayRef<int64_t> dilation = adaptor.getDilation();
  llvm::ArrayRef<int64_t> stride = adaptor.getStride();
  llvm::ArrayRef<int64_t> padding = adaptor.getPad();

  if (!ShapedType::isDynamic(inputHeight) &&
      !ShapedType::isDynamic(weightHeight)) {
    int64_t inputSize = inputHeight + padding[0] + padding[1];
    int64_t filterSize = (weightHeight - 1) * dilation[0] + 1;
    int64_t unstridedResult = inputSize - filterSize + 1;
    outputShape[1] = (unstridedResult - 1) / stride[0] + 1;
  }

  if (!ShapedType::isDynamic(inputWidth) &&
      !ShapedType::isDynamic(weightWidth)) {
    int64_t inputSize = inputWidth + padding[2] + padding[3];
    int64_t filterSize = (weightWidth - 1) * dilation[1] + 1;
    int64_t unstridedResult = inputSize - filterSize + 1;
    outputShape[2] = (unstridedResult - 1) / stride[1] + 1;
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult Conv2DOp::verify() { return verifyConvOp(*this); }

LogicalResult Conv3DOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<int64_t> outputShape(5, ShapedType::kDynamic);
  Conv3DOp::Adaptor adaptor(operands.getValues(), attributes);

  int64_t inputWidth = ShapedType::kDynamic;
  int64_t inputHeight = ShapedType::kDynamic;
  int64_t inputDepth = ShapedType::kDynamic;

  int64_t weightWidth = ShapedType::kDynamic;
  int64_t weightHeight = ShapedType::kDynamic;
  int64_t weightDepth = ShapedType::kDynamic;

  // Input shape describes input width/height and batch.
  ShapeAdaptor inputShape = operands.getShape(adaptor.getInput());
  if (inputShape.hasRank()) {
    outputShape[0] = inputShape.getDimSize(0);
    inputDepth = inputShape.getDimSize(1);
    inputHeight = inputShape.getDimSize(2);
    inputWidth = inputShape.getDimSize(3);
  }

  // Weight shapes describes the filter width/height and the output channels.
  ShapeAdaptor weightShape = operands.getShape(adaptor.getWeight());
  if (weightShape.hasRank()) {
    outputShape[4] = weightShape.getDimSize(0);
    weightDepth = weightShape.getDimSize(1);
    weightHeight = weightShape.getDimSize(2);
    weightWidth = weightShape.getDimSize(3);
  }

  // Bias shape can describe the output channels.
  ShapeAdaptor biasShape = operands.getShape(adaptor.getBias());
  if (biasShape.hasRank() && ShapedType::isDynamic(outputShape[4])) {
    outputShape[4] = biasShape.getDimSize(0);
  }

  llvm::ArrayRef<int64_t> dilation = adaptor.getDilation();
  llvm::ArrayRef<int64_t> stride = adaptor.getStride();
  llvm::ArrayRef<int64_t> pad = adaptor.getPad();

  if (!ShapedType::isDynamic(inputDepth) &&
      !ShapedType::isDynamic(weightDepth)) {
    int32_t inputSize = inputDepth + pad[0] + pad[1];
    int32_t filterSize = (weightDepth - 1) * dilation[0] + 1;
    int32_t unstridedResult = inputSize - filterSize + 1;
    outputShape[1] = (unstridedResult - 1) / stride[0] + 1;
  }

  if (!ShapedType::isDynamic(inputHeight) &&
      !ShapedType::isDynamic(weightHeight)) {
    int32_t inputSize = inputHeight + pad[2] + pad[3];
    int32_t filterSize = (weightHeight - 1) * dilation[1] + 1;
    int32_t unstridedResult = inputSize - filterSize + 1;
    outputShape[2] = (unstridedResult - 1) / stride[1] + 1;
  }

  if (!ShapedType::isDynamic(inputWidth) &&
      !ShapedType::isDynamic(weightWidth)) {
    int32_t inputSize = inputWidth + pad[4] + pad[5];
    int32_t filterSize = (weightWidth - 1) * dilation[2] + 1;
    int32_t unstridedResult = inputSize - filterSize + 1;
    outputShape[3] = (unstridedResult - 1) / stride[2] + 1;
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult Conv3DOp::verify() { return verifyConvOp(*this); }

LogicalResult AvgPool2dOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  return poolingInferReturnTypes(operands, attributes, inferredReturnShapes);
}

LogicalResult MaxPool2dOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  return poolingInferReturnTypes(operands, attributes, inferredReturnShapes);
}

LogicalResult DepthwiseConv2DOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<int64_t> outputShape(4, ShapedType::kDynamic);
  DepthwiseConv2DOp::Adaptor adaptor(operands.getValues(), attributes);

  int64_t inputWidth = ShapedType::kDynamic;
  int64_t inputHeight = ShapedType::kDynamic;
  int64_t inputChannels = ShapedType::kDynamic;

  int64_t weightWidth = ShapedType::kDynamic;
  int64_t weightHeight = ShapedType::kDynamic;
  int64_t depthChannels = ShapedType::kDynamic;

  // Input shape describes input width/height and batch.
  ShapeAdaptor inputShape = operands.getShape(adaptor.getInput());
  if (inputShape.hasRank()) {
    outputShape[0] = inputShape.getDimSize(0);
    inputHeight = inputShape.getDimSize(1);
    inputWidth = inputShape.getDimSize(2);
    inputChannels = inputShape.getDimSize(3);
  }

  // Weight shapes describes the filter width/height and the output channels.
  ShapeAdaptor weightShape = operands.getShape(adaptor.getWeight());
  if (weightShape.hasRank()) {
    weightHeight = weightShape.getDimSize(0);
    weightWidth = weightShape.getDimSize(1);
    inputChannels = ShapedType::isDynamic(inputChannels)
                        ? weightShape.getDimSize(2)
                        : inputChannels;
    depthChannels = weightShape.getDimSize(3);
  }

  // If both inputChannels and depthChannels are available we can determine
  // the output channels.
  if (!ShapedType::isDynamic(inputChannels) &&
      !ShapedType::isDynamic(depthChannels)) {
    outputShape[3] = inputChannels * depthChannels;
  }

  // Bias shape can describe the output channels.
  ShapeAdaptor biasShape = operands.getShape(adaptor.getBias());
  if (biasShape.hasRank()) {
    outputShape[3] = ShapedType::isDynamic(outputShape[3])
                         ? biasShape.getDimSize(0)
                         : outputShape[3];
  }

  llvm::ArrayRef<int64_t> dilation = adaptor.getDilation();
  llvm::ArrayRef<int64_t> padding = adaptor.getPad();
  llvm::ArrayRef<int64_t> stride = adaptor.getStride();

  if (!ShapedType::isDynamic(inputHeight) &&
      !ShapedType::isDynamic(weightHeight)) {
    int64_t inputSize = inputHeight + padding[0] + padding[1];
    int64_t filterSize = (weightHeight - 1) * dilation[0] + 1;
    int64_t unstridedResult = inputSize - filterSize + 1;
    outputShape[1] = (unstridedResult - 1) / stride[0] + 1;
  }

  if (!ShapedType::isDynamic(inputWidth) &&
      !ShapedType::isDynamic(weightWidth)) {
    int64_t inputSize = inputWidth + padding[2] + padding[3];
    int64_t filterSize = (weightWidth - 1) * dilation[1] + 1;
    int64_t unstridedResult = inputSize - filterSize + 1;
    outputShape[2] = (unstridedResult - 1) / stride[1] + 1;
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult DepthwiseConv2DOp::verify() { return verifyConvOp(*this); }

LogicalResult TransposeConv2DOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  TransposeConv2DOp::Adaptor adaptor(operands.getValues(), attributes);
  // outputShape is mutable.
  llvm::SmallVector<int64_t> outputShape =
      convertToMlirShape(adaptor.getOutShape());

  int64_t inputWidth = ShapedType::kDynamic;
  int64_t inputHeight = ShapedType::kDynamic;
  int64_t weightWidth = ShapedType::kDynamic;
  int64_t weightHeight = ShapedType::kDynamic;

  // Input shape describes input width/height and batch.
  ShapeAdaptor inputShape = operands.getShape(adaptor.getInput());
  if (inputShape.hasRank()) {
    outputShape[0] = ShapedType::isDynamic(outputShape[0])
                         ? inputShape.getDimSize(0)
                         : outputShape[0];
    inputHeight = inputShape.getDimSize(1);
    inputWidth = inputShape.getDimSize(2);
  }

  // Weight shapes describes the filter width/height and the output channels.
  ShapeAdaptor weightShape = operands.getShape(adaptor.getFilter());
  if (weightShape.hasRank()) {
    outputShape[3] = ShapedType::isDynamic(outputShape[3])
                         ? weightShape.getDimSize(0)
                         : outputShape[3];
    weightHeight = weightShape.getDimSize(1);
    weightWidth = weightShape.getDimSize(2);
  }

  // Bias shape can describe the output channels.
  ShapeAdaptor biasShape = operands.getShape(adaptor.getInput());
  if (biasShape.hasRank()) {
    outputShape[3] = ShapedType::isDynamic(outputShape[3])
                         ? biasShape.getDimSize(0)
                         : outputShape[3];
  }

  llvm::ArrayRef<int64_t> padding = adaptor.getOutPad();
  llvm::ArrayRef<int64_t> stride = adaptor.getStride();

  if (!ShapedType::isDynamic(inputHeight) &&
      !ShapedType::isDynamic(weightHeight)) {
    int64_t calculateSize =
        (inputHeight - 1) * stride[0] + padding[0] + padding[1] + weightHeight;
    outputShape[1] =
        ShapedType::isDynamic(outputShape[1]) ? calculateSize : outputShape[1];
  }

  if (!ShapedType::isDynamic(inputWidth) &&
      !ShapedType::isDynamic(weightWidth)) {
    int64_t calculateSize =
        (inputWidth - 1) * stride[1] + padding[2] + padding[3] + weightWidth;
    outputShape[2] =
        ShapedType::isDynamic(outputShape[2]) ? calculateSize : outputShape[2];
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult IfOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<tosa::YieldOp> yieldOps;
  for (Region *region : regions) {
    for (auto &block : *region)
      if (auto returnOp = dyn_cast<tosa::YieldOp>(block.getTerminator()))
        yieldOps.push_back(returnOp);
  }

  if (yieldOps.empty())
    return failure();

  // Get the initial type information for the yield op.
  llvm::SmallVector<ValueKnowledge> resultKnowledge;
  resultKnowledge.reserve(yieldOps.front().getNumOperands());
  for (auto operand : yieldOps.front().getOperands()) {
    resultKnowledge.push_back(
        ValueKnowledge::getKnowledgeFromType(operand.getType()));
  }

  for (auto yieldOp : yieldOps) {
    if (resultKnowledge.size() != yieldOp.getNumOperands())
      return failure();

    for (const auto &it : llvm::enumerate(yieldOp.getOperands())) {
      int32_t index = it.index();
      auto meet = ValueKnowledge::meet(
          resultKnowledge[index],
          ValueKnowledge::getKnowledgeFromType(it.value().getType()));
      if (!meet)
        continue;
      resultKnowledge[index] = meet;
    }
  }

  for (const ValueKnowledge &result : resultKnowledge) {
    inferredReturnShapes.push_back(result.getShapedTypeComponents());
  }

  return success();
}

LogicalResult WhileOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<tosa::YieldOp> yieldOps;
  for (auto &block : *regions[1])
    if (auto returnOp = dyn_cast<tosa::YieldOp>(block.getTerminator()))
      yieldOps.push_back(returnOp);

  // TOSA's while must have a tosa.yield as its terminator. If not found this
  // tosa.while is invalid.
  if (yieldOps.empty())
    return failure();

  // Get the initial type information from the operand types.
  llvm::SmallVector<ValueKnowledge> resultKnowledge;
  resultKnowledge.reserve(yieldOps.front().getNumOperands());
  for (auto operand : yieldOps.front().getOperands()) {
    resultKnowledge.push_back(
        ValueKnowledge::getKnowledgeFromType(operand.getType()));
  }

  for (auto yieldOp : yieldOps) {
    if (resultKnowledge.size() != yieldOp.getNumOperands())
      return failure();

    for (const auto &it : llvm::enumerate(yieldOp.getOperands())) {
      int32_t index = it.index();
      if (auto meet = ValueKnowledge::meet(
              resultKnowledge[index],
              ValueKnowledge::getKnowledgeFromType(it.value().getType()))) {
        resultKnowledge[index] = meet;
      }
    }
  }

  for (const ValueKnowledge &result : resultKnowledge) {
    inferredReturnShapes.push_back(result.getShapedTypeComponents());
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TOSA Attribute Definitions.
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Tosa/IR/TosaAttributes.cpp.inc"

//===----------------------------------------------------------------------===//
// TOSA Operator Definitions.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Tosa/IR/TosaOps.cpp.inc"
