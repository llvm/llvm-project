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
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/RegionUtils.h"

using namespace mlir;
using namespace mlir::tosa;

#include "mlir/Dialect/Tosa/IR/TosaOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Tosa dialect structs and interface includes.
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Tosa/IR/TosaInterfaces.cpp.inc"
#include "mlir/Dialect/Tosa/IR/TosaStructs.cpp.inc"

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
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// TOSA control flow support.
//===----------------------------------------------------------------------===//

/// Returns the while loop body.
Region &tosa::WhileOp::getLoopBody() { return body(); }

bool tosa::WhileOp::isDefinedOutsideOfLoop(Value value) {
  return !body().isAncestor(value.getParentRegion());
}

LogicalResult WhileOp::moveOutOfLoop(ArrayRef<mlir::Operation *> ops) {
  if (ops.empty())
    return success();

  Operation *tosaWhileOp = this->getOperation();
  for (auto *op : ops)
    op->moveBefore(tosaWhileOp);

  return success();
}

//===----------------------------------------------------------------------===//
// Tosa dialect initialization.
//===----------------------------------------------------------------------===//

void TosaDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Tosa/IR/TosaOps.cpp.inc"
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
// Operator Canonicalizers.
//===----------------------------------------------------------------------===//

struct RemoveReshapeNoop : public OpRewritePattern<tosa::ReshapeOp> {
  using OpRewritePattern<tosa::ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    if (op.input1().getType() != op.getType())
      return failure();

    rewriter.replaceOp(op, op.input1());
    return success();
  }
};

struct ReshapeReshapeOptimization : public OpRewritePattern<tosa::ReshapeOp> {
  using OpRewritePattern<tosa::ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.input1();
    Operation *definingOp = input.getDefiningOp();
    if (!definingOp)
      return failure();

    if (tosa::ReshapeOp reshapeOp = dyn_cast<tosa::ReshapeOp>(definingOp)) {
      rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(
          op, op.getType(), reshapeOp.input1(), op.new_shape());
      return success();
    }

    return failure();
  }
};

void ReshapeOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                            MLIRContext *context) {
  results.insert<ReshapeReshapeOptimization, RemoveReshapeNoop>(context);
}

//===----------------------------------------------------------------------===//
// Operator Folders.
//===----------------------------------------------------------------------===//

OpFoldResult ConstOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return valueAttr();
}

//===----------------------------------------------------------------------===//
// TOSA Operator Verifiers.
//===----------------------------------------------------------------------===//

template <typename T>
static LogicalResult verifyConvOp(T op) {
  // All TOSA conv ops have an input() and weight().
  auto inputType = op.input().getType().template dyn_cast<RankedTensorType>();
  auto weightType = op.weight().getType().template dyn_cast<RankedTensorType>();

  // Must be ranked tensor types
  if (!inputType || !weightType)
    return failure();

  auto inputEType = inputType.getElementType();
  auto weightEType = weightType.getElementType();

  bool inputIsQuant = !inputEType.template isa<FloatType>();
  bool weightIsQuant = !weightEType.template isa<FloatType>();

  // Either both must be quantized or both unquantized.
  if (inputIsQuant != weightIsQuant)
    return failure();

  // Quantized type must have constructed the quantizationattr, and unquantized
  // types should not have a quantizationattr.
  if ((inputIsQuant && !op.quantization_info()) ||
      (!inputIsQuant && op.quantization_info()))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// TOSA Operator Quantization Builders.
//===----------------------------------------------------------------------===//

/// This builder is called on all convolution operators except TransposeConv,
/// which has specialized output shape semantics. The builder also defines the
/// bitwidth of the output given the bit width of the input & weight content.
static void buildConvOpWithQuantInfo(OpBuilder &builder, OperationState &result,
                                     Type outputType, Value input, Value weight,
                                     Value bias, ArrayAttr pad,
                                     ArrayAttr stride, ArrayAttr dilation) {

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
static void
buildTransConvOpWithQuantInfo(OpBuilder &builder, OperationState &result,
                              Type outputType, Value input, Value weight,
                              Value bias, ArrayAttr outpad, ArrayAttr stride,
                              ArrayAttr dilation, ArrayAttr outputShape) {
  result.addOperands({input, weight, bias});
  result.addAttribute("out_pad", outpad);
  result.addAttribute("stride", stride);
  result.addAttribute("dilation", dilation);
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

    auto inputType = a.getType().dyn_cast<RankedTensorType>();
    assert(inputType && "Input must be a ranked tensor type!");

    auto inputQType = inputType.getElementType()
                          .dyn_cast<mlir::quant::UniformQuantizedType>();
    assert(inputQType && "Tensor must have quantized datatype!");

    unsigned inputBits = inputQType.getStorageTypeIntegralWidth();

    auto outputShapedType = outputType.dyn_cast<RankedTensorType>();
    assert(outputShapedType && "Output must be a ranked tensor type");

    auto outputShape = outputShapedType.getShape();

    IntegerType accElementType;
    if (inputBits == 16)
      accElementType = builder.getIntegerType(48);
    else
      accElementType = builder.getI32Type();
    auto accType = RankedTensorType::get(outputShape, accElementType);
    result.addTypes(accType);
  } else {
    result.addTypes(outputType);
  }
}

/// Both the tosa.avg_pool2d and unary ops use the same UnaruOpQuantizationAttr
/// but avg_pool operator has its own builder as it has additional parameters
/// not part of the unary ops.
static void buildAvgPool2dOpWithQuantInfo(OpBuilder &builder,
                                          OperationState &result,
                                          Type outputType, Value input,
                                          ArrayAttr kernel, ArrayAttr stride,
                                          ArrayAttr pad) {
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
/// correctly.
static void buildPadOpWithQuantInfo(OpBuilder &builder, OperationState &result,
                                    Type outputType, Value input,
                                    Value paddings) {
  result.addOperands({input, paddings});
  auto quantAttr = buildPadOpQuantizationAttr(builder, input);
  if (quantAttr)
    result.addAttribute("quantization_info", quantAttr);
  result.types.push_back(outputType);
}

//===----------------------------------------------------------------------===//
// TOSA Operator Return Type Inference.
//===----------------------------------------------------------------------===//

static void getI64Values(ArrayAttr arrayAttr, SmallVector<int64_t> &values) {
  for (auto it : arrayAttr) {
    values.push_back(it.cast<IntegerAttr>().getValue().getSExtValue());
  }
}

static void getF64Values(ArrayAttr arrayAttr, SmallVector<double> &values) {
  for (auto it : arrayAttr) {
    values.push_back(it.cast<FloatAttr>().getValueAsDouble());
  }
}

LogicalResult tosa::ArgMaxOp::inferReturnTypeComponents(
    MLIRContext *context, ::llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapedType inputTy = operands[0].getType().cast<ShapedType>();
  IntegerAttr axis = attributes.get("axis").cast<IntegerAttr>();
  int32_t axisVal = axis.getValue().getSExtValue();

  if (!inputTy.hasRank()) {
    inferredReturnShapes.push_back(ShapedTypeComponents());
    return success();
  }

  SmallVector<int64_t> outShape;
  outShape.reserve(inputTy.getRank() - 1);
  for (int i = 0, s = inputTy.getRank(); i < s; i++) {
    if (i == axisVal)
      continue;
    outShape.push_back(inputTy.getDimSize(i));
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outShape));
  return success();
}

LogicalResult tosa::ConcatOp::inferReturnTypeComponents(
    MLIRContext *context, ::llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  // Infer all dimension sizes by reducing based on inputs.
  int32_t axis =
      attributes.get("axis").cast<IntegerAttr>().getValue().getSExtValue();
  llvm::SmallVector<int64_t> outputShape;
  bool hasRankedInput = false;
  for (auto operand : operands) {
    ShapedType operandTy = operand.getType().cast<ShapedType>();
    if (!operandTy.hasRank())
      continue;

    // Copy the Operand's rank.
    if (!hasRankedInput)
      outputShape.resize(operandTy.getRank(), ShapedType::kDynamicSize);

    // Copy shapes until the dim is non-dynamic.
    for (int i = 0, s = operandTy.getRank(); i < s; i++) {
      if (i == axis || operandTy.isDynamicDim(i))
        continue;
      if (outputShape[i] == ShapedType::kDynamicSize)
        outputShape[i] = operandTy.getDimSize(i);
      if (outputShape[i] != operandTy.getDimSize(i))
        return failure();
    }

    hasRankedInput = true;
  }

  if (!hasRankedInput) {
    inferredReturnShapes.push_back(ShapedTypeComponents());
    return success();
  }

  // Determine the dimension size along the concatenation axis.
  int concatDimSize = 0;
  for (auto operand : operands) {
    ShapedType operandTy = operand.getType().cast<ShapedType>();

    // We need to know the length of the concatenation axis of all inputs to
    // determine the dimension size of the output shape.
    if (!operandTy.hasRank() || operandTy.isDynamicDim(axis)) {
      concatDimSize = ShapedType::kDynamicSize;
      break;
    }

    concatDimSize += operandTy.getDimSize(axis);
  }

  outputShape[axis] = concatDimSize;

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult tosa::FullyConnectedOp::inferReturnTypeComponents(
    MLIRContext *context, ::llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapedType inputTy = operands[0].getType().cast<ShapedType>();
  ShapedType weightTy = operands[1].getType().cast<ShapedType>();
  ShapedType biasTy = operands[2].getType().cast<ShapedType>();

  // All shapes are dynamic.
  SmallVector<int64_t> outShape;
  outShape.resize(2, ShapedType::kDynamicSize);

  if (inputTy.hasRank()) {
    outShape[0] = inputTy.getDimSize(0);
  }

  if (weightTy.hasRank()) {
    outShape[1] = weightTy.getDimSize(0);
  }

  if (biasTy.hasRank()) {
    outShape[1] = outShape[1] == ShapedType::kDynamicSize ? biasTy.getDimSize(0)
                                                          : outShape[1];
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outShape));
  return success();
}

LogicalResult tosa::MatMulOp::inferReturnTypeComponents(
    MLIRContext *context, ::llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapedType lhsTy = operands[0].getType().cast<ShapedType>();
  ShapedType rhsTy = operands[1].getType().cast<ShapedType>();

  // All shapes are dynamic.
  SmallVector<int64_t> outShape;
  outShape.resize(3, ShapedType::kDynamicSize);

  if (lhsTy.hasRank()) {
    outShape[0] = lhsTy.getDimSize(0);
    outShape[1] = lhsTy.getDimSize(1);
  }

  if (rhsTy.hasRank()) {
    outShape[0] = outShape[0] == ShapedType::kDynamicSize ? rhsTy.getDimSize(0)
                                                          : outShape[0];
    outShape[2] = rhsTy.getDimSize(2);
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outShape));
  return success();
}

LogicalResult tosa::PadOp::inferReturnTypeComponents(
    MLIRContext *context, ::llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapedType inputTy = operands[0].getType().cast<ShapedType>();
  ShapedType paddingTy = operands[1].getType().cast<ShapedType>();
  SmallVector<int64_t> outputShape;

  // If both inputs have unknown shape, we cannot determine the shape of the
  // output.
  if (!inputTy.hasRank() && !paddingTy.hasRank()) {
    inferredReturnShapes.push_back(ShapedTypeComponents());
    return success();
  }

  // If the input rank is unknown we can info the output rank using the padding
  // shape's first dim.
  if (!inputTy.hasRank()) {
    if (paddingTy.isDynamicDim(0)) {
      inferredReturnShapes.push_back(ShapedTypeComponents());
      return success();
    }

    outputShape.resize(paddingTy.getDimSize(0), ShapedType::kDynamicSize);
    inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
    return success();
  }

  DenseIntElementsAttr paddings;
  // If the paddings value is not a constant, all dimensions must be dynamic.
  if (!matchPattern(operands[1], m_Constant(&paddings))) {
    outputShape.resize(inputTy.getRank(), ShapedType::kDynamicSize);
    inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
    return success();
  }

  SmallVector<int64_t> paddingValues;
  for (auto val : paddings) {
    paddingValues.push_back(val.getSExtValue());
  }

  outputShape.reserve(inputTy.getRank());
  for (int i = 0, s = inputTy.getRank(); i < s; i++) {
    if (inputTy.isDynamicDim(i)) {
      outputShape.push_back(ShapedType::kDynamicSize);
      continue;
    }

    outputShape.push_back(inputTy.getDimSize(i) + paddingValues[i * 2] +
                          paddingValues[i * 2 + 1]);
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult tosa::SliceOp::inferReturnTypeComponents(
    MLIRContext *context, ::llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  auto sizes = attributes.get("size").cast<ArrayAttr>().getValue();
  SmallVector<int64_t> outputShape;
  outputShape.reserve(sizes.size());
  for (auto val : sizes) {
    outputShape.push_back(val.cast<IntegerAttr>().getValue().getSExtValue());
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult tosa::TableOp::inferReturnTypeComponents(
    MLIRContext *context, ::llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapedType inputTy = operands[0].getType().cast<ShapedType>();

  if (!inputTy.hasRank()) {
    inferredReturnShapes.push_back(ShapedTypeComponents());
    return success();
  }

  inferredReturnShapes.push_back(inputTy.getShape());
  return success();
}

LogicalResult tosa::TileOp::inferReturnTypeComponents(
    MLIRContext *context, ::llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  auto multiples = attributes.get("multiples").cast<ArrayAttr>().getValue();
  ShapedType inputTy = operands[0].getType().cast<ShapedType>();
  SmallVector<int64_t> outputShape;
  if (!inputTy.hasRank()) {
    outputShape.resize(multiples.size(), ShapedType::kDynamicSize);
    inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
    return success();
  }

  // We need the multiple values to determine the output shape.
  SmallVector<int64_t> multipleValues;
  multipleValues.reserve(multiples.size());
  for (auto val : multiples) {
    multipleValues.push_back(val.cast<IntegerAttr>().getValue().getSExtValue());
  }

  // Any non dynamic dimension can be multiplied to a known size.
  outputShape.reserve(multiples.size());
  for (int i = 0, s = inputTy.getRank(); i < s; i++) {
    int dim = inputTy.getDimSize(i);
    if (dim != ShapedType::kDynamicSize)
      dim *= multipleValues[i];
    outputShape.push_back(dim);
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult tosa::ReshapeOp::inferReturnTypeComponents(
    MLIRContext *context, ::llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapedType type = operands.front().getType().cast<ShapedType>();

  auto newShape = attributes.get("new_shape").cast<ArrayAttr>();
  llvm::SmallVector<int64_t> newShapeValue;
  getI64Values(newShape, newShapeValue);

  // We cannot infer from the total number of elements so we must take the
  // shape attribute as exact.
  if (!type.hasRank() || !type.hasStaticShape()) {
    inferredReturnShapes.push_back(ShapedTypeComponents(newShapeValue));
    return success();
  }

  // Determine the number of elements covered by the slice of all static
  // dimensions. This allows us to infer the length of the remaining dynamic
  // dimension.
  int64_t numElements = type.getNumElements();
  int64_t staticMul = 1;
  for (auto val : newShapeValue) {
    if (val != ShapedType::kDynamicSize) {
      staticMul *= val;
    }
  }

  // Determine the length of the dynamic dimension.
  for (auto &val : newShapeValue) {
    if (val == ShapedType::kDynamicSize)
      val = numElements / staticMul;
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(newShapeValue));
  return success();
}

LogicalResult tosa::TransposeOp::inferReturnTypeComponents(
    MLIRContext *context, ::llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapedType inputTy = operands[0].getType().cast<ShapedType>();
  ShapedType permsTy = operands[1].getType().cast<ShapedType>();

  // If input rank and permutation length is unknown, the output rank is
  // unknown.
  if (!inputTy.hasRank() && (!permsTy.hasRank() || permsTy.isDynamicDim(0))) {
    inferredReturnShapes.push_back(ShapedTypeComponents());
    return success();
  }

  // Without the input dims we cannot determine the output dim sizes but we
  // can determine the output rank.
  SmallVector<int64_t> outputShape;
  if (!inputTy.hasRank()) {
    outputShape.resize(permsTy.getDimSize(0), ShapedType::kDynamicSize);
    inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
    return success();
  }

  // Rank-0 means no permutations matter.
  if (inputTy.getRank() == 0) {
    inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
    return success();
  }

  // Check whether the input dimensions are all the same.
  bool allTheSame = true;
  for (int i = 1, s = inputTy.getRank(); i < s; i++) {
    if (inputTy.getDimSize(0) != inputTy.getDimSize(i)) {
      allTheSame = false;
      break;
    }
  }

  // If all of the input dimensions are the same we don't care about the
  // permutation.
  if (allTheSame) {
    outputShape.resize(inputTy.getRank(), inputTy.getDimSize(0));
    inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
    return success();
  }

  DenseIntElementsAttr perms;
  outputShape.resize(inputTy.getRank(), ShapedType::kDynamicSize);
  // If the permuations are a constant we can directly determine the output
  // shape.
  if (matchPattern(operands[1], m_Constant(&perms))) {
    llvm::SmallVector<int64_t> permValues;
    for (auto val : perms) {
      permValues.push_back(val.getSExtValue());
    }

    outputShape.reserve(inputTy.getRank());
    for (int i = 0, s = inputTy.getRank(); i < s; i++) {
      outputShape[i] = inputTy.getDimSize(permValues[i]);
    }
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult tosa::GatherOp::inferReturnTypeComponents(
    MLIRContext *context, ::llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<int64_t> outputShape;
  outputShape.resize(3, ShapedType::kDynamicSize);

  if (auto ty = operands[0].getType().dyn_cast<RankedTensorType>()) {
    outputShape[0] = ty.getDimSize(0);
    outputShape[2] = ty.getDimSize(2);
  }

  if (auto ty = operands[1].getType().dyn_cast<RankedTensorType>()) {
    if (outputShape[0] == ShapedType::kDynamicSize)
      outputShape[0] = ty.getDimSize(0);
    if (outputShape[1] == ShapedType::kDynamicSize)
      outputShape[1] = ty.getDimSize(1);
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult tosa::ResizeOp::inferReturnTypeComponents(
    MLIRContext *context, ::llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<int64_t, 4> outputShape;
  outputShape.resize(4, ShapedType::kDynamicSize);

  int32_t inHeight = ShapedType::kDynamicSize;
  int32_t inWidth = ShapedType::kDynamicSize;

  if (auto ty = operands[0].getType().dyn_cast<RankedTensorType>()) {
    outputShape[0] = ty.getDimSize(0);
    outputShape[3] = ty.getDimSize(3);

    inHeight = ty.getDimSize(1);
    inWidth = ty.getDimSize(2);
  }

  int32_t shift =
      attributes.get("shift").cast<IntegerAttr>().getValue().getSExtValue();
  llvm::SmallVector<int64_t> newShape;
  getI64Values(attributes.get("output_size").cast<ArrayAttr>(), newShape);
  outputShape[1] = newShape[0];
  outputShape[2] = newShape[1];

  llvm::SmallVector<int64_t> strideInt;
  llvm::SmallVector<int64_t> offsetInt;
  llvm::SmallVector<double> strideFp;
  llvm::SmallVector<double> offsetFp;
  getI64Values(attributes.get("offset").cast<ArrayAttr>(), offsetInt);
  getF64Values(attributes.get("offset_fp").cast<ArrayAttr>(), offsetFp);
  getI64Values(attributes.get("stride").cast<ArrayAttr>(), strideInt);
  getF64Values(attributes.get("stride_fp").cast<ArrayAttr>(), strideFp);

  // If we have a 0 zero in integers we know that the resize indexing needs to
  // be performed in floating point. Use the floating point varient to compute
  // the resize shape.
  bool fpMode = strideInt[0] == 0;

  // We can compute the output shape if attribute specifies unknown dimensions
  // based on the offset and stride. If we perfectly line up to the last index
  // we need to round up the size to include it.
  if (outputShape[1] == ShapedType::kDynamicSize && inHeight >= 0 && fpMode) {
    float sizeFp = (inHeight - offsetFp[0] - 1) / strideFp[0];
    float round = std::floor(sizeFp) == sizeFp ? 1 : 0;
    outputShape[1] = std::ceil(sizeFp) + round;
  }

  if (outputShape[2] == ShapedType::kDynamicSize && inWidth >= 0 && fpMode) {
    float sizeFp = (inWidth - offsetFp[1] - 1) / strideFp[1];
    float round = std::floor(sizeFp) == sizeFp ? 1 : 0;
    outputShape[2] = std::ceil(sizeFp) + round;
  }

  if (outputShape[1] == ShapedType::kDynamicSize && inHeight >= 0 && !fpMode) {
    int64_t size = (inHeight - 1);
    size = ((size << shift) - offsetInt[0]) / strideInt[0];
    outputShape[1] = size + 1;
  }

  if (outputShape[2] == ShapedType::kDynamicSize && inWidth >= 0 && !fpMode) {
    int64_t size = (inWidth - 1);
    size = ((size << shift) - offsetInt[1]) / strideInt[1];
    outputShape[2] = size + 1;
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult tosa::ScatterOp::inferReturnTypeComponents(
    MLIRContext *context, ::llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<int64_t> outputShape;
  outputShape.resize(3, ShapedType::kDynamicSize);

  if (auto ty = operands[0].getType().dyn_cast<RankedTensorType>()) {
    outputShape[0] = ty.getDimSize(0);
    outputShape[1] = ty.getDimSize(1);
    outputShape[2] = ty.getDimSize(2);
  }

  if (auto ty = operands[1].getType().dyn_cast<RankedTensorType>()) {
    if (outputShape[0] == ShapedType::kDynamicSize)
      outputShape[0] = ty.getDimSize(0);
  }

  if (auto ty = operands[2].getType().dyn_cast<RankedTensorType>()) {
    if (outputShape[0] == ShapedType::kDynamicSize)
      outputShape[0] = ty.getDimSize(0);
    if (outputShape[2] == ShapedType::kDynamicSize)
      outputShape[2] = ty.getDimSize(2);
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

static LogicalResult ReduceInferReturnTypes(
    Value operand, IntegerAttr axis,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  auto operandTy = operand.getType().cast<ShapedType>();
  if (!operandTy.hasRank()) {
    inferredReturnShapes.push_back(ShapedTypeComponents());
    return success();
  }

  int64_t axisVal = axis.getValue().getSExtValue();
  SmallVector<int64_t> outputShape;
  outputShape.reserve(operandTy.getRank());
  for (auto dim : operandTy.getShape()) {
    outputShape.push_back(dim);
  }

  outputShape[axisVal] = 1;
  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

#define REDUCE_SHAPE_INFER(OP)                                                 \
  LogicalResult OP::inferReturnTypeComponents(                                 \
      MLIRContext *context, ::llvm::Optional<Location> location,               \
      ValueShapeRange operands, DictionaryAttr attributes,                     \
      RegionRange regions,                                                     \
      SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {           \
    return ReduceInferReturnTypes(operands[0],                                 \
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

static LogicalResult resolveBroadcastShape(ValueRange operands,
                                           SmallVector<int64_t> &outShape) {
  int64_t outRank = 0;
  for (auto operand : operands) {
    auto type = operand.getType().cast<ShapedType>();
    if (!type.hasRank())
      return failure();
    outRank = std::max<int64_t>(outRank, type.getRank());
  }

  outShape.resize(outRank, 1);

  for (auto operand : operands) {
    auto type = operand.getType().cast<ShapedType>();
    auto shape = type.getShape();
    auto rankDiff = outShape.size() - shape.size();

    for (size_t i = 0; i < shape.size(); i++) {
      auto dim1 = outShape[i + rankDiff];
      auto dim2 = shape[i];
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

static LogicalResult NAryInferReturnTypes(
    ValueRange operands,
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
      MLIRContext *context, ::llvm::Optional<Location> location,               \
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
NARY_SHAPE_INFER(tosa::EqualOp)
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
NARY_SHAPE_INFER(tosa::ReluNOp)
NARY_SHAPE_INFER(tosa::RescaleOp)
NARY_SHAPE_INFER(tosa::ReverseOp)
NARY_SHAPE_INFER(tosa::RsqrtOp)
NARY_SHAPE_INFER(tosa::SelectOp)
NARY_SHAPE_INFER(tosa::SubOp)
NARY_SHAPE_INFER(tosa::TanhOp)
NARY_SHAPE_INFER(tosa::SigmoidOp)
#undef PRED_SHAPE_INFER

static LogicalResult poolingInferReturnTypes(
    ValueRange operands, DictionaryAttr attributes,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  RankedTensorType inputTy = operands[0].getType().dyn_cast<RankedTensorType>();
  llvm::SmallVector<int64_t> outputShape;
  outputShape.resize(4, -1);

  // We only know the rank if the input type is unranked.
  if (!inputTy) {
    inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
    return success();
  }

  // Batch and number of channels are identical for pooling layer.
  outputShape[0] = inputTy.getDimSize(0);
  outputShape[3] = inputTy.getDimSize(3);

  int32_t height = inputTy.getDimSize(1);
  int32_t width = inputTy.getDimSize(2);

  llvm::SmallVector<int64_t> kernel;
  llvm::SmallVector<int64_t> stride;
  llvm::SmallVector<int64_t> pad;

  getI64Values(attributes.get("kernel").cast<ArrayAttr>(), kernel);
  getI64Values(attributes.get("stride").cast<ArrayAttr>(), stride);
  getI64Values(attributes.get("pad").cast<ArrayAttr>(), pad);

  if (height != -1) {
    int32_t padded = height + pad[0] + pad[1] - kernel[0];
    outputShape[1] = padded / stride[0] + 1;
  }

  if (width != -1) {
    int32_t padded = width + pad[2] + pad[3] - kernel[1];
    outputShape[2] = padded / stride[1] + 1;
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult Conv2DOp::inferReturnTypeComponents(
    MLIRContext *context, ::llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<int64_t> outputShape(4, ShapedType::kDynamicSize);
  Conv2DOp::Adaptor adaptor(operands.getValues());

  int32_t inputWidth = ShapedType::kDynamicSize;
  int32_t inputHeight = ShapedType::kDynamicSize;
  int32_t weightWidth = ShapedType::kDynamicSize;
  int32_t weightHeight = ShapedType::kDynamicSize;

  // Input shape describes input width/height and batch.
  if (auto inputTy = adaptor.input().getType().dyn_cast<RankedTensorType>()) {
    outputShape[0] = inputTy.getDimSize(0);
    inputHeight = inputTy.getDimSize(1);
    inputWidth = inputTy.getDimSize(2);
  }

  // Weight shapes describes the filter width/height and the output channels.
  if (auto weightTy = adaptor.weight().getType().dyn_cast<RankedTensorType>()) {
    outputShape[3] = weightTy.getDimSize(0);
    weightHeight = weightTy.getDimSize(1);
    weightWidth = weightTy.getDimSize(2);
  }

  // Bias shape can describe the output channels.
  if (auto biasTy = adaptor.bias().getType().dyn_cast<RankedTensorType>()) {
    outputShape[3] = ShapedType::isDynamic(outputShape[3])
                         ? biasTy.getDimSize(0)
                         : outputShape[3];
  }

  llvm::SmallVector<int64_t> dilation;
  llvm::SmallVector<int64_t> padding;
  llvm::SmallVector<int64_t> stride;

  getI64Values(attributes.get("dilation").cast<ArrayAttr>(), dilation);
  getI64Values(attributes.get("pad").cast<ArrayAttr>(), padding);
  getI64Values(attributes.get("stride").cast<ArrayAttr>(), stride);

  if (!ShapedType::isDynamic(inputHeight) &&
      !ShapedType::isDynamic(weightHeight)) {
    int32_t inputSize = inputHeight + padding[0] + padding[1];
    int32_t filterSize = (weightHeight - 1) * dilation[0] + 1;
    int32_t unstridedResult = inputSize - filterSize + 1;
    outputShape[1] = (unstridedResult - 1) / stride[0] + 1;
  }

  if (!ShapedType::isDynamic(inputWidth) &&
      !ShapedType::isDynamic(weightWidth)) {
    int32_t inputSize = inputWidth + padding[2] + padding[3];
    int32_t filterSize = (weightWidth - 1) * dilation[1] + 1;
    int32_t unstridedResult = inputSize - filterSize + 1;
    outputShape[2] = (unstridedResult - 1) / stride[1] + 1;
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult Conv3DOp::inferReturnTypeComponents(
    MLIRContext *context, ::llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<int64_t> outputShape(5, ShapedType::kDynamicSize);
  Conv2DOp::Adaptor adaptor(operands.getValues());

  int32_t inputWidth = ShapedType::kDynamicSize;
  int32_t inputHeight = ShapedType::kDynamicSize;
  int32_t inputDepth = ShapedType::kDynamicSize;

  int32_t weightWidth = ShapedType::kDynamicSize;
  int32_t weightHeight = ShapedType::kDynamicSize;
  int32_t weightDepth = ShapedType::kDynamicSize;

  // Input shape describes input width/height and batch.
  if (auto inputTy = adaptor.input().getType().dyn_cast<RankedTensorType>()) {
    outputShape[0] = inputTy.getDimSize(0);
    inputHeight = inputTy.getDimSize(1);
    inputWidth = inputTy.getDimSize(2);
    inputDepth = inputTy.getDimSize(3);
  }

  // Weight shapes describes the filter width/height and the output channels.
  if (auto weightTy = adaptor.weight().getType().dyn_cast<RankedTensorType>()) {
    outputShape[4] = weightTy.getDimSize(0);
    weightHeight = weightTy.getDimSize(1);
    weightWidth = weightTy.getDimSize(2);
    weightDepth = weightTy.getDimSize(3);
  }

  // Bias shape can describe the output channels.
  if (auto biasTy = adaptor.bias().getType().dyn_cast<RankedTensorType>()) {
    outputShape[4] =
        (outputShape[4] == -1) ? biasTy.getDimSize(0) : outputShape[4];
  }

  llvm::SmallVector<int64_t> dilation;
  llvm::SmallVector<int64_t> padding;
  llvm::SmallVector<int64_t> stride;

  getI64Values(attributes.get("dilation").cast<ArrayAttr>(), dilation);
  getI64Values(attributes.get("pad").cast<ArrayAttr>(), padding);
  getI64Values(attributes.get("stride").cast<ArrayAttr>(), stride);

  if (!ShapedType::isDynamic(inputHeight) &&
      !ShapedType::isDynamic(weightHeight)) {
    int32_t inputSize = inputHeight + padding[0] + padding[1];
    int32_t filterSize = (weightHeight - 1) * dilation[0] + 1;
    int32_t unstridedResult = inputSize - filterSize + 1;
    outputShape[1] = (unstridedResult - 1) / stride[0] + 1;
  }

  if (!ShapedType::isDynamic(inputWidth) &&
      !ShapedType::isDynamic(weightWidth)) {
    int32_t inputSize = inputWidth + padding[2] + padding[3];
    int32_t filterSize = (weightWidth - 1) * dilation[1] + 1;
    int32_t unstridedResult = inputSize - filterSize + 1;
    outputShape[2] = (unstridedResult - 1) / stride[1] + 1;
  }

  if (!ShapedType::isDynamic(inputDepth) &&
      !ShapedType::isDynamic(weightDepth)) {
    int32_t inputSize = inputDepth + padding[4] + padding[5];
    int32_t filterSize = (weightDepth - 1) * dilation[2] + 1;
    int32_t unstridedResult = inputSize - filterSize + 1;
    outputShape[3] = (unstridedResult - 1) / stride[2] + 1;
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult AvgPool2dOp::inferReturnTypeComponents(
    MLIRContext *context, ::llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  return poolingInferReturnTypes(operands, attributes, inferredReturnShapes);
}

LogicalResult MaxPool2dOp::inferReturnTypeComponents(
    MLIRContext *context, ::llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  return poolingInferReturnTypes(operands, attributes, inferredReturnShapes);
}

LogicalResult DepthwiseConv2DOp::inferReturnTypeComponents(
    MLIRContext *context, ::llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<int64_t> outputShape(4, ShapedType::kDynamicSize);
  DepthwiseConv2DOp::Adaptor adaptor(operands.getValues());

  int32_t inputWidth = ShapedType::kDynamicSize;
  int32_t inputHeight = ShapedType::kDynamicSize;
  int32_t inputChannels = ShapedType::kDynamicSize;

  int32_t weightWidth = ShapedType::kDynamicSize;
  int32_t weightHeight = ShapedType::kDynamicSize;
  int32_t depthChannels = ShapedType::kDynamicSize;

  // Input shape describes input width/height and batch.
  if (auto inputTy = adaptor.input().getType().dyn_cast<RankedTensorType>()) {
    outputShape[0] = inputTy.getDimSize(0);
    inputHeight = inputTy.getDimSize(1);
    inputWidth = inputTy.getDimSize(2);
    inputChannels = inputTy.getDimSize(3);
  }

  // Weight shapes describes the filter width/height and the output channels.
  if (auto weightTy = adaptor.weight().getType().dyn_cast<RankedTensorType>()) {
    weightHeight = weightTy.getDimSize(0);
    weightWidth = weightTy.getDimSize(1);
    inputChannels = ShapedType::isDynamic(inputChannels)
                        ? weightTy.getDimSize(2)
                        : inputChannels;
    depthChannels = weightTy.getDimSize(3);
  }

  // If both inputChannels and depthChannels are available we can determine
  // the output channels.
  if (!ShapedType::isDynamic(inputChannels) &&
      !ShapedType::isDynamic(depthChannels)) {
    outputShape[3] = inputChannels * depthChannels;
  }

  // Bias shape can describe the output channels.
  if (auto biasTy = adaptor.bias().getType().dyn_cast<RankedTensorType>()) {
    outputShape[3] = ShapedType::isDynamic(outputShape[3])
                         ? biasTy.getDimSize(0)
                         : outputShape[3];
  }

  llvm::SmallVector<int64_t> dilation;
  llvm::SmallVector<int64_t> padding;
  llvm::SmallVector<int64_t> stride;

  getI64Values(attributes.get("dilation").cast<ArrayAttr>(), dilation);
  getI64Values(attributes.get("pad").cast<ArrayAttr>(), padding);
  getI64Values(attributes.get("stride").cast<ArrayAttr>(), stride);

  if (!ShapedType::isDynamic(inputHeight) &&
      !ShapedType::isDynamic(weightHeight)) {
    int32_t inputSize = inputHeight + padding[0] + padding[1];
    int32_t filterSize = (weightHeight - 1) * dilation[0] + 1;
    int32_t unstridedResult = inputSize - filterSize + 1;
    outputShape[1] = (unstridedResult - 1) / stride[0] + 1;
  }

  if (!ShapedType::isDynamic(inputWidth) &&
      !ShapedType::isDynamic(weightWidth)) {
    int32_t inputSize = inputWidth + padding[2] + padding[3];
    int32_t filterSize = (weightWidth - 1) * dilation[1] + 1;
    int32_t unstridedResult = inputSize - filterSize + 1;
    outputShape[2] = (unstridedResult - 1) / stride[1] + 1;
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult TransposeConv2DOp::inferReturnTypeComponents(
    MLIRContext *context, ::llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  TransposeConv2DOp::Adaptor adaptor(operands.getValues());
  llvm::SmallVector<int64_t> outputShape;
  getI64Values(attributes.get("out_shape").cast<ArrayAttr>(), outputShape);

  int32_t inputWidth = ShapedType::kDynamicSize;
  int32_t inputHeight = ShapedType::kDynamicSize;
  int32_t weightWidth = ShapedType::kDynamicSize;
  int32_t weightHeight = ShapedType::kDynamicSize;

  // Input shape describes input width/height and batch.
  if (auto inputTy = adaptor.input().getType().dyn_cast<RankedTensorType>()) {
    outputShape[0] = ShapedType::isDynamic(outputShape[0])
                         ? inputTy.getDimSize(0)
                         : outputShape[0];
    inputHeight = inputTy.getDimSize(1);
    inputWidth = inputTy.getDimSize(2);
  }

  // Weight shapes describes the filter width/height and the output channels.
  if (auto weightTy = adaptor.filter().getType().dyn_cast<RankedTensorType>()) {
    outputShape[3] = ShapedType::isDynamic(outputShape[3])
                         ? weightTy.getDimSize(0)
                         : outputShape[3];
    weightHeight = weightTy.getDimSize(1);
    weightWidth = weightTy.getDimSize(2);
  }

  // Bias shape can describe the output channels.
  if (auto biasTy = adaptor.bias().getType().dyn_cast<RankedTensorType>()) {
    outputShape[3] = ShapedType::isDynamic(outputShape[3])
                         ? biasTy.getDimSize(0)
                         : outputShape[3];
  }

  llvm::SmallVector<int64_t> dilation;
  llvm::SmallVector<int64_t> padding;
  llvm::SmallVector<int64_t> stride;

  getI64Values(attributes.get("dilation").cast<ArrayAttr>(), dilation);
  getI64Values(attributes.get("out_pad").cast<ArrayAttr>(), padding);
  getI64Values(attributes.get("stride").cast<ArrayAttr>(), stride);

  if (!ShapedType::isDynamic(inputHeight) &&
      !ShapedType::isDynamic(weightHeight)) {
    int32_t dilated = (weightHeight - 1) * dilation[0] + 1;
    int32_t calculateSize =
        (inputHeight - 1) * stride[0] - padding[0] + dilated;
    outputShape[1] = outputShape[1] == -1 ? calculateSize : outputShape[1];
  }

  if (!ShapedType::isDynamic(inputWidth) &&
      !ShapedType::isDynamic(weightWidth)) {
    int32_t dilated = (weightWidth - 1) * dilation[1] + 1;
    int32_t calculateSize = (inputWidth - 1) * stride[1] - padding[1] + dilated;
    outputShape[2] = outputShape[2] == -1 ? calculateSize : outputShape[2];
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

//===----------------------------------------------------------------------===//
// TOSA Operator Definitions.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Tosa/IR/TosaOps.cpp.inc"
