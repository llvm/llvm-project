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
#include "mlir/Dialect/Mesh/Interfaces/ShardingInterface.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
#include "mlir/Dialect/Tosa/Utils/ShapeUtils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/APFloat.h"
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
#include "mlir/Dialect/Tosa/IR/TosaDialectBytecode.cpp.inc"

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
                       IRMapping &map) const final {
    return true;
  }

  /// All regions with If and While parent operators can be inlined.
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &map) const final {
    return (isa<tosa::IfOp>(dest->getParentOp()) ||
            isa<tosa::WhileOp>(dest->getParentOp()));
  }
};

/// This class implements the bytecode interface for the Tosa dialect.
struct TosaDialectBytecodeInterface : public BytecodeDialectInterface {
  TosaDialectBytecodeInterface(Dialect *dialect)
      : BytecodeDialectInterface(dialect) {}

  //===--------------------------------------------------------------------===//
  // Attributes

  Attribute readAttribute(DialectBytecodeReader &reader) const override {
    return ::readAttribute(getContext(), reader);
  }

  LogicalResult writeAttribute(Attribute attr,
                               DialectBytecodeWriter &writer) const override {
    return ::writeAttribute(attr, writer);
  }

  //===--------------------------------------------------------------------===//
  // Types

  Type readType(DialectBytecodeReader &reader) const override {
    return ::readType(getContext(), reader);
  }

  LogicalResult writeType(Type type,
                          DialectBytecodeWriter &writer) const override {
    return ::writeType(type, writer);
  }

  void writeVersion(DialectBytecodeWriter &writer) const final {
    // TODO: Populate.
  }

  std::unique_ptr<DialectVersion>
  readVersion(DialectBytecodeReader &reader) const final {
    // TODO: Populate
    reader.emitError("Dialect does not support versioning");
    return nullptr;
  }

  LogicalResult upgradeFromVersion(Operation *topLevelOp,
                                   const DialectVersion &version) const final {
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// TOSA control flow support.
//===----------------------------------------------------------------------===//

/// Returns the while loop body.
SmallVector<Region *> tosa::WhileOp::getLoopRegions() { return {&getBody()}; }

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
  addInterfaces<TosaDialectBytecodeInterface, TosaInlinerInterface>();
  declarePromisedInterfaces<
      mesh::ShardingInterface, ClampOp, SigmoidOp, TanhOp, AddOp,
      ArithmeticRightShiftOp, BitwiseAndOp, BitwiseOrOp, BitwiseXorOp, IntDivOp,
      LogicalAndOp, LogicalLeftShiftOp, LogicalRightShiftOp, LogicalOrOp,
      LogicalXorOp, MaximumOp, MinimumOp, MulOp, PowOp, SubOp, AbsOp,
      BitwiseNotOp, CeilOp, ClzOp, ExpOp, FloorOp, LogOp, LogicalNotOp,
      NegateOp, ReciprocalOp, RsqrtOp, SelectOp, EqualOp, GreaterOp,
      GreaterEqualOp, MatMulOp>();
}

Operation *TosaDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  // Tosa dialect constants only support ElementsAttr unlike standard dialect
  // constant which supports all attributes.
  if (llvm::isa<ElementsAttr>(value))
    return builder.create<tosa::ConstOp>(loc, type,
                                         llvm::cast<ElementsAttr>(value));
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Parsers and printers
//===----------------------------------------------------------------------===//

ParseResult mlir::tosa::parseTypeOrAttr(OpAsmParser &parser, TypeAttr &typeAttr,
                                        Attribute &attr) {
  if (succeeded(parser.parseOptionalEqual())) {
    if (failed(parser.parseAttribute(attr))) {
      return parser.emitError(parser.getCurrentLocation())
             << "expected attribute";
    }
    if (auto typedAttr = dyn_cast<TypedAttr>(attr)) {
      typeAttr = TypeAttr::get(typedAttr.getType());
    }
    return success();
  }

  Type type;
  if (failed(parser.parseColonType(type))) {
    return parser.emitError(parser.getCurrentLocation()) << "expected type";
  }
  typeAttr = TypeAttr::get(type);

  return success();
}

void mlir::tosa::printTypeOrAttr(OpAsmPrinter &p, Operation *op, TypeAttr type,
                                 Attribute attr) {
  bool needsSpace = false;
  auto typedAttr = dyn_cast_or_null<TypedAttr>(attr);
  if (!typedAttr || typedAttr.getType() != type.getValue()) {
    p << ": ";
    p.printAttribute(type);
    needsSpace = true; // subsequent attr value needs a space separator
  }
  if (attr) {
    if (needsSpace)
      p << ' ';
    p << "= ";
    p.printAttribute(attr);
  }
}

//===----------------------------------------------------------------------===//
// TOSA Operator Verifiers.
//===----------------------------------------------------------------------===//

template <typename T>
static LogicalResult verifyConvOp(T op) {
  // All TOSA conv ops have an input() and weight().
  auto inputType = llvm::dyn_cast<RankedTensorType>(op.getInput().getType());
  auto weightType = llvm::dyn_cast<RankedTensorType>(op.getWeight().getType());

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

  bool inputIsQuant = !llvm::isa<FloatType>(inputEType);
  bool weightIsQuant = !llvm::isa<FloatType>(weightEType);

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

LogicalResult tosa::ConstOp::verify() {

  auto attrType = llvm::dyn_cast<TensorType>(getValueAttr().getType());
  auto outputType = llvm::dyn_cast<TensorType>(getOutput().getType());

  if (!attrType || !outputType) {
    emitOpError("expected tensors for attr/result type");
    return failure();
  }

  if (auto result = llvm::dyn_cast<mlir::quant::QuantizedType>(
          outputType.getElementType())) {
    if (result.getStorageType() == attrType.getElementType())
      return success();
  }

  if (attrType.getElementType() != outputType.getElementType()) {
    emitOpError("expected same attr/result element types");
    return failure();
  }

  return success();
}

LogicalResult tosa::ArgMaxOp::verify() {
  // Ensure output is of 32-bit integer
  const auto resultETy = llvm::cast<ShapedType>(getType()).getElementType();
  if (!resultETy.isIntOrIndex())
    return emitOpError("result tensor is not of integer type");

  // Ensure axis is within the tensor rank
  const auto inputType = llvm::cast<ShapedType>(getInput().getType());
  const int64_t axis = getAxisAttr().getInt();
  if (inputType.hasRank() && ((axis < 0) || axis >= inputType.getRank()))
    return emitOpError("specified axis is outside the rank of the tensor");

  return success();
}

LogicalResult tosa::AvgPool2dOp::verify() {
  auto inputType = llvm::cast<ShapedType>(getInput().getType());

  auto inputETy = inputType.getElementType();
  auto resultETy = llvm::cast<ShapedType>(getType()).getElementType();

  if (auto quantType =
          llvm::dyn_cast<mlir::quant::UniformQuantizedType>(inputETy))
    inputETy = quantType.getStorageType();

  if (auto quantType =
          llvm::dyn_cast<mlir::quant::UniformQuantizedType>(resultETy))
    resultETy = quantType.getStorageType();

  auto accType = getAccType();
  if (llvm::isa<IntegerType>(inputETy) && !accType.isInteger(32))
    return emitOpError("accumulator type for integer tensor is not i32");

  if (inputETy.isF16() && !(accType.isF16() || accType.isF32()))
    return emitOpError("accumulator type for f16 tensor is not f16/f32");

  if (inputETy.isBF16() && !accType.isF32())
    return emitOpError("accumulator type for bf16 tensor is not f32");

  if (inputETy.isF32() && !accType.isF32())
    return emitOpError("accumulator type for f32 tensor is not f32");

  if ((inputETy.isF32() && resultETy.isF32()) ||
      (inputETy.isF16() && resultETy.isF16()) ||
      (inputETy.isBF16() && resultETy.isBF16()) ||
      (inputETy.isInteger(8) && resultETy.isInteger(8)) ||
      (inputETy.isInteger(16) && resultETy.isInteger(16)))
    return success();

  return emitOpError("input/output element types are incompatible.");
}

LogicalResult tosa::ClampOp::verify() {
  mlir::Type inputETy =
      llvm::cast<ShapedType>(getInput().getType()).getElementType();
  if (auto quantType =
          llvm::dyn_cast<mlir::quant::UniformQuantizedType>(inputETy)) {
    inputETy = quantType.getStorageType();
  }
  mlir::Type maxFpType = getMaxFpAttr().getType();
  mlir::Type minFpType = getMinFpAttr().getType();
  mlir::Type outputETy =
      llvm::cast<ShapedType>(getOutput().getType()).getElementType();
  if (auto quantType =
          llvm::dyn_cast<mlir::quant::UniformQuantizedType>(outputETy)) {
    outputETy = quantType.getStorageType();
  }
  unsigned dataTypeBitWidth = inputETy.getIntOrFloatBitWidth();

  if (inputETy != outputETy)
    return emitOpError("input/output element types are incompatible.");

  // If input datatype is float, check that the two min/max_fp attributes
  // share the same type and that their type is either the same of the input's
  // datatype, or a float type whose bitwidth > input datatype bitwidth.
  if (!inputETy.isInteger(dataTypeBitWidth)) {
    if (((maxFpType != minFpType) ||
         (maxFpType != inputETy && maxFpType.getIntOrFloatBitWidth() <=
                                       inputETy.getIntOrFloatBitWidth())))
      return emitOpError("min/max attributes types are incompatible with "
                         "input/output element types.");
  }

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

/// Handles tosa.transpose_conv2d which has outpad and output shape
/// attributes.
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

/// The tosa.matmul op is also intended to be generated where a
/// fully_connected op must be constructed where the weight is not a constant.
/// In this case, the fully_connected op must be expressed using matmul.
/// TODO: Add link to the leglization document explaining this.
static void buildMatMulOpWithQuantInfo(OpBuilder &builder,
                                       OperationState &result, Type outputType,
                                       Value a, Value b) {
  result.addOperands({a, b});
  auto quantAttr = ::buildMatMulOpQuantizationAttr(builder, a, b);

  if (quantAttr) {
    result.addAttribute("quantization_info", quantAttr);

    auto inputType = llvm::dyn_cast<ShapedType>(a.getType());
    assert(inputType && "Input must be a shaped tensor type!");

    auto inputQType = llvm::dyn_cast<mlir::quant::UniformQuantizedType>(
        inputType.getElementType());
    assert(inputQType && "Tensor must have quantized datatype!");

    unsigned inputBits = inputQType.getStorageTypeIntegralWidth();

    auto outputShapedType = llvm::dyn_cast<ShapedType>(outputType);
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

/// Both the tosa.avg_pool2d and unary ops use the same
/// UnaruOpQuantizationAttr but avg_pool operator has its own builder as it
/// has additional parameters not part of the unary ops.
static void
buildAvgPool2dOpWithQuantInfo(OpBuilder &builder, OperationState &result,
                              Type outputType, Value input,
                              DenseArrayAttr kernel, DenseArrayAttr stride,
                              DenseArrayAttr pad, TypeAttr accType) {
  result.addOperands(input);
  result.addAttribute("kernel", kernel);
  result.addAttribute("stride", stride);
  result.addAttribute("pad", pad);
  result.addAttribute("acc_type", accType);
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
      // TODO(jennik): Update function to have better case handling for
      // invalid operands and for ranked tensors.
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
    ArgMaxOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapeAdaptor inputShape(adaptor.getInput().getType());
  IntegerAttr axis = adaptor.getProperties().axis;
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

LogicalResult tosa::RFFT2dOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    RFFT2dOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapeAdaptor inputShape(adaptor.getInput().getType());

  if (!inputShape.hasRank())
    return failure();

  llvm::SmallVector<int64_t> outputShape;
  outputShape.resize(3, ShapedType::kDynamic);
  outputShape[0] = inputShape.getDimSize(0);
  outputShape[1] = inputShape.getDimSize(1);
  int64_t inWidth = inputShape.getDimSize(2);

  // Note that we can support this calculation symbolically
  // in the future e.g. [x, y, z] -> [x, y, z / 2 - 1]
  if (inWidth != ShapedType::kDynamic)
    outputShape[2] = inWidth / 2 + 1;

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));

  return success();
}

LogicalResult tosa::FFT2dOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    FFT2dOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  inferredReturnShapes.push_back(
      ShapedTypeComponents(ShapeAdaptor(adaptor.getInputReal().getType())));
  inferredReturnShapes.push_back(
      ShapedTypeComponents(ShapeAdaptor(adaptor.getInputImag().getType())));
  return success();
}

LogicalResult tosa::ConcatOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    ConcatOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  // Infer all dimension sizes by reducing based on inputs.
  const Properties &prop = adaptor.getProperties();
  int32_t axis = prop.axis.getValue().getSExtValue();
  llvm::SmallVector<int64_t> outputShape;
  bool hasRankedInput = false;
  for (auto operand : adaptor.getOperands()) {
    ShapeAdaptor operandShape(operand.getType());
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
        return emitOptionalError(location,
                                 "Cannot concat tensors with different sizes"
                                 " on the non-axis dimension ",
                                 i);
    }

    hasRankedInput = true;
  }
  Type inputType =
      llvm::cast<TensorType>(adaptor.getInput1().getType()[0]).getElementType();
  if (!hasRankedInput) {
    inferredReturnShapes.push_back(ShapedTypeComponents(inputType));
    return success();
  }

  // Determine the dimension size along the concatenation axis.
  int64_t concatDimSize = 0;
  for (auto operand : adaptor.getOperands()) {
    ShapeAdaptor operandShape(operand.getType());

    // We need to know the length of the concatenation axis of all inputs to
    // determine the dimension size of the output shape.
    if (!operandShape.hasRank() || operandShape.isDynamicDim(axis)) {
      concatDimSize = ShapedType::kDynamic;
      break;
    }

    concatDimSize += operandShape.getDimSize(axis);
  }

  outputShape[axis] = concatDimSize;

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape, inputType));
  return success();
}

LogicalResult tosa::EqualOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  auto elementType = IntegerType::get(context, /*width=*/1);

  llvm::SmallVector<int64_t> outShape;
  if (resolveBroadcastShape(operands, outShape).failed()) {
    inferredReturnShapes.push_back(ShapedTypeComponents(elementType));
    return success();
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outShape, elementType));
  return success();
}

bool tosa::EqualOp::isCompatibleReturnTypes(TypeRange l, TypeRange r) {
  if (l.size() != r.size() || l.size() != 1)
    return false;
  return succeeded(verifyCompatibleShape(l[0], r[0]));
}

LogicalResult tosa::FullyConnectedOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    FullyConnectedOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapeAdaptor inputShape(adaptor.getInput().getType());
  ShapeAdaptor weightShape(adaptor.getWeight().getType());
  ShapeAdaptor biasShape(adaptor.getBias().getType());

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
    MatMulOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapeAdaptor lhsShape(adaptor.getA().getType());
  ShapeAdaptor rhsShape(adaptor.getB().getType());

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
    PadOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapeAdaptor inputShape(adaptor.getInput1().getType());
  ShapeAdaptor paddingShape(adaptor.getPadding().getType());
  SmallVector<int64_t> outputShape;

  // If both inputs have unknown shape, we cannot determine the shape of the
  // output.
  if (!inputShape.hasRank() && !paddingShape.hasRank()) {
    inferredReturnShapes.push_back(ShapedTypeComponents());
    return success();
  }

  // If the input rank is unknown we can info the output rank using the
  // padding shape's first dim.
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
  if (!matchPattern(adaptor.getPadding(), m_Constant(&paddings))) {
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

LogicalResult tosa::PadOp::verify() {
  RankedTensorType inputType = getInput1().getType();
  RankedTensorType outputType = getOutput().getType();
  TensorType paddingType = getPadding().getType();

  if (inputType.getRank() != outputType.getRank())
    return emitOpError() << "expect same input and output tensor rank.";

  if (paddingType.hasRank() && paddingType.getRank() != 2)
    return emitOpError() << "expect 'padding' tensor rank equal to 2.";

  return success();
}

static SmallVector<int64_t> convertToMlirShape(ArrayRef<int64_t> shape) {
  return to_vector(llvm::map_range(shape, [](int64_t dim) {
    return dim == -1 ? ShapedType::kDynamic : dim;
  }));
}

LogicalResult tosa::SliceOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    SliceOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  inferredReturnShapes.push_back(
      ShapedTypeComponents(convertToMlirShape(adaptor.getSize())));
  return success();
}

LogicalResult tosa::SliceOp::verify() {
  auto inputType = llvm::dyn_cast<RankedTensorType>(getInput().getType());
  if (!inputType)
    return success();

  if (static_cast<size_t>(inputType.getRank()) != getStart().size())
    return emitOpError(
        "length of start attribute is not equal rank of input shape");

  if (static_cast<size_t>(inputType.getRank()) != getSize().size())
    return emitOpError(
        "length of size attribute is not equal rank of input shape");

  return success();
}

LogicalResult tosa::TableOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    TableOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapeAdaptor inputShape(adaptor.getInput().getType());

  if (!inputShape.hasRank()) {
    inferredReturnShapes.push_back(ShapedTypeComponents());
    return success();
  }

  inferredReturnShapes.resize(1);
  inputShape.getDims(inferredReturnShapes[0]);
  return success();
}

LogicalResult tosa::TableOp::verify() {
  TensorType inputType = getInput().getType();
  TensorType outputType = getOutput().getType();

  if (inputType.hasRank() && outputType.hasRank() &&
      inputType.getRank() != outputType.getRank())
    return emitOpError()
           << "expected input tensor rank to equal result tensor rank";

  auto inputDims = inputType.getShape();
  auto outputDims = outputType.getShape();
  for (auto it : llvm::enumerate(llvm::zip(inputDims, outputDims))) {
    int64_t dim = it.index();
    auto [inputDim, outputDim] = it.value();
    if (!ShapedType::isDynamic(outputDim) && outputDim != inputDim) {
      return emitOpError() << "dim(result, " << dim << ") = " << outputDim
                           << " doesn't match dim(input, " << dim
                           << ") = " << inputDim;
    }
  }
  return success();
}

LogicalResult tosa::TileOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    TileOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ArrayRef<int64_t> multiples = adaptor.getMultiples();
  ShapeAdaptor inputShape(adaptor.getInput1().getType());
  SmallVector<int64_t> outputShape;
  if (!inputShape.hasRank()) {
    outputShape.resize(multiples.size(), ShapedType::kDynamic);
    inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
    return success();
  } else if (static_cast<size_t>(inputShape.getRank()) != multiples.size())
    return failure();

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

LogicalResult tosa::TileOp::verify() {
  ShapedType inputType = llvm::cast<ShapedType>(getInput1().getType());
  ShapedType outputType = llvm::cast<ShapedType>(getType());
  auto multiples = getMultiples();

  if (inputType.hasRank()) {
    if (static_cast<size_t>(inputType.getRank()) != multiples.size())
      return emitOpError("expect 'multiples' array to have length ")
             << inputType.getRank() << " but got " << multiples.size() << ".";
    if (outputType.hasRank() && inputType.getRank() != outputType.getRank())
      return emitOpError("expect same input and output tensor rank.");
  } else if (outputType.hasRank() &&
             static_cast<size_t>(outputType.getRank()) != multiples.size())
    return emitOpError("expect 'multiples' array to have length ")
           << outputType.getRank() << " but got " << multiples.size() << ".";

  if (llvm::any_of(multiples, [](int64_t v) { return v <= 0 && v != -1; }))
    return emitOpError(
        "expect element of 'multiples' to be positive integer or -1.");

  return success();
}

bool tosa::ReshapeOp::isCompatibleReturnTypes(TypeRange l, TypeRange r) {
  if (l.size() != r.size() || l.size() != 1)
    return false;
  return getElementTypeOrSelf(l[0]) == getElementTypeOrSelf(r[0]);
}

LogicalResult tosa::ReshapeOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    ReshapeOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapeAdaptor inputShape(adaptor.getInput1().getType());
  Type inputType = getElementTypeOrSelf(adaptor.getInput1().getType());
  llvm::SmallVector<int64_t> newShapeValue =
      convertToMlirShape(adaptor.getNewShape());

  // We cannot infer from the total number of elements so we must take the
  // shape attribute as exact.
  if (!inputShape.hasRank() || !inputShape.hasStaticShape()) {
    inferredReturnShapes.push_back(
        ShapedTypeComponents(newShapeValue, inputType));
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

  inferredReturnShapes.push_back(
      ShapedTypeComponents(newShapeValue, inputType));
  return success();
}

llvm::LogicalResult tosa::ReshapeOp::verify() {
  TensorType inputType = getInput1().getType();
  RankedTensorType outputType = getType();

  if ((int64_t)getNewShape().size() != outputType.getRank())
    return emitOpError() << "new shape does not match result rank";

  for (auto [newShapeDim, outputShapeDim] :
       zip(getNewShape(), outputType.getShape())) {
    if (newShapeDim != -1 && outputShapeDim != ShapedType::kDynamic &&
        newShapeDim != outputShapeDim)
      return emitOpError() << "new shape is inconsistent with result shape";

    if (newShapeDim != ShapedType::kDynamic && newShapeDim < -1)
      return emitOpError() << "new shape has invalid tensor dimension size "
                           << newShapeDim;
  }

  if (inputType.hasStaticShape() && outputType.hasStaticShape()) {
    int64_t inputElementsNum = inputType.getNumElements();
    int64_t outputElementsNum = outputType.getNumElements();
    if (inputElementsNum != outputElementsNum) {
      return emitOpError() << "cannot reshape " << inputElementsNum
                           << " elements into " << outputElementsNum;
    }
  }

  int missingDims = llvm::count(getNewShape(), -1);
  if (missingDims > 1)
    return emitOpError() << "expected at most one target dimension to be -1";

  return mlir::success();
}

LogicalResult tosa::TransposeOp::getConstantPerms(SmallVector<int32_t> &perms) {
  // Perms must be constants.
  DenseIntElementsAttr permsAttr;
  if (!matchPattern(getPerms(), m_Constant(&permsAttr)))
    return failure();

  perms.clear();
  for (auto v : permsAttr.getValues<APInt>())
    perms.push_back(v.getSExtValue());

  return success();
}

LogicalResult tosa::TransposeOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    TransposeOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapeAdaptor inputShape(adaptor.getInput1().getType());
  ShapeAdaptor permsShape(adaptor.getPerms().getType());

  // We cannot infer anything from a rank-0 "permutation" tensor.
  if (permsShape.hasRank() && permsShape.getRank() == 0)
    return failure();

  // If input rank and permutation length is unknown, the output rank is
  // unknown.
  if (!inputShape.hasRank() || !permsShape.hasRank() ||
      permsShape.isDynamicDim(0)) {
    inferredReturnShapes.push_back(ShapedTypeComponents());
    return success();
  }

  // This would imply the number of permutations does not match the rank of
  // the input which is illegal.
  if (permsShape.getDimSize(0) != inputShape.getRank()) {
    return failure();
  }

  SmallVector<int64_t> outputShape;
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
  DenseIntElementsAttr attr;
  if (matchPattern(adaptor.getPerms(), m_Constant(&attr)) &&
      attr.getType().getRank() == 1) {
    ShapeAdaptor permShape = attr;
    // Constant permutation must be the same length as the input rank.
    if (inputShape.getRank() != permShape.getRank())
      return emitOptionalError(location,
                               "constant permutation must be the same length"
                               " as the input rank");

    // Constant permutation values must be within the input rank.
    for (int i = 0, e = inputShape.getRank(); i < e; i++) {
      if (inputShape.getRank() <= permShape.getDimSize(i))
        return failure();
    }

    outputShape.reserve(inputShape.getRank());
    for (int i = 0, s = inputShape.getRank(); i < s; i++) {
      outputShape[i] = inputShape.getDimSize(permShape.getDimSize(i));
    }
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult tosa::TransposeOp::verify() {
  TensorType inputType = getInput1().getType();
  TensorType permType = getPerms().getType();
  TensorType outputType = getOutput().getType();

  if (permType.hasRank() && permType.getRank() != 1)
    return emitOpError()
           << "expected permutation tensor to be rank 1 but got rank "
           << permType.getRank();
  if (inputType.hasRank() && permType.hasRank())
    if (!permType.isDynamicDim(0) &&
        permType.getDimSize(0) != inputType.getRank())
      return emitOpError() << "expected permutation tensor dim 0 to have size "
                           << inputType.getRank()
                           << " (input rank) but got size "
                           << permType.getDimSize(0);
  if (inputType.hasRank() && outputType.hasRank() &&
      inputType.getRank() != outputType.getRank())
    return emitOpError()
           << "expected input tensor rank to equal result tensor rank";
  if (outputType.hasRank() && permType.hasRank())
    if (!permType.isDynamicDim(0) &&
        permType.getDimSize(0) != outputType.getRank())
      return emitOpError() << "expected permutation tensor dim 0 to have size "
                           << outputType.getRank()
                           << " (output rank) but got size "
                           << permType.getDimSize(0);

  SmallVector<int32_t> constantPerms;
  if (succeeded(getConstantPerms(constantPerms))) {
    // Assert that the permutation tensor has a rank, which means that the
    // rank has been verified above.
    assert(permType.hasRank() &&
           "Unexpectedly found permutation tensor without rank");
    if (!llvm::all_of(constantPerms,
                      [&constantPerms](int32_t s) {
                        return s >= 0 &&
                               static_cast<size_t>(s) < constantPerms.size();
                      }) ||
        !isPermutationVector(llvm::to_vector(llvm::map_range(
            constantPerms, [](int32_t v) -> int64_t { return v; }))))
      return emitOpError() << "expected valid permutation tensor";

    // Verify that the types of the input and output tensors are properly
    // permuted.
    if (inputType.hasRank() && outputType.hasRank()) {
      assert(constantPerms.size() == static_cast<size_t>(inputType.getRank()) &&
             inputType.getRank() == outputType.getRank());

      for (auto i = 0; i < outputType.getRank(); i++) {
        if (inputType.isDynamicDim(constantPerms[i]) ||
            outputType.isDynamicDim(i))
          continue;

        if (inputType.getDimSize(constantPerms[i]) != outputType.getDimSize(i))
          return emitOpError()
                 << "expected output tensor dim " << i << " to match "
                 << "input dim " << constantPerms[i] << " with value of "
                 << inputType.getDimSize(constantPerms[i]);
      }
    }
  }
  return success();
}

LogicalResult TransposeOp::reifyResultShapes(
    OpBuilder &builder, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {

  SmallVector<int32_t> transposePerms;
  if (getConstantPerms(transposePerms).failed())
    return failure();

  Value input = getInput1();
  auto inputType = cast<TensorType>(input.getType());

  SmallVector<OpFoldResult> returnedDims(inputType.getRank());
  for (auto dim : transposePerms) {
    int32_t dimInInput = transposePerms[dim];
    if (inputType.isDynamicDim(dimInInput))
      returnedDims[dim] =
          builder.create<tensor::DimOp>(getLoc(), input, dimInInput)
              .getResult();
    else
      returnedDims[dim] =
          builder.getIndexAttr(inputType.getDimSize(dimInInput));
  }

  reifiedReturnShapes.emplace_back(std::move(returnedDims));
  return success();
}

LogicalResult tosa::GatherOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    GatherOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<int64_t> outputShape;
  outputShape.resize(3, ShapedType::kDynamic);

  ShapeAdaptor valuesShape(adaptor.getValues().getType());
  if (valuesShape.hasRank()) {
    outputShape[0] = valuesShape.getDimSize(0);
    outputShape[2] = valuesShape.getDimSize(2);
  }

  ShapeAdaptor indicesShape(adaptor.getIndices().getType());
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
    ResizeOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<int64_t, 4> outputShape;
  outputShape.resize(4, ShapedType::kDynamic);

  ShapeAdaptor inputShape(adaptor.getInput().getType());
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
    ScatterOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<int64_t> outputShape;
  outputShape.resize(3, ShapedType::kDynamic);

  ShapeAdaptor valuesInShape(adaptor.getValuesIn().getType());
  if (valuesInShape.hasRank()) {
    outputShape[0] = valuesInShape.getDimSize(0);
    outputShape[1] = valuesInShape.getDimSize(1);
    outputShape[2] = valuesInShape.getDimSize(2);
  }

  ShapeAdaptor indicesShape(adaptor.getIndices().getType());
  if (indicesShape.hasRank()) {
    if (outputShape[0] == ShapedType::kDynamic)
      outputShape[0] = indicesShape.getDimSize(0);
  }

  ShapeAdaptor inputShape(adaptor.getInput().getType());
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
    ShapeAdaptor operandShape, Type inputType, IntegerAttr axis,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  int64_t axisVal = axis.getValue().getSExtValue();
  if (!operandShape.hasRank() || operandShape.getRank() <= axisVal) {
    inferredReturnShapes.push_back(ShapedTypeComponents(inputType));
    return success();
  }

  SmallVector<int64_t> outputShape;
  operandShape.getDims(outputShape);
  outputShape[axisVal] = 1;
  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape, inputType));
  return success();
}

#define COMPATIBLE_RETURN_TYPES(OP)                                            \
  bool OP::isCompatibleReturnTypes(TypeRange l, TypeRange r) {                 \
    if (l.size() != r.size() || l.size() != 1)                                 \
      return false;                                                            \
    if (getElementTypeOrSelf(l[0]) != getElementTypeOrSelf(r[0]))              \
      return false;                                                            \
    return succeeded(verifyCompatibleShape(l[0], r[0]));                       \
  }

#define REDUCE_SHAPE_INFER(OP)                                                 \
  LogicalResult OP::inferReturnTypeComponents(                                 \
      MLIRContext *context, ::std::optional<Location> location,                \
      OP::Adaptor adaptor,                                                     \
      SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {           \
    Type inputType =                                                           \
        llvm::cast<TensorType>(adaptor.getInput().getType()).getElementType(); \
    ShapeAdaptor inputShape(adaptor.getInput().getType());                     \
    const Properties &prop = adaptor.getProperties();                          \
    return ReduceInferReturnTypes(inputShape, inputType, prop.axis,            \
                                  inferredReturnShapes);                       \
  }                                                                            \
  COMPATIBLE_RETURN_TYPES(OP)

REDUCE_SHAPE_INFER(tosa::ReduceAllOp)
REDUCE_SHAPE_INFER(tosa::ReduceAnyOp)
REDUCE_SHAPE_INFER(tosa::ReduceMaxOp)
REDUCE_SHAPE_INFER(tosa::ReduceMinOp)
REDUCE_SHAPE_INFER(tosa::ReduceProdOp)
REDUCE_SHAPE_INFER(tosa::ReduceSumOp)
#undef REDUCE_SHAPE_INFER
COMPATIBLE_RETURN_TYPES(tosa::ConcatOp)
#undef COMPATIBLE_RETURN_TYPES

template <typename T>
static LogicalResult verifyReduceOp(T op) {
  // All TOSA reduce Ops have input, output and axis.
  TensorType inputType = op.getInput().getType();
  TensorType outputType = op.getOutput().getType();
  int32_t reduceAxis = op.getAxis();

  if (reduceAxis < 0) {
    op.emitOpError("reduce axis must not be negative");
    return failure();
  }
  if (inputType.hasRank()) {
    int64_t inputRank = inputType.getRank();
    // We allow for a special case where the input/output shape has rank 0 and
    // axis is also 0.
    if (reduceAxis >= inputRank && !(reduceAxis == 0 && inputRank == 0)) {
      op.emitOpError("expect input tensor rank (")
          << inputRank << ") to be larger than reduce axis (" << reduceAxis
          << ")";
      return failure();
    }
  }
  if (outputType.hasRank()) {
    int64_t outputRank = outputType.getRank();
    if (inputType.hasRank() && outputRank != inputType.getRank()) {
      op.emitOpError(
          "expect output tensor rank to be equal to input tensor rank");
      return failure();
    }
    if (reduceAxis >= outputRank && !(reduceAxis == 0 && outputRank == 0)) {
      op.emitOpError("expect output tensor rank (")
          << outputRank << ") to be larger than reduce axis (" << reduceAxis
          << ")";
      return failure();
    }
    // We can only verify the reduced dimension size to be 1 if this is not
    // the special case of output rank == 0.
    if (outputRank != 0) {
      auto outputShape = outputType.getShape();
      if (!outputType.isDynamicDim(reduceAxis) &&
          outputShape[reduceAxis] != 1) {
        op.emitOpError("expect reduced dimension size to be 1, got ")
            << outputShape[reduceAxis];
        return failure();
      }
    }
  }
  return success();
}

LogicalResult tosa::ReduceAllOp::verify() { return verifyReduceOp(*this); }
LogicalResult tosa::ReduceAnyOp::verify() { return verifyReduceOp(*this); }
LogicalResult tosa::ReduceMaxOp::verify() { return verifyReduceOp(*this); }
LogicalResult tosa::ReduceMinOp::verify() { return verifyReduceOp(*this); }
LogicalResult tosa::ReduceProdOp::verify() { return verifyReduceOp(*this); }
LogicalResult tosa::ReduceSumOp::verify() { return verifyReduceOp(*this); }

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
      OpaqueProperties properties, RegionRange regions,                        \
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
NARY_SHAPE_INFER(tosa::CosOp)
NARY_SHAPE_INFER(tosa::ExpOp)
NARY_SHAPE_INFER(tosa::FloorOp)
NARY_SHAPE_INFER(tosa::GreaterEqualOp)
NARY_SHAPE_INFER(tosa::GreaterOp)
NARY_SHAPE_INFER(tosa::IdentityOp)
NARY_SHAPE_INFER(tosa::IntDivOp)
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
NARY_SHAPE_INFER(tosa::SinOp)
NARY_SHAPE_INFER(tosa::SelectOp)
NARY_SHAPE_INFER(tosa::SubOp)
NARY_SHAPE_INFER(tosa::TanhOp)
NARY_SHAPE_INFER(tosa::ErfOp)
NARY_SHAPE_INFER(tosa::SigmoidOp)
#undef PRED_SHAPE_INFER

static LogicalResult poolingInferReturnTypes(
    ShapeAdaptor inputShape, ArrayRef<int64_t> kernel, ArrayRef<int64_t> stride,
    ArrayRef<int64_t> pad,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
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
    Conv2DOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<int64_t> outputShape(4, ShapedType::kDynamic);

  int64_t inputWidth = ShapedType::kDynamic;
  int64_t inputHeight = ShapedType::kDynamic;
  int64_t weightWidth = ShapedType::kDynamic;
  int64_t weightHeight = ShapedType::kDynamic;

  // Input shape describes input width/height and batch.

  ShapeAdaptor inputShape(adaptor.getInput().getType());
  if (inputShape.hasRank()) {
    outputShape[0] = inputShape.getDimSize(0);
    inputHeight = inputShape.getDimSize(1);
    inputWidth = inputShape.getDimSize(2);
  }

  // Weight shapes describes the filter width/height and the output channels.
  ShapeAdaptor weightShape(adaptor.getWeight().getType());
  if (weightShape.hasRank()) {
    outputShape[3] = weightShape.getDimSize(0);
    weightHeight = weightShape.getDimSize(1);
    weightWidth = weightShape.getDimSize(2);
  }

  // Bias shape can describe the output channels.
  ShapeAdaptor biasShape(adaptor.getBias().getType());
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
    Conv3DOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<int64_t> outputShape(5, ShapedType::kDynamic);

  int64_t inputWidth = ShapedType::kDynamic;
  int64_t inputHeight = ShapedType::kDynamic;
  int64_t inputDepth = ShapedType::kDynamic;

  int64_t weightWidth = ShapedType::kDynamic;
  int64_t weightHeight = ShapedType::kDynamic;
  int64_t weightDepth = ShapedType::kDynamic;

  // Input shape describes input width/height and batch.
  ShapeAdaptor inputShape(adaptor.getInput().getType());
  if (inputShape.hasRank()) {
    outputShape[0] = inputShape.getDimSize(0);
    inputDepth = inputShape.getDimSize(1);
    inputHeight = inputShape.getDimSize(2);
    inputWidth = inputShape.getDimSize(3);
  }

  // Weight shapes describes the filter width/height and the output channels.
  ShapeAdaptor weightShape(adaptor.getWeight().getType());
  if (weightShape.hasRank()) {
    outputShape[4] = weightShape.getDimSize(0);
    weightDepth = weightShape.getDimSize(1);
    weightHeight = weightShape.getDimSize(2);
    weightWidth = weightShape.getDimSize(3);
  }

  // Bias shape can describe the output channels.
  ShapeAdaptor biasShape(adaptor.getBias().getType());
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
    AvgPool2dOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapeAdaptor inputShape(adaptor.getInput().getType());
  const Properties &prop = adaptor.getProperties();
  return poolingInferReturnTypes(inputShape, prop.kernel, prop.stride, prop.pad,
                                 inferredReturnShapes);
}

LogicalResult MaxPool2dOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    MaxPool2dOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapeAdaptor inputShape(adaptor.getInput().getType());
  const Properties &prop = adaptor.getProperties();
  return poolingInferReturnTypes(inputShape, prop.kernel, prop.stride, prop.pad,
                                 inferredReturnShapes);
}

LogicalResult DepthwiseConv2DOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    DepthwiseConv2DOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<int64_t> outputShape(4, ShapedType::kDynamic);

  int64_t inputWidth = ShapedType::kDynamic;
  int64_t inputHeight = ShapedType::kDynamic;
  int64_t inputChannels = ShapedType::kDynamic;

  int64_t weightWidth = ShapedType::kDynamic;
  int64_t weightHeight = ShapedType::kDynamic;
  int64_t depthChannels = ShapedType::kDynamic;

  // Input shape describes input width/height and batch.
  ShapeAdaptor inputShape(adaptor.getInput().getType());
  if (inputShape.hasRank()) {
    outputShape[0] = inputShape.getDimSize(0);
    inputHeight = inputShape.getDimSize(1);
    inputWidth = inputShape.getDimSize(2);
    inputChannels = inputShape.getDimSize(3);
  }

  // Weight shapes describes the filter width/height and the output channels.
  ShapeAdaptor weightShape(adaptor.getWeight().getType());
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
  ShapeAdaptor biasShape(adaptor.getBias().getType());
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
    TransposeConv2DOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  // outputShape is mutable.
  llvm::SmallVector<int64_t> outputShape =
      convertToMlirShape(adaptor.getOutShape());

  int64_t inputWidth = ShapedType::kDynamic;
  int64_t inputHeight = ShapedType::kDynamic;
  int64_t weightWidth = ShapedType::kDynamic;
  int64_t weightHeight = ShapedType::kDynamic;

  // Input shape describes input width/height and batch.
  ShapeAdaptor inputShape(adaptor.getInput().getType());
  if (inputShape.hasRank()) {
    outputShape[0] = ShapedType::isDynamic(outputShape[0])
                         ? inputShape.getDimSize(0)
                         : outputShape[0];
    inputHeight = inputShape.getDimSize(1);
    inputWidth = inputShape.getDimSize(2);
  }

  // Weight shapes describes the filter width/height and the output channels.
  ShapeAdaptor weightShape(adaptor.getFilter().getType());
  if (weightShape.hasRank()) {
    outputShape[3] = ShapedType::isDynamic(outputShape[3])
                         ? weightShape.getDimSize(0)
                         : outputShape[3];
    weightHeight = weightShape.getDimSize(1);
    weightWidth = weightShape.getDimSize(2);
  }

  // Bias shape can describe the output channels.
  ShapeAdaptor biasShape(adaptor.getInput().getType());
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
    IfOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<tosa::YieldOp> yieldOps;
  for (Region *region : adaptor.getRegions()) {
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
    WhileOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<tosa::YieldOp> yieldOps;
  for (auto &block : adaptor.getBody())
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

std::optional<SmallVector<int64_t, 4>> ApplyScaleOp::getShapeForUnroll() {
  if (auto vt = llvm::dyn_cast<VectorType>(getType()))
    return llvm::to_vector<4>(vt.getShape());
  return std::nullopt;
}

// parse and print of IfOp refer to the implementation of SCF dialect.
ParseResult IfOp::parse(OpAsmParser &parser, OperationState &result) {
  // Create the regions for 'then'.
  result.regions.reserve(2);
  Region *thenRegion = result.addRegion();
  Region *elseRegion = result.addRegion();

  auto &builder = parser.getBuilder();
  OpAsmParser::UnresolvedOperand cond;
  // Create a i1 tensor type for the boolean condition.
  Type i1Type = RankedTensorType::get({}, builder.getIntegerType(1));
  if (parser.parseOperand(cond) ||
      parser.resolveOperand(cond, i1Type, result.operands))
    return failure();
  // Parse optional results type list.
  if (parser.parseOptionalArrowTypeList(result.types))
    return failure();
  // Parse the 'then' region.
  if (parser.parseRegion(*thenRegion, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();

  // If we find an 'else' keyword then parse the 'else' region.
  if (!parser.parseOptionalKeyword("else")) {
    if (parser.parseRegion(*elseRegion, /*arguments=*/{}, /*argTypes=*/{}))
      return failure();
  }

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

void IfOp::print(OpAsmPrinter &p) {
  bool printBlockTerminators = false;

  p << " " << getCond();
  if (!getResults().empty()) {
    p << " -> (" << getResultTypes() << ")";
    // Print yield explicitly if the op defines values.
    printBlockTerminators = true;
  }
  p << ' ';
  p.printRegion(getThenBranch(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/printBlockTerminators);

  // Print the 'else' regions if it exists and has a block.
  auto &elseRegion = getElseBranch();
  if (!elseRegion.empty()) {
    p << " else ";
    p.printRegion(elseRegion,
                  /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/printBlockTerminators);
  }

  p.printOptionalAttrDict((*this)->getAttrs());
}

LogicalResult ReverseOp::verify() {
  TensorType inputType = getInput().getType();
  TensorType outputType = getOutput().getType();
  int32_t reverseAxis = getAxis();

  if (reverseAxis < 0)
    return emitOpError("expected non-negative reverse axis");
  if (inputType.hasRank()) {
    int64_t inputRank = inputType.getRank();
    // We allow for a special case where the input/output shape has rank 0 and
    // axis is also 0.
    if (reverseAxis >= inputRank && !(reverseAxis == 0 && inputRank == 0))
      return emitOpError("expect input tensor rank (")
             << inputRank << ") to be larger than reverse axis (" << reverseAxis
             << ")";
  }
  if (outputType.hasRank()) {
    int64_t outputRank = outputType.getRank();
    if (inputType.hasRank() && outputRank != inputType.getRank())
      return emitOpError(
          "expect output tensor rank to be equal to input tensor rank");
    if (reverseAxis >= outputRank && !(reverseAxis == 0 && outputRank == 0))
      return emitOpError("expect output tensor rank (")
             << outputRank << ") to be larger than reverse axis ("
             << reverseAxis << ")";
  }
  return success();
}

// parse and print of WhileOp refer to the implementation of SCF dialect.
ParseResult WhileOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::Argument, 4> regionArgs;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
  Region *cond = result.addRegion();
  Region *body = result.addRegion();

  OptionalParseResult listResult =
      parser.parseOptionalAssignmentList(regionArgs, operands);
  if (listResult.has_value() && failed(listResult.value()))
    return failure();

  FunctionType functionType;
  SMLoc typeLoc = parser.getCurrentLocation();
  if (failed(parser.parseColonType(functionType)))
    return failure();

  result.addTypes(functionType.getResults());

  if (functionType.getNumInputs() != operands.size()) {
    return parser.emitError(typeLoc)
           << "expected as many input types as operands "
           << "(expected " << operands.size() << " got "
           << functionType.getNumInputs() << ")";
  }

  // Resolve input operands.
  if (failed(parser.resolveOperands(operands, functionType.getInputs(),
                                    parser.getCurrentLocation(),
                                    result.operands)))
    return failure();

  // Propagate the types into the region arguments.
  for (size_t i = 0, e = regionArgs.size(); i != e; ++i)
    regionArgs[i].type = functionType.getInput(i);

  return failure(parser.parseRegion(*cond, regionArgs) ||
                 parser.parseKeyword("do") || parser.parseRegion(*body) ||
                 parser.parseOptionalAttrDictWithKeyword(result.attributes));
}

static void printInitializationList(OpAsmPrinter &parser,
                                    Block::BlockArgListType blocksArgs,
                                    ValueRange initializers,
                                    StringRef prefix = "") {
  assert(blocksArgs.size() == initializers.size() &&
         "expected same length of arguments and initializers");
  if (initializers.empty())
    return;

  parser << prefix << '(';
  llvm::interleaveComma(
      llvm::zip(blocksArgs, initializers), parser,
      [&](auto it) { parser << std::get<0>(it) << " = " << std::get<1>(it); });
  parser << ")";
}

void WhileOp::print(OpAsmPrinter &parser) {
  printInitializationList(parser, getCond().front().getArguments(), getInputs(),
                          " ");
  parser << " : ";
  parser.printFunctionalType(getInputs().getTypes(), getResults().getTypes());
  parser << ' ';
  parser.printRegion(getCond(), /*printEntryBlockArgs=*/false);
  parser << " do ";
  parser.printRegion(getBody());
  parser.printOptionalAttrDictWithKeyword((*this)->getAttrs());
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
