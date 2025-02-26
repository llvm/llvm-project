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
#include "mlir/Dialect/Quant/IR/Quant.h"
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

#include <numeric>

using namespace mlir;
using namespace mlir::tosa;

#include "mlir/Dialect/Tosa/IR/TosaOpsDialect.cpp.inc"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"

//===----------------------------------------------------------------------===//
// Tosa dialect interface includes.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaAvailability.cpp.inc"
#include "mlir/Dialect/Tosa/IR/TosaEnums.cpp.inc"
#include "mlir/Dialect/Tosa/IR/TosaInterfaces.cpp.inc"
#include "mlir/Dialect/Tosa/IR/TosaOpAvailabilityImpl.inc"

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
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/Tosa/IR/TosaOpsTypesBase.cpp.inc"
      >();
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
  if (llvm::isa<shapeType>(type) && llvm::isa<DenseIntElementsAttr>(value)) {
    return builder.create<tosa::ConstShapeOp>(
        loc, type, llvm::cast<DenseIntElementsAttr>(value));
  }
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
// Tosa utilities.
//===----------------------------------------------------------------------===//

std::optional<int64_t> idivCheck(const int64_t lhs, const int64_t rhs) {
  if (lhs % rhs != 0)
    return std::nullopt;
  return lhs / rhs;
}

//===----------------------------------------------------------------------===//
// Tosa utilities.
//===----------------------------------------------------------------------===//

static Type getStorageElementTypeOrSelf(Type type) {
  auto elementType = getElementTypeOrSelf(type);
  if (auto quantType = llvm::dyn_cast<mlir::quant::QuantizedType>(elementType))
    elementType = quantType.getStorageType();

  return elementType;
}

//===----------------------------------------------------------------------===//
// TOSA Operator Verifiers.
//===----------------------------------------------------------------------===//

template <typename T>
static LogicalResult verifyConvOp(T op) {
  // All TOSA conv ops have an input and weight arguments which must be ranked
  // tensors.
  auto inputType = llvm::dyn_cast<RankedTensorType>(op.getInput().getType());
  if (!inputType) {
    op.emitOpError("expect a ranked tensor for input, got ") << op.getInput();
    return failure();
  }

  auto weightType = llvm::dyn_cast<RankedTensorType>(op.getWeight().getType());
  if (!weightType) {
    op.emitOpError("expect a ranked tensor for weight, got ") << op.getWeight();
    return failure();
  }

  auto inputEType = inputType.getElementType();
  auto weightEType = weightType.getElementType();
  auto biasEType =
      llvm::cast<ShapedType>(op.getBias().getType()).getElementType();
  auto resultEType =
      llvm::cast<ShapedType>(op.getResult().getType()).getElementType();
  bool biasIsFloat = llvm::isa<FloatType>(biasEType);
  bool resultIsFloat = llvm::isa<FloatType>(resultEType);

  if (auto quantType = llvm::dyn_cast<mlir::quant::QuantizedType>(inputEType))
    inputEType = quantType.getStorageType();

  if (auto quantType = llvm::dyn_cast<mlir::quant::QuantizedType>(weightEType))
    weightEType = quantType.getStorageType();

  if (auto quantType = llvm::dyn_cast<mlir::quant::QuantizedType>(biasEType))
    biasEType = quantType.getStorageType();

  if (auto quantType = llvm::dyn_cast<mlir::quant::QuantizedType>(resultEType))
    resultEType = quantType.getStorageType();

  if (biasIsFloat && resultIsFloat && (biasEType != resultEType)) {
    // for now, only enforce bias element type == result element type for
    // float types.
    op.emitOpError(
        "expect both bias and result to have same element type, got ")
        << biasEType << " and " << resultEType;
    return failure();
  }

  if (isa<Float8E5M2Type>(inputEType) || isa<Float8E4M3FNType>(inputEType) ||
      isa<Float8E5M2Type>(weightEType) || isa<Float8E4M3FNType>(weightEType)) {
    if (inputEType != weightEType) {
      op.emitOpError(
          "expect both input and weight to have same element type, got ")
          << inputEType << " and " << weightEType;
      return failure();
    }
  }

  bool inputIsFloat = llvm::isa<FloatType>(inputEType);
  bool weightIsFloat = llvm::isa<FloatType>(weightEType);

  // Either both must be float or both non-float.
  if (inputIsFloat != weightIsFloat) {
    op.emitOpError(
        "expect both input and weight to be float or not together, got ")
        << inputEType << " and " << weightEType;
    return failure();
  }

  auto inputZpEType = getStorageElementTypeOrSelf(op.getInputZp().getType());
  if (inputEType != inputZpEType) {
    return op.emitOpError("expect both input and its zero point are the same "
                          "element type, got ")
           << inputEType << " and " << inputZpEType;
  }

  auto weightZpEType = getStorageElementTypeOrSelf(op.getWeightZp().getType());
  if (weightEType != weightZpEType) {
    return op.emitOpError("expect both weight and its zero point are the same "
                          "element type, got ")
           << weightEType << " and " << weightZpEType;
  }

  int64_t inputZpVal;
  if (op.getInputZeroPoint(inputZpVal).succeeded() &&
      op.verifyInputZeroPoint(inputZpVal).failed())
    return op.emitOpError(
        "input zero point must be zero for non-int8 integer types");

  int64_t weightZpVal;
  if (op.getWeightZeroPoint(weightZpVal).succeeded() &&
      op.verifyWeightZeroPoint(weightZpVal).failed())
    return op.emitOpError(
        "weight zero point must be zero for non-int8 integer types");

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

template <typename T>
static LogicalResult verifyConvOpModes(T op) {
  auto inputEType =
      llvm::cast<ShapedType>(op.getInput().getType()).getElementType();

  if (auto quantType = llvm::dyn_cast<mlir::quant::QuantizedType>(inputEType))
    inputEType = quantType.getStorageType();

  auto accType = op.getAccType();
  if (inputEType.isInteger(8) && !accType.isInteger(32))
    return op.emitOpError("accumulator type for i8 tensor is not i32");

  if (inputEType.isInteger(16) && !accType.isInteger(48))
    return op.emitOpError("accumulator type for i16 tensor is not i48");

  if (isa<Float8E5M2Type, Float8E4M3Type>(inputEType) && !accType.isF16())
    return op.emitOpError("accumulator type for f8 tensor is not f16");

  if (inputEType.isF16() && !(accType.isF16() || accType.isF32()))
    return op.emitOpError("accumulator type for f16 tensor is not f16/f32");

  if (inputEType.isBF16() && !accType.isF32())
    return op.emitOpError("accumulator type for bf16 tensor is not f32");

  if (inputEType.isF32() && !accType.isF32())
    return op.emitOpError("accumulator type for f32 tensor is not f32");

  auto resultEType =
      llvm::cast<ShapedType>(op.getResult().getType()).getElementType();

  if (auto quantType = llvm::dyn_cast<mlir::quant::QuantizedType>(resultEType))
    resultEType = quantType.getStorageType();

  // check allowed input/result element types combinations
  if ((inputEType.isInteger(8) && resultEType.isInteger(32)) ||
      (inputEType.isInteger(16) && resultEType.isInteger(48)) ||
      (isa<Float8E5M2Type>(inputEType) && resultEType.isF16()) ||
      (isa<Float8E4M3FNType>(inputEType) && resultEType.isF16()) ||
      (inputEType.isF16() && resultEType.isF16()) ||
      (inputEType.isBF16() && resultEType.isBF16()) ||
      (inputEType.isF32() && resultEType.isF32()))
    return success();

  return op.emitOpError("input/output element types are incompatible.");
}

// verify that inType and outType have same element types
template <typename T>
static LogicalResult verifySameElementTypes(T op, Type inType, Type outType) {
  auto inputType = llvm::dyn_cast<TensorType>(inType);
  auto outputType = llvm::dyn_cast<TensorType>(outType);
  if (!inputType) {
    op.emitOpError("expect shaped tensor for input, got ") << inType;
    return failure();
  }
  if (!outputType) {
    op.emitOpError("expect shaped tensor for output, got ") << outType;
    return failure();
  }
  auto inputElementType = inputType.getElementType();
  auto outputElementType = outputType.getElementType();
  auto inputQuantType =
      llvm::dyn_cast<mlir::quant::UniformQuantizedType>(inputElementType);
  auto outputQuantType =
      llvm::dyn_cast<mlir::quant::UniformQuantizedType>(outputElementType);
  if ((inputElementType.isIntOrIndexOrFloat() || inputQuantType) &&
      (outputElementType.isIntOrIndexOrFloat() || outputQuantType) &&
      inputElementType != outputElementType) {
    // only check if both element types are int/index/float/UniformQuantized
    // eg, not sure how to check quant::QuantizedType
    // this happens in test_conv2d_q_grouped_convolution in
    // tfl-to-tosa-pipeline.mlir
    op.emitOpError("expect input and output to have same element type, got ")
        << inputElementType << " and " << outputElementType;
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
  mlir::Type outputETy =
      llvm::cast<ShapedType>(getOutput().getType()).getElementType();
  if (auto quantType =
          llvm::dyn_cast<mlir::quant::UniformQuantizedType>(outputETy)) {
    outputETy = quantType.getStorageType();
  }
  if (inputETy != outputETy)
    return emitOpError("input/output element types are incompatible.");

  auto maxValAttr = getMaxValAttr();
  auto minValAttr = getMinValAttr();

  unsigned dataTypeBitWidth = inputETy.getIntOrFloatBitWidth();

  if (inputETy.isInteger(dataTypeBitWidth)) {
    // if input datatype is integer, check that the min_val/max_val attributes
    // are integer attributes, and that their type is the same as the input's
    // datatype
    auto intMaxValAttr = mlir::dyn_cast<mlir::IntegerAttr>(maxValAttr);
    auto intMinValAttr = mlir::dyn_cast<mlir::IntegerAttr>(minValAttr);
    if (!intMaxValAttr || !intMinValAttr ||
        (intMaxValAttr.getType() != intMinValAttr.getType()) ||
        (intMaxValAttr.getType() != inputETy))
      return emitOpError("min/max attributes types are incompatible with "
                         "input/output element types.");
  } else {
    // otherwise, input datatype is float, check that the min_val/max_val
    // attributes share the same type and that their type is the same as the
    // input's datatype
    auto floatMaxValAttr = mlir::dyn_cast<mlir::FloatAttr>(maxValAttr);
    auto floatMinValAttr = mlir::dyn_cast<mlir::FloatAttr>(minValAttr);
    if (!floatMaxValAttr || !floatMinValAttr ||
        (floatMaxValAttr.getType() != floatMinValAttr.getType()) ||
        (floatMaxValAttr.getType() != inputETy))
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
                                     DenseI64ArrayAttr dilation,
                                     TypeAttr accType) {
  auto zps = createZPsAsConst(builder, input, weight);
  result.addOperands({input, weight, bias, zps.first, zps.second});
  result.addAttribute("pad", pad);
  result.addAttribute("stride", stride);
  result.addAttribute("dilation", dilation);
  result.addAttribute("acc_type", accType);
  Type finalOutputType = outputType;
  auto quantAttr = buildConvOpQuantizationAttr(builder, input, weight);
  if (quantAttr) {
    finalOutputType =
        buildConvOpResultTypeInfo(builder, outputType, input, weight);
  }
  result.addTypes(finalOutputType);
}

/// Handles tosa.transpose_conv2d which has outpad and output shape
/// attributes.
static void buildTransConvOpWithQuantInfo(
    OpBuilder &builder, OperationState &result, Type outputType, Value input,
    Value weight, Value bias, DenseI64ArrayAttr outpad,
    DenseI64ArrayAttr stride, DenseI64ArrayAttr outputShape, TypeAttr accType) {
  auto zps = createZPsAsConst(builder, input, weight);
  result.addOperands({input, weight, bias, zps.first, zps.second});
  result.addAttribute("out_pad", outpad);
  result.addAttribute("stride", stride);
  result.addAttribute("out_shape", outputShape);
  result.addAttribute("acc_type", accType);
  Type finalOutputType = outputType;
  auto quantAttr = buildConvOpQuantizationAttr(builder, input, weight);
  if (quantAttr) {
    finalOutputType =
        buildConvOpResultTypeInfo(builder, outputType, input, weight);
  }
  result.addTypes(finalOutputType);
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
    result.addAttribute("a_zp", builder.getI32IntegerAttr(
                                    static_cast<int32_t>(quantAttr.getAZp())));
    result.addAttribute("b_zp", builder.getI32IntegerAttr(
                                    static_cast<int32_t>(quantAttr.getBZp())));

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
  if (quantAttr) {
    result.addAttribute("input_zp",
                        builder.getI32IntegerAttr(
                            static_cast<int32_t>(quantAttr.getInputZp())));
    result.addAttribute("output_zp",
                        builder.getI32IntegerAttr(
                            static_cast<int32_t>(quantAttr.getOutputZp())));
  }
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
  if (quantAttr) {
    // note: negateOp has attributes input1_zp and output_zp
    result.addAttribute("input1_zp",
                        builder.getI32IntegerAttr(
                            static_cast<int32_t>(quantAttr.getInputZp())));
    result.addAttribute("output_zp",
                        builder.getI32IntegerAttr(
                            static_cast<int32_t>(quantAttr.getOutputZp())));
  }
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
  if (quantAttr) {
    result.addAttribute("input_zp",
                        builder.getI32IntegerAttr(
                            static_cast<int32_t>(quantAttr.getInputZp())));
  }
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
  if (quantAttr) {
    result.addAttribute("input_zp",
                        builder.getI32IntegerAttr(
                            static_cast<int32_t>(quantAttr.getInputZp())));
  }
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
  // in the future e.g. [x, y, z] -> [x, y, z / 2 + 1]
  if (inWidth != ShapedType::kDynamic)
    outputShape[2] = inWidth / 2 + 1;

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));

  return success();
}

static LogicalResult verifyDimIsPowerOfTwo(Operation *op, const int64_t dimSize,
                                           const llvm::StringRef dimName) {
  const bool isPowerOfTwo = (dimSize & (dimSize - 1)) == 0 && dimSize > 0;
  if (!isPowerOfTwo)
    return op->emitOpError("expected ")
           << dimName << " to be a power of two, got " << dimSize;

  return success();
}

LogicalResult tosa::RFFT2dOp::verify() {
  const auto outputTypes = getResultTypes();
  if (failed(verifyCompatibleShapes(outputTypes)))
    return emitOpError("expected output shapes to match, got ") << outputTypes;

  const auto inputType = llvm::dyn_cast<RankedTensorType>(getInput().getType());
  if (!inputType)
    return success();

  const int64_t height = inputType.getDimSize(1);
  if (!ShapedType::isDynamic(height) &&
      failed(verifyDimIsPowerOfTwo(*this, height, "height")))
    return failure();

  const int64_t width = inputType.getDimSize(2);
  if (!ShapedType::isDynamic(width) &&
      failed(verifyDimIsPowerOfTwo(*this, width, "width")))
    return failure();

  const auto outputType = llvm::dyn_cast<RankedTensorType>(outputTypes[0]);
  if (!outputType)
    return success();

  // Batch and height input/output dimensions should match
  if (failed(verifyCompatibleShape(inputType.getShape().drop_back(),
                                   outputType.getShape().drop_back())))
    return emitOpError("expected batch and height dimensions of input/output "
                       "to match, got input=")
           << inputType << " output=" << outputType;

  // Output width dimension expected to be input_width / 2 + 1
  const int64_t outputWidth = outputType.getDimSize(2);
  if (!ShapedType::isDynamic(width) && !ShapedType::isDynamic(outputWidth) &&
      (outputWidth - 1) * 2 != width)
    return emitOpError(
               "expected output width to be equal to input_width / 2 + 1, got ")
           << outputWidth;

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

LogicalResult tosa::FFT2dOp::verify() {
  const auto inputRealType =
      llvm::dyn_cast<RankedTensorType>(getInputReal().getType());
  const auto inputImagType =
      llvm::dyn_cast<RankedTensorType>(getInputImag().getType());
  if (!inputRealType || !inputImagType)
    return success();

  const auto trySelectStaticDim = [](const int64_t a, const int64_t b) {
    return ShapedType::isDynamic(a) ? a : b;
  };

  const int64_t height = trySelectStaticDim(inputRealType.getDimSize(1),
                                            inputImagType.getDimSize(1));
  if (!ShapedType::isDynamic(height) &&
      failed(verifyDimIsPowerOfTwo(*this, height, "height")))
    return failure();

  const int64_t width = trySelectStaticDim(inputRealType.getDimSize(2),
                                           inputImagType.getDimSize(2));
  if (!ShapedType::isDynamic(width) &&
      failed(verifyDimIsPowerOfTwo(*this, width, "width")))
    return failure();

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
  auto paddingRank =
      cast<tosa::shapeType>(adaptor.getPadding().getType()).getRank();
  SmallVector<int64_t> outputShape;

  // If the input rank is unknown, we can infer the output rank using the
  // padding shape's rank divided by 2.
  if (!inputShape.hasRank()) {
    outputShape.resize(paddingRank / 2, ShapedType::kDynamic);
    inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
    return success();
  }

  SmallVector<int64_t> paddingValues;
  // If the paddings value is not a constant, all dimensions must be dynamic.
  if (!tosa::getConstShapeValue(adaptor.getPadding().getDefiningOp(),
                                paddingValues)) {
    outputShape.resize(inputShape.getRank(), ShapedType::kDynamic);
    inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
    return success();
  }

  outputShape.reserve(inputShape.getRank());
  for (int i = 0, s = inputShape.getRank(); i < s; i++) {
    if (inputShape.isDynamicDim(i)) {
      outputShape.push_back(ShapedType::kDynamic);
      continue;
    }
    auto padFront = paddingValues[i * 2];
    auto padBack = paddingValues[i * 2 + 1];
    if (padFront < 0 || padBack < 0) {
      // if either padding for dim i is -1, output dim is unknown
      outputShape.push_back(ShapedType::kDynamic);
      continue;
    }

    outputShape.push_back(inputShape.getDimSize(i) + padFront + padBack);
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult tosa::PadOp::verify() {
  RankedTensorType inputType = getInput1().getType();
  RankedTensorType outputType = getOutput().getType();
  auto paddingRank = cast<tosa::shapeType>(getPadding().getType()).getRank();

  if (inputType.getRank() != outputType.getRank())
    return emitOpError() << "expect same input and output tensor rank.";

  if (paddingRank != inputType.getRank() * 2)
    return emitOpError() << "expected padding tensor dim 0 to have size "
                         << inputType.getRank() * 2
                         << " (2*rank(shape1)) but got size " << paddingRank;

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

  Type inputType = getElementTypeOrSelf(adaptor.getInput1().getType());
  SmallVector<int64_t> start;
  SmallVector<int64_t> size;

  if (!tosa::getConstShapeValue(adaptor.getStart().getDefiningOp(), start) ||
      !tosa::getConstShapeValue(adaptor.getSize().getDefiningOp(), size)) {
    auto rank = cast<tosa::shapeType>(adaptor.getSize().getType()).getRank();
    SmallVector<int64_t> fallback(rank, ShapedType::kDynamic);
    inferredReturnShapes.push_back(ShapedTypeComponents(fallback, inputType));
    return success();
  }

  // if size[i] is -1, all remaining elements in dimension i are included
  // in the slice, similar to TF.
  ShapeAdaptor inputShape(adaptor.getInput1().getType());
  // initialize outputShape to all unknown
  SmallVector<int64_t> outputShape(size.size(), ShapedType::kDynamic);
  if (inputShape.hasRank()) {
    for (size_t i = 0; i < size.size(); i++) {
      if (size[i] != 0 && size[i] >= -1 && start[i] >= 0 &&
          (ShapedType::isDynamic(inputShape.getDimSize(i)) ||
           start[i] < inputShape.getDimSize(i))) {
        // size[i] is not 0 and not < -1, and start[i] is in valid range
        if (ShapedType::isDynamic(inputShape.getDimSize(i))) {
          // input shape has unknown dim[i] - only valid if size[i] > 0
          if (size[i] > 0) {
            outputShape[i] = size[i];
          }
        } else {
          // input shape has known dim[i]
          if (size[i] == -1) {
            outputShape[i] = inputShape.getDimSize(i) - start[i];
          } else if (start[i] + size[i] <= inputShape.getDimSize(i)) {
            // start[i] + size[i] is within bound of input shape's dim[i]
            outputShape[i] = size[i];
          }
        }
      }
    }
  } else {
    outputShape = convertToMlirShape(size);
  }
  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult tosa::SliceOp::verify() {
  auto inputType = llvm::dyn_cast<RankedTensorType>(getInput1().getType());
  if (!inputType)
    return success();

  auto startShapeRank =
      llvm::cast<tosa::shapeType>(getStart().getType()).getRank();
  if (inputType.getRank() != startShapeRank)
    return emitOpError(
        "length of start attribute is not equal rank of input shape");

  auto sizeShapeRank =
      llvm::cast<tosa::shapeType>(getSize().getType()).getRank();
  if (inputType.getRank() != sizeShapeRank)
    return emitOpError(
        "length of size attribute is not equal rank of input shape");

  return success();
}

LogicalResult tosa::MulOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  // mul op's output shape only depend on input1 and input2, not on shift
  ValueShapeRange twoInputs = operands.drop_back();
  llvm::SmallVector<int64_t> outShape;
  if (resolveBroadcastShape(twoInputs, outShape).failed()) {
    inferredReturnShapes.push_back(ShapedTypeComponents());
  } else {
    inferredReturnShapes.push_back(ShapedTypeComponents(outShape));
  }
  return success();
}

LogicalResult tosa::MulOp::verify() {
  auto resElemType = getElementTypeOrSelf(getOutput());

  // Verify if the element type among operands and result match tosa
  // specification.
  if (auto resIntType = dyn_cast<IntegerType>(resElemType)) {
    IntegerType lhsIntType =
        cast<IntegerType>(getElementTypeOrSelf(getInput1()));
    IntegerType rhsIntType =
        cast<IntegerType>(getElementTypeOrSelf(getInput2()));
    if (lhsIntType != rhsIntType)
      return emitOpError("requires the same element type for all operands");

    // Though the spec requires the element type of result to be i32, a more
    // relaxed way is provided at dialect level for easier cooperating with
    // other dialects.
    if (lhsIntType.getWidth() > resIntType.getWidth())
      return emitOpError("invalid data type size for operands or result");

  } else {
    // For other supported type, the spec requires requires the same element
    // type for all operands (excludes `shift` operand) and results.
    for (int i = 0; i < 2; ++i) {
      if (getElementTypeOrSelf(getOperand(i)) != resElemType)
        return emitOpError(
            "requires the same element type for all operands and results");
    }

    // verify shift has value 0 for non-integer types
    ElementsAttr shift_elem;
    if (matchPattern(getShift(), m_Constant(&shift_elem))) {
      int32_t shift = shift_elem.getValues<IntegerAttr>()[0].getInt();
      if (shift != 0) {
        return emitOpError() << "require shift to be 0 for float type";
      }
    }
  }

  // Verify the op has same ranks for all main operands (excludes extra operands
  // such as shift of mul op, so this is the only difference with the built-in
  // `SameOperandsAndResultRank` trait) and results types, if known.

  // delegate function that returns true if type is a shaped type with known
  // rank
  auto hasRank = [](const Type type) {
    if (auto shaped_type = dyn_cast<ShapedType>(type))
      return shaped_type.hasRank();

    return false;
  };

  auto rankedOperandTypes =
      llvm::to_vector(llvm::make_filter_range(getOperandTypes(), hasRank));

  auto rankedResultTypes =
      llvm::make_filter_range(getOperation()->getResultTypes(), hasRank);

  // If all operands and results are unranked, then no further verification.
  if (rankedOperandTypes.empty() && rankedResultTypes.empty())
    return success();

  // delegate function that returns rank of shaped type with known rank
  auto getRank = [](const Type type) {
    return cast<ShapedType>(type).getRank();
  };

  auto rank = !rankedOperandTypes.empty() ? getRank(*rankedOperandTypes.begin())
                                          : getRank(*rankedResultTypes.begin());

  for (size_t i = 0; i < 2; ++i) {
    if (rank != getRank(rankedOperandTypes[i])) {
      return emitOpError("operands don't have matching ranks");
    }
  }

  for (const auto type : rankedResultTypes) {
    if (rank != getRank(type)) {
      return emitOpError("result type has different rank than operands");
    }
  }

  // check for broadcast compatible shapes in first two operands (ignoring
  // shift)

  // delegate function that returns shape of shaped type
  auto getShape = [](const Type type) {
    return mlir::cast<ShapedType>(type).getShape();
  };
  SmallVector<int64_t> resultShape;
  if (!mlir::OpTrait::util::getBroadcastedShape(getShape(rankedOperandTypes[0]),
                                                getShape(rankedOperandTypes[1]),
                                                resultShape)) {
    return emitOpError("operands don't have broadcast-compatible shapes");
  }

  return success();
}

LogicalResult tosa::TableOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    TableOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapeAdaptor inputShape(adaptor.getInput1().getType());

  if (!inputShape.hasRank()) {
    inferredReturnShapes.push_back(ShapedTypeComponents());
    return success();
  }

  inferredReturnShapes.resize(1);
  inputShape.getDims(inferredReturnShapes[0]);
  return success();
}

LogicalResult tosa::TableOp::verify() {
  TensorType inputType = getInput1().getType();
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

LogicalResult
tosa::TileOp::getConstantMultiples(SmallVector<int64_t> &multiples) {
  // Multiples must be constants.
  DenseIntElementsAttr multiplesAttr;
  if (!matchPattern(getMultiples(), m_Constant(&multiplesAttr)))
    return failure();
  multiples = llvm::to_vector(
      llvm::map_range(multiplesAttr.getValues<APInt>(),
                      [](const APInt &val) { return val.getSExtValue(); }));
  return success();
}

LogicalResult tosa::TileOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    TileOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  DenseIntElementsAttr multiplesAttr;
  if (!matchPattern(adaptor.getMultiples(), m_Constant(&multiplesAttr)))
    return failure();

  SmallVector<int64_t> multiples = llvm::to_vector(
      llvm::map_range(multiplesAttr.getValues<APInt>(),
                      [](const APInt &val) { return val.getSExtValue(); }));

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

  shapeType multiplesType =
      llvm::cast<tosa::shapeType>(getMultiples().getType());

  auto multiplesRank = multiplesType.getRank();

  if (inputType.hasRank()) {
    if (inputType.getRank() != multiplesRank)
      return emitOpError("expect 'multiples' to have rank ")
             << inputType.getRank() << " but got " << multiplesRank << ".";
    if (outputType.hasRank() && inputType.getRank() != outputType.getRank())
      return emitOpError("expect same input and output tensor rank.");
  } else if (outputType.hasRank() && outputType.getRank() != multiplesRank)
    return emitOpError("expect 'multiples' array to have length ")
           << outputType.getRank() << " but got " << multiplesRank << ".";

  SmallVector<int64_t> multiples;
  if (getConstantMultiples(multiples).succeeded() &&
      llvm::any_of(multiples, [](int64_t v) { return v <= 0 && v != -1; }))
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
  llvm::SmallVector<int64_t> newShapeValue;
  if (!tosa::getConstShapeValue(adaptor.getShape().getDefiningOp(),
                                newShapeValue)) {
    auto rank = cast<tosa::shapeType>(adaptor.getShape().getType()).getRank();
    SmallVector<int64_t> fallback(rank, ShapedType::kDynamic);
    inferredReturnShapes.push_back(ShapedTypeComponents(fallback, inputType));
    return success();
  } else {
    newShapeValue = convertToMlirShape(newShapeValue);
  }

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

  SmallVector<int64_t> shapeValues;
  if (!tosa::getConstShapeValue(getShape().getDefiningOp(), shapeValues)) {
    // skip following checks if shape is not constant
    return mlir::success();
  }

  if ((int64_t)shapeValues.size() != outputType.getRank())
    return emitOpError() << "new shape does not match result rank";

  for (auto [newShapeDim, outputShapeDim] :
       zip(shapeValues, outputType.getShape())) {
    if (newShapeDim != -1 && newShapeDim != ShapedType::kDynamic &&
        outputShapeDim != ShapedType::kDynamic && newShapeDim != outputShapeDim)
      return emitOpError() << "new shape is inconsistent with result shape";

    if (newShapeDim != ShapedType::kDynamic && newShapeDim < -1)
      return emitOpError() << "new shape has invalid tensor dimension size "
                           << newShapeDim;
  }

  if (inputType.hasStaticShape()) {
    int64_t inputElementsNum = inputType.getNumElements();
    if (outputType.hasStaticShape()) {
      int64_t outputElementsNum = outputType.getNumElements();
      if (inputElementsNum != outputElementsNum) {
        return emitOpError() << "cannot reshape " << inputElementsNum
                             << " elements into " << outputElementsNum;
      }
    }

    int64_t newShapeElementsNum = std::accumulate(
        shapeValues.begin(), shapeValues.end(), 1LL,
        [](int64_t acc, int64_t dim) { return (dim > 0) ? acc * dim : acc; });
    bool isStaticNewShape =
        llvm::all_of(shapeValues, [](int64_t s) { return s > 0; });
    if ((isStaticNewShape && inputElementsNum != newShapeElementsNum) ||
        (!isStaticNewShape && newShapeElementsNum > inputElementsNum)) {
      return emitOpError() << "cannot reshape " << inputElementsNum
                           << " elements into " << newShapeElementsNum;
    }
  }

  int missingDims = llvm::count(shapeValues, -1);
  if (missingDims > 1)
    return emitOpError() << "expected at most one target dimension to be -1";

  return mlir::success();
}

template <typename T>
static LogicalResult getZeroPoint(T op, Value val, int64_t &zp) {
  ElementsAttr zpAttr;
  if (!matchPattern(val, m_Constant(&zpAttr))) {
    return failure();
  }

  Type zpElemType = zpAttr.getElementType();
  if (auto quantType =
          llvm::dyn_cast<mlir::quant::UniformQuantizedType>(zpElemType)) {
    zp = quantType.getZeroPoint();
    return success();
  }

  if (llvm::isa<FloatType>(zpElemType)) {
    if (!zpAttr.getValues<APFloat>()[0].isZero())
      return op.emitOpError(
          "non-zero zero point is not allowed for float types");
    zp = 0;
    return success();
  }

  if (llvm::isa<IntegerType>(zpElemType)) {
    zp = zpAttr.getValues<APInt>()[0].getSExtValue();
    return success();
  }

  return op.emitOpError("zero point is not allowed for unsupported types");
}

template <typename T>
static LogicalResult verifyZeroPoint(T op, Value val, int64_t &zp) {
  // TODO clean it up when the entire zero point (attribute -> input tensor
  // type) change is done. Remaining Matmul, Rescale, Negate, and AvgPool2D.
  if constexpr (!std::is_same_v<T, Conv2DOp> && !std::is_same_v<T, Conv3DOp> &&
                !std::is_same_v<T, DepthwiseConv2DOp> &&
                !std::is_same_v<T, TransposeConv2DOp>)
    return failure();

  Type zpElemType = getElementTypeOrSelf(val);

  if (!zpElemType.isIntOrFloat())
    return op.emitOpError("zero point is not integer or float typss");

  if (!zpElemType.isInteger(8) && zp != 0)
    return op.emitOpError("zero point must be zero for non-int8 integer types");

  if (zp < -128 || zp > 127)
    return failure();

  return success();
}

#define ZERO_POINT_HELPER(OP)                                                  \
  LogicalResult tosa::OP::getInputZeroPoint(int64_t &zp) {                     \
    return getZeroPoint(*this, getInputZp(), zp);                              \
  }                                                                            \
  LogicalResult tosa::OP::getWeightZeroPoint(int64_t &zp) {                    \
    return getZeroPoint(*this, getWeightZp(), zp);                             \
  }                                                                            \
  LogicalResult tosa::OP::verifyInputZeroPoint(int64_t zp) {                   \
    return verifyZeroPoint(*this, getInputZp(), zp);                           \
  }                                                                            \
  LogicalResult tosa::OP::verifyWeightZeroPoint(int64_t zp) {                  \
    return verifyZeroPoint(*this, getWeightZp(), zp);                          \
  }

ZERO_POINT_HELPER(Conv2DOp)
ZERO_POINT_HELPER(Conv3DOp)
ZERO_POINT_HELPER(DepthwiseConv2DOp)
ZERO_POINT_HELPER(TransposeConv2DOp)
#undef ZERO_POINT_HELPER

LogicalResult tosa::TransposeOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    TransposeOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapeAdaptor inputShape(adaptor.getInput1().getType());

  // If input rank and permutation length is unknown, the output rank is
  // unknown.
  if (!inputShape.hasRank()) {
    inferredReturnShapes.push_back(ShapedTypeComponents());
    return success();
  }

  const auto inputRank = inputShape.getRank();

  // This would imply the number of permutations does not match the rank of
  // the input which is illegal.
  if (adaptor.getPerms().size() != static_cast<size_t>(inputRank)) {
    return failure();
  }

  SmallVector<int64_t> outputShape;
  // Rank-0 means no permutations matter.
  if (inputRank == 0) {
    inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
    return success();
  }

  // Check whether the input dimensions are all the same.
  bool allTheSame = true;
  for (int i = 1, s = inputRank; i < s; i++) {
    if (inputShape.getDimSize(0) != inputShape.getDimSize(i)) {
      allTheSame = false;
      break;
    }
  }

  // If all of the input dimensions are the same we don't care about the
  // permutation.
  if (allTheSame) {
    outputShape.resize(inputRank, inputShape.getDimSize(0));
    inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
    return success();
  }

  outputShape.resize(inputRank, ShapedType::kDynamic);

  // Constant permutation values must be within the input rank.
  if (llvm::any_of(adaptor.getPerms(),
                   [inputRank](const auto i) { return i >= inputRank; }))
    return failure();

  outputShape.reserve(inputRank);
  for (int i = 0, s = inputRank; i < s; i++) {
    outputShape[i] = inputShape.getDimSize(adaptor.getPerms()[i]);
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult tosa::TransposeOp::verify() {
  TensorType inputType = getInput1().getType();
  TensorType outputType = getOutput().getType();
  const llvm::ArrayRef<int32_t> constantPerms = getPerms();

  if (inputType.hasRank() &&
      constantPerms.size() != static_cast<size_t>(inputType.getRank()))
    return emitOpError() << "expected perms attribute to have size "
                         << inputType.getRank() << " (input rank) but got size "
                         << constantPerms.size();
  if (inputType.hasRank() && outputType.hasRank() &&
      inputType.getRank() != outputType.getRank())
    return emitOpError()
           << "expected input tensor rank to equal result tensor rank";
  if (outputType.hasRank() &&
      constantPerms.size() != static_cast<size_t>(outputType.getRank()))
    return emitOpError() << "expected perms attribute to have size "
                         << outputType.getRank()
                         << " (output rank) but got size "
                         << constantPerms.size();

  if (!llvm::all_of(constantPerms,
                    [&constantPerms](int32_t s) {
                      return s >= 0 &&
                             static_cast<size_t>(s) < constantPerms.size();
                    }) ||
      !isPermutationVector(llvm::to_vector(llvm::map_range(
          constantPerms, [](int32_t v) -> int64_t { return v; }))))
    return emitOpError() << "expected valid permutation indices";

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

  return success();
}

LogicalResult TransposeOp::reifyResultShapes(
    OpBuilder &builder, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {

  const llvm::ArrayRef<int32_t> transposePerms = getPerms();

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

  SmallVector<int64_t> scaleInt, offsetInt, borderInt;
  if (!tosa::getConstShapeValue(adaptor.getScale().getDefiningOp(), scaleInt) ||
      !tosa::getConstShapeValue(adaptor.getOffset().getDefiningOp(),
                                offsetInt) ||
      !tosa::getConstShapeValue(adaptor.getBorder().getDefiningOp(),
                                borderInt)) {
    return failure();
  }

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

LogicalResult tosa::ResizeOp::verify() {
  const Value input = getInput();
  const Value output = getOutput();
  const RankedTensorType inputType =
      llvm::dyn_cast<RankedTensorType>(input.getType());
  const RankedTensorType outputType =
      llvm::dyn_cast<RankedTensorType>(output.getType());

  if (!inputType)
    return emitOpError("expect a ranked input tensor");
  if (!outputType)
    return emitOpError("expect a ranked output tensor");

  const int64_t oh = outputType.getDimSize(1);
  const int64_t ow = outputType.getDimSize(2);
  const int64_t ih = inputType.getDimSize(1);
  const int64_t iw = inputType.getDimSize(2);

  SmallVector<int64_t> scaleValues;
  SmallVector<int64_t> offsetValues;
  SmallVector<int64_t> borderValues;
  if (!tosa::getConstShapeValue(getScale().getDefiningOp(), scaleValues) ||
      !tosa::getConstShapeValue(getOffset().getDefiningOp(), offsetValues) ||
      !tosa::getConstShapeValue(getBorder().getDefiningOp(), borderValues)) {
    // Skip following checks if shape is not constant
    return success();
  }

  if (llvm::any_of(scaleValues, [](int64_t s) { return s <= 0; }))
    return emitOpError("expect all scale values to be > 0, got ")
           << scaleValues;

  const int64_t scaleYN = scaleValues[0];
  const int64_t scaleYD = scaleValues[1];
  const int64_t scaleXN = scaleValues[2];
  const int64_t scaleXD = scaleValues[3];

  const int64_t offsetY = offsetValues[0];
  const int64_t offsetX = offsetValues[1];

  const int64_t borderY = borderValues[0];
  const int64_t borderX = borderValues[1];

  // Don't check with input height that could be broadcast (ih != 1)
  // since Linalg, a consumer of TOSA, expects broadcasting support
  // in resize to be available. Taking the cautious approach for now,
  // we can consider removing support for broadcasting later.
  if (ih != ShapedType::kDynamic && ih != 1) {
    const std::optional<int64_t> calculatedOutHeightMinusOne =
        idivCheck((ih - 1) * scaleYN - offsetY + borderY, scaleYD);
    if (!calculatedOutHeightMinusOne.has_value())
      return emitOpError("expected (input_height - 1) * scale_y_n - offset_y + "
                         "border_y ")
             << "to be wholly divisible by scale_y_d, got ((" << ih
             << " - 1) * " << scaleYN << " - " << offsetY << " + " << borderY
             << ") / " << scaleYD;
    const int64_t calculatedOutHeight = calculatedOutHeightMinusOne.value() + 1;
    if (oh != ShapedType::kDynamic && calculatedOutHeight != oh)
      return emitOpError("calculated output height did not match expected: ")
             << "calculated=" << calculatedOutHeight << ", expected=" << oh;
  }

  // Don't check with input width that could be broadcast (iw != 1)
  // since Linalg, a consumer of TOSA, expects broadcasting support
  // in resize to be available. Taking the cautious approach for now,
  // we can consider removing support for broadcasting later.
  if (iw != ShapedType::kDynamic && iw != 1) {
    const int64_t scaledInWidth = (iw - 1) * scaleXN - offsetX + borderX;
    const std::optional<int64_t> calculatedOutWidthMinusOne =
        idivCheck(scaledInWidth, scaleXD);
    if (!calculatedOutWidthMinusOne.has_value())
      return emitOpError("expected (input_width - 1) * scale_x_n - offset_x + "
                         "border_x ")
             << "to be wholly divisible by scale_x_d, got ((" << iw
             << " - 1) * " << scaleXN << " - " << offsetX << " + " << borderX
             << ") / " << scaleXD;
    const int64_t calculatedOutWidth = calculatedOutWidthMinusOne.value() + 1;
    if (ow != ShapedType::kDynamic && calculatedOutWidth != ow)
      return emitOpError("calculated output width did not match expected: ")
             << "calculated=" << calculatedOutWidth << ", expected=" << ow;
  }

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
REDUCE_SHAPE_INFER(tosa::ReduceProductOp)
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
LogicalResult tosa::ReduceProductOp::verify() { return verifyReduceOp(*this); }
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

LogicalResult Conv2DOp::verify() {
  if (verifyConvOp(*this).failed() || verifyConvOpModes(*this).failed())
    return failure();

  llvm::ArrayRef<int64_t> padding = getPad();
  if (llvm::any_of(padding, [](int64_t p) { return p < 0; }))
    return emitOpError("expect all padding values to be >= 0, got ") << padding;

  llvm::ArrayRef<int64_t> strides = getStride();
  if (llvm::any_of(strides, [](int64_t s) { return s < 1; }))
    return emitOpError("expect all stride values to be >= 1, got ") << strides;

  llvm::ArrayRef<int64_t> dilations = getDilation();
  if (llvm::any_of(dilations, [](int64_t d) { return d < 1; }))
    return emitOpError("expect all dilation values to be >= 1, got ")
           << dilations;

  const RankedTensorType outputType =
      llvm::dyn_cast<RankedTensorType>(getOutput().getType());
  if (!outputType)
    // Skip following checks if output is not ranked
    return success();

  const RankedTensorType inputType =
      llvm::dyn_cast<RankedTensorType>(getInput().getType());
  const RankedTensorType weightType =
      llvm::dyn_cast<RankedTensorType>(getWeight().getType());

  if (inputType && weightType) {
    const auto verifyOutputSize =
        [this](const int64_t inputSize, const int64_t kernelSize,
               const int64_t outputSize, const int64_t padBefore,
               const int64_t padAfter, const int64_t stride,
               const int64_t dilation, const llvm::StringRef dimName,
               const llvm::StringRef dimAxis,
               const llvm::StringRef padBeforeName,
               const llvm::StringRef padAfterName) -> LogicalResult {
      if (inputSize == ShapedType::kDynamic ||
          kernelSize == ShapedType::kDynamic)
        return success();

      const std::optional<int64_t> calculatedOutSizeMinusOne = idivCheck(
          inputSize - 1 + padBefore + padAfter - (kernelSize - 1) * dilation,
          stride);
      if (!calculatedOutSizeMinusOne.has_value())
        return emitOpError("expected input_")
               << dimName << " - 1 + pad_" << padBeforeName << " + pad_"
               << padAfterName << " - (kernel_" << dimName
               << " - 1) * dilation_" << dimAxis
               << " to be wholly divisible by stride_" << dimAxis << ", got ("
               << inputSize << " - 1 + " << padBefore << " + " << padAfter
               << " - (" << kernelSize << " - 1) * " << dilation << ") / "
               << stride;

      const int64_t calculatedOutSize = calculatedOutSizeMinusOne.value() + 1;
      if (outputSize != ShapedType::kDynamic && calculatedOutSize != outputSize)
        return emitOpError("calculated output ")
               << dimName << " did not match expected: "
               << "calculated=" << calculatedOutSize
               << ", expected=" << outputSize;

      return success();
    };

    if (failed(verifyOutputSize(
            inputType.getDimSize(1), weightType.getDimSize(1),
            outputType.getDimSize(1), padding[0], padding[1], strides[0],
            dilations[0], "height", "y", "top", "bottom")))
      return failure();

    if (failed(verifyOutputSize(
            inputType.getDimSize(2), weightType.getDimSize(2),
            outputType.getDimSize(2), padding[2], padding[3], strides[1],
            dilations[1], "width", "x", "left", "right")))
      return failure();
  }

  const RankedTensorType biasType =
      llvm::dyn_cast<RankedTensorType>(getBias().getType());
  if (!biasType)
    // Skip following checks if bias is not ranked
    return success();

  const int64_t biasChannels = biasType.getDimSize(0);
  const int64_t outputChannels = outputType.getDimSize(3);
  if (biasChannels == ShapedType::kDynamic ||
      outputChannels == ShapedType::kDynamic)
    // Skip following checks if biasChannels or outputChannels is dynamic dim
    return success();

  if (biasChannels != outputChannels && biasChannels != 1)
    return emitOpError(
               "bias channels expected to be equal to output channels (")
           << outputChannels << ") or 1, got " << biasChannels;
  return success();
}

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

LogicalResult Conv3DOp::verify() {
  if (verifyConvOp(*this).failed() || verifyConvOpModes(*this).failed())
    return failure();
  return success();
}

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

LogicalResult DepthwiseConv2DOp::verify() {
  if (verifyConvOp(*this).failed() || verifyConvOpModes(*this).failed())
    return failure();
  return success();
}

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
  ShapeAdaptor weightShape(adaptor.getWeight().getType());
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

LogicalResult TransposeConv2DOp::verify() {
  if (verifyConvOp(*this).failed() || verifyConvOpModes(*this).failed())
    return failure();
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
  TensorType inputType = getInput1().getType();
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

// Create a rank-1 const tensor for zero point of the source tensor.
std::optional<Value> mlir::tosa::createZeroPointTensor(OpBuilder &builder,
                                                       Location loc,
                                                       Type srcElemType,
                                                       int64_t zp) {
  srcElemType = getStorageElementTypeOrSelf(srcElemType);
  auto zpType = mlir::RankedTensorType::get({1}, srcElemType);
  if (llvm::isa<FloatType>(srcElemType)) {
    auto zpAttr = DenseElementsAttr::get(
        zpType, builder.getFloatAttr(srcElemType, static_cast<double>(zp)));
    return builder.create<tosa::ConstOp>(loc, zpType, zpAttr);
  }
  if (llvm::isa<IntegerType>(srcElemType)) {
    auto zpAttr =
        DenseElementsAttr::get(zpType, builder.getIntegerAttr(srcElemType, zp));
    return builder.create<tosa::ConstOp>(loc, zpType, zpAttr);
  }
  llvm::errs() << "zero point is not allowed for unsupported data types\n";
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// TOSA Shape and Shape Operators Helper functions.
//===----------------------------------------------------------------------===//

bool mlir::tosa::isa_tosa_shape_type(mlir::Type t) {
  return mlir::isa<tosa::shapeType>(t);
}

LogicalResult
mlir::tosa::shapeType::verify(function_ref<InFlightDiagnostic()> emitError,
                              int rank) {
  if (rank < 0)
    return emitError() << "invalid rank (must be >= 0): " << rank;
  return success();
}

LogicalResult OpTrait::tosa::verifyTosaResolvableShapeOperands(Operation *op) {
  for (auto v : op->getOperands()) {
    if (mlir::isa<::mlir::tosa::shapeType>(v.getType())) {
      Operation *definingOp = v.getDefiningOp();
      if (!definingOp || !definingOp->hasTrait<TosaShapeOperator>()) {
        return op->emitOpError("shape operand is not compile time resolvable");
      }
    }
  }
  return success();
}

LogicalResult OpTrait::tosa::verifyTosaShapeOperator(Operation *op) {
  for (auto type : op->getOperandTypes()) {
    if (!mlir::isa<mlir::tosa::shapeType>(type)) {
      return op->emitOpError("must have operands with tosa shape type");
    }
  }
  for (auto type : op->getResultTypes()) {
    if (!mlir::isa<mlir::tosa::shapeType>(type)) {
      return op->emitOpError("must have result with tosa shape type");
    }
  }
  return success();
}

LogicalResult
OpTrait::tosa::verifyTosaShapeOperatorWithSameRanks(Operation *op) {
  if (failed(OpTrait::impl::verifyAtLeastNOperands(op, 1)) ||
      failed(verifyTosaShapeOperator(op)))
    return failure();

  // delegate function that returns rank of shape type
  auto getRank = [](const Type type) {
    return mlir::cast<mlir::tosa::shapeType>(type).getRank();
  };
  auto operandTypes = op->getOperandTypes();
  auto resultTypes = op->getResultTypes();

  auto rank = getRank(*op->getOperandTypes().begin());
  for (auto type : operandTypes) {
    if (getRank(type) != rank) {
      return op->emitOpError("operands don't have matching ranks");
    }
  }
  for (auto type : resultTypes) {
    if (getRank(type) != rank) {
      return op->emitOpError("result shape has different rank than operands");
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// TOSA Shape Operators verify functions.
//===----------------------------------------------------------------------===//

LogicalResult tosa::ConstShapeOp::verify() {
  // check one dimensional rank
  auto valuesRank = getValue().getType().getRank();
  if (valuesRank != 1)
    return emitOpError("expect elements in attribute value with rank 1");
  // check that number of elements in value attr equal to rank of result shape
  auto count = getValue().getNumElements();
  auto rank = (cast<tosa::shapeType>(getResult().getType())).getRank();
  if (!(count == rank || (count == 1 && rank == 0))) {
    return emitOpError("expect number of elements in attribute value (")
           << count << ") to be equal to the rank (" << rank
           << ") for the result shape type";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// TOSA Attribute Definitions.
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Tosa/IR/TosaAttributes.cpp.inc"

//===----------------------------------------------------------------------===//
// TOSA Type Definitions.
//===----------------------------------------------------------------------===//
#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Tosa/IR/TosaOpsTypesBase.cpp.inc"

//===----------------------------------------------------------------------===//
// TOSA Operator Definitions.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Tosa/IR/TosaOps.cpp.inc"
