//===- SPIRVTosaOps.cpp - MLIR SPIR-V Tosa operations ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Tosa operations in the SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir::spirv {

//===----------------------------------------------------------------------===//
// TOSA Operator Verifiers.
//===----------------------------------------------------------------------===//

namespace {

template <typename T>
static LogicalResult verifyConvOp(T op) {
  ShapedType inputTy = op.getInputType();
  ShapedType weightTy = op.getWeightType();
  ShapedType biasTy = op.getBiasType();
  ShapedType resultTy = op.getResultType();

  Type inputETy = inputTy.getElementType();
  Type weightETy = weightTy.getElementType();
  Type biasETy = biasTy.getElementType();
  Type resultETy = resultTy.getElementType();

  if (inputETy.isInteger(8) && !resultETy.isInteger(32)) {
    return op.emitOpError("expect result type to be i32, got ") << resultETy;
  }

  if (inputETy.isInteger(16) && !resultETy.isInteger(64)) {
    return op.emitOpError("expect result type to be i64, got ") << resultETy;
  }

  if (inputETy.isF16() && !resultETy.isF16()) {
    return op.emitOpError("expect result type to be f16, got ") << resultETy;
  }

  if (inputETy.isF32() && !resultETy.isF32()) {
    return op.emitOpError("expect result type to be f32, got ") << resultETy;
  }

  if (biasETy != resultETy) {
    return op.emitOpError("element types of bias and result must be the same");
  }

  TosaExtAccType accType = op.getAccType();
  if (inputETy.isInteger(8) && accType != TosaExtAccType::INT32) {
    return op.emitOpError("accumulator type for i8 tensorARM is not i32");
  }

  if (inputETy.isInteger(16) && accType != TosaExtAccType::INT48) {
    return op.emitOpError("accumulator type for i16 tensorARM is not i48");
  }

  if (inputETy.isF16() &&
      !(accType == TosaExtAccType::FP16 || accType == TosaExtAccType::FP32)) {
    return op.emitOpError(
        "accumulator type for f16 tensorARM is not f16 or f32");
  }

  if (inputETy.isBF16() && accType != TosaExtAccType::FP32) {
    return op.emitOpError("accumulator type for bf16 tensorARM is not f32");
  }

  if (inputETy.isF32() && accType != TosaExtAccType::FP32) {
    return op.emitOpError("accumulator type for f32 tensorARM is not f32");
  }

  DenseIntOrFPElementsAttr inputZpAttr;
  if (!matchPattern(op.getInputZp(), m_Constant(&inputZpAttr))) {
    return op.emitOpError(
        "input_zp must be a tensorARM of an integer/float constant");
  }

  if (inputZpAttr.size() != 1) {
    return op.emitOpError("input_zp must have a single element");
  }

  Type inputZpETy = inputZpAttr.getElementType();
  if (inputZpETy != inputETy) {
    return op.emitOpError(
        "element types of input_zp and input must be the same");
  }

  DenseIntOrFPElementsAttr weightZpAttr;
  if (!matchPattern(op.getWeightZp(), m_Constant(&weightZpAttr))) {
    return op.emitOpError(
        "weight_zp must be a tensorARM of an integer/float constant");
  }

  if (weightZpAttr.size() != 1) {
    return op.emitOpError("weight_zp must have a single element");
  }

  Type weightZpETy = weightZpAttr.getElementType();
  if (weightZpETy != weightETy) {
    return op.emitOpError(
        "element types of weight_zp and weight must be the same");
  }

  if (isa<IntegerType>(inputZpETy)) {
    if ((inputZpETy.getIntOrFloatBitWidth() != 8) &&
        !inputZpAttr.getValues<APInt>()[0].isZero()) {
      return op.emitOpError(
          "input_zp element value must be zero for non-int8 types.");
    }
  } else {
    if (!inputZpAttr.getValues<APFloat>()[0].isZero()) {
      return op.emitOpError(
          "input_zp element value must be zero for non-int8 types.");
    }
  }

  if (isa<IntegerType>(weightZpETy)) {
    if ((weightZpETy.getIntOrFloatBitWidth() != 8) &&
        !weightZpAttr.getValues<APInt>()[0].isZero()) {
      return op.emitOpError(
          "weight_zp element value must be zero for non-int8 types.");
    }
  } else {
    if (!weightZpAttr.getValues<APFloat>()[0].isZero()) {
      return op.emitOpError(
          "weight_zp element value must be zero for non-int8 types.");
    }
  }

  return success();
}

// Verify that inType and outType have same element types
template <typename TOp>
static LogicalResult verifySameElementTypes(TOp op, ShapedType inputType,
                                            ShapedType outputType) {
  if (!inputType) {
    op.emitOpError("expect shaped tensorARM for input, got ") << inputType;
    return failure();
  }
  if (!outputType) {
    op.emitOpError("expect shaped tensorARM for output, got ") << outputType;
    return failure();
  }

  Type inputElementType = inputType.getElementType();
  Type outputElementType = outputType.getElementType();
  if (inputElementType != outputElementType) {
    op.emitOpError("expect input and output to have same element type, got ")
        << inputElementType << " and " << outputElementType;
    return failure();
  }
  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// spirv.TosaArgmaxOp
//===----------------------------------------------------------------------===//

LogicalResult TosaArgMaxOp::verify() {
  ShapedType inputTy = getInputType();
  ShapedType resultTy = getResultType();

  if (inputTy.hasRank() && resultTy.hasRank() &&
      resultTy.getRank() !=
          (inputTy.getRank() > 1 ? inputTy.getRank() - 1 : 1)) {
    return emitOpError(
               "result rank must be max of 1 and (input rank - 1), got ")
           << resultTy.getRank();
  }

  const uint32_t axis = getAxis();
  if (inputTy.hasRank() && axis >= inputTy.getRank()) {
    return emitOpError(
               "specified axis is greater than the rank of input, got axis = ")
           << axis << " and input rank = " << inputTy.getRank();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// spirv.TosaAvgPool2DOp
//===----------------------------------------------------------------------===//

LogicalResult TosaAvgPool2DOp::verify() {
  ShapedType inputTy = getInputType();
  ShapedType resultTy = getResultType();
  Type inputETy = inputTy.getElementType();
  Type resultETy = resultTy.getElementType();

  TosaExtAccType accType = getAccType();
  if (isa<IntegerType>(inputETy) && accType != TosaExtAccType::INT32) {
    return emitOpError("accumulator type for integer tensorARM is not i32");
  }

  if (inputETy.isF16() &&
      !(accType == TosaExtAccType::FP16 || accType == TosaExtAccType::FP32)) {
    return emitOpError("accumulator type for f16 tensorARM is not f16/f32");
  }

  if (inputETy.isF32() && accType != TosaExtAccType::FP32) {
    return emitOpError("accumulator type for f32 tensorARM is not f32");
  }

  if (inputETy != resultETy) {
    return emitOpError("input and output element types must be the same");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// spirv.TosaConv2DOp
//===----------------------------------------------------------------------===//

LogicalResult TosaConv2DOp::verify() { return verifyConvOp(*this); }

//===----------------------------------------------------------------------===//
// spirv.TosaConv3DOp
//===----------------------------------------------------------------------===//

LogicalResult TosaConv3DOp::verify() { return verifyConvOp(*this); }

//===----------------------------------------------------------------------===//
// SPIRV Tosa DepthwiseConv2D Ops:
//===----------------------------------------------------------------------===//

LogicalResult TosaDepthwiseConv2DOp::verify() { return verifyConvOp(*this); }

//===----------------------------------------------------------------------===//
// spirv.TosaFFT2DOp
//===----------------------------------------------------------------------===//

LogicalResult TosaFFT2DOp::verify() {
  ShapedType inputRealTy = getInputRealType();
  ShapedType inputImagTy = getInputImagType();
  ShapedType resultRealTy = getResultRealType();
  ShapedType resultImagTy = getResultImagType();

  if (inputRealTy != inputImagTy || inputRealTy != resultRealTy ||
      inputImagTy != resultImagTy) {
    return emitOpError("real input type, imaginary input type, and types of "
                       "real and imaginary parts of result must be the same");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// spirv.TosaMatMulOp
//===----------------------------------------------------------------------===//

LogicalResult TosaMatMulOp::verify() {
  ShapedType aTy = getAType();
  ShapedType bTy = getBType();
  ShapedType resultTy = getResultType();

  Type aETy = aTy.getElementType();
  Type bETy = bTy.getElementType();
  Type resultETy = resultTy.getElementType();

  if (aETy != bETy) {
    return emitOpError("expect same element type for inputs a and b, got ")
           << aETy << " and " << bETy;
  }

  if (aETy.isInteger(8) && !resultETy.isInteger(32)) {
    return emitOpError("expect result element type to be i32, got ")
           << resultETy;
  }

  if (aETy.isInteger(16) && !resultETy.isInteger(64)) {
    return emitOpError("expect result element type to be i64, got ")
           << resultETy;
  }

  if (aETy.isF16() && !(resultETy.isF16() || resultETy.isF32())) {
    return emitOpError("expect result element type to be f16 or f32, got ")
           << resultETy;
  }

  if (aETy.isF32() && !resultETy.isF32()) {
    return emitOpError("expect result element type to be f32, got ")
           << resultETy;
  }

  DenseIntOrFPElementsAttr aZpAttr;
  if (!matchPattern(getAZp(), m_Constant(&aZpAttr))) {
    return emitOpError("a_zp must be a tensorARM of an integer/float constant");
  }

  if (aZpAttr.size() != 1) {
    return emitOpError("a_zp must have a single element");
  }

  DenseIntOrFPElementsAttr bZpAttr;
  if (!matchPattern(getBZp(), m_Constant(&bZpAttr))) {
    return emitOpError("b_zp must be a tensorARM of an integer/float constant");
  }

  if (bZpAttr.size() != 1) {
    return emitOpError("b_zp must have a single element");
  }

  if (isa<IntegerType>(aETy)) {
    if ((aETy.getIntOrFloatBitWidth() != 8) &&
        (!aZpAttr.getValues<APInt>()[0].isZero() ||
         !bZpAttr.getValues<APInt>()[0].isZero())) {
      return emitOpError("a_zp and b_zp must be zero for non-int8 types.");
    }
  } else {
    if (!aZpAttr.getValues<APFloat>()[0].isZero() ||
        !bZpAttr.getValues<APFloat>()[0].isZero()) {
      return emitOpError("a_zp and b_zp must be zero for non-int8 types.");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// spirv.TosaMaxPool2DOp
//===----------------------------------------------------------------------===//

LogicalResult TosaMaxPool2DOp::verify() {
  return verifySameElementTypes(*this, getInputType(), getResultType());
}

//===----------------------------------------------------------------------===//
// spirv.TosaRFFT2DOp
//===----------------------------------------------------------------------===//

LogicalResult TosaRFFT2DOp::verify() {
  ShapedType inputTy = getInputRealType();
  ShapedType resultRealTy = getResultRealType();
  ShapedType resultImagTy = getResultImagType();

  if (inputTy.getElementType() != resultRealTy.getElementType() ||
      inputTy.getElementType() != resultImagTy.getElementType()) {
    return emitOpError(
        "input element type and element types of real and imaginary parts of "
        "result must be the same");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SPIRV Tosa TransposeConv2D Ops:
//===----------------------------------------------------------------------===//

LogicalResult TosaTransposeConv2DOp::verify() { return verifyConvOp(*this); }

//===----------------------------------------------------------------------===//
// spirv.TosaMulOp
//===----------------------------------------------------------------------===//
LogicalResult TosaMulOp::verify() {
  ShapedType resType = getResultType();
  Type resElemType = resType.getElementType();

  // Verify if the element type amoung operands and result match tosa
  // specification.
  if (auto resIntType = dyn_cast<IntegerType>(resElemType)) {
    auto lhsIntType = getInput1Type().getElementType();
    auto rhsIntType = getInput2Type().getElementType();
    if (lhsIntType != rhsIntType)
      return emitOpError(
          "requires the same element type for all input operands");

    // Though the spec requires the element type of result to be i32, a more
    // relaxed way is provided at dialect level for easier cooperating with
    // other dialects.
    if (!lhsIntType.isInteger() ||
        cast<IntegerType>(lhsIntType).getWidth() > resIntType.getWidth())
      return emitOpError("invalid data type size for operands or result");
  } else {
    // For other supported type, the spec requires requires the same element
    // type for all operands (excludes `shift` operand) and results.
    for (int i = 0; i < 2; ++i) {
      if (getElementTypeOrSelf(getOperand(i)) != resElemType)
        return emitOpError(
            "requires the same element type for all operands and results");
    }
  }

  auto compareRank = [](const ShapedType type, const ShapedType against) {
    return type.hasRank() && against.hasRank() &&
           type.getRank() == against.getRank();
  };

  for (int i = 0; i < 2; ++i) {
    if (!compareRank(cast<ShapedType>(getOperand(i).getType()), resType))
      return emitOpError("result type has different rank than operands");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// spirv.TosaCastOp
//===----------------------------------------------------------------------===//
LogicalResult TosaCastOp::verify() {
  Type inputETy = getInputType().getElementType();
  Type outputETy = getResultType().getElementType();

  // input element type: bool
  if (inputETy.isInteger(1)) {
    if (outputETy.isInteger(8) || outputETy.isInteger(16) ||
        outputETy.isInteger(32)) {
      return success();
    }
  }
  // input element type: int8
  if (inputETy.isInteger(8)) {
    if (outputETy.isInteger(1) || outputETy.isInteger(16) ||
        outputETy.isInteger(32) || outputETy.isF16() || outputETy.isBF16() ||
        outputETy.isF32()) {
      return success();
    }
  }
  // input element type: int16
  if (inputETy.isInteger(16)) {
    if (outputETy.isInteger(1) || outputETy.isInteger(8) ||
        outputETy.isInteger(32) || outputETy.isF16() || outputETy.isBF16() ||
        outputETy.isF32()) {
      return success();
    }
  }
  // input element type: int32
  if (inputETy.isInteger(32)) {
    if (outputETy.isInteger(1) || outputETy.isInteger(8) ||
        outputETy.isInteger(16) || outputETy.isF16() || outputETy.isBF16() ||
        outputETy.isF32()) {
      return success();
    }
  }
  // input element type: bf16 or fp16
  if (inputETy.isBF16() || inputETy.isF16()) {
    if (outputETy.isInteger(8) || outputETy.isInteger(16) ||
        outputETy.isInteger(32) || outputETy.isF32()) {
      return success();
    }
  }
  // input element type: fp32
  if (inputETy.isF32()) {
    if (outputETy.isInteger(8) || outputETy.isInteger(16) ||
        outputETy.isInteger(32) || outputETy.isF16() || outputETy.isBF16()) {
      return success();
    }
  }

  return emitOpError("input/output element types are incompatible: ")
         << inputETy << " and " << outputETy;
}

//===----------------------------------------------------------------------===//
// spirv.TosaClampOp
//===----------------------------------------------------------------------===//
LogicalResult TosaClampOp::verify() {
  Type inputETy = getInputType().getElementType();
  Type outputETy = getResultType().getElementType();

  if (inputETy != outputETy)
    return emitOpError("input/output element types are incompatible");

  auto minValAttr = dyn_cast<TypedAttr>(getMinVal());
  auto maxValAttr = dyn_cast<TypedAttr>(getMaxVal());
  if (!minValAttr || !maxValAttr ||
      (minValAttr.getType() != maxValAttr.getType()) ||
      (minValAttr.getType() != inputETy)) {

    return emitOpError("min/max attributes types are incompatible with "
                       "input/output element types.");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// spirv.TosaConcatOp
//===----------------------------------------------------------------------===//
LogicalResult TosaConcatOp::verify() {
  ShapedType outType = getResultType();
  for (Value input : getInput1()) {
    if (verifySameElementTypes(*this, cast<ShapedType>(input.getType()),
                               outType)
            .failed()) {
      return failure();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// spirv.TosaPadOp
//===----------------------------------------------------------------------===//
LogicalResult TosaPadOp::verify() {
  if (verifySameElementTypes(*this, getInput1Type(), getResultType())
          .failed()) {
    return failure();
  }

  auto inputETy = cast<ShapedType>(getInput1().getType()).getElementType();

  DenseIntOrFPElementsAttr padConstAttr;
  if (!matchPattern(getPadConst(), m_Constant(&padConstAttr)) ||
      (padConstAttr.getElementType() != inputETy)) {
    return emitOpError(
        "PadConst element type is not same as input element type.");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// spirv.TosaSliceOp
//===----------------------------------------------------------------------===//
LogicalResult TosaSliceOp::verify() {
  ShapedType inputTy = getInput1Type();
  ShapedType outputTy = getResultType();
  ShapedType startTy = cast<ShapedType>(getStart().getType());
  ShapedType sizeTy = cast<ShapedType>(getSize().getType());

  if (verifySameElementTypes(*this, inputTy, outputTy).failed()) {
    return failure();
  }

  if (inputTy.hasRank() && startTy.hasRank() &&
      startTy.getShape()[0] != ShapedType::kDynamic &&
      inputTy.getRank() != startTy.getShape()[0]) {
    return emitOpError("length of start is not equal to rank of input shape");
  }

  if (inputTy.hasRank() && sizeTy.hasRank() &&
      sizeTy.getShape()[0] != ShapedType::kDynamic &&
      inputTy.getRank() != sizeTy.getShape()[0]) {
    return emitOpError("length of size is not equal to rank of input shape");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// spirv.TosaTileOp
//===----------------------------------------------------------------------===//
LogicalResult TosaTileOp::verify() {
  ShapedType inputTy = getInput1Type();
  ShapedType outputTy = getResultType();
  ShapedType multiplesTy = cast<ShapedType>(getMultiples().getType());

  if (verifySameElementTypes(*this, inputTy, outputTy).failed()) {
    return failure();
  }

  if (inputTy.hasRank() && outputTy.hasRank() &&
      inputTy.getRank() != outputTy.getRank()) {
    return emitOpError("expect same input and output tensorARM rank");
  }

  if (inputTy.hasRank() && multiplesTy.hasRank() &&
      multiplesTy.getShape()[0] != ShapedType::kDynamic &&
      inputTy.getRank() != multiplesTy.getShape()[0]) {
    return emitOpError("expect 'multiples' array to have length ")
           << inputTy.getRank() << " but got " << multiplesTy.getShape()[0];
  }
  return success();
}

//===----------------------------------------------------------------------===//
// spirv.TosaTransposeOp
//===----------------------------------------------------------------------===//
LogicalResult TosaTransposeOp::verify() {
  ShapedType inputTy = getInput1Type();
  ShapedType outputTy = getResultType();
  ShapedType permsTy = cast<DenseElementsAttr>(getPerms()).getType();

  if (verifySameElementTypes(*this, inputTy, outputTy).failed()) {
    return failure();
  }

  if (inputTy.hasRank() && outputTy.hasRank() &&
      inputTy.getRank() != outputTy.getRank()) {
    return emitOpError("expect same input and output tensorARM rank");
  }

  if (inputTy.hasRank() && permsTy.getShape()[0] != ShapedType::kDynamic &&
      inputTy.getRank() != permsTy.getShape()[0]) {
    return emitOpError("expect permutation tensorARM to have length ")
           << inputTy.getRank() << " but got " << permsTy.getShape()[0];
  }
  return success();
}

//===----------------------------------------------------------------------===//
// spirv.TosaGatherOp
//===----------------------------------------------------------------------===//
LogicalResult TosaGatherOp::verify() {
  return verifySameElementTypes(*this, getValues().getType(), getResultType());
}

//===----------------------------------------------------------------------===//
// spirv.TosaScatterOp
//===----------------------------------------------------------------------===//
LogicalResult TosaScatterOp::verify() {
  if (verifySameElementTypes(*this, getValuesIn().getType(),
                             getValuesOut().getType())
          .failed()) {
    return failure();
  }
  if (verifySameElementTypes(*this, getInputType(), getValuesOut().getType())
          .failed()) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// spirv.TosaRescaleOp
//===----------------------------------------------------------------------===//

LogicalResult TosaRescaleOp::verify() {
  ShapedType inputTy = getInputType();
  ShapedType outputTy = getResultType();

  Type inputETy = inputTy.getElementType();
  if (!isa<IntegerType>(inputETy)) {
    return emitOpError("expect input to have integer element type, got ")
           << inputETy;
  }

  if (inputTy.hasRank() != outputTy.hasRank() ||
      (inputTy.hasRank() && inputTy.getShape() != outputTy.getShape())) {
    return emitOpError("Shape of input and output must be same");
  }

  Type outputETy = outputTy.getElementType();
  if (!isa<IntegerType>(outputETy)) {
    return emitOpError("expect output to have integer element type, got ")
           << outputETy;
  }

  DenseIntElementsAttr inputZpAttr;
  if (!matchPattern(getInputZp(), m_Constant(&inputZpAttr))) {
    return emitOpError(
        "input_zp must be single element tensorARM of an integer constant");
  }

  if (inputZpAttr.size() != 1) {
    return emitOpError("input_zp must have a single element");
  }

  Type inputZPETy = inputZpAttr.getElementType();
  if (inputZPETy != inputETy) {
    return emitOpError(
        "input_zp element type is not same as input element type");
  }

  if (auto inputAPInt = inputZpAttr.getValues<APInt>()[0];
      !inputAPInt.isZero()) {
    if (!inputETy.isInteger(8) &&
        !(inputETy.isInteger(16) && getInputUnsigned())) {
      return emitOpError("expect input_zp of 0, got ")
             << inputAPInt.getZExtValue();
    }
    if (inputETy.isInteger(16) && getInputUnsigned()) {
      if (uint64_t input_zp = inputAPInt.getZExtValue(); input_zp != 32768u) {
        return emitOpError("expect input_zp of 0 or 32768 for unsigned int16 "
                           "input, got ")
               << input_zp;
      }
    }
  }

  DenseIntElementsAttr outputZpAttr;
  if (!matchPattern(getOutputZp(), m_Constant(&outputZpAttr))) {
    return emitOpError(
        "output_zp must be single element tensorARM of an integer constant");
  }

  if (outputZpAttr.size() != 1) {
    return emitOpError("output_zp must have a single element");
  }

  auto outputZPETy = outputZpAttr.getElementType();
  if (outputZPETy != outputETy) {
    return emitOpError(
        "output_zp element type is not same as output element type");
  }

  if (auto outputAPInt = outputZpAttr.getValues<APInt>()[0];
      !outputAPInt.isZero()) {
    if (!outputETy.isInteger(8) &&
        !(outputETy.isInteger(16) && getOutputUnsigned())) {
      return emitOpError("expect output_zp of 0, got ")
             << outputAPInt.getZExtValue();
    }
    if (outputETy.isInteger(16) && getOutputUnsigned()) {
      if (auto output_zp = outputAPInt.getZExtValue(); output_zp != 32768u) {
        return emitOpError("expect output_zp of 0 or 32768 for unsigned int16 "
                           "output, got ")
               << output_zp;
      }
    }
  }

  auto shiftTy = cast<ShapedType>(getShift().getType());
  auto multiplierTy = cast<ShapedType>(getMultiplier().getType());

  Type shiftETy = shiftTy.getElementType();
  if (!shiftETy.isInteger(8)) {
    return emitOpError("shift element type must be i8");
  }

  bool scale32 = getScale32();
  auto multiplierETy = multiplierTy.getElementType();
  if (scale32 && !multiplierETy.isInteger(32)) {
    return emitOpError("expect i32 element type for multiplier for "
                       "scale32=true, got ")
           << multiplierETy;
  }

  if (!scale32 && !multiplierETy.isInteger(16)) {
    return emitOpError("expect i16 element type for multiplier for "
                       "scale32=false, got ")
           << multiplierETy;
  }

  // multiplier/shift must have shape = {numChannels},
  // where numChannel is 1 if per_channel = false
  // otherwise numChannel is dimension in input shape's last axis
  int64_t numChannels = 1;
  if (getPerChannel()) {
    ArrayRef<int64_t> inputShape = inputTy.getShape();
    numChannels = inputTy.hasRank() ? inputShape[inputShape.size() - 1]
                                    : ShapedType::kDynamic;
  }

  if (multiplierTy.hasRank() &&
      multiplierTy.getShape()[0] != ShapedType::kDynamic &&
      numChannels != ShapedType::kDynamic &&
      multiplierTy.getShape()[0] != numChannels) {
    return emitOpError("expect shape of { ")
           << numChannels << " } for multiplier input, got { "
           << multiplierTy.getShape()[0] << " }";
  }

  if (shiftTy.hasRank() && shiftTy.getShape()[0] != ShapedType::kDynamic &&
      numChannels != ShapedType::kDynamic &&
      shiftTy.getShape()[0] != numChannels) {
    return emitOpError("expect shape of { ")
           << numChannels << " } for shift input, got { "
           << shiftTy.getShape()[0] << " }";
  }

  if (inputETy.isInteger(8) || inputETy.isInteger(16) ||
      inputETy.isInteger(32) || inputETy.isInteger(64)) {
    if (outputETy.isInteger(8) || outputETy.isInteger(16) ||
        outputETy.isInteger(32)) {
      return success();
    }
  }

  return emitOpError("input/output element types are incompatible: ")
         << inputETy << " and " << outputETy;
}

//===----------------------------------------------------------------------===//
// spirv.TosaReverseOp
//===----------------------------------------------------------------------===//

LogicalResult TosaReverseOp::verify() {
  ShapedType inputTy = getInput1Type();
  ShapedType outputTy = getResultType();

  if (verifySameElementTypes(*this, inputTy, outputTy).failed()) {
    return failure();
  }

  if (inputTy.getRank() != outputTy.getRank()) {
    return emitOpError(
        "expect output tensorARM rank to be equal to input rank");
  }

  const uint32_t axis = getAxis();
  if (inputTy.hasRank() && axis >= inputTy.getRank()) {
    return emitOpError("specified axis is greater than the rank of input");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// spirv.TosaSelectOp
//===----------------------------------------------------------------------===//
LogicalResult TosaSelectOp::verify() {
  if (verifySameElementTypes(*this, getInput2Type(), getResultType())
          .failed()) {
    return failure();
  }
  if (verifySameElementTypes(*this, getInput3Type(), getResultType())
          .failed()) {
    return failure();
  }

  auto predicateType = getInput1Type();
  if (!predicateType) {
    emitOpError("expect shaped tensorARM for input1, got ") << getInput1Type();
    return failure();
  }

  Type predicateElementType = predicateType.getElementType();
  if (!predicateElementType.isInteger(1)) {
    emitOpError("expect element type of bool for input1, got ")
        << predicateElementType;
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// spirv.TosaReshapeOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult TosaReshapeOp::verify() {
  ShapedType inputType = getInput1Type();
  ShapedType outputType = getResultType();
  if (verifySameElementTypes(*this, inputType, outputType).failed()) {
    return failure();
  }

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

} // namespace mlir::spirv
