//===- TosaOps.cpp - MLIR SPIR-V operations -------------------------------===//
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
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// TOSA Operator Verifiers.
//===----------------------------------------------------------------------===//

// Get value attr from spirv::ConstantOp or
// spirv::EXTConstantCompositeReplicateOp
template <typename TAttr>
static LogicalResult getConstAttr(Value value, TAttr &valAttr) {
  if (auto constOp = value.template getDefiningOp<spirv::ConstantOp>()) {
    valAttr = dyn_cast<TAttr>(constOp.getValue());
  } else if (auto constCompositeReplicateOp =
                 value.template getDefiningOp<
                     spirv::EXTConstantCompositeReplicateOp>()) {
    auto splatAttr = constCompositeReplicateOp.getValue();
    auto denseValAttr = SplatElementsAttr::get(
        cast<ShapedType>(constCompositeReplicateOp.getType()), splatAttr);
    valAttr = dyn_cast<TAttr>(denseValAttr);
  }

  return valAttr ? success() : failure();
}

template <typename T, typename TAdaptor>
static LogicalResult verifyConvOp(T op, TAdaptor adaptor) {
  auto inputTy = cast<ShapedType>(op.getInput().getType());
  auto weightTy = cast<ShapedType>(op.getWeight().getType());
  auto biasTy = cast<ShapedType>(op.getBias().getType());
  auto resultTy = cast<ShapedType>(op.getType());

  if constexpr (std::is_same_v<T, spirv::TosaConv3DOp>) {
    if (inputTy.hasRank() && inputTy.getRank() != 5) {
      return op.emitOpError("input rank must be 5");
    }

    if (weightTy.hasRank() && weightTy.getRank() != 5) {
      return op.emitOpError("weight rank must be 5");
    }

    if (resultTy.hasRank() && resultTy.getRank() != 5) {
      return op.emitOpError("result rank must be 5");
    }
  } else {
    if (inputTy.getRank() != 4) {
      return op.emitOpError("input rank must be 4");
    }

    if (weightTy.hasRank() && weightTy.getRank() != 4) {
      return op.emitOpError("weight rank must be 4");
    }

    if (resultTy.hasRank() && resultTy.getRank() != 4) {
      return op.emitOpError("result rank must be 4");
    }
  }

  if (biasTy.hasRank() && biasTy.getRank() != 1) {
    return op.emitOpError("bias rank must be 1");
  }

  auto inputETy = inputTy.getElementType();
  auto weightETy = weightTy.getElementType();
  auto biasETy = biasTy.getElementType();
  auto resultETy = resultTy.getElementType();

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

  DenseIntOrFPElementsAttr inputZpAttr;
  if (getConstAttr(adaptor.getInputZp(), inputZpAttr).failed()) {
    return op.emitOpError(
        "input_zp must be a tensorARM of an integer/float constant");
  }

  if (inputZpAttr.size() != 1) {
    return op.emitOpError("input_zp must have a single element");
  }

  auto inputZpETy = inputZpAttr.getElementType();
  if (inputZpETy != inputETy) {
    return op.emitOpError(
        "element types of input_zp and input must be the same");
  }

  DenseIntOrFPElementsAttr weightZpAttr;
  if (getConstAttr(adaptor.getWeightZp(), weightZpAttr).failed()) {
    return op.emitOpError(
        "weight_zp must be a tensorARM of an integer/float constant");
  }

  if (weightZpAttr.size() != 1) {
    return op.emitOpError("weight_zp must have a single element");
  }

  auto weightZpETy = weightZpAttr.getElementType();
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

  BoolAttr localBoundAttr;
  if ((getConstAttr(adaptor.getLocalBound(), localBoundAttr).failed())) {
    return op.emitOpError("local bound must be a constant boolean");
  }

  return success();
}

template <typename T>
static LogicalResult verifyConvOpModes(T op) {
  IntegerAttr accTypeAttr;
  if (getConstAttr(op.getAccType(), accTypeAttr).failed()) {
    return op.emitOpError("accumulator type must be a constant integer");
  }

  int accType = accTypeAttr.getInt();
  if (accType != 1 && accType != 2 && accType != 3 && accType != 4) {
    return op.emitOpError("accumulator type can only have values 1/2/3/4 "
                          "corresponding to i32/f16/f32/i48");
  }

  auto inputTy = cast<ShapedType>(op.getInput().getType());
  auto inputETy = inputTy.getElementType();

  if (inputETy.isInteger(8) && accType != 1) {
    return op.emitOpError("accumulator type for i8 tensorARM is not i32");
  }

  if (inputETy.isInteger(16) && accType != 4) {
    return op.emitOpError("accumulator type for i16 tensorARM is not i48");
  }

  if (inputETy.isF16() && !(accType == 2 || accType == 3)) {
    return op.emitOpError(
        "accumulator type for f16 tensorARM is not f16 or f32");
  }

  if (inputETy.isBF16() && accType != 3) {
    return op.emitOpError("accumulator type for bf16 tensorARM is not f32");
  }

  if (inputETy.isF32() && accType != 3) {
    return op.emitOpError("accumulator type for f32 tensorARM is not f32");
  }

  return success();
}

// Verify that inType and outType have same element types
template <typename TOp>
static LogicalResult verifySameElementTypes(TOp op, Type inType, Type outType) {
  auto inputType = dyn_cast<ShapedType>(inType);
  auto outputType = dyn_cast<ShapedType>(outType);

  if (!inputType) {
    op.emitOpError("expect shaped tensorARM for input, got ") << inType;
    return failure();
  }
  if (!outputType) {
    op.emitOpError("expect shaped tensorARM for output, got ") << outType;
    return failure();
  }
  auto inputElementType = inputType.getElementType();
  auto outputElementType = outputType.getElementType();

  if (inputElementType != outputElementType) {
    op.emitOpError("expect input and output to have same element type, got ")
        << inputElementType << " and " << outputElementType;
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// spirv.TosaArgmaxOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::TosaArgMaxOp::verify() {
  auto inputTy = cast<ShapedType>(getInput().getType());
  auto resultTy = cast<ShapedType>(getType());

  if (inputTy.hasRank() && resultTy.hasRank() &&
      resultTy.getRank() !=
          (inputTy.getRank() > 1 ? inputTy.getRank() - 1 : 1)) {
    return emitOpError("result rank must be max of 1 and (input rank - 1)");
  }

  auto resultETy = resultTy.getElementType();
  if (!resultETy.isIntOrIndex()) {
    return emitOpError("result is not of integer type");
  }

  IntegerAttr axisAttr;
  if (getConstAttr(getAxis(), axisAttr).failed()) {
    return emitOpError("axis type must be a constant integer");
  }

  const int axis = axisAttr.getInt();
  if (inputTy.hasRank() && ((axis < 0) || axis >= inputTy.getRank())) {
    return emitOpError("specified axis is outside the rank of input");
  }

  IntegerAttr nanModeAttr;
  if (getConstAttr(getNanMode(), nanModeAttr).failed()) {
    return emitOpError("nan_mode type must be a constant integer");
  }

  int nanMode = nanModeAttr.getInt();
  if (nanMode != 1 && nanMode != 2) {
    return emitOpError("nan_mode can only have values 1 and 2 corresponding to "
                       "PROPAGATE/IGNORE");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// spirv.TosaAvgPool2DOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::TosaAvgPool2DOp::verify() {
  auto inputTy = cast<ShapedType>(getInput().getType());
  if (inputTy.hasRank() && inputTy.getRank() != 4) {
    return emitOpError("input rank must be 4");
  }

  auto resultTy = cast<ShapedType>(getType());
  if (resultTy.hasRank() && resultTy.getRank() != 4) {
    return emitOpError("result rank must be 4");
  }

  IntegerAttr accTypeAttr;
  if (getConstAttr(getAccType(), accTypeAttr).failed()) {
    return emitOpError("accumulator type must be a constant integer");
  }

  int accType = accTypeAttr.getInt();
  if (accType != 1 && accType != 2 && accType != 3) {
    return emitOpError("accumulator type can only have values 1/2/3 "
                       "corresponding to i32/f16/f32");
  }

  auto inputETy = inputTy.getElementType();
  auto resultETy = resultTy.getElementType();

  if (isa<IntegerType>(inputETy) && accType != 1) {
    return emitOpError("accumulator type for integer tensorARM is not i32");
  }

  if (inputETy.isF16() && !(accType == 2 || accType == 3)) {
    return emitOpError("accumulator type for f16 tensorARM is not f16/f32");
  }

  if (inputETy.isF32() && accType != 3) {
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

LogicalResult spirv::TosaConv2DOp::verify() {
  if (verifyConvOp(*this, TosaConv2DOp::Adaptor(*this)).failed() ||
      verifyConvOpModes(*this).failed())
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// spirv.TosaConv3DOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::TosaConv3DOp::verify() {
  if (verifyConvOp(*this, TosaConv3DOp::Adaptor(*this)).failed() ||
      verifyConvOpModes(*this).failed())
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// SPIRV Tosa DepthwiseConv2D Ops:
//===----------------------------------------------------------------------===//

LogicalResult spirv::TosaDepthwiseConv2DOp::verify() {
  if (verifyConvOp(*this, TosaDepthwiseConv2DOp::Adaptor(*this)).failed() ||
      verifyConvOpModes(*this).failed())
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// spirv.TosaFFT2DOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::TosaFFT2DOp::verify() {
  auto inputRealTy = cast<ShapedType>(getInputReal().getType());
  auto inputImagTy = cast<ShapedType>(getInputImag().getType());
  auto resultTy = cast<StructType>(getType());
  auto resultRealTy = cast<ShapedType>(resultTy.getElementType(0));
  auto resultImagTy = cast<ShapedType>(resultTy.getElementType(1));

  if (inputRealTy.hasRank() && inputRealTy.getRank() != 3) {
    return emitOpError("real input rank must be 3");
  }

  if (inputImagTy.hasRank() && inputImagTy.getRank() != 3) {
    return emitOpError("imaginary input rank must be 3");
  }

  if (resultRealTy.hasRank() && resultRealTy.getRank() != 3) {
    return emitOpError("real result rank must be 3");
  }

  if (resultImagTy.hasRank() && resultImagTy.getRank() != 3) {
    return emitOpError("imaginary result rank must be 3");
  }

  if (inputRealTy != inputImagTy || inputRealTy != resultRealTy ||
      inputImagTy != resultImagTy) {
    return emitOpError("real input type, imaginary input type, and types of "
                       "real and imaginary parts of result must be the same");
  }

  BoolAttr inverseAttr;
  if ((getConstAttr(getInverse(), inverseAttr).failed())) {
    return emitOpError("inverse must be a constant boolean");
  }

  BoolAttr localBoundAttr;
  if ((getConstAttr(getLocalBound(), localBoundAttr).failed())) {
    return emitOpError("local bound must be a constant boolean");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// spirv.TosaMatMulOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::TosaMatMulOp::verify() {
  auto aTy = cast<ShapedType>(getA().getType());
  auto bTy = cast<ShapedType>(getB().getType());
  auto resultTy = cast<ShapedType>(getType());

  if (!aTy || !bTy) {
    return emitOpError("expected shaped tensors for inputs, got ")
           << getA().getType() << " and " << getB().getType();
  }

  if (aTy.hasRank() && aTy.getRank() != 3) {
    return emitOpError("A rank must be 3");
  }

  if (bTy.hasRank() && bTy.getRank() != 3) {
    return emitOpError("B rank must be 3");
  }

  if (resultTy.hasRank() && resultTy.getRank() != 3) {
    return emitOpError("result rank must be 3");
  }

  auto aETy = aTy.getElementType();
  auto bETy = bTy.getElementType();
  auto resultETy = resultTy.getElementType();

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
  if (getConstAttr(getAZp(), aZpAttr).failed()) {
    return emitOpError("a_zp must be a tensorARM of an integer/float constant");
  }

  if (aZpAttr.size() != 1) {
    return emitOpError("a_zp must have a single element");
  }

  DenseIntOrFPElementsAttr bZpAttr;
  if (getConstAttr(getBZp(), bZpAttr).failed()) {
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

LogicalResult spirv::TosaMaxPool2DOp::verify() {
  auto inputTy = cast<ShapedType>(getInput().getType());
  auto resultTy = cast<ShapedType>(getType());

  if (inputTy.hasRank() && inputTy.getRank() != 4) {
    return emitOpError("input rank must be 4");
  }

  if (resultTy.hasRank() && resultTy.getRank() != 4) {
    return emitOpError("result rank must be 4");
  }

  IntegerAttr nanModeAttr;
  if (getConstAttr(getNanMode(), nanModeAttr).failed()) {
    return emitOpError("nan_mode type must be a constant integer");
  }

  int nanMode = nanModeAttr.getInt();
  if (nanMode != 1 && nanMode != 2) {
    return emitOpError("nan_mode can only have values 1 and 2 corresponding to "
                       "PROPAGATE/IGNORE");
  }

  return verifySameElementTypes(*this, getInput().getType(),
                                getOutput().getType());
}

//===----------------------------------------------------------------------===//
// spirv.TosaRFFT2DOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::TosaRFFT2DOp::verify() {
  auto inputTy = cast<ShapedType>(getInputReal().getType());
  auto resultTy = cast<StructType>(getType());
  auto resultRealTy = cast<ShapedType>(resultTy.getElementType(0));
  auto resultImagTy = cast<ShapedType>(resultTy.getElementType(1));

  if (inputTy.hasRank() && inputTy.getRank() != 3) {
    return emitOpError("input rank must be 3");
  }

  if (resultRealTy.hasRank() && resultRealTy.getRank() != 3) {
    return emitOpError("real result rank must be 3");
  }

  if (resultImagTy.hasRank() && resultImagTy.getRank() != 3) {
    return emitOpError("imaginary result rank must be 3");
  }

  if (inputTy.getElementType() != resultRealTy.getElementType() ||
      inputTy.getElementType() != resultImagTy.getElementType()) {
    return emitOpError(
        "input element type and element types of real and imaginary parts of "
        "result must be the same");
  }

  BoolAttr localBoundAttr;
  if ((getConstAttr(getLocalBound(), localBoundAttr).failed())) {
    return emitOpError("local bound must be a constant boolean");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SPIRV Tosa TransposeConv2D Ops:
//===----------------------------------------------------------------------===//

LogicalResult spirv::TosaTransposeConv2DOp::verify() {
  if (verifyConvOp(*this, TosaTransposeConv2DOp::Adaptor(*this)).failed() ||
      verifyConvOpModes(*this).failed())
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// spirv.TosaMulOp
//===----------------------------------------------------------------------===//
LogicalResult spirv::TosaMulOp::verify() {
  auto resElemType = getElementTypeOrSelf(getOutput());

  // Verify if the element type amoung operands and result match tosa
  // specification.
  if (auto resIntType = dyn_cast<IntegerType>(resElemType)) {
    IntegerType lhsIntType =
        cast<IntegerType>(getElementTypeOrSelf(getInput1()));
    IntegerType rhsIntType =
        cast<IntegerType>(getElementTypeOrSelf(getInput2()));
    if (lhsIntType != rhsIntType)
      return emitOpError(
          "requires the same element type for all input operands");

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
  }

  auto compareRank = [](const ShapedType type, const ShapedType against) {
    return type.hasRank() && against.hasRank() &&
           type.getRank() == against.getRank();
  };
  ShapedType opType = cast<ShapedType>(getType());
  for (int i = 0; i < 2; ++i) {
    if (!compareRank(cast<ShapedType>(getOperand(i).getType()), opType))
      return emitOpError("result type has different rank than operands");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// spirv.TosaCastOp
//===----------------------------------------------------------------------===//
LogicalResult spirv::TosaCastOp::verify() {
  auto inputETy = cast<ShapedType>(getInput().getType()).getElementType();
  auto outputETy = cast<ShapedType>(getType()).getElementType();

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
LogicalResult spirv::TosaClampOp::verify() {
  auto inputETy = cast<ShapedType>(getInput().getType()).getElementType();
  auto outputETy = cast<ShapedType>(getOutput().getType()).getElementType();

  if (inputETy != outputETy)
    return emitOpError("input/output element types are incompatible");

  unsigned dataTypeBitWidth = inputETy.getIntOrFloatBitWidth();

  if (inputETy.isInteger(dataTypeBitWidth)) {
    IntegerAttr minValAttr, maxValAttr;
    if ((getConstAttr(getMinVal(), minValAttr).failed()) ||
        (getConstAttr(getMaxVal(), maxValAttr).failed()) ||
        (minValAttr.getType() != maxValAttr.getType()) ||
        (minValAttr.getType() != inputETy))

      return emitOpError("min/max attributes types are incompatible with "
                         "input/output element types.");
  } else {
    FloatAttr minValAttr, maxValAttr;
    if ((getConstAttr(getMinVal(), minValAttr).failed()) ||
        (getConstAttr(getMaxVal(), maxValAttr).failed()) ||
        (minValAttr.getType() != maxValAttr.getType()) ||
        (minValAttr.getType() != inputETy))

      return emitOpError("min/max attributes types are incompatible with "
                         "input/output element types.");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// spirv.TosaConcatOp
//===----------------------------------------------------------------------===//
LogicalResult spirv::TosaConcatOp::verify() {
  auto outType = getOutput().getType();
  for (auto input : getInput1()) {
    if (verifySameElementTypes(*this, input.getType(), outType).failed()) {
      return failure();
    }
  }
  IntegerAttr axisAttr;
  if (getConstAttr(getAxis(), axisAttr).failed()) {
    return emitOpError("Axis must be an integer constant");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// spirv.TosaPadOp
//===----------------------------------------------------------------------===//
LogicalResult spirv::TosaPadOp::verify() {
  if (verifySameElementTypes(*this, getInput1().getType(),
                             getOutput().getType())
          .failed()) {
    return failure();
  }

  auto inputETy = cast<ShapedType>(getInput1().getType()).getElementType();

  DenseIntOrFPElementsAttr padConstAttr;
  if ((getConstAttr(getPadConst(), padConstAttr).failed()) ||
      (padConstAttr.getElementType() != inputETy)) {
    return emitOpError(
        "PadConst element type is not same as input element type.");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// spirv.TosaSliceOp
//===----------------------------------------------------------------------===//
LogicalResult spirv::TosaSliceOp::verify() {
  auto inputTy = cast<ShapedType>(getInput1().getType());
  auto outputTy = cast<ShapedType>(getOutput().getType());
  auto startTy = cast<ShapedType>(getStart().getType());
  auto sizeTy = cast<ShapedType>(getSize().getType());

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
LogicalResult spirv::TosaTileOp::verify() {
  auto inputTy = cast<ShapedType>(getInput1().getType());
  auto outputTy = cast<ShapedType>(getOutput().getType());
  auto multiplesTy = cast<ShapedType>(getMultiples().getType());

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
LogicalResult spirv::TosaTransposeOp::verify() {
  auto inputTy = cast<ShapedType>(getInput1().getType());
  auto outputTy = cast<ShapedType>(getOutput().getType());
  auto permsTy = cast<ShapedType>(getPerms().getType());

  if (verifySameElementTypes(*this, inputTy, outputTy).failed()) {
    return failure();
  }

  if (inputTy.hasRank() && outputTy.hasRank() &&
      inputTy.getRank() != outputTy.getRank()) {
    return emitOpError("expect same input and output tensorARM rank");
  }

  if (permsTy.hasRank() && permsTy.getRank() != 1) {
    return emitOpError(
               "expected permutation tensorARM to be rank 1 but got rank ")
           << permsTy.getRank();
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
LogicalResult spirv::TosaGatherOp::verify() {
  return verifySameElementTypes(*this, getValues().getType(),
                                getOutput().getType());
}

//===----------------------------------------------------------------------===//
// spirv.TosaScatterOp
//===----------------------------------------------------------------------===//
LogicalResult spirv::TosaScatterOp::verify() {
  if (verifySameElementTypes(*this, getValuesIn().getType(),
                             getValuesOut().getType())
          .failed()) {
    return failure();
  }
  if (verifySameElementTypes(*this, getInput().getType(),
                             getValuesOut().getType())
          .failed()) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// spirv.TosaRescaleOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::TosaRescaleOp::verify() {
  auto inputTy = cast<ShapedType>(getInput().getType());
  auto outputTy = cast<ShapedType>(getOutput().getType());

  auto inputETy = inputTy.getElementType();
  if (!isa<IntegerType>(inputETy)) {
    return emitOpError("expect input to have integer element type, got ")
           << inputETy;
  }

  if (inputTy.hasRank() != outputTy.hasRank() ||
      (inputTy.hasRank() && inputTy.getShape() != outputTy.getShape())) {
    return emitOpError("Shape of input and output must be same");
  }

  auto outputETy = outputTy.getElementType();
  if (!isa<IntegerType>(outputETy)) {
    return emitOpError("expect output to have integer element type, got ")
           << outputETy;
  }

  DenseIntElementsAttr inputZpAttr;
  if ((getConstAttr(getInputZp(), inputZpAttr).failed())) {
    return emitOpError(
        "input_zp must be single element tensorARM of an integer constant");
  }

  if (inputZpAttr.size() != 1) {
    return emitOpError("input_zp must have a single element");
  }

  auto inputZPETy = inputZpAttr.getElementType();
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
      if (auto input_zp = inputAPInt.getZExtValue(); input_zp != 32768u) {
        return emitOpError("expect input_zp of 0 or 32768 for unsigned int16 "
                           "input, got ")
               << input_zp;
      }
    }
  }

  DenseIntElementsAttr outputZpAttr;
  if ((getConstAttr(getOutputZp(), outputZpAttr).failed())) {
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

  auto shiftETy = shiftTy.getElementType();
  if (!shiftETy.isInteger(8)) {
    return emitOpError("shift element type must be i8");
  }

  BoolAttr scale32Attr;
  if ((getConstAttr(getScale32(), scale32Attr).failed())) {
    return emitOpError("scale32 must be a constant boolean");
  }

  auto multiplierETy = multiplierTy.getElementType();
  if (scale32Attr.getValue() && !multiplierETy.isInteger(32)) {
    return emitOpError(
               "expect i32 element type for multiplier for scale32=true, got ")
           << multiplierETy;
  }

  if (!scale32Attr.getValue() && !multiplierETy.isInteger(16)) {
    return emitOpError(
               "expect i16 element type for multiplier for scale32=false, got ")
           << multiplierETy;
  }

  IntegerAttr roundingModeAttr;
  if ((getConstAttr(getRoundingMode(), roundingModeAttr).failed())) {
    return emitOpError("rounding_mode must be a constant integer");
  }

  if (auto roundingMode = roundingModeAttr.getInt();
      (roundingMode != 1 && roundingMode != 2 && roundingMode != 3)) {
    return emitOpError(
               "rounding mode must be an integer of value 1/2/3 "
               "corresponding to SINGLE_ROUND/INEXACT_ROUND/DOUBLE_ROUND, got ")
           << roundingMode;
  }

  BoolAttr perChannelAttr;
  if ((getConstAttr(getPerChannel(), perChannelAttr).failed())) {
    return emitOpError("per_channel must be a constant boolean");
  }

  // multiplier/shift must have shape = {numChannels},
  // where numChannel is 1 if per_channel = false
  // otherwise numChannel is dimension in input shape's last axis
  int64_t numChannels = 1;
  if (perChannelAttr.getValue()) {
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

  BoolAttr inputUnsignedAttr;
  if ((getConstAttr(getInputUnsigned(), inputUnsignedAttr).failed())) {
    return emitOpError("input_unsigned must be a constant boolean");
  }

  BoolAttr outputUnsignedAttr;
  if ((getConstAttr(getOutputUnsigned(), outputUnsignedAttr).failed())) {
    return emitOpError("output_unsigned must be a constant boolean");
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

LogicalResult spirv::TosaReverseOp::verify() {
  auto inputTy = cast<ShapedType>(getInput1().getType());
  auto outputTy = cast<ShapedType>(getOutput().getType());

  if (verifySameElementTypes(*this, inputTy, outputTy).failed()) {
    return failure();
  }

  if (inputTy.getRank() != outputTy.getRank()) {
    return emitOpError(
        "expect output tensorARM rank to be equal to input rank");
  }

  IntegerAttr axisAttr;
  if ((getConstAttr(getAxis(), axisAttr).failed())) {
    return emitOpError("axis must be a constant integer");
  }

  int32_t reverseAxis = axisAttr.getInt();
  if (reverseAxis < 0) {
    return emitOpError("expected non-negative reverse axis");
  }

  if (inputTy.hasRank() && reverseAxis >= inputTy.getRank() &&
      !(reverseAxis == 0 && inputTy.getRank() == 0)) {
    return emitOpError("expect input rank (")
           << inputTy.getRank() << ") to be larger than reverse axis ("
           << reverseAxis << ")";
  }

  if (outputTy.hasRank() && reverseAxis >= outputTy.getRank() &&
      !(reverseAxis == 0 && outputTy.getRank() == 0)) {
    return emitOpError("expect output tensorARM rank (")
           << outputTy.getRank() << ") to be larger than reverse axis ("
           << reverseAxis << ")";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// spirv.TosaSelectOp
//===----------------------------------------------------------------------===//
LogicalResult spirv::TosaSelectOp::verify() {
  if (verifySameElementTypes(*this, getInput2().getType(),
                             getOutput().getType())
          .failed()) {
    return failure();
  }
  if (verifySameElementTypes(*this, getInput3().getType(),
                             getOutput().getType())
          .failed()) {
    return failure();
  }

  auto predicateType = cast<ShapedType>(getInput1().getType());
  if (!predicateType) {
    emitOpError("expect shaped tensorARM for input1, got ")
        << getInput1().getType();
    return failure();
  }

  auto predicateElementType = predicateType.getElementType();
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

mlir::LogicalResult spirv::TosaReshapeOp::verify() {
  if (verifySameElementTypes(*this, getInput1().getType(),
                             getOutput().getType())
          .failed()) {
    return failure();
  }
  ShapedType inputType = cast<ShapedType>(getInput1().getType());
  ShapedType outputType = cast<ShapedType>(getType());

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
