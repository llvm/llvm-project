//===-- QuantUtils.h - TOSA numerical support declarations ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Function declarations for TOSA numerical support functions and quantization
// attribute builders
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TOSA_UTILS_QUANTUTILS_H
#define MLIR_DIALECT_TOSA_UTILS_QUANTUTILS_H

#include "mlir/Dialect/Tosa/IR/TosaOps.h"

#include "mlir/Dialect/Quant/Utils/FakeQuantSupport.h"
#include "mlir/Dialect/Quant/Utils/UniformSupport.h"

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// Utility functions to support quantization handling in Tosa.
//===----------------------------------------------------------------------===//

/// From a scale value, computes multiplier and shift values
/// for 16 or 32-bit scale widths.
bool computeMultiplierAndShift(double scale, int32_t &multiplier,
                               int32_t &shift, int32_t scaleWidth);

// Return a const value for array of IntType vec
template <typename IntType>
Value getConstTensorInt(OpBuilder &builder, Location loc,
                        ArrayRef<IntType> vec) {
  static_assert(
      std::is_same<IntType, int8_t>::value ||
          std::is_same<IntType, int16_t>::value ||
          std::is_same<IntType, int32_t>::value,
      "getConstTensorInt only supports int8_t, int16_t, and int32_t types.");

  int64_t count = vec.size();
  assert(count > 0 && "Vector must not be empty");
  auto element_type = builder.getIntegerType(sizeof(IntType) * 8);
  mlir::RankedTensorType const_type =
      RankedTensorType::get({count}, element_type);
  mlir::DenseElementsAttr const_attr = DenseElementsAttr::get(const_type, vec);
  auto const_op = builder.create<tosa::ConstOp>(loc, const_type, const_attr);
  return const_op.getResult();
}

//// Builds ConvOpQuantizationAttr from input and weight.
ConvOpQuantizationAttr buildConvOpQuantizationAttr(OpBuilder &builder,
                                                   Value input, Value weight);

std::pair<Value, Value> createZPsAsConst(OpBuilder &builder, Value input,
                                         Value weight);

//// Builds MatMulOpQuantizationAttr for MatMul operations from A and B.
MatMulOpQuantizationAttr buildMatMulOpQuantizationAttr(OpBuilder &builder,
                                                       Value a, Value b);

//// Builds UnaryOpQuantizationAttr for unary operations from input values.
UnaryOpQuantizationAttr buildUnaryOpQuantizationAttr(OpBuilder &builder,
                                                     Value input,
                                                     Type outputRawType);

//// Builds PadOpQuantizationAttr for pad operations from input values.
PadOpQuantizationAttr buildPadOpQuantizationAttr(OpBuilder &builder,
                                                 Value input);

//// construct ConvOp output type with correct bitwidth based on input/weight
/// width.
Type buildConvOpResultTypeInfo(OpBuilder &builder, Type outputType, Value input,
                               Value weight);

/// Builds Tosa quantization attributes from min/max values.
Type buildQTypeFromMinMax(OpBuilder builder, Type inputDType, Attribute minAttr,
                          Attribute maxAttr, IntegerAttr quantBits,
                          int filterQuantDim, bool isSigned,
                          BoolAttr narrowRange);

/// Builds Tosa quantization attributes from min/max values.
TypeAttr buildQTypeAttrFromMinMax(OpBuilder builder, Type inputDType,
                                  Attribute minAttr, Attribute maxAttr,
                                  IntegerAttr quantBits, int filterQuantDim,
                                  bool isSigned, BoolAttr narrowRange);

} // namespace tosa
} // namespace mlir

#endif // MLIR_DIALECT_TOSA_UTILS_QUANTUTILS_H
