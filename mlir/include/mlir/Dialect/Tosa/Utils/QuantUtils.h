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

#include "mlir/Dialect/Quant/FakeQuantSupport.h"
#include "mlir/Dialect/Quant/UniformSupport.h"

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// Utility functions to support quantization handling in Tosa.
//===----------------------------------------------------------------------===//

/// From a scale value, computes multiplier and shift values
/// for 16 or 32-bit scale widths.
void computeMultiplierAndShift(double scale, int32_t &multiplier,
                               int32_t &shift, int32_t scaleWidth);

//// Builds ConvOpQuantizationAttr from input and weight.
ConvOpQuantizationAttr buildConvOpQuantizationAttr(OpBuilder &builder,
                                                   Value input, Value weight);

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
