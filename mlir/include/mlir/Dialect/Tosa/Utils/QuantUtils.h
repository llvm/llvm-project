//===-- QuantUtils.h - TOSA numerical support declarations *- C++ -*-===//
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

#ifndef MLIR_DIALECT_TOSA_UTILS_QUANT_UTILS_H
#define MLIR_DIALECT_TOSA_UTILS_QUANT_UTILS_H

// Utils to support quantization handling in Tosa

#include <climits>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>

#include "mlir/Dialect/Quant/FakeQuantSupport.h"
#include "mlir/Dialect/Quant/UniformSupport.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"

#include "mlir/Dialect/Tosa/IR/TosaOps.h"

namespace mlir {
namespace tosa {

void computeMultiplierAndShift(double scale, int32_t &multiplier,
                               int32_t &shift, int32_t scale_width);
void computeMultiplierAndShiftGtOne(double scale, int32_t &multiplier,
                                    int32_t &shift, int32_t scale_width);
void computeMultiplierAndShiftLtOneExp(double scale, int32_t &multiplier,
                                       int32_t &shift, int32_t scale_width);

ConvOpQuantizationAttr buildConvOpQuantizationAttr(mlir::OpBuilder &builder,
                                                   Value input, Value weight);

MatMulOpQuantizationAttr buildMatMulOpQuantizationAttr(mlir::OpBuilder &builder,
                                                       Value a, Value b);

UnaryOpQuantizationAttr buildUnaryOpQuantizationAttr(mlir::OpBuilder &builder,
                                                     Value input,
                                                     Type output_raw_type);

PadOpQuantizationAttr buildPadOpQuantizationAttr(mlir::OpBuilder &builder,
                                                 Value input);

Type buildQTypeFromMinMax(OpBuilder builder, Type input_dtype,
                          Attribute minattr, Attribute maxattr,
                          IntegerAttr quant_bits, int filter_quantdim,
                          bool issigned, BoolAttr narrow_range);

TypeAttr buildQTypeAttrFromMinMax(OpBuilder builder, Type input_dtype,
                                  Attribute minattr, Attribute maxattr,
                                  IntegerAttr quant_bits, int filter_quantdim,
                                  bool issigned, BoolAttr narrow_range);

} // namespace tosa
} // namespace mlir

#endif // MLIR_DIALECT_TOSA_UTILS_QUANT_UTILS_H
