//===-- TosaToTensor.h - TOSA to Tensor legalization ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the passes for the TOSA to Standard Dialect conversion.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_TOSATOTENSOR_TOSATOTENSOR_H
#define MLIR_CONVERSION_TOSATOTENSOR_TOSATOTENSOR_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class TypeConverter;

#define GEN_PASS_DECL_TOSATOTENSOR
#include "mlir/Conversion/Passes.h.inc"

namespace tosa {

std::unique_ptr<Pass> createTosaToTensor();

void populateTosaToTensorConversionPatterns(TypeConverter &converter,
                                            RewritePatternSet *patterns);

} // namespace tosa
} // namespace mlir

#endif // MLIR_CONVERSION_TOSATOTENSOR_TOSATOTENSOR_H
