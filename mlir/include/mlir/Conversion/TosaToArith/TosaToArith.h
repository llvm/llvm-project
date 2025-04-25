//===-- TosaToArith.h - TOSA optimization pass declarations --*- C++ -*-===//
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

#ifndef MLIR_CONVERSION_TOSATOARITH_TOSATOARITH_H
#define MLIR_CONVERSION_TOSATOARITH_TOSATOARITH_H

#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_DECL_TOSATOARITH
#include "mlir/Conversion/Passes.h.inc"

namespace tosa {

std::unique_ptr<Pass> createTosaToArith(bool includeApplyRescale = false,
                                        bool use32BitApplyRescale = false);

void populateTosaToArithConversionPatterns(RewritePatternSet *patterns);

void populateTosaRescaleToArithConversionPatterns(RewritePatternSet *patterns,
                                                  bool include32Bit = false);

} // namespace tosa
} // namespace mlir

#endif // MLIR_CONVERSION_TOSATOARITH_TOSATOARITH_H
