//===-- TosaToStandard.h - TOSA optimization pass declarations --*- C++ -*-===//
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

#ifndef MLIR_CONVERSION_TOSATOSTANDARD_TOSATOSTANDARD_H
#define MLIR_CONVERSION_TOSATOSTANDARD_TOSATOSTANDARD_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace tosa {

std::unique_ptr<Pass> createTosaToStandard();

void populateTosaToStandardConversionPatterns(RewritePatternSet *patterns);

void populateTosaRescaleToStandardConversionPatterns(
    RewritePatternSet *patterns);

/// Populates passes to convert from TOSA to Standard.
void addTosaToStandardPasses(OpPassManager &pm);

} // namespace tosa
} // namespace mlir

#endif // MLIR_CONVERSION_TOSATOSTANDARD_TOSATOSTANDARD_H
