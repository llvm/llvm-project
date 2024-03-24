//===-- TosaToMLProgram.h - TOSA to MLProgram dialect lowerings-*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the passes for the TOSA to MLProgram Dialect conversion.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_TOSATOMLPROGRAM_TOSATOMLPROGRAM_H
#define MLIR_CONVERSION_TOSATOMLPROGRAM_TOSATOMLPROGRAM_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

#define GEN_PASS_DECL_TOSATOMLPROGRAM

namespace tosa {

void populateTosaToMLProgramConversionPatterns(RewritePatternSet *patterns);

} // namespace tosa
} // namespace mlir

#endif // MLIR_CONVERSION_TOSATOMLPROGRAM_TOSATOMLPROGRAM_H
