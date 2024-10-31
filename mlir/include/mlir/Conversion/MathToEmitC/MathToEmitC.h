//===- MathToEmitC.h - Math to EmitCPatterns -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_MATHTOEMITC_MATHTOEMITC_H
#define MLIR_CONVERSION_MATHTOEMITC_MATHTOEMITC_H
#include "mlir/Dialect/EmitC/IR/EmitC.h"

namespace mlir {
class RewritePatternSet;

void populateConvertMathToEmitCPatterns(
    RewritePatternSet &patterns,
    emitc::MathToEmitCLanguageTarget languageTarget);
} // namespace mlir

#endif // MLIR_CONVERSION_MATHTOEMITC_MATHTOEMITC_H
