//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MATH_TRANSFORMS_PASSES_H_
#define MLIR_DIALECT_MATH_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Bufferize.h"

namespace mlir {

class OwningRewritePatternList;

void populateExpandTanhPattern(OwningRewritePatternList &patterns,
                               MLIRContext *ctx);

void populateMathPolynomialApproximationPatterns(
    OwningRewritePatternList &patterns, MLIRContext *ctx);

} // namespace mlir

#endif // MLIR_DIALECT_MATH_TRANSFORMS_PASSES_H_
