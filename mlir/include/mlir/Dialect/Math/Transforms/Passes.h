//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MATH_TRANSFORMS_PASSES_H_
#define MLIR_DIALECT_MATH_TRANSFORMS_PASSES_H_

namespace mlir {

class RewritePatternSet;

void populateExpandCtlzPattern(RewritePatternSet &patterns);
void populateExpandTanPattern(RewritePatternSet &patterns);
void populateExpandTanhPattern(RewritePatternSet &patterns);
void populateExpandFmaFPattern(RewritePatternSet &patterns);
void populateExpandFloorFPattern(RewritePatternSet &patterns);
void populateExpandCeilFPattern(RewritePatternSet &patterns);
void populateExpandExp2FPattern(RewritePatternSet &patterns);
void populateExpandPowFPattern(RewritePatternSet &patterns);
void populateExpandRoundFPattern(RewritePatternSet &patterns);
void populateMathAlgebraicSimplificationPatterns(RewritePatternSet &patterns);

struct MathPolynomialApproximationOptions {
  // Enables the use of AVX2 intrinsics in some of the approximations.
  bool enableAvx2 = false;
};

void populateMathPolynomialApproximationPatterns(
    RewritePatternSet &patterns,
    const MathPolynomialApproximationOptions &options = {});

} // namespace mlir

#endif // MLIR_DIALECT_MATH_TRANSFORMS_PASSES_H_
