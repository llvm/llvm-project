//===- MathTransformOps.cpp - Implementation of Math transform ops --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Math/TransformOps/MathTransformOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/X86/X86Dialect.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Apply...PatternsOp
//===----------------------------------------------------------------------===//

void transform::ApplyF32ExpansionPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  populateMathF32ExpansionPatterns(patterns, [](StringRef) { return true; });
}

void transform::ApplyPolynomialApproximationPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  bool enableAvx2 = getEnableAvx2();
  populateMathPolynomialApproximationPatterns(patterns, [&](StringRef name) {
    // `rsqrt` requires AVX2 so always gate it.
    return enableAvx2 || name != math::RsqrtOp::getOperationName();
  });
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
class MathTransformDialectExtension
    : public transform::TransformDialectExtension<
          MathTransformDialectExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MathTransformDialectExtension)

  MathTransformDialectExtension() {
    declareGeneratedDialect<arith::ArithDialect>();
    declareGeneratedDialect<math::MathDialect>();
    declareGeneratedDialect<vector::VectorDialect>();
    declareGeneratedDialect<x86::X86Dialect>();
    registerTransformOps<
#define GET_OP_LIST
#include "mlir/Dialect/Math/TransformOps/MathTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "mlir/Dialect/Math/TransformOps/MathTransformOps.cpp.inc"

void mlir::math::registerTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<MathTransformDialectExtension>();
}
