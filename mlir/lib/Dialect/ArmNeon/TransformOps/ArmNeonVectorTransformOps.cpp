//===- ArmNeonVectorTransformOps.cpp - Implementation transform ops -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/ArmNeon/TransformOps/ArmNeonVectorTransformOps.h"

#include "mlir/Dialect/ArmNeon/ArmNeonDialect.h"
#include "mlir/Dialect/ArmNeon/Transforms.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Apply...PatternsOp
//===----------------------------------------------------------------------===//

void transform::ApplyArmNeonContractionToI8MMPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  arm_neon::populateLowerContractionToNeonI8MMPatterns(patterns);
}

void transform::ApplyArmNeonContractionToBFMMLAPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  arm_neon::populateLowerContractionToNeonBFMMLAPatterns(patterns);
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
class ArmNeonVectorTransformDialectExtension
    : public transform::TransformDialectExtension<
          ArmNeonVectorTransformDialectExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      ArmNeonVectorTransformDialectExtension)

  ArmNeonVectorTransformDialectExtension() {
    declareGeneratedDialect<arm_neon::ArmNeonDialect>();
    registerTransformOps<
#define GET_OP_LIST
#include "mlir/Dialect/ArmNeon/TransformOps/ArmNeonVectorTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "mlir/Dialect/ArmNeon/TransformOps/ArmNeonVectorTransformOps.cpp.inc"

void mlir::arm_neon::registerTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<ArmNeonVectorTransformDialectExtension>();
}
