//===- ArmSVEVectorTransformOps.cpp - Implementation transform ops -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/ArmSVE/TransformOps/ArmSVEVectorTransformOps.h"

#include "mlir/Dialect/ArmSVE/IR/ArmSVEDialect.h"
#include "mlir/Dialect/ArmSVE/Transforms/Transforms.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Apply...PatternsOp
//===----------------------------------------------------------------------===//

void transform::ApplyArmSVELowerContractionPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  mlir::populateLowerContractionToSVEI8MMPatternPatterns(patterns);
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
class ArmSVEVectorTransformDialectExtension
    : public transform::TransformDialectExtension<
          ArmSVEVectorTransformDialectExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      ArmSVEVectorTransformDialectExtension)

  ArmSVEVectorTransformDialectExtension() {
    declareGeneratedDialect<arm_sve::ArmSVEDialect>();
    registerTransformOps<
#define GET_OP_LIST
#include "mlir/Dialect/ArmSVE/TransformOps/ArmSVEVectorTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "mlir/Dialect/ArmSVE/TransformOps/ArmSVEVectorTransformOps.cpp.inc"

void mlir::arm_sve::registerTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<ArmSVEVectorTransformDialectExtension>();
}
