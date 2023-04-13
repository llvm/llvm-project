//===- VectorTransformOps.cpp - Implementation of Vector transform ops ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Dialect/X86Vector/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::vector;
using namespace mlir::transform;

//===----------------------------------------------------------------------===//
// ApplyRankReducingSubviewPatternsOp
//===----------------------------------------------------------------------===//

void transform::ApplyRankReducingSubviewPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  vector::populateVectorTransferDropUnitDimsPatterns(patterns);
}

//===----------------------------------------------------------------------===//
// ApplyTransferPermutationPatternsOp
//===----------------------------------------------------------------------===//

void transform::ApplyTransferPermutationPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
}

//===----------------------------------------------------------------------===//
// LowerBroadcastOp
//===----------------------------------------------------------------------===//

void transform::LowerBroadcastOp::populatePatterns(
    RewritePatternSet &patterns) {
  populateVectorBroadcastLoweringPatterns(patterns);
}

//===----------------------------------------------------------------------===//
// LowerContractionOp
//===----------------------------------------------------------------------===//

void transform::LowerContractionOp::populatePatterns(
    RewritePatternSet &patterns) {
  vector::VectorTransformsOptions vectorTransformOptions;
  vectorTransformOptions.setVectorTransformsOptions(getLoweringStrategy());
  populateVectorContractLoweringPatterns(patterns, vectorTransformOptions,
                                         /*benefit=*/1,
                                         /*disableOuterProductLowering=*/true);
}

//===----------------------------------------------------------------------===//
// LowerMasksOp
//===----------------------------------------------------------------------===//

void transform::LowerMasksOp::populatePatterns(RewritePatternSet &patterns) {
  populateVectorMaskOpLoweringPatterns(patterns);
}

//===----------------------------------------------------------------------===//
// LowerMaskedTransfersOp
//===----------------------------------------------------------------------===//

void transform::LowerMaskedTransfersOp::populatePatterns(
    RewritePatternSet &patterns) {
  populateVectorMaskLoweringPatternsForSideEffectingOps(patterns);
}

//===----------------------------------------------------------------------===//
// LowerMultiReductionOp
//===----------------------------------------------------------------------===//

void transform::LowerMultiReductionOp::populatePatterns(
    RewritePatternSet &patterns) {
  vector::VectorTransformsOptions vectorTransformOptions;
  vectorTransformOptions.setVectorMultiReductionLowering(getLoweringStrategy());
  vector::populateVectorMultiReductionLoweringPatterns(
      patterns, vectorTransformOptions.vectorMultiReductionLowering);
}

//===----------------------------------------------------------------------===//
// LowerOuterProductOp
//===----------------------------------------------------------------------===//

void transform::LowerOuterProductOp::populatePatterns(
    RewritePatternSet &patterns) {
  populateVectorOuterProductLoweringPatterns(patterns);
}

//===----------------------------------------------------------------------===//
// LowerShapeCastOp
//===----------------------------------------------------------------------===//

void transform::LowerShapeCastOp::populatePatterns(
    RewritePatternSet &patterns) {
  vector::populateVectorShapeCastLoweringPatterns(patterns);
}

//===----------------------------------------------------------------------===//
// LowerTransferOp
//===----------------------------------------------------------------------===//

void transform::LowerTransferOp::populatePatterns(RewritePatternSet &patterns) {
  vector::populateVectorTransferLoweringPatterns(patterns,
                                                 getMaxTransferRank());
}

//===----------------------------------------------------------------------===//
// LowerTransposeOp
//===----------------------------------------------------------------------===//

void transform::LowerTransposeOp::populatePatterns(
    RewritePatternSet &patterns) {
  vector::populateVectorTransposeLoweringPatterns(
      patterns, vector::VectorTransformsOptions().setVectorTransposeLowering(
                    getLoweringStrategy()));
  if (getAvx2LoweringStrategy()) {
    auto avx2LoweringOptions =
        x86vector::avx2::LoweringOptions().setTransposeOptions(
            x86vector::avx2::TransposeLoweringOptions()
                .lower4x8xf32(true)
                .lower8x8xf32(true));
    x86vector::avx2::populateSpecializedTransposeLoweringPatterns(
        patterns, avx2LoweringOptions, /*benefit=*/10);
  }
}

//===----------------------------------------------------------------------===//
// SplitTransferFullPartialOp
//===----------------------------------------------------------------------===//

void transform::SplitTransferFullPartialOp::populatePatterns(
    RewritePatternSet &patterns) {
  vector::VectorTransformsOptions vectorTransformOptions;
  vectorTransformOptions.setVectorTransferSplit(getSplitTransferStrategy());
  populateVectorTransferFullPartialPatterns(patterns, vectorTransformOptions);
}

//===----------------------------------------------------------------------===//
// TransferToScfOp
//===----------------------------------------------------------------------===//

void transform::TransferToScfOp::populatePatterns(RewritePatternSet &patterns) {
  VectorTransferToSCFOptions vectorTransferToSCFOptions =
      VectorTransferToSCFOptions()
          .enableFullUnroll(getFullUnroll())
          .setTargetRank(getMaxTransferRank());
  populateVectorToSCFConversionPatterns(patterns, vectorTransferToSCFOptions);
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
/// Registers new ops and declares PDL as dependent dialect since the additional
/// ops are using PDL types for operands and results.
class VectorTransformDialectExtension
    : public transform::TransformDialectExtension<
          VectorTransformDialectExtension> {
public:
  VectorTransformDialectExtension() {
    declareGeneratedDialect<vector::VectorDialect>();
    declareGeneratedDialect<LLVM::LLVMDialect>();
    registerTransformOps<
#define GET_OP_LIST
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.cpp.inc"

void mlir::vector::registerTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<VectorTransformDialectExtension>();
}
