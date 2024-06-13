//===- VectorTransformOps.cpp - Implementation of Vector transform ops ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
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
// Apply...ConversionPatternsOp
//===----------------------------------------------------------------------===//

void transform::ApplyVectorToLLVMConversionPatternsOp::populatePatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns) {
  populateVectorToLLVMConversionPatterns(
      static_cast<LLVMTypeConverter &>(typeConverter), patterns,
      getReassociateFpReductions(), getForce_32bitVectorIndices());
}

LogicalResult
transform::ApplyVectorToLLVMConversionPatternsOp::verifyTypeConverter(
    transform::TypeConverterBuilderOpInterface builder) {
  if (builder.getTypeConverterType() != "LLVMTypeConverter")
    return emitOpError("expected LLVMTypeConverter");
  return success();
}

//===----------------------------------------------------------------------===//
// Apply...PatternsOp
//===----------------------------------------------------------------------===//

void transform::ApplyCastAwayVectorLeadingOneDimPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
}

void transform::ApplyFoldArithExtensionPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  vector::populateFoldArithExtensionPatterns(patterns);
}

void transform::ApplyVectorReductionToContractPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  vector::populateVectorReductionToContractPatterns(patterns);
}

void transform::ApplyLowerCreateMaskPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  vector::populateVectorMaskOpLoweringPatterns(patterns);
}

void transform::ApplyRankReducingSubviewPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  vector::populateVectorTransferDropUnitDimsPatterns(patterns);
}

void transform::ApplyTransferPermutationPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
}

void transform::ApplyLowerBitCastPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  vector::populateVectorBitCastLoweringPatterns(patterns);
}

void transform::ApplyLowerBroadcastPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  populateVectorBroadcastLoweringPatterns(patterns);
}

void transform::ApplyLowerContractionPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  vector::VectorTransformsOptions vectorTransformOptions;
  vectorTransformOptions.setVectorTransformsOptions(getLoweringStrategy());
  populateVectorContractLoweringPatterns(patterns, vectorTransformOptions,
                                         /*benefit=*/1,
                                         /*disableOuterProductLowering=*/true);
}

void transform::ApplyLowerMasksPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  populateVectorMaskOpLoweringPatterns(patterns);
}

void transform::ApplyLowerMaskedTransfersPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  populateVectorMaskLoweringPatternsForSideEffectingOps(patterns);
}

void transform::ApplyMaterializeMasksPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  populateVectorMaskMaterializationPatterns(patterns,
                                            /*force32BitVectorIndices=*/false);
}

void transform::ApplyLowerMultiReductionPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  vector::VectorTransformsOptions vectorTransformOptions;
  vectorTransformOptions.setVectorMultiReductionLowering(getLoweringStrategy());
  vector::populateVectorMultiReductionLoweringPatterns(
      patterns, vectorTransformOptions.vectorMultiReductionLowering);
}

void transform::ApplyLowerOuterProductPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  populateVectorOuterProductLoweringPatterns(patterns);
}

void transform::ApplyLowerGatherPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  vector::populateVectorGatherLoweringPatterns(patterns);
}

void transform::ApplyLowerScanPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  vector::populateVectorScanLoweringPatterns(patterns);
}

void transform::ApplyLowerShapeCastPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  vector::populateVectorShapeCastLoweringPatterns(patterns);
}

void transform::ApplyLowerTransferPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  vector::populateVectorTransferLoweringPatterns(patterns,
                                                 getMaxTransferRank());
}

void transform::ApplyLowerTransposePatternsOp::populatePatterns(
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

void transform::ApplyLowerInterleavePatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  vector::populateVectorInterleaveLoweringPatterns(patterns);
}

void transform::ApplyInterleaveToShufflePatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  vector::populateVectorInterleaveToShufflePatterns(patterns);
}

void transform::ApplyRewriteNarrowTypePatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  populateVectorNarrowTypeRewritePatterns(patterns);
  populateVectorTransposeNarrowTypeRewritePatterns(patterns);
}

void transform::ApplySplitTransferFullPartialPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  vector::VectorTransformsOptions vectorTransformOptions;
  vectorTransformOptions.setVectorTransferSplit(getSplitTransferStrategy());
  populateVectorTransferFullPartialPatterns(patterns, vectorTransformOptions);
}

void transform::ApplyTransferToScfPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
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
