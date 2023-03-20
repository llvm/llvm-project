//===- VectorTransformOps.cpp - Implementation of Vector transform ops ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"

#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Dialect/X86Vector/Transforms.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::vector;
using namespace mlir::transform;

//===----------------------------------------------------------------------===//
// LowerVectorsOp
//===----------------------------------------------------------------------===//

void transform::LowerVectorsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTarget(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure transform::LowerVectorsOp::applyToOne(
    ::mlir::Operation *target,
    ::mlir::transform::ApplyToEachResultList &results,
    ::mlir::transform::TransformState &state) {

  // This check can't be part of the verifier because payload IR is
  // independent from transform IR and may not even exist.
  if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    return mlir::emitDefiniteFailure(target,
                                     "applies only to isolated-from-above "
                                     "targets because it needs to apply "
                                     "patterns greedily");
  }

  MLIRContext *ctx = getContext();
  RewritePatternSet patterns(ctx);
  vector::VectorTransposeLowering vectorTransposeLowering =
      getTransposeLowering();
  vector::VectorMultiReductionLowering vectorMultiReductionLowering =
      getMultireductionLowering();
  vector::VectorContractLowering vectorContractLowering =
      getContractionLowering();
  vector::VectorTransferSplit vectorTransferSplit = getSplitTransfers();

  vector::VectorTransformsOptions vectorTransformOptions;
  vectorTransformOptions.setVectorTransformsOptions(vectorContractLowering)
      .setVectorMultiReductionLowering(vectorMultiReductionLowering)
      .setVectorTransposeLowering(vectorTransposeLowering)
      .setVectorTransferSplit(vectorTransferSplit);

  VectorTransferToSCFOptions vectorTransferToSCFOptions =
      VectorTransferToSCFOptions().enableFullUnroll(getUnrollVectorTransfers());

  int maxTransferRank = 1;

  auto avx2LoweringOptions =
      x86vector::avx2::LoweringOptions().setTransposeOptions(
          x86vector::avx2::TransposeLoweringOptions()
              .lower4x8xf32(getTransposeAvx2Lowering())
              .lower8x8xf32(getTransposeAvx2Lowering()));

  vector::populateVectorToVectorCanonicalizationPatterns(patterns);

  // In the future we may want to more finely select particular stages.
  // Stage 1: contraction lowerings.
  patterns.add<mlir::vector::ContractionOpToOuterProductOpLowering,
               mlir::vector::ContractionOpToMatmulOpLowering,
               mlir::vector::ContractionOpLowering>(vectorTransformOptions,
                                                    ctx);
  vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);

  // Stage 2: multi-reduction lowerings.
  vector::populateVectorMultiReductionLoweringPatterns(
      patterns, vectorTransformOptions.vectorMultiReductionLowering);

  // Stage 3: Rewrite vector.transfer into full and partial parts.
  patterns.add<vector::VectorTransferFullPartialRewriter>(
      ctx, vectorTransformOptions);

  // Stage 4: Lower vector transfers.
  vector::populateVectorTransferLoweringPatterns(patterns, maxTransferRank);

  // Stage 5: Vector to scf patterns.
  populateVectorToSCFConversionPatterns(
      patterns, vectorTransferToSCFOptions.setTargetRank(maxTransferRank));

  // Stage 6: Lower vector.shape_cast.
  vector::populateVectorShapeCastLoweringPatterns(patterns);

  // Stage 7: Lower vector.transpose.
  vector::populateVectorTransposeLoweringPatterns(patterns,
                                                  vectorTransformOptions);
  if (getTransposeAvx2Lowering())
    x86vector::avx2::populateSpecializedTransposeLoweringPatterns(
        patterns, avx2LoweringOptions, /*benefit=*/10);

  // Apply everything.
  if (failed(applyPatternsAndFoldGreedily(target, std::move(patterns))))
    return DiagnosedSilenceableFailure::definiteFailure();

  return DiagnosedSilenceableFailure::success();
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
    declareDependentDialect<pdl::PDLDialect>();
    declareDependentDialect<vector::VectorDialect>();
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
