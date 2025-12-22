//===- VectorToLLVM.cpp - Conversion from Vector to the LLVM dialect ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/AMX/AMXDialect.h"
#include "mlir/Dialect/AMX/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ArmNeon/ArmNeonDialect.h"
#include "mlir/Dialect/ArmNeon/Transforms.h"
#include "mlir/Dialect/ArmSVE/IR/ArmSVEDialect.h"
#include "mlir/Dialect/ArmSVE/Transforms/Transforms.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/X86Vector/Transforms.h"
#include "mlir/Dialect/X86Vector/X86VectorDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTVECTORTOLLVMPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::vector;

namespace {
struct ConvertVectorToLLVMPass
    : public impl::ConvertVectorToLLVMPassBase<ConvertVectorToLLVMPass> {

  using Base::Base;

  // Override explicitly to allow conditional dialect dependence.
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<tensor::TensorDialect>();
    if (armNeon)
      registry.insert<arm_neon::ArmNeonDialect>();
    if (armSVE)
      registry.insert<arm_sve::ArmSVEDialect>();
    if (amx)
      registry.insert<amx::AMXDialect>();
    if (x86Vector)
      registry.insert<x86vector::X86VectorDialect>();
  }
  void runOnOperation() override;
};
} // namespace

void ConvertVectorToLLVMPass::runOnOperation() {
  // Perform progressive lowering of operations on slices and all contraction
  // operations. Also materializes masks, lowers vector.step, rank-reduces FMA,
  // applies folding and DCE.
  {
    RewritePatternSet patterns(&getContext());
    populateVectorToVectorCanonicalizationPatterns(patterns);
    populateVectorBitCastLoweringPatterns(patterns);
    populateVectorBroadcastLoweringPatterns(patterns);
    populateVectorContractLoweringPatterns(patterns, vectorContractLowering);
    if (vectorContractLowering == vector::VectorContractLowering::LLVMIntr) {
      // This pattern creates a dependency on the LLVM dialect, hence we don't
      // include it in `populateVectorContractLoweringPatterns` that is part of
      // the Vector dialect (and should not depend on LLVM).
      populateVectorContractToMatrixMultiply(patterns);
    }
    populateVectorMaskOpLoweringPatterns(patterns);
    populateVectorShapeCastLoweringPatterns(patterns);
    populateVectorInterleaveLoweringPatterns(patterns);
    populateVectorTransposeLoweringPatterns(patterns, vectorTransposeLowering);
    if (vectorTransposeLowering == vector::VectorTransposeLowering::LLVMIntr) {
      // This pattern creates a dependency on the LLVM dialect, hence we don't
      // include it in `populateVectorTransposeLoweringPatterns` that is part of
      // the Vector dialect (and should not depend on LLVM).
      populateVectorTransposeToFlatTranspose(patterns);
    }
    // Vector transfer ops with rank > 1 should be lowered with VectorToSCF.
    populateVectorTransferLoweringPatterns(patterns, /*maxTransferRank=*/1);
    populateVectorMaskMaterializationPatterns(patterns,
                                              force32BitVectorIndices);
    populateVectorInsertExtractStridedSliceTransforms(patterns);
    populateVectorStepLoweringPatterns(patterns);
    populateVectorRankReducingFMAPattern(patterns);
    populateVectorGatherLoweringPatterns(patterns);
    populateVectorFromElementsUnrollPatterns(patterns);
    populateVectorToElementsUnrollPatterns(patterns);
    if (armI8MM) {
      if (armNeon)
        arm_neon::populateLowerContractionToNeonI8MMPatterns(patterns);
      if (armSVE)
        populateLowerContractionToSVEI8MMPatterns(patterns);
    }
    if (armBF16) {
      if (armNeon)
        arm_neon::populateLowerContractionToNeonBFMMLAPatterns(patterns);
      if (armSVE)
        populateLowerContractionToSVEBFMMLAPatterns(patterns);
    }
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }

  // Convert to the LLVM IR dialect.
  LowerToLLVMOptions options(&getContext());
  LLVMTypeConverter converter(&getContext(), options);

  if (enableOneToNConversion) {

    converter.addConversion(
        [&](VectorType type,
            SmallVectorImpl<Type> &result) -> std::optional<LogicalResult> {
          auto elementType = converter.convertType(type.getElementType());
          if (!elementType)
            return failure();
          if (type.getShape().empty()) {
            result.push_back(VectorType::get({1}, elementType));
            return success();
          }
          Type vectorType = VectorType::get(type.getShape().back(), elementType,
                                            type.getScalableDims().back());
          assert(LLVM::isCompatibleVectorType(vectorType) &&
                 "expected vector type compatible with the LLVM dialect");
          // For n-D vector types for which a _non-trailing_ dim is scalable,
          // return a failure. Supporting such cases would require LLVM
          // to support something akin "scalable arrays" of vectors.
          if (llvm::is_contained(type.getScalableDims().drop_back(), true))
            return failure();

          ArrayRef<int64_t> shapeLeadingDims = type.getShape().drop_back();
          int64_t numVectors = ShapedType::getNumElements(shapeLeadingDims);
          for (int64_t i = 0; i < numVectors; i++)
            result.push_back(vectorType);

          return success();
        });

    converter.addTargetMaterialization(
        [&](OpBuilder &builder, TypeRange resultTypes, ValueRange inputs,
            Location loc) -> SmallVector<Value> {
          // from ('vector<4x4xf32>')
          // to ('vector<4xf32>', 'vector<4xf32>', 'vector<4xf32>',
          // 'vector<4xf32>')
          Type ty = resultTypes[0];
          for (Type ithTy : resultTypes)
            if (ithTy != ty)
              return {};

          if (!isa<VectorType>(ty))
            return {};

          if (inputs.size() != 1)
            return {};

          Type inputTy = inputs[0].getType();
          if (!isa<VectorType>(inputTy))
            return {};

          VectorType inputVectorTy = cast<VectorType>(inputTy);
          ArrayRef<int64_t> inputShape = inputVectorTy.getShape();
          size_t numElements =
              ShapedType::getNumElements(inputShape.drop_back());
          if (numElements != resultTypes.size())
            return {};

          return UnrealizedConversionCastOp::create(builder, loc, resultTypes,
                                                    inputs)
              .getResults();
        });
  }

  RewritePatternSet patterns(&getContext());
  populateVectorTransferLoweringPatterns(patterns);
  populateVectorToLLVMConversionPatterns(
      converter, patterns, reassociateFPReductions, force32BitVectorIndices,
      useVectorAlignment);

  // Architecture specific augmentations.
  LLVMConversionTarget target(getContext());
  target.addLegalDialect<arith::ArithDialect>();
  target.addLegalDialect<memref::MemRefDialect>();
  target.addLegalOp<UnrealizedConversionCastOp>();

  if (armNeon) {
    // TODO: we may or may not want to include in-dialect lowering to
    // LLVM-compatible operations here. So far, all operations in the dialect
    // can be translated to LLVM IR so there is no conversion necessary.
    target.addLegalDialect<arm_neon::ArmNeonDialect>();
  }
  if (armSVE) {
    configureArmSVELegalizeForExportTarget(target);
    populateArmSVELegalizeForLLVMExportPatterns(converter, patterns);
  }
  if (amx) {
    configureAMXLegalizeForExportTarget(target);
    populateAMXLegalizeForLLVMExportPatterns(converter, patterns);
  }
  if (x86Vector) {
    configureX86VectorLegalizeForExportTarget(target);
    populateX86VectorLegalizeForLLVMExportPatterns(converter, patterns);
  }

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
