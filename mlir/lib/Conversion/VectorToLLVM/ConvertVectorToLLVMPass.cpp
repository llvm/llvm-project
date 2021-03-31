//===- VectorToLLVM.cpp - Conversion from Vector to the LLVM dialect ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"

#include "../PassDetail.h"

#include "mlir/Conversion/ArmSVEToLLVM/ArmSVEToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/AMX/AMXDialect.h"
#include "mlir/Dialect/AMX/Transforms.h"
#include "mlir/Dialect/AVX512/AVX512Dialect.h"
#include "mlir/Dialect/AVX512/Transforms.h"
#include "mlir/Dialect/ArmNeon/ArmNeonDialect.h"
#include "mlir/Dialect/ArmSVE/ArmSVEDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMArmSVEDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::vector;

namespace {
struct LowerVectorToLLVMPass
    : public ConvertVectorToLLVMBase<LowerVectorToLLVMPass> {
  LowerVectorToLLVMPass(const LowerVectorToLLVMOptions &options) {
    this->reassociateFPReductions = options.reassociateFPReductions;
    this->enableIndexOptimizations = options.enableIndexOptimizations;
    this->enableArmNeon = options.enableArmNeon;
    this->enableArmSVE = options.enableArmSVE;
    this->enableAMX = options.enableAMX;
    this->enableAVX512 = options.enableAVX512;
  }
  // Override explicitly to allow conditional dialect dependence.
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<memref::MemRefDialect>();
    if (enableArmNeon)
      registry.insert<arm_neon::ArmNeonDialect>();
    if (enableArmSVE)
      registry.insert<LLVM::LLVMArmSVEDialect>();
    if (enableAMX)
      registry.insert<amx::AMXDialect>();
    if (enableAVX512)
      registry.insert<avx512::AVX512Dialect>();
  }
  void runOnOperation() override;
};
} // namespace

void LowerVectorToLLVMPass::runOnOperation() {
  // Perform progressive lowering of operations on slices and
  // all contraction operations. Also applies folding and DCE.
  {
    RewritePatternSet patterns(&getContext());
    populateVectorToVectorCanonicalizationPatterns(patterns);
    populateVectorSlicesLoweringPatterns(patterns);
    populateVectorContractLoweringPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }

  // Convert to the LLVM IR dialect.
  LLVMTypeConverter converter(&getContext());
  RewritePatternSet patterns(&getContext());
  populateVectorToLLVMMatrixConversionPatterns(converter, patterns);
  populateVectorToLLVMConversionPatterns(
      converter, patterns, reassociateFPReductions, enableIndexOptimizations);
  populateVectorToLLVMMatrixConversionPatterns(converter, patterns);

  // Architecture specific augmentations.
  LLVMConversionTarget target(getContext());
  target.addLegalOp<LLVM::DialectCastOp>();
  target.addLegalDialect<memref::MemRefDialect>();
  target.addLegalDialect<StandardOpsDialect>();
  target.addLegalOp<UnrealizedConversionCastOp>();
  if (enableArmNeon) {
    // TODO: we may or may not want to include in-dialect lowering to
    // LLVM-compatible operations here. So far, all operations in the dialect
    // can be translated to LLVM IR so there is no conversion necessary.
    target.addLegalDialect<arm_neon::ArmNeonDialect>();
  }
  if (enableArmSVE) {
    target.addLegalDialect<LLVM::LLVMArmSVEDialect>();
    target.addIllegalDialect<arm_sve::ArmSVEDialect>();
    auto hasScalableVectorType = [](TypeRange types) {
      for (Type type : types)
        if (type.isa<arm_sve::ScalableVectorType>())
          return true;
      return false;
    };
    // Remove any ArmSVE-specific types from function signatures and results.
    populateFuncOpTypeConversionPattern(patterns, converter);
    target.addDynamicallyLegalOp<FuncOp>([hasScalableVectorType](FuncOp op) {
      return !hasScalableVectorType(op.getType().getInputs()) &&
             !hasScalableVectorType(op.getType().getResults());
    });
    target.addDynamicallyLegalOp<CallOp, CallIndirectOp, ReturnOp>(
        [hasScalableVectorType](Operation *op) {
          return !hasScalableVectorType(op->getOperandTypes()) &&
                 !hasScalableVectorType(op->getResultTypes());
        });
    populateArmSVEToLLVMConversionPatterns(converter, patterns);
  }
  if (enableAMX) {
    configureAMXLegalizeForExportTarget(target);
    populateAMXLegalizeForLLVMExportPatterns(converter, patterns);
  }
  if (enableAVX512) {
    configureAVX512LegalizeForExportTarget(target);
    populateAVX512LegalizeForLLVMExportPatterns(converter, patterns);
  }

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertVectorToLLVMPass(const LowerVectorToLLVMOptions &options) {
  return std::make_unique<LowerVectorToLLVMPass>(options);
}
