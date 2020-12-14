//===- VectorToLLVM.cpp - Conversion from Vector to the LLVM dialect ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"

#include "../PassDetail.h"

#include "mlir/Conversion/AVX512ToLLVM/ConvertAVX512ToLLVM.h"
#include "mlir/Conversion/ArmNeonToLLVM/ArmNeonToLLVM.h"
#include "mlir/Conversion/ArmSVEToLLVM/ArmSVEToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/AVX512/AVX512Dialect.h"
#include "mlir/Dialect/ArmNeon/ArmNeonDialect.h"
#include "mlir/Dialect/ArmSVE/ArmSVEDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMAVX512Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMArmNeonDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMArmSVEDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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
    this->enableAVX512 = options.enableAVX512;
  }
  // Override explicitly to allow conditional dialect dependence.
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
    if (enableArmNeon)
      registry.insert<LLVM::LLVMArmNeonDialect>();
    if (enableArmSVE)
      registry.insert<LLVM::LLVMArmSVEDialect>();
    if (enableAVX512)
      registry.insert<LLVM::LLVMAVX512Dialect>();
  }
  void runOnOperation() override;
};
} // namespace

void LowerVectorToLLVMPass::runOnOperation() {
  // Perform progressive lowering of operations on slices and
  // all contraction operations. Also applies folding and DCE.
  {
    OwningRewritePatternList patterns;
    populateVectorToVectorCanonicalizationPatterns(patterns, &getContext());
    populateVectorSlicesLoweringPatterns(patterns, &getContext());
    populateVectorContractLoweringPatterns(patterns, &getContext());
    applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }

  // Convert to the LLVM IR dialect.
  LLVMTypeConverter converter(&getContext());
  OwningRewritePatternList patterns;
  populateVectorToLLVMMatrixConversionPatterns(converter, patterns);
  populateVectorToLLVMConversionPatterns(
      converter, patterns, reassociateFPReductions, enableIndexOptimizations);
  populateVectorToLLVMMatrixConversionPatterns(converter, patterns);
  populateStdToLLVMConversionPatterns(converter, patterns);

  // Architecture specific augmentations.
  LLVMConversionTarget target(getContext());
  if (enableArmNeon) {
    target.addLegalDialect<LLVM::LLVMArmNeonDialect>();
    target.addIllegalDialect<arm_neon::ArmNeonDialect>();
    populateArmNeonToLLVMConversionPatterns(converter, patterns);
  }
  if (enableArmSVE) {
    target.addLegalDialect<LLVM::LLVMArmSVEDialect>();
    target.addIllegalDialect<arm_sve::ArmSVEDialect>();
    populateArmSVEToLLVMConversionPatterns(converter, patterns);
  }
  if (enableAVX512) {
    target.addLegalDialect<LLVM::LLVMAVX512Dialect>();
    target.addIllegalDialect<avx512::AVX512Dialect>();
    populateAVX512ToLLVMConversionPatterns(converter, patterns);
  }

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertVectorToLLVMPass(const LowerVectorToLLVMOptions &options) {
  return std::make_unique<LowerVectorToLLVMPass>(options);
}
