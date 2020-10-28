//===- OpenMPToLLVM.cpp - conversion from OpenMP to LLVM dialect ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"

#include "../PassDetail.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"

using namespace mlir;

namespace {
struct ParallelOpConversion : public ConvertToLLVMPattern {
  explicit ParallelOpConversion(MLIRContext *context,
                                LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(omp::ParallelOp::getOperationName(), context,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto curOp = cast<omp::ParallelOp>(op);
    auto newOp = rewriter.create<omp::ParallelOp>(
        curOp.getLoc(), ArrayRef<Type>(), operands, curOp.getAttrs());
    rewriter.inlineRegionBefore(curOp.region(), newOp.region(),
                                newOp.region().end());
    if (failed(rewriter.convertRegionTypes(&newOp.region(), typeConverter)))
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

void mlir::populateOpenMPToLLVMConversionPatterns(
    MLIRContext *context, LLVMTypeConverter &converter,
    OwningRewritePatternList &patterns) {
  patterns.insert<ParallelOpConversion>(context, converter);
}

namespace {
struct ConvertOpenMPToLLVMPass
    : public ConvertOpenMPToLLVMBase<ConvertOpenMPToLLVMPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertOpenMPToLLVMPass::runOnOperation() {
  auto module = getOperation();
  MLIRContext *context = &getContext();

  // Convert to OpenMP operations with LLVM IR dialect
  OwningRewritePatternList patterns;
  LLVMTypeConverter converter(&getContext());
  populateStdToLLVMConversionPatterns(converter, patterns);
  populateOpenMPToLLVMConversionPatterns(context, converter, patterns);

  LLVMConversionTarget target(getContext());
  target.addDynamicallyLegalOp<omp::ParallelOp>(
      [&](omp::ParallelOp op) { return converter.isLegal(&op.getRegion()); });
  target.addLegalOp<omp::TerminatorOp, omp::TaskyieldOp, omp::FlushOp,
                    omp::BarrierOp, omp::TaskwaitOp>();
  if (failed(applyPartialConversion(module, target, patterns)))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createConvertOpenMPToLLVMPass() {
  return std::make_unique<ConvertOpenMPToLLVMPass>();
}
