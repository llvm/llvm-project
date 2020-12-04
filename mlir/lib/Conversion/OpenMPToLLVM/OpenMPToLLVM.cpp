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
/// A pattern that converts the region arguments in a single-region OpenMP
/// operation to the LLVM dialect. The body of the region is not modified and is
/// expected to either be processed by the conversion infrastructure or already
/// contain ops compatible with LLVM dialect types.
template <typename OpType>
struct RegionOpConversion : public ConvertToLLVMPattern {
  explicit RegionOpConversion(MLIRContext *context,
                              LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(OpType::getOperationName(), context,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto curOp = cast<OpType>(op);
    auto newOp = rewriter.create<OpType>(curOp.getLoc(), TypeRange(), operands,
                                         curOp.getAttrs());
    rewriter.inlineRegionBefore(curOp.region(), newOp.region(),
                                newOp.region().end());
    if (failed(rewriter.convertRegionTypes(&newOp.region(), *typeConverter)))
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

void mlir::populateOpenMPToLLVMConversionPatterns(
    MLIRContext *context, LLVMTypeConverter &converter,
    OwningRewritePatternList &patterns) {
  patterns.insert<RegionOpConversion<omp::ParallelOp>,
                  RegionOpConversion<omp::WsLoopOp>>(context, converter);
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
  target.addDynamicallyLegalOp<omp::ParallelOp, omp::WsLoopOp>(
      [&](Operation *op) { return converter.isLegal(&op->getRegion(0)); });
  target.addLegalOp<omp::TerminatorOp, omp::TaskyieldOp, omp::FlushOp,
                    omp::BarrierOp, omp::TaskwaitOp>();
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createConvertOpenMPToLLVMPass() {
  return std::make_unique<ConvertOpenMPToLLVMPass>();
}
