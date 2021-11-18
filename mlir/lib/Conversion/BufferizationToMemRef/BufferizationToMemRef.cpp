//===- BufferizationToMemRef.cpp - Bufferization to MemRef conversion -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert Bufferization dialect to MemRef
// dialect.
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
/// The CloneOpConversion transforms all bufferization clone operations into
/// memref alloc and memref copy operations. In the dynamic-shape case, it also
/// emits additional dim and constant operations to determine the shape. This
/// conversion does not resolve memory leaks if it is used alone.
struct CloneOpConversion : public OpConversionPattern<bufferization::CloneOp> {
  using OpConversionPattern<bufferization::CloneOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(bufferization::CloneOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Check for unranked memref types which are currently not supported.
    Type type = op.getType();
    if (type.isa<UnrankedMemRefType>()) {
      return rewriter.notifyMatchFailure(
          op, "UnrankedMemRefType is not supported.");
    }

    // Transform a clone operation into alloc + copy operation and pay
    // attention to the shape dimensions.
    MemRefType memrefType = type.cast<MemRefType>();
    Location loc = op->getLoc();
    SmallVector<Value, 4> dynamicOperands;
    for (int i = 0; i < memrefType.getRank(); ++i) {
      if (!memrefType.isDynamicDim(i))
        continue;
      Value size = rewriter.createOrFold<arith::ConstantIndexOp>(loc, i);
      Value dim = rewriter.createOrFold<memref::DimOp>(loc, op.input(), size);
      dynamicOperands.push_back(dim);
    }
    Value alloc = rewriter.replaceOpWithNewOp<memref::AllocOp>(op, memrefType,
                                                               dynamicOperands);
    rewriter.create<memref::CopyOp>(loc, op.input(), alloc);
    return success();
  }
};
} // namespace

void mlir::populateBufferizationToMemRefConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<CloneOpConversion>(patterns.getContext());
}

namespace {
struct BufferizationToMemRefPass
    : public ConvertBufferizationToMemRefBase<BufferizationToMemRefPass> {
  BufferizationToMemRefPass() = default;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateBufferizationToMemRefConversionPatterns(patterns);

    ConversionTarget target(getContext());
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalOp<arith::ConstantOp>();
    target.addIllegalDialect<bufferization::BufferizationDialect>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createBufferizationToMemRefPass() {
  return std::make_unique<BufferizationToMemRefPass>();
}
