//===- StdExpandDivs.cpp - Code to prepare Std for lowering Divs to LLVM  -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file Std transformations to expand Divs operation to help for the
// lowering to LLVM. Currently implemented transformations are Ceil and Floor
// for Signed Integers.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

/// Converts `atomic_rmw` that cannot be lowered to a simple atomic op with
/// AtomicRMWOpLowering pattern, e.g. with "minf" or "maxf" attributes, to
/// `generic_atomic_rmw` with the expanded code.
///
/// %x = atomic_rmw "maxf" %fval, %F[%i] : (f32, memref<10xf32>) -> f32
///
/// will be lowered to
///
/// %x = std.generic_atomic_rmw %F[%i] : memref<10xf32> {
/// ^bb0(%current: f32):
///   %cmp = arith.cmpf "ogt", %current, %fval : f32
///   %new_value = select %cmp, %current, %fval : f32
///   atomic_yield %new_value : f32
/// }
struct AtomicRMWOpConverter : public OpRewritePattern<AtomicRMWOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AtomicRMWOp op,
                                PatternRewriter &rewriter) const final {
    arith::CmpFPredicate predicate;
    switch (op.getKind()) {
    case AtomicRMWKind::maxf:
      predicate = arith::CmpFPredicate::OGT;
      break;
    case AtomicRMWKind::minf:
      predicate = arith::CmpFPredicate::OLT;
      break;
    default:
      return failure();
    }

    auto loc = op.getLoc();
    auto genericOp = rewriter.create<GenericAtomicRMWOp>(loc, op.getMemref(),
                                                         op.getIndices());
    OpBuilder bodyBuilder =
        OpBuilder::atBlockEnd(genericOp.getBody(), rewriter.getListener());

    Value lhs = genericOp.getCurrentValue();
    Value rhs = op.getValue();
    Value cmp = bodyBuilder.create<arith::CmpFOp>(loc, predicate, lhs, rhs);
    Value select = bodyBuilder.create<SelectOp>(loc, cmp, lhs, rhs);
    bodyBuilder.create<AtomicYieldOp>(loc, select);

    rewriter.replaceOp(op, genericOp.getResult());
    return success();
  }
};

/// Converts `memref.reshape` that has a target shape of a statically-known
/// size to `memref.reinterpret_cast`.
struct MemRefReshapeOpConverter : public OpRewritePattern<memref::ReshapeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::ReshapeOp op,
                                PatternRewriter &rewriter) const final {
    auto shapeType = op.shape().getType().cast<MemRefType>();
    if (!shapeType.hasStaticShape())
      return failure();

    int64_t rank = shapeType.cast<MemRefType>().getDimSize(0);
    SmallVector<OpFoldResult, 4> sizes, strides;
    sizes.resize(rank);
    strides.resize(rank);

    Location loc = op.getLoc();
    Value stride = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    for (int i = rank - 1; i >= 0; --i) {
      Value size;
      // Load dynamic sizes from the shape input, use constants for static dims.
      if (op.getType().isDynamicDim(i)) {
        Value index = rewriter.create<arith::ConstantIndexOp>(loc, i);
        size = rewriter.create<memref::LoadOp>(loc, op.shape(), index);
        if (!size.getType().isa<IndexType>())
          size = rewriter.create<arith::IndexCastOp>(loc, size,
                                                     rewriter.getIndexType());
        sizes[i] = size;
      } else {
        sizes[i] = rewriter.getIndexAttr(op.getType().getDimSize(i));
        size =
            rewriter.create<arith::ConstantOp>(loc, sizes[i].get<Attribute>());
      }
      strides[i] = stride;
      if (i > 0)
        stride = rewriter.create<arith::MulIOp>(loc, stride, size);
    }
    rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(
        op, op.getType(), op.source(), /*offset=*/rewriter.getIndexAttr(0),
        sizes, strides);
    return success();
  }
};

struct StdExpandOpsPass : public StdExpandOpsBase<StdExpandOpsPass> {
  void runOnFunction() override {
    MLIRContext &ctx = getContext();

    RewritePatternSet patterns(&ctx);
    populateStdExpandOpsPatterns(patterns);
    ConversionTarget target(getContext());

    target.addLegalDialect<arith::ArithmeticDialect, memref::MemRefDialect,
                           StandardOpsDialect>();
    target.addDynamicallyLegalOp<AtomicRMWOp>([](AtomicRMWOp op) {
      return op.getKind() != AtomicRMWKind::maxf &&
             op.getKind() != AtomicRMWKind::minf;
    });
    target.addDynamicallyLegalOp<memref::ReshapeOp>([](memref::ReshapeOp op) {
      return !op.shape().getType().cast<MemRefType>().hasStaticShape();
    });
    if (failed(
            applyPartialConversion(getFunction(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::populateStdExpandOpsPatterns(RewritePatternSet &patterns) {
  patterns.add<AtomicRMWOpConverter, MemRefReshapeOpConverter>(
      patterns.getContext());
}

std::unique_ptr<Pass> mlir::createStdExpandOpsPass() {
  return std::make_unique<StdExpandOpsPass>();
}
